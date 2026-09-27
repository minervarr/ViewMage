#include "viewmage_app.hh"

#include "png_export.hh"
#include "text_util.hh"    // truncateToWidth: status text must never overlap buttons
#include "text_font.hh"    // FontStyle::Roman (full definition, not the forward decl)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>

#include "log.hh"
#ifdef __ANDROID__
#include "activity_bridge.hh"
#endif

namespace {

// The viewer's whole palette. A photo is the subject; anything else on screen
// competes with it, so there is nothing else on screen.
constexpr Color kBackground{0.06f, 0.06f, 0.07f, 1.0f};
constexpr Color kMessage   {0.85f, 0.85f, 0.88f, 1.0f};
constexpr Color kBar       {0.04f, 0.04f, 0.05f, 0.82f};
constexpr Color kButton    {0.16f, 0.17f, 0.20f, 1.0f};
constexpr Color kButtonOff {0.10f, 0.10f, 0.12f, 1.0f};
constexpr Color kMessageOff{0.42f, 0.42f, 0.45f, 1.0f};

constexpr float  kTapSlopPx      = 24.0f;
constexpr double kTapMaxSeconds  = 0.35;
constexpr double kDoubleTapGap   = 0.35;
constexpr float  kDoubleTapSlop  = 60.0f;

// IEEE binary32 -> binary16. Only used by the memory fallback, so it is
// written for clarity rather than speed: if we are here at all, an allocation
// has already failed and one pass over the image is not the problem.
uint16_t to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;                       // underflow -> +/-0
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);          // overflow  -> +/-inf
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

float srgb_encode(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x <= 0.0031308f ? x * 12.92f
                           : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

float distance(float ax, float ay, float bx, float by) {
    const float dx = ax - bx, dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

ViewMageApp::ViewMageApp(std::unique_ptr<Host> host) : host_(std::move(host)) {}

ViewMageApp::~ViewMageApp() {
    joinScan();
    joinEnhancement();
    joinRefinement();
    releaseTexture();
    renderer_.reset();
}

double ViewMageApp::nowSeconds() const {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

bool ViewMageApp::create() {
    if (!host_ || !host_->init(this)) return false;
    // Hdr10PQ. Both it and ExtendedLinearScrgb are real HDR targets and
    // vk_canvas falls back to SDR if neither is available, so this is a request
    // and not an assumption -- see hdrActive() below.
    //
    // PQ is chosen because it is the target Android actually grants on a phone:
    // the 10-bit ST 2084 pair is what surfaces enumerate, and its absolute
    // luminance means "1000 nits" in the file arrives as 1000 nits on the panel
    // instead of as a number scaled by whatever the OS thinks SDR white is.
    //
    // ACCEPTED CAVEAT, the one scRGB was picked for: under PQ the fixed-function
    // blend mixes PQ code values by a coverage weight, and PQ is steep enough
    // that mixing code values is not mixing the luminances they stand for.
    // Antialiased edges and the letterbox seam are therefore slightly wrong.
    // That is a sub-pixel error on a viewer whose entire screen is one opaque
    // photograph over a flat background -- there is almost nothing to blend --
    // and it is worth paying for correct absolute brightness on the photo
    // itself. See vk_canvas USAGE_hdr_output.md.
    //
    // PQ is BT.2020 primaries, and the decode has to be told so: see
    // loadFromSource(), which passes activeTarget() down to decode_jxl(). Those
    // two must agree or every colour is stretched onto the wrong gamut.
    renderer_ = std::make_unique<Renderer>(host_->surfaceProvider(), host_->assetReader(),
                                           /*images=*/4, OutputTarget::Hdr10PQ);

    // ASK, never assume. The request above can be refused by the driver, by the
    // compositor, or by the window never having been put into HDR colour mode,
    // and none of those say so. hdrActive() is the only honest answer, and the
    // headroom is meaningless without it.
    hdr_ = renderer_->hdrActive();
    refreshHeadroom();
    VCE_LOGI("ViewMage", "output target: %s (hdr=%s) headroom=%.2fx",
             outputTargetName(renderer_->activeTarget()), hdr_ ? "yes" : "no", headroom_);

    syncViewport();

    // Load a font from APK assets for text rendering. The stroke-fallback
    // glyphs work but look mechanical; a real face is a few hundred KB and
    // one read.
    {
        std::vector<uint8_t> fontBuf;
        if (host_->dataReader().read("fonts/newcomputermodern/NewCMSans10-Regular.otf",
                                     fontBuf) &&
            !fontBuf.empty()) {
            if (font_.loadFromMemory(fontBuf.data(), fontBuf.size()))
                VCE_LOGI("ViewMage", "font loaded from assets");
            else
                VCE_LOGI("ViewMage", "font load failed; using stroke fallback");
        } else {
            VCE_LOGI("ViewMage", "no font in assets; using stroke fallback");
        }
    }

    return true;
}

void ViewMageApp::run() {
    while (running_) {
        const bool refining = refine_thread_.joinable() && !refined_ready_.load();
        const bool enhancing = enhance_thread_.joinable() && !enhanced_ready_.load();
        const bool scanning = scan_thread_.joinable() && !scan_ready_.load();
        const bool busy = refining || enhancing || scanning;
        // Also force a draw when a result is ready to pick up — without this,
        // refined_ready_==true makes refining==false, busy==false, pump blocks,
        // and pollRefinement() is never reached.
        const bool hasResult = refined_ready_.load() || enhanced_ready_.load() ||
                               scan_ready_.load();
        if (busy || hasResult) dirty_ = true;
        // Hold the screen on for the whole job. A denoise is long enough that
        // the panel would otherwise sleep, and Android then freezes or kills
        // the process with the photo only in memory.
        const bool awake = busy || hasResult;
        if (awake != keepAwake_) {
            keepAwake_ = awake;
            host_->setKeepAwake(awake);
        }
        host_->pump(/*haveWork=*/dirty_ || busy || hasResult);
        // Don't exit while a worker is in flight.
        if (host_->quitRequested() && !busy) break;
        if (!running_) break;

        pollRefinement();
        pollEnhancement();
        pollScan();

        if (dirty_ && surfaceOk_ && renderer_) {
            draw();
            dirty_ = false;
        }
    }
}

void ViewMageApp::shutdown() {
    running_ = false;
    joinScan();
    joinEnhancement();
    joinRefinement();
    if (keepAwake_) {
        keepAwake_ = false;
        if (host_) host_->setKeepAwake(false);
    }
}

// ── Progressive refinement ──────────────────────────────────────────────────

void ViewMageApp::joinRefinement() {
    refine_abort_.store(true);
    if (refine_thread_.joinable()) refine_thread_.join();
}

void ViewMageApp::startRefinement() {
    if (model_bytes_.empty() || refine_thread_.joinable()) return;
    if (!looks_like_dng(source_.data(), source_.size())) return;

    const uint32_t maxDim = renderer_ ? renderer_->caps().max_image_dim_2d : 0;
    const WorkingPrimaries prim =
        (renderer_ && renderer_->activeTarget() == OutputTarget::Hdr10PQ)
            ? WorkingPrimaries::Bt2100 : WorkingPrimaries::Srgb;

    refined_ready_.store(false);
    refine_abort_.store(false);
    VCE_LOGI("ViewMage", "neural denoise: started in the background");

    refine_thread_ = std::thread([this, maxDim, prim] {
        // Every exit stores an image and sets refined_ready_. A bare return
        // leaves the thread joinable with the flag still false, and the UI
        // treats that as "still denoising" with every button disabled.
        auto publish = [this](DecodedImage img) {
            {
                std::lock_guard<std::mutex> lock(refined_mutex_);
                refined_ = std::move(img);
            }
            refined_ready_.store(true);
        };
        auto fail = [](const std::string& why) {
            DecodedImage img;
            img.error = why;
            return img;
        };
        try {
            // Building the session is the expensive half -- on an S23 Ultra it is
            // minutes, not seconds, which is exactly why it lives on this thread.
            if (!denoiser_) {
                const double t0 = nowSeconds();
                std::string err;
                denoiser_ = RawDenoiser::load(model_bytes_.data(), model_bytes_.size(), err);
                if (!denoiser_) {
                    VCE_LOGE("ViewMage", "denoiser unavailable: %s", err.c_str());
                    publish(fail(err.empty() ? "denoiser unavailable" : err));
                    return;
                }
                VCE_LOGI("ViewMage", "denoiser ready (%s) in %.1f s",
                         denoiser_->provider().c_str(), nowSeconds() - t0);
            }
            if (refine_abort_.load()) {
                publish(fail("cancelled"));
                return;
            }
            const double t1 = nowSeconds();

            // decode_dng again rather than trying to re-enter the develop halfway:
            // the develop is a few hundred milliseconds against the denoise's tens
            // of seconds, and a second full decode is far easier to reason about
            // than a pipeline that can be resumed from the middle.
            DecodedImage out = decode_dng(source_.data(), source_.size(), maxDim,
                                          prim, nullptr, denoiser_.get(),
                                          [this](float f) {
                                              refine_progress_.store(f);
                                              return !refine_abort_.load();
                                          });
            if (refine_abort_.load() || out.error == "cancelled") {
                publish(fail("cancelled"));
                return;
            }
            if (!out.ok()) {
                VCE_LOGE("ViewMage", "neural denoise failed: %s", out.error.c_str());
                publish(std::move(out));
                return;
            }
            // decode_dng falls back to Malvar and still returns a picture. The
            // screen already shows that develop; swapping in a second copy
            // only risks the allocation that just failed.
            bool fellBack = false;
            for (const Diagnostic& n : out.notes) {
                if (n.title == "Shown without neural denoise") fellBack = true;
            }
            if (fellBack) {
                VCE_LOGE("ViewMage", "neural denoise fell back; keeping the develop");
                publish(fail("denoise failed"));
                return;
            }
            VCE_LOGI("ViewMage", "neural denoise: %.1f s for %ux%u",
                     nowSeconds() - t1, out.w, out.h);
            publish(std::move(out));
        } catch (const std::bad_alloc&) {
            VCE_LOGE("ViewMage", "neural denoise: out of memory");
            publish(fail("not enough memory to denoise"));
        } catch (const std::exception& e) {
            VCE_LOGE("ViewMage", "neural denoise: %s", e.what());
            publish(fail(e.what()));
        } catch (...) {
            VCE_LOGE("ViewMage", "neural denoise: unknown failure");
            publish(fail("denoise failed"));
        }
    });
}


// A cheap local-roughness probe: mean |2c - left - right| on the green channel.
// It is the ONE number that tells a denoised buffer from a raw-demosaic one at
// a glance -- measured on this scene, Malvar reads ~0.05 and the neural output
// ~0.005, a full order of magnitude apart. Sampled, not exhaustive, because it
// runs on the UI thread.
static float roughness_probe(const std::vector<float>& linear, uint32_t w, uint32_t h) {
    if (linear.size() < size_t(w) * h * 4 || w < 4 || h < 4) return -1.0f;
    double acc = 0.0;
    size_t n = 0;
    for (uint32_t y = 1; y + 1 < h; y += 7) {
        for (uint32_t x = 1; x + 1 < w; x += 7) {
            const size_t i = (size_t(y) * w + x) * 4 + 1;
            acc += std::fabs(2.0 * linear[i] - linear[i - 4] - linear[i + 4]);
            ++n;
        }
    }
    return n ? float(acc / double(n)) : -1.0f;
}

std::string ViewMageApp::statusLine() const {
    if (!export_note_.empty()) return export_note_;
    if (exporting_)            return "Saving...";
    if (refine_thread_.joinable() && !refined_ready_.load()) {
        if (refine_abort_.load()) return "Cancelling...";
        const int pct = (int)(refine_progress_.load() * 100.0f + 0.5f);
        return "Denoising " + std::to_string(pct) + "%";
    }
    if (enhance_thread_.joinable() && !enhanced_ready_.load())
        return "Enhancing...";
    if (scan_thread_.joinable() && !scan_ready_.load())
        return scan_flattening_ ? "Straightening..." : "Finding the page...";
    if (scan_adjust_) {
        if (placing_) return "Release to add a point";
        if (drag_corner_ >= 0 &&
            drag_corner_ < (int)adjust_outline_.points.size()) {
            if (drag_handle_) {
                const PagePointKind k = adjust_outline_.points[drag_corner_].kind;
                if (k == PagePointKind::Fold) return "Release to straighten";
                return "Drag onto the point to straighten";
            }
            if (pointRemoveArmed()) return "Release to remove";
            const PagePointKind k = adjust_outline_.points[drag_corner_].kind;
            if (k == PagePointKind::Corner) return "Corner";
            return "Drag away to remove";
        }
        return scan_page_ ? "Four corners are the page" : "Tap a line to add";
    }
    if (enhanced_)             return "Enhanced";
    if (scanned_)              return "Page flattened";
    if (denoised_)             return "Denoised";
    return "Developed from RAW";
}

void ViewMageApp::doExport(bool jpeg) {
    if (exporting_) return;
    if (refine_thread_.joinable() && !refined_ready_.load()) {
        export_note_ = "Still denoising";
        dirty_ = true;
        return;
    }
    if (scan_thread_.joinable() && !scan_ready_.load()) {
        export_note_ = "Still straightening";
        dirty_ = true;
        return;
    }
    if (enhance_thread_.joinable() && !enhanced_ready_.load()) {
        export_note_ = "Still enhancing";
        dirty_ = true;
        return;
    }
    VCE_LOGI("ViewMage", "export requested (%ux%u, %zu floats, roughness %.5f, %s)",
             pixels_.w, pixels_.h, pixels_.linear.size(),
             roughness_probe(pixels_.linear, pixels_.w, pixels_.h),
             jpeg ? "jpeg" : "png");

    // The pixels are needed to write anything. On the DNG path they were kept
    // for exactly this; anywhere else, say so plainly rather than exporting a
    // blank file.
    if (pixels_.linear.empty()) {
        VCE_LOGE("ViewMage", "export: no pixels held");
        export_note_ = "Nothing to save";
        dirty_ = true;
        return;
    }

    exporting_ = true;
    dirty_ = true;
    draw();          // paint "Saving..." BEFORE the write blocks the thread

    // The source's own filename never reaches us -- the app is handed BYTES
    // through a ContentResolver, not a path -- so the export is timestamped.
    const std::string name = export_name_for(std::string(),
                                             jpeg ? ".jpg" : ".png");
    std::string where, err;
    const bool ok = jpeg
        ? export_jpeg(pixels_.linear, pixels_.w, pixels_.h, ev_, white_,
                      pixels_.widePrimaries, name, where, err)
        : export_png(pixels_.linear, pixels_.w, pixels_.h, ev_, white_,
                     pixels_.widePrimaries, name, where, err);
    if (ok) {
        // Name the place it ACTUALLY went: saying "Pictures" when it fell back
        // elsewhere is how a photo gets lost.
        export_note_ = std::string("Saved ") + (jpeg ? "JPG" : "PNG")
                     + " to " + where;
    } else {
        export_note_ = "Could not save: " + err;
        VCE_LOGE("ViewMage", "export failed: %s", err.c_str());
    }
    exporting_ = false;
    dirty_ = true;
}

void ViewMageApp::pollRefinement() {
    if (!refined_ready_.load()) return;
    refined_ready_.store(false);
    if (refine_thread_.joinable()) refine_thread_.join();

    DecodedImage got;
    {
        std::lock_guard<std::mutex> lock(refined_mutex_);
        got = std::move(refined_);
    }
    if (!got.ok()) {
        // The develop already on screen is the recovered state. Replacing it,
        // or clearing the texture, is how a failed denoise becomes a black
        // window the user cannot leave. If the surface was recreated mid-job
        // the texture handle is gone and has to be put back from pixels_.
        VCE_LOGE("ViewMage", "neural denoise failed: %s", got.error.c_str());
        if (surfaceOk_ && renderer_ && texture_ == kInvalidTexture &&
            !pixels_.linear.empty() && !uploadTexture()) {
            VCE_LOGE("ViewMage", "could not restore the develop after denoise");
        }
        export_note_ = (got.error == "cancelled")
            ? "Denoise cancelled"
            : "Denoise failed — photo unchanged";
        dirty_ = true;
        return;
    }

    // Surface is gone (app backgrounded): keep the denoised pixels so
    // onSurfaceRecreated() can upload them without re-denoising.
    if (!surfaceOk_ || !renderer_) {
        VCE_LOGI("ViewMage", "neural denoise done while surface is lost; "
                 "storing for later upload");
        pendingDenoised_ = std::move(got);
        return;
    }

    DecodedImage previous = std::move(pixels_);
    const TextureHandle old = texture_;
    texture_ = kInvalidTexture;
    pixels_  = std::move(got);

    if (!uploadTexture()) {
        VCE_LOGE("ViewMage", "denoised upload failed; keeping the first render");
        releaseTexture();
        pixels_ = std::move(previous);
        texture_ = old;
        export_note_ = "Denoise failed — photo unchanged";
        dirty_ = true;
        return;
    }
    if (old != kInvalidTexture) renderer_->destroy_texture(old);
    if (!keep_pixels_) releasePixels();
    export_note_.clear();

    ev_       = pixels_.autoEv;
    white_    = pixels_.white;
    denoised_ = true;
    dirty_ = true;
    VCE_LOGI("ViewMage", "neural denoise: swapped in (autoEV=%.2f white=%.2f, "
             "roughness %.5f)", ev_, white_,
             roughness_probe(pixels_.linear, pixels_.w, pixels_.h));
}

// ── Enhance ─────────────────────────────────────────────────────────────────
//
// Same worker-thread pattern as the denoise: snapshot the current pixels
// under the mutex, process on a thread, swap in the result on the main thread.

void ViewMageApp::joinEnhancement() {
    enhance_abort_.store(true);
    if (enhance_thread_.joinable()) enhance_thread_.join();
}

void ViewMageApp::startEnhance() {
    if (enhance_thread_.joinable() || enhanced_) return;
    if (pixels_.linear.empty() || !pixels_.ok()) return;

    // Snapshot the current pixels so the worker does not race with the main
    // thread.  At 16 bytes/pixel this is the largest allocation in the app,
    // but enhance is a one-shot like denoise, not a per-frame operation.
    DecodedImage snapshot;
    snapshot.w = pixels_.w;
    snapshot.h = pixels_.h;
    snapshot.linear = pixels_.linear;   // copy — the worker must not touch pixels_
    snapshot.autoEv = pixels_.autoEv;
    snapshot.white  = pixels_.white;
    snapshot.hdrTransfer   = pixels_.hdrTransfer;
    snapshot.widePrimaries = pixels_.widePrimaries;

    enhanced_ready_.store(false);
    enhance_abort_.store(false);
    VCE_LOGI("ViewMage", "enhance: started in the background");

    const EnhanceParams params = enhance_params_;
    // The neural denoise already reconstructs edge detail the sensor never
    // had; unsharp-masking its output re-invents halos around every light.
    // Sharpening is for the plain develop only. A flattened page takes a
    // different finish: even paper, dark ink, colour kept.
    EnhanceParams effective = params;
    if (denoised_) effective.sharpen = 0.0f;
    const bool document = scanned_;

    enhance_thread_ = std::thread([this, snap = std::move(snapshot), effective, document]() mutable {
        const double t0 = nowSeconds();
        if (document) {
            enhanceDocument(snap.linear, snap.w, snap.h, snap.widePrimaries);
            snap.autoEv = 0.0f;
            snap.white = 1.0f;
        } else {
            enhanceImage(snap.linear, snap.w, snap.h, effective,
                         snap.widePrimaries);
        }
        if (enhance_abort_.load()) return;
        VCE_LOGI("ViewMage", "enhance: %.1f s for %ux%u",
                 nowSeconds() - t0, snap.w, snap.h);
        {
            std::lock_guard<std::mutex> lock(enhanced_mutex_);
            enhance_result_ = std::move(snap);
        }
        enhanced_ready_.store(true);
    });
}

void ViewMageApp::pollEnhancement() {
    if (!enhanced_ready_.load()) return;
    enhanced_ready_.store(false);
    if (enhance_thread_.joinable()) enhance_thread_.join();

    DecodedImage got;
    {
        std::lock_guard<std::mutex> lock(enhanced_mutex_);
        got = std::move(enhance_result_);
    }
    if (!got.ok()) {
        VCE_LOGE("ViewMage", "enhance failed: %s", got.error.c_str());
        return;
    }

    // Surface is gone: keep for later upload.
    if (!surfaceOk_ || !renderer_) {
        VCE_LOGI("ViewMage", "enhance done while surface is lost; "
                 "storing for later upload");
        pendingEnhanced_ = std::move(got);
        return;
    }

    DecodedImage previous = std::move(pixels_);
    const TextureHandle old = texture_;
    texture_ = kInvalidTexture;
    pixels_  = std::move(got);

    if (!uploadTexture()) {
        VCE_LOGE("ViewMage", "enhanced upload failed; keeping the previous render");
        releaseTexture();
        pixels_ = std::move(previous);
        texture_ = old;
        return;
    }
    if (old != kInvalidTexture) renderer_->destroy_texture(old);
    if (!keep_pixels_) releasePixels();
    export_note_.clear();

    ev_       = pixels_.autoEv;
    white_    = pixels_.white;
    enhanced_ = true;
    dirty_ = true;
    VCE_LOGI("ViewMage", "enhance: swapped in (autoEV=%.2f white=%.2f)",
             ev_, white_);
}

void ViewMageApp::joinScan() {
    scan_abort_.store(true);
    if (scan_thread_.joinable()) scan_thread_.join();
}

void ViewMageApp::startScan(bool rectangle) {
    if (scan_thread_.joinable() || scan_adjust_) return;
    if (pixels_.linear.empty() || !pixels_.ok()) {
        export_note_ = "Nothing to scan";
        dirty_ = true;
        return;
    }

    DecodedImage snapshot;
    snapshot.w = pixels_.w;
    snapshot.h = pixels_.h;
    snapshot.linear = pixels_.linear;

    scan_ready_.store(false);
    scan_abort_.store(false);
    scan_flattening_ = false;
    scan_page_ = rectangle;
    VCE_LOGI("ViewMage", "scan: looking for the page");

    std::vector<uint8_t> pageModel;
    if (rectangle)
        host_->dataReader().read("models/page_quad.onnx", pageModel);

    scan_thread_ = std::thread([this, snap = std::move(snapshot), rectangle,
                                pageModel = std::move(pageModel)]() mutable {
        DocumentDetect found = rectangle
            ? detectRectangle(snap.linear.data(), snap.w, snap.h,
                              pageModel.empty() ? nullptr : pageModel.data(),
                              pageModel.size())
            : detectDocument(snap.linear.data(), snap.w, snap.h);
        if (scan_abort_.load()) return;
        std::lock_guard<std::mutex> lock(scan_mutex_);
        scan_result_ = {};
        if (rectangle && found.found) {
            DocumentQuad q = found.quad;
            int nCorner = 0;
            PagePoint corners[4];
            for (const PagePoint& p : found.outline.points) {
                if (p.kind != PagePointKind::Corner || nCorner >= 4) continue;
                corners[nCorner++] = p;
            }
            if (found.outline.shape != DocumentShape::Quad) {
                if (nCorner != 4) {
                    found.found = false;
                    found.error = "No rectangle found";
                } else {
                    for (int i = 0; i < 4; ++i) {
                        q.x[i] = corners[i].x;
                        q.y[i] = corners[i].y;
                    }
                    found.outline = DocumentOutline{};
                    found.outline.shape = DocumentShape::Quad;
                    for (int i = 0; i < 4; ++i) {
                        PagePoint c;
                        c.kind = PagePointKind::Corner;
                        c.x = q.x[i];
                        c.y = q.y[i];
                        found.outline.points.push_back(c);
                        found.outline.x.push_back(c.x);
                        found.outline.y.push_back(c.y);
                    }
                    found.quad = q;
                }
            }
        }
        if (!found.found) {
            scan_has_quad_ = false;
            scan_error_ = found.error.empty() ? "No document found" : found.error;
            VCE_LOGI("ViewMage", "scan: %s", scan_error_.c_str());
        } else {
            scan_has_quad_ = true;
            scan_quad_ = found.quad;
            scan_outline_ = found.outline;
            scan_error_.clear();
            VCE_LOGI("ViewMage", "scan: corners ready");
        }
        scan_ready_.store(true);
    });
}

void ViewMageApp::confirmScan() {
    if (scan_thread_.joinable() || !scan_adjust_) return;
    if (pixels_.linear.empty() || !pixels_.ok()) return;

    DecodedImage snapshot;
    snapshot.w = pixels_.w;
    snapshot.h = pixels_.h;
    snapshot.linear = pixels_.linear;
    snapshot.autoEv = pixels_.autoEv;
    snapshot.white  = pixels_.white;
    snapshot.hdrTransfer   = pixels_.hdrTransfer;
    snapshot.widePrimaries = pixels_.widePrimaries;
    snapshot.bitsPerSample = pixels_.bitsPerSample;
    DocumentQuad quad;
    DocumentOutline outline = adjust_outline_;
    const bool straight = outline.points.size() == 4 &&
                          outline.shape == DocumentShape::Quad;
    if (straight) {
        for (int i = 0; i < 4; ++i) {
            quad.x[i] = outline.points[i].x;
            quad.y[i] = outline.points[i].y;
        }
    } else if (outline.points.empty()) {
        for (int i = 0; i < 4; ++i) {
            quad.x[i] = corner_x_[i];
            quad.y[i] = corner_y_[i];
        }
    }
    const DocumentShape shape = straight || outline.points.empty()
                                    ? adjust_shape_ : outline.shape;

    drag_corner_ = -1;
    drag_handle_ = false;
    placing_ = false;
    scan_adjust_ = false;
    scan_ready_.store(false);
    scan_abort_.store(false);
    scan_flattening_ = true;
    VCE_LOGI("ViewMage", "scan: straightening");

    scan_thread_ = std::thread([this, snap = std::move(snapshot), quad, shape, outline]() mutable {
        const double t0 = nowSeconds();
        const bool useQuad = shape == DocumentShape::Quad &&
                             (outline.points.empty() || outline.points.size() == 4);
        // Rect always rebuilds the four-corner page. A diamond or a circle
        // on it is a bend, not a new corner. Shape follows the outline.
        DocumentScan scan = scan_page_
            ? reconstructRectangle(snap.linear.data(), snap.w, snap.h, outline)
            : (useQuad
                   ? flattenDocument(snap.linear.data(), snap.w, snap.h, quad)
                   : flattenOutline(snap.linear.data(), snap.w, snap.h, outline));
        if (scan_abort_.load()) return;
        std::lock_guard<std::mutex> lock(scan_mutex_);
        scan_has_quad_ = false;
        if (!scan.found) {
            scan_error_ = scan.error.empty() ? "Could not straighten the page" : scan.error;
            scan_result_ = {};
            VCE_LOGI("ViewMage", "scan: %s", scan_error_.c_str());
        } else {
            snap.linear = std::move(scan.linear);
            snap.w = scan.w;
            snap.h = scan.h;
            // The photo's exposure was for the whole frame. Applied again it
            // blows the paper out, because the paper was the bright part.
            snap.autoEv = 0.0f;
            snap.white = 1.0f;
            scan_error_.clear();
            scan_result_ = std::move(snap);
            VCE_LOGI("ViewMage", "scan: %.2f s, page %ux%u",
                     nowSeconds() - t0, scan_result_.w, scan_result_.h);
        }
        scan_ready_.store(true);
    });
}

void ViewMageApp::pollScan() {
    if (!scan_ready_.load()) return;
    const bool flattening = scan_flattening_;
    scan_ready_.store(false);
    if (scan_thread_.joinable()) scan_thread_.join();

    DocumentQuad quad;
    bool hasQuad = false;
    DecodedImage got;
    std::string err;
    {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        hasQuad = scan_has_quad_;
        quad = scan_quad_;
        adjust_outline_ = scan_outline_;
        adjust_shape_ = scan_outline_.shape;
        got = std::move(scan_result_);
        err = std::move(scan_error_);
        scan_has_quad_ = false;
    }
    if (!flattening) {
        if (!hasQuad) {
            export_note_ = err.empty() ? "No document found" : err;
            dirty_ = true;
            return;
        }
        for (int i = 0; i < 4; ++i) {
            corner_x_[i] = quad.x[i];
            corner_y_[i] = quad.y[i];
        }
        drag_corner_ = -1;
        scan_adjust_ = true;
        export_note_.clear();
        dirty_ = true;
        return;
    }
    if (!got.ok()) {
        scan_adjust_ = true;
        export_note_ = err.empty() ? "Could not straighten the page" : err;
        dirty_ = true;
        return;
    }
    if (!surfaceOk_ || !renderer_) {
        pendingScan_ = std::move(got);
        return;
    }

    DecodedImage previous = std::move(pixels_);
    const TextureHandle old = texture_;
    texture_ = kInvalidTexture;
    pixels_  = std::move(got);

    if (!uploadTexture()) {
        VCE_LOGE("ViewMage", "scan upload failed; keeping the previous picture");
        releaseTexture();
        pixels_ = std::move(previous);
        texture_ = old;
        export_note_ = "Could not show the page";
        dirty_ = true;
        return;
    }
    if (old != kInvalidTexture) renderer_->destroy_texture(old);
    keep_pixels_ = true;
    view_.setImage((int)pixels_.w, (int)pixels_.h);
    syncViewport();
    view_.fit();
    scanned_ = true;
    enhanced_ = false;
    ev_ = pixels_.autoEv;
    white_ = pixels_.white;
    export_note_.clear();
    dirty_ = true;
    VCE_LOGI("ViewMage", "scan: swapped in %ux%u", pixels_.w, pixels_.h);
}

void ViewMageApp::imageToScreen(float ix, float iy, float& sx, float& sy) const {
    const ViewTransform::Quad q = view_.quad();
    if (pixels_.w == 0 || pixels_.h == 0 || q.w <= 0.0f || q.h <= 0.0f) {
        sx = sy = 0.0f;
        return;
    }
    sx = q.x + ix / (float)pixels_.w * q.w;
    sy = q.y + iy / (float)pixels_.h * q.h;
}

void ViewMageApp::screenToImage(float sx, float sy, float& ix, float& iy) const {
    const ViewTransform::Quad q = view_.quad();
    if (pixels_.w == 0 || pixels_.h == 0 || q.w <= 0.0f || q.h <= 0.0f) {
        ix = iy = 0.0f;
        return;
    }
    ix = (sx - q.x) / q.w * (float)pixels_.w;
    iy = (sy - q.y) / q.h * (float)pixels_.h;
}



void ViewMageApp::handleVector(int index, float& hx, float& hy) const {
    hx = hy = 0.0f;
    const auto& pts = adjust_outline_.points;
    if (index < 0 || index >= (int)pts.size()) return;
    const PagePoint& p = pts[index];
    if (p.kind == PagePointKind::Corner) return;
    const float len = std::sqrt(p.hx * p.hx + p.hy * p.hy);
    if (p.kind == PagePointKind::Curve && len > 2.0f) {
        hx = p.hx;
        hy = p.hy;
        return;
    }
    const int n = (int)pts.size();
    const PagePoint& prev = pts[(index + n - 1) % n];
    const PagePoint& next = pts[(index + 1) % n];
    float tx = next.x - prev.x, ty = next.y - prev.y;
    const float tl = std::sqrt(tx * tx + ty * ty);
    if (tl < 1e-3f) return;
    // A source-pixel arm is a few screen pixels once the page is fitted, and
    // the stick disappears into the dot. Keep it a finger's width on screen.
    const float scale = std::max(view_.scale(), 1e-3f);
    // Long enough that the tip sits past the point's own grab, so a press
    // on the diamond moves it instead of bending it.
    const float arm = 96.0f / scale;
    hx = tx / tl * arm;
    hy = ty / tl * arm;
}

void ViewMageApp::syncShapeFromPoints() {
    int folds = 0, curves = 0;
    for (PagePoint& p : adjust_outline_.points) {
        if (p.kind == PagePointKind::Curve) {
            const float len = std::sqrt(p.hx * p.hx + p.hy * p.hy);
            if (len < 1.0f) {
                p.kind = PagePointKind::Fold;
                p.hx = p.hy = 0.0f;
            }
        }
        if (p.kind == PagePointKind::Fold) ++folds;
        else if (p.kind == PagePointKind::Curve) ++curves;
    }
    if (curves) adjust_shape_ = DocumentShape::Curve;
    else if (folds) adjust_shape_ = DocumentShape::Fold;
    else if (adjust_outline_.points.size() == 3) adjust_shape_ = DocumentShape::Triangle;
    else adjust_shape_ = DocumentShape::Quad;
    adjust_outline_.shape = adjust_shape_;
    adjust_outline_.ax.clear();
    adjust_outline_.ay.clear();
    adjust_outline_.bx.clear();
    adjust_outline_.by.clear();
    adjust_outline_.x.clear();
    adjust_outline_.y.clear();
    for (const PagePoint& p : adjust_outline_.points) {
        adjust_outline_.x.push_back(p.x);
        adjust_outline_.y.push_back(p.y);
    }
}

void ViewMageApp::ensureAdjustPoints() {
    if (!adjust_outline_.points.empty()) return;
    for (int i = 0; i < 4; ++i) {
        PagePoint p;
        p.kind = PagePointKind::Corner;
        p.x = corner_x_[i];
        p.y = corner_y_[i];
        adjust_outline_.points.push_back(p);
    }
    syncShapeFromPoints();
}

bool ViewMageApp::segmentScreen(int after, float& x0, float& y0,
                                float& x1, float& y1) const {
    const auto& pts = adjust_outline_.points;
    if (!pts.empty()) {
        if (after < 0 || after >= (int)pts.size()) return false;
        const int j = (after + 1) % (int)pts.size();
        imageToScreen(pts[after].x, pts[after].y, x0, y0);
        imageToScreen(pts[j].x, pts[j].y, x1, y1);
        return true;
    }
    if (after < 0 || after >= 4) return false;
    imageToScreen(corner_x_[after], corner_y_[after], x0, y0);
    imageToScreen(corner_x_[(after + 1) % 4], corner_y_[(after + 1) % 4], x1, y1);
    return true;
}

bool ViewMageApp::barContains(float x, float y) const {
    return next_rect_.contains(x, y) || export_rect_.contains(x, y) ||
           jpg_rect_.contains(x, y) || denoise_rect_.contains(x, y) ||
           enhance_rect_.contains(x, y) || scan_rect_.contains(x, y) ||
           shape_rect_.contains(x, y);
}

void ViewMageApp::updatePlacement(float fx, float fy) {
    float x0, y0, x1, y1;
    if (!segmentScreen(place_after_, x0, y0, x1, y1)) return;
    const OutlineAim a = aimOnSegment(fx, fy, x0, y0, x1, y1);
    if (!a.ok) return;
    place_ghost_sx_ = a.ghostX;
    place_ghost_sy_ = a.ghostY;
    finger_x_ = fx;
    finger_y_ = fy;
    float ix, iy;
    screenToImage(a.ghostX, a.ghostY, ix, iy);
    place_ix_ = std::clamp(ix, 0.0f, std::max(0.0f, (float)pixels_.w - 1.0f));
    place_iy_ = std::clamp(iy, 0.0f, std::max(0.0f, (float)pixels_.h - 1.0f));
}

void ViewMageApp::insertOutlinePoint(int after, float ix, float iy,
                                     PagePointKind kind, float hx, float hy) {
    auto& pts = adjust_outline_.points;
    if (after < 0 || after >= (int)pts.size()) return;
    PagePoint knot;
    knot.kind = kind;
    knot.x = ix;
    knot.y = iy;
    knot.hx = kind == PagePointKind::Curve ? hx : 0.0f;
    knot.hy = kind == PagePointKind::Curve ? hy : 0.0f;
    pts.insert(pts.begin() + after + 1, knot);
    syncShapeFromPoints();
}

void ViewMageApp::commitPlacement() {
    if (!placing_) return;
    ensureAdjustPoints();
    // Always a straight point. Bending is the stick, which is a separate
    // grab, so a finger that is not perfectly on the line does not invent a bow.
    insertOutlinePoint(place_after_, place_ix_, place_iy_,
                       PagePointKind::Fold, 0.0f, 0.0f);
}

bool ViewMageApp::pointRemoveArmed() const {
    if (drag_handle_ || drag_corner_ < 0) return false;
    const auto& pts = adjust_outline_.points;
    if (drag_corner_ >= (int)pts.size() || pts.size() <= 3) return false;
    if (pts[drag_corner_].kind == PagePointKind::Corner) return false;
    const int n = (int)pts.size();
    const int prev = (drag_corner_ + n - 1) % n;
    const int next = (drag_corner_ + 1) % n;
    float sx, sy;
    imageToScreen(pts[drag_corner_].x, pts[drag_corner_].y, sx, sy);
    auto nearScreen = [&](int i) {
        if (pts[i].kind != PagePointKind::Corner) return false;
        float kx, ky;
        imageToScreen(pts[i].x, pts[i].y, kx, ky);
        const float dx = sx - kx, dy = sy - ky;
        return dx * dx + dy * dy < 80.0f * 80.0f;
    };
    if (nearScreen(prev) || nearScreen(next)) return true;
    float x0, y0, x1, y1;
    imageToScreen(pts[prev].x, pts[prev].y, x0, y0);
    imageToScreen(pts[next].x, pts[next].y, x1, y1);
    const float dx = x1 - x0, dy = y1 - y0;
    const float len2 = dx * dx + dy * dy;
    float dist;
    if (len2 < 1.0f) {
        dist = std::sqrt((sx - x0) * (sx - x0) + (sy - y0) * (sy - y0));
    } else {
        float t = ((sx - x0) * dx + (sy - y0) * dy) / len2;
        t = std::clamp(t, 0.0f, 1.0f);
        const float px = x0 + dx * t, py = y0 + dy * t;
        dist = std::sqrt((sx - px) * (sx - px) + (sy - py) * (sy - py));
    }
    return dist > 72.0f;
}

int ViewMageApp::pointAt(float x, float y) const {
    int best = -1;
    float bestD = 56.0f * 56.0f;
    for (int i = 0; i < (int)adjust_outline_.points.size(); ++i) {
        float sx, sy;
        imageToScreen(adjust_outline_.points[i].x, adjust_outline_.points[i].y, sx, sy);
        const float dx = x - sx, dy = y - sy;
        const float d = dx * dx + dy * dy;
        if (d <= bestD) { bestD = d; best = i; }
    }
    if (!adjust_outline_.points.empty()) return best;
    return cornerAt(x, y);
}

int ViewMageApp::handleAt(float x, float y) const {
    int best = -1;
    // Smaller than the stick, so the circle around the tip does not cover
    // the point. The point and the tip still meet; nearerHandle splits them.
    float bestD = 40.0f * 40.0f;
    for (int i = 0; i < (int)adjust_outline_.points.size(); ++i) {
        if (adjust_outline_.points[i].kind == PagePointKind::Corner) continue;
        float hx, hy;
        handleVector(i, hx, hy);
        if (hx == 0.0f && hy == 0.0f) continue;
        float sx, sy;
        imageToScreen(adjust_outline_.points[i].x + hx,
                      adjust_outline_.points[i].y + hy, sx, sy);
        const float dx = x - sx, dy = y - sy;
        const float d = dx * dx + dy * dy;
        if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
}

bool ViewMageApp::segmentAt(float sx, float sy, int& after, float& ix, float& iy) const {
    const auto& pts = adjust_outline_.points;
    const bool have = !pts.empty();
    const int n = have ? (int)pts.size() : 4;
    if (n < 2) return false;
    float best = kOutlineHitPx;
    bool hit = false;
    for (int i = 0; i < n; ++i) {
        float x0, y0, x1, y1;
        if (!segmentScreen(i, x0, y0, x1, y1)) continue;
        const OutlineAim a = aimOnSegment(sx, sy, x0, y0, x1, y1);
        if (!a.ok || a.fingerDist > best) continue;
        bool onKnob = false;
        const int knobs = have ? n : 4;
        for (int k = 0; k < knobs; ++k) {
            float kx, ky;
            if (have) imageToScreen(pts[k].x, pts[k].y, kx, ky);
            else      imageToScreen(corner_x_[k], corner_y_[k], kx, ky);
            const float ox = sx - kx, oy = sy - ky;
            if (ox * ox + oy * oy < 56.0f * 56.0f) onKnob = true;
        }
        if (onKnob) continue;
        best = a.fingerDist;
        hit = true;
        after = i;
        screenToImage(a.ghostX, a.ghostY, ix, iy);
    }
    return hit;
}

int ViewMageApp::cornerAt(float x, float y) const {
    int best = -1;
    float bestD = 56.0f * 56.0f;
    for (int i = 0; i < 4; ++i) {
        float sx, sy;
        imageToScreen(corner_x_[i], corner_y_[i], sx, sy);
        const float dx = x - sx, dy = y - sy;
        const float d = dx * dx + dy * dy;
        if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
}

void ViewMageApp::drawScanChrome(Canvas& canvas, float barTop) {
    const Color edge{0.96f, 0.86f, 0.20f, 0.95f};
    auto stroke = [&](float x0, float y0, float x1, float y1) {
        const float dx = x1 - x0, dy = y1 - y0;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len < 0.5f) return;
        const float px = -dy / len * 1.5f, py = dx / len * 1.5f;
        canvas.triangle(x0 + px, y0 + py, x0 - px, y0 - py, x1 + px, y1 + py, edge);
        canvas.triangle(x0 - px, y0 - py, x1 + px, y1 + py, x1 - px, y1 - py, edge);
    };
    auto strokePoly = [&](const std::vector<float>& xs, const std::vector<float>& ys, bool close) {
        const size_t n = std::min(xs.size(), ys.size());
        if (n < 2) return;
        float x0, y0, px, py;
        imageToScreen(xs[0], ys[0], x0, y0);
        px = x0; py = y0;
        for (size_t i = 1; i < n; ++i) {
            float x1, y1;
            imageToScreen(xs[i], ys[i], x1, y1);
            stroke(px, py, x1, y1);
            px = x1; py = y1;
        }
        if (close) stroke(px, py, x0, y0);
    };
    const std::vector<PagePoint>& pts = adjust_outline_.points;
    const Color cornerInk{0.96f, 0.86f, 0.20f, 1.0f};
    const Color guideInk{0.25f, 0.72f, 0.95f, 1.0f};
    const Color removeInk{0.86f, 0.16f, 0.12f, 1.0f};
    const bool removing = pointRemoveArmed();
    auto knob = [&](float cx, float cy, float fill, bool round, Color fillC) {
        const float outer = fill + 4.0f;
        canvas.rect(cx - outer, cy - outer, outer * 2.0f, outer * 2.0f,
                    Color{0.0f, 0.0f, 0.0f, 1.0f}, round ? outer : 3.0f);
        canvas.rect(cx - fill, cy - fill, fill * 2.0f, fill * 2.0f,
                    fillC, round ? fill : 2.0f);
    };
    auto diamond = [&](float cx, float cy, float r, Color fillC) {
        canvas.triangle(cx, cy - r, cx - r, cy, cx + r, cy, Color{0.0f, 0.0f, 0.0f, 1.0f});
        canvas.triangle(cx, cy + r, cx - r, cy, cx + r, cy, Color{0.0f, 0.0f, 0.0f, 1.0f});
        const float ir = r - 4.0f;
        canvas.triangle(cx, cy - ir, cx - ir, cy, cx + ir, cy, fillC);
        canvas.triangle(cx, cy + ir, cx - ir, cy, cx + ir, cy, fillC);
    };
    if (!pts.empty()) {
        const int n = (int)pts.size();
        for (int i = 0; i < n; ++i) {
            const PagePoint& a = pts[i];
            const PagePoint& b = pts[(i + 1) % n];
            const bool curved = a.kind == PagePointKind::Curve || b.kind == PagePointKind::Curve;
            const int steps = curved ? 12 : 1;
            float x0, y0;
            imageToScreen(a.x, a.y, x0, y0);
            for (int k = 1; k <= steps; ++k) {
                const float t = (float)k / (float)steps;
                const float u = 1.0f - t;
                const float hx0 = a.kind == PagePointKind::Curve ? a.hx : 0.0f;
                const float hy0 = a.kind == PagePointKind::Curve ? a.hy : 0.0f;
                const float hx1 = b.kind == PagePointKind::Curve ? -b.hx : 0.0f;
                const float hy1 = b.kind == PagePointKind::Curve ? -b.hy : 0.0f;
                const float ix = u*u*u*a.x + 3*u*u*t*(a.x+hx0) + 3*u*t*t*(b.x+hx1) + t*t*t*b.x;
                const float iy = u*u*u*a.y + 3*u*u*t*(a.y+hy0) + 3*u*t*t*(b.y+hy1) + t*t*t*b.y;
                float x1, y1;
                imageToScreen(ix, iy, x1, y1);
                stroke(x0, y0, x1, y1);
                x0 = x1; y0 = y1;
            }
        }
        // A tick at the middle of each side is the affordance: the line takes a dot.
        for (int i = 0; i < n; ++i) {
            float x0, y0, x1, y1;
            imageToScreen(pts[i].x, pts[i].y, x0, y0);
            imageToScreen(pts[(i + 1) % n].x, pts[(i + 1) % n].y, x1, y1);
            const float dx = x1 - x0, dy = y1 - y0;
            const float len = std::sqrt(dx * dx + dy * dy);
            if (len < 96.0f) continue;
            const float mx = (x0 + x1) * 0.5f, my = (y0 + y1) * 0.5f;
            const float nx = -dy / len, ny = dx / len;
            stroke(mx - nx * 8.0f, my - ny * 8.0f, mx + nx * 8.0f, my + ny * 8.0f);
        }
        for (int i = 0; i < n; ++i) {
            float cx, cy;
            imageToScreen(pts[i].x, pts[i].y, cx, cy);
            const float fill = (i == drag_corner_ && !drag_handle_) ? 20.0f : 16.0f;
            const bool pageCorner = pts[i].kind == PagePointKind::Corner;
            const Color mark = (removing && i == drag_corner_) ? removeInk
                             : pageCorner ? cornerInk : guideInk;
            if (pts[i].kind == PagePointKind::Fold) diamond(cx, cy, fill + 4.0f, mark);
            else knob(cx, cy, fill, pts[i].kind == PagePointKind::Curve, mark);
            if (pts[i].kind == PagePointKind::Corner) continue;
            float hx, hy;
            handleVector(i, hx, hy);
            if (hx == 0.0f && hy == 0.0f) continue;
            float tx, ty;
            imageToScreen(pts[i].x + hx, pts[i].y + hy, tx, ty);
            stroke(cx, cy, tx, ty);
            knob(tx, ty, 14.0f, true, guideInk);
        }
    } else if (adjust_shape_ != DocumentShape::Quad) {
        if (adjust_shape_ == DocumentShape::Curve) {
            strokePoly(adjust_outline_.ax, adjust_outline_.ay, false);
            strokePoly(adjust_outline_.bx, adjust_outline_.by, false);
            if (!adjust_outline_.ax.empty() && !adjust_outline_.bx.empty()) {
                float x0, y0, x1, y1;
                imageToScreen(adjust_outline_.ax.front(), adjust_outline_.ay.front(), x0, y0);
                imageToScreen(adjust_outline_.bx.front(), adjust_outline_.by.front(), x1, y1);
                stroke(x0, y0, x1, y1);
                imageToScreen(adjust_outline_.ax.back(), adjust_outline_.ay.back(), x0, y0);
                imageToScreen(adjust_outline_.bx.back(), adjust_outline_.by.back(), x1, y1);
                stroke(x0, y0, x1, y1);
            }
        } else {
            strokePoly(adjust_outline_.x, adjust_outline_.y, true);
        }
        return;
    } else {
        float sx[4], sy[4];
        for (int i = 0; i < 4; ++i)
            imageToScreen(corner_x_[i], corner_y_[i], sx[i], sy[i]);
        for (int i = 0; i < 4; ++i)
            stroke(sx[i], sy[i], sx[(i + 1) % 4], sy[(i + 1) % 4]);
        for (int i = 0; i < 4; ++i) {
            const float dx = sx[(i + 1) % 4] - sx[i];
            const float dy = sy[(i + 1) % 4] - sy[i];
            const float len = std::sqrt(dx * dx + dy * dy);
            if (len < 96.0f) continue;
            const float mx = (sx[i] + sx[(i + 1) % 4]) * 0.5f;
            const float my = (sy[i] + sy[(i + 1) % 4]) * 0.5f;
            const float nx = -dy / len, ny = dx / len;
            stroke(mx - nx * 8.0f, my - ny * 8.0f, mx + nx * 8.0f, my + ny * 8.0f);
        }
        for (int i = 0; i < 4; ++i)
            knob(sx[i], sy[i], (i == drag_corner_) ? 20.0f : 16.0f, false, cornerInk);
    }

    if (placing_) {
        stroke(finger_x_, finger_y_, place_ghost_sx_, place_ghost_sy_);
        diamond(place_ghost_sx_, place_ghost_sy_, 20.0f, guideInk);
    }

    const bool aiming = placing_ || drag_corner_ >= 0;
    if (!aiming || pixels_.linear.empty()) return;
    // Past about 4x fit the photograph is already the close view. A second
    // zoom on top of it is how the page disappears.
    const float fit = view_.fitScale();
    if (!(fit > 0.0f) || view_.scale() > fit * 4.0f) return;

    // 25 source pixels, 8 screen pixels each. The aim pixel is left clear:
    // the cross stops short of it, so the centre of the window is the photo.
    constexpr int kHalf = 12;
    constexpr float kCell = 8.0f;
    const float loupe = kCell * (float)(kHalf * 2 + 1);
    const float gap = 96.0f;
    const float margin = 8.0f;
    const float screenW = (float)renderer_->width();
    struct Cand { float x, y; };
    const Cand cands[4] = {
        {finger_x_ - loupe * 0.5f, finger_y_ - gap - loupe},
        {finger_x_ - loupe * 0.5f, finger_y_ + gap},
        {finger_x_ - gap - loupe, finger_y_ - loupe * 0.5f},
        {finger_x_ + gap, finger_y_ - loupe * 0.5f},
    };
    auto clearOfFinger = [&](Cand c) {
        const float nx = std::clamp(finger_x_, c.x, c.x + loupe);
        const float ny = std::clamp(finger_y_, c.y, c.y + loupe);
        const float dx = finger_x_ - nx, dy = finger_y_ - ny;
        return dx * dx + dy * dy >= 64.0f * 64.0f;
    };
    auto fits = [&](Cand c) {
        return c.x >= margin && c.y >= margin &&
               c.x + loupe <= screenW - margin &&
               c.y + loupe <= barTop - margin;
    };
    Cand place = cands[0];
    bool placed = false;
    for (Cand c : cands) {
        if (fits(c) && clearOfFinger(c)) { place = c; placed = true; break; }
    }
    if (!placed) {
        place.x = std::clamp(place.x, margin, std::max(margin, screenW - loupe - margin));
        place.y = std::clamp(place.y, margin, std::max(margin, barTop - loupe - margin));
    }

    float loupeX = 0, loupeY = 0;
    if (placing_) {
        loupeX = place_ix_;
        loupeY = place_iy_;
    } else if (drag_corner_ >= 0 && drag_corner_ < (int)adjust_outline_.points.size()) {
        loupeX = adjust_outline_.points[drag_corner_].x;
        loupeY = adjust_outline_.points[drag_corner_].y;
    } else if (drag_corner_ >= 0 && drag_corner_ < 4) {
        loupeX = corner_x_[drag_corner_];
        loupeY = corner_y_[drag_corner_];
    }
    const int pixX = (int)std::floor(loupeX);
    const int pixY = (int)std::floor(loupeY);
    const float gain = std::exp2(ev_);
    const Color back{0.02f, 0.02f, 0.025f, 0.94f};
    canvas.rect(place.x, place.y, loupe, loupe, back);

    const auto clip = canvas.saveClip();
    canvas.setClip(place.x, place.y, loupe, loupe);

    for (int dy = -kHalf; dy <= kHalf; ++dy) {
        for (int dx = -kHalf; dx <= kHalf; ++dx) {
            const int ix = pixX + dx, iy = pixY + dy;
            Color col{0.12f, 0.12f, 0.13f, 1.0f};
            if (ix >= 0 && iy >= 0 && ix < (int)pixels_.w && iy < (int)pixels_.h) {
                const float* p = pixels_.linear.data() +
                    ((size_t)iy * pixels_.w + (size_t)ix) * 4;
                col = Color{srgb_encode(std::clamp(p[0] * gain, 0.0f, 1.0f)),
                            srgb_encode(std::clamp(p[1] * gain, 0.0f, 1.0f)),
                            srgb_encode(std::clamp(p[2] * gain, 0.0f, 1.0f)), 1.0f};
            }
            const float cellCx = place.x + loupe * 0.5f +
                                 ((float)ix + 0.5f - loupeX) * kCell;
            const float cellCy = place.y + loupe * 0.5f +
                                 ((float)iy + 0.5f - loupeY) * kCell;
            canvas.rect(cellCx - kCell * 0.5f, cellCy - kCell * 0.5f, kCell, kCell, col);
        }
    }

    auto toWin = [&](float ix, float iy, float& wx, float& wy) {
        wx = place.x + loupe * 0.5f + (ix - loupeX) * kCell;
        wy = place.y + loupe * 0.5f + (iy - loupeY) * kCell;
    };
    auto strokeWin = [&](float ax, float ay, float bx, float by) {
        float x0, y0, x1, y1;
        toWin(ax, ay, x0, y0);
        toWin(bx, by, x1, y1);
        stroke(x0, y0, x1, y1);
    };
    const std::vector<PagePoint>& wpts = adjust_outline_.points;
    if (!wpts.empty()) {
        const int n = (int)wpts.size();
        for (int i = 0; i < n; ++i) {
            const PagePoint& a = wpts[i];
            const PagePoint& b = wpts[(i + 1) % n];
            const bool curved = a.kind == PagePointKind::Curve || b.kind == PagePointKind::Curve;
            const int steps = curved ? 8 : 1;
            float px = a.x, py = a.y;
            for (int k = 1; k <= steps; ++k) {
                const float t = (float)k / (float)steps;
                const float u = 1.0f - t;
                const float hx0 = a.kind == PagePointKind::Curve ? a.hx : 0.0f;
                const float hy0 = a.kind == PagePointKind::Curve ? a.hy : 0.0f;
                const float hx1 = b.kind == PagePointKind::Curve ? -b.hx : 0.0f;
                const float hy1 = b.kind == PagePointKind::Curve ? -b.hy : 0.0f;
                const float ix = u*u*u*a.x + 3*u*u*t*(a.x+hx0) + 3*u*t*t*(b.x+hx1) + t*t*t*b.x;
                const float iy = u*u*u*a.y + 3*u*u*t*(a.y+hy0) + 3*u*t*t*(b.y+hy1) + t*t*t*b.y;
                strokeWin(px, py, ix, iy);
                px = ix;
                py = iy;
            }
        }
    } else {
        for (int i = 0; i < 4; ++i)
            strokeWin(corner_x_[i], corner_y_[i],
                      corner_x_[(i + 1) % 4], corner_y_[(i + 1) % 4]);
    }
    // Paint the aim pixel again, on top of the outline, then a cross that
    // stops short of it. A line through the middle hides the very pixel
    // being chosen, and a white hairline disappears on paper.
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int ix = pixX + dx, iy = pixY + dy;
            Color col{0.12f, 0.12f, 0.13f, 1.0f};
            if (ix >= 0 && iy >= 0 && ix < (int)pixels_.w && iy < (int)pixels_.h) {
                const float* p = pixels_.linear.data() +
                    ((size_t)iy * pixels_.w + (size_t)ix) * 4;
                col = Color{srgb_encode(std::clamp(p[0] * gain, 0.0f, 1.0f)),
                            srgb_encode(std::clamp(p[1] * gain, 0.0f, 1.0f)),
                            srgb_encode(std::clamp(p[2] * gain, 0.0f, 1.0f)), 1.0f};
            }
            const float cellCx = place.x + loupe * 0.5f +
                                 ((float)ix + 0.5f - loupeX) * kCell;
            const float cellCy = place.y + loupe * 0.5f +
                                 ((float)iy + 0.5f - loupeY) * kCell;
            canvas.rect(cellCx - kCell * 0.5f, cellCy - kCell * 0.5f, kCell, kCell, col);
        }
    }
    const float cx = place.x + loupe * 0.5f;
    const float cy = place.y + loupe * 0.5f;
    const float hole = 14.0f;
    const Color crossDark{0.0f, 0.0f, 0.0f, 0.95f};
    const Color crossLite{1.0f, 1.0f, 1.0f, 0.95f};
    auto bar = [&](float x, float y, float w, float h, Color c) {
        if (w > 0.5f && h > 0.5f) canvas.rect(x, y, w, h, c);
    };
    const float dark = 4.0f, lite = 2.0f;
    bar(place.x, cy - dark * 0.5f, cx - hole - place.x, dark, crossDark);
    bar(cx + hole, cy - dark * 0.5f, place.x + loupe - (cx + hole), dark, crossDark);
    bar(cx - dark * 0.5f, place.y, dark, cy - hole - place.y, crossDark);
    bar(cx - dark * 0.5f, cy + hole, dark, place.y + loupe - (cy + hole), crossDark);
    bar(place.x, cy - lite * 0.5f, cx - hole - place.x, lite, crossLite);
    bar(cx + hole, cy - lite * 0.5f, place.x + loupe - (cx + hole), lite, crossLite);
    bar(cx - lite * 0.5f, place.y, lite, cy - hole - place.y, crossLite);
    bar(cx - lite * 0.5f, cy + hole, lite, place.y + loupe - (cy + hole), crossLite);
    canvas.restoreClip(clip);

    const Color frame{1.0f, 1.0f, 1.0f, 0.85f};
    canvas.rect(place.x, place.y, loupe, 2.0f, frame);
    canvas.rect(place.x, place.y + loupe - 2.0f, loupe, 2.0f, frame);
    canvas.rect(place.x, place.y, 2.0f, loupe, frame);
    canvas.rect(place.x + loupe - 2.0f, place.y, 2.0f, loupe, frame);
}

void ViewMageApp::onHostResized() {
    syncViewport();
    dirty_ = true;
}

void ViewMageApp::syncViewport() {
    if (!renderer_) return;
    const int w = (int)renderer_->width();
    const int h = (int)renderer_->height();
    if (w > 0 && h > 0) view_.setViewport(w, h);
}

// The image is loaded HERE rather than in create(), because the ceiling the
// decode has to respect is the GPU's max 2D image dimension and there is no
// GPU until the Renderer has chosen a device.
void ViewMageApp::onHostReady() {
    loadFromSource();
    dirty_ = true;
}

void ViewMageApp::loadFromSource() {
    if (source_.empty()) {
        state_   = State::kError;
        message_ = "No image to show";
        return;
    }

    uint32_t maxDim = 0;
    if (renderer_) maxDim = renderer_->caps().max_image_dim_2d;

    // ASK THE RENDERER, do not repeat the request from create(): a PQ request
    // can fall back to SDR/scRGB, and those are sRGB primaries. Decoding to
    // BT.2100 for a surface the compositor reads as sRGB is the same error as
    // the reverse, just pointing the other way.
    const WorkingPrimaries prim =
        (renderer_ && renderer_->activeTarget() == OutputTarget::Hdr10PQ)
            ? WorkingPrimaries::Bt2100 : WorkingPrimaries::Srgb;

    // Route on the file's own signature, not on the extension or the MIME type
    // the opener claimed: a file manager's guess is not evidence, and both
    // decoders can say "not mine" cheaply from the first four bytes.
    //
    // DNG is not a second format bolted onto a viewer — since the camera became
    // capture-only it is the ONLY format the camera produces, and developing it
    // is the whole reason this app exists now. See dng_image.hh.
    if (looks_like_dng(source_.data(), source_.size())) {
        pixels_ = decode_dng(source_.data(), source_.size(), maxDim, prim, &noise_);
        if (noise_.valid)
            VCE_LOGI("ViewMage", "noise profile: ch0 S=%.3e O=%.3e", noise_.so[0], noise_.so[1]);
        else
            VCE_LOGI("ViewMage", "no NoiseProfile tag — a denoiser would have to guess");
    } else {
        pixels_ = decode_jxl(source_.data(), source_.size(), maxDim, prim);
    }
    if (!pixels_.ok()) {
        state_   = State::kError;
        message_ = pixels_.error.empty() ? "Could not decode this image" : pixels_.error;
        VCE_LOGE("ViewMage", "decode failed: %s", message_.c_str());
        return;
    }
    if (pixels_.downsampleFactor > 1) {
        VCE_LOGI("ViewMage",
                 "image exceeded the GPU limit (%u px); downsampled %dx to %ux%u",
                 maxDim, pixels_.downsampleFactor, pixels_.w, pixels_.h);
    }

    // The decoder worked out what this image needs; adopt it before the first
    // frame so the picture is right the instant it appears rather than being
    // corrected a frame later.
    ev_    = pixels_.autoEv;
    white_ = pixels_.white;
    // kRolloff only where there is something to roll off. For an ordinary SDR
    // photo white_ is 1.0 and the curve is an exact identity, but going through
    // kPassthrough instead keeps that case provably byte-identical to before.
    const bool needsTone = pixels_.hdrTransfer || pixels_.widePrimaries ||
                           pixels_.bitsPerSample > 8 || pixels_.exponentBits > 0 ||
                           white_ > 1.0f || std::fabs(ev_) > 0.01f;
    toneMode_ = needsTone ? ToneMode::kRolloff : ToneMode::kClip;

    // On an HDR swapchain the rolloff knee aims at the panel's headroom rather
    // than at display white (vk_canvas's rolloffCurve() takes that ceiling).
    // Nothing here changes on SDR, where the ceiling is 1.0 and the curve is
    // bit-identical to what the tests already assert.
    //
    // autoEV is KEPT. The alternative -- pinning 0 EV because the panel is
    // capable -- opens a dim file dim and calls it honesty; the exposure exists
    // so an image lands where it was meant to land, and a display with headroom
    // does not make that decision wrong. The full float range is still in the
    // texture either way, so nothing the panel could have shown is discarded
    // before the exposure control can reach it.

    if (!uploadTexture()) {
        state_   = State::kError;
        message_ = "Not enough graphics memory";
        return;
    }
    // Keep them for "Save PNG" when this is a RAW file -- see keep_pixels_.
    keep_pixels_ = looks_like_dng(source_.data(), source_.size());
    if (!keep_pixels_) releasePixels();

    view_.setImage((int)pixels_.w, (int)pixels_.h);
    syncViewport();
    view_.fit();
    state_ = State::kReady;

    // Read the model's bytes here but do NOT parse them here: the read is I/O
    // and quick, while building the ONNX session is minutes of CPU on a phone.
    // Both used to happen on this thread and the app sat frozen for six minutes
    // with the picture already drawn behind it -- measured, not theorised.
    if (!denoiser_ && looks_like_dng(source_.data(), source_.size())) {
        const double t0 = nowSeconds();
        if (host_->dataReader().read("models/model_bayer.onnx", model_bytes_) &&
            !model_bytes_.empty()) {
            VCE_LOGI("ViewMage", "denoise model read: %zu MB in %.0f ms",
                     model_bytes_.size() / (1024 * 1024),
                     (nowSeconds() - t0) * 1000.0);
            // NOT started here. Denoising is minutes of CPU and a deliberate
            // step taken before exporting -- not something to spend on every
            // photo merely opened. The Denoise button starts it.
        } else {
            VCE_LOGI("ViewMage", "no denoise model shipped; ordinary develop only");
        }
    }

    VCE_LOGI("ViewMage", "showing %ux%u  transfer=%s primaries=%s  autoEV=%.2f white=%.2f",
             pixels_.w, pixels_.h,
             pixels_.transferName.empty() ? "?" : pixels_.transferName.c_str(),
             pixels_.primariesName.empty() ? "?" : pixels_.primariesName.c_str(),
             ev_, white_);
}

// Upload at the best precision this device will actually accept, and SAY when
// it is not the best one.
//
// The chain is full float -> half float -> 8-bit, and each step is a real
// concession: full float loses nothing, half float loses about a twentieth of
// an 8-bit code step, and 8-bit loses the range that this whole feature exists
// to preserve -- at which point exposure has to be baked in on the CPU and the
// slider stops being able to reveal anything.
//
// Degrading silently would be the worst option available: the viewer would
// still show a picture, and the user would have no way to know the data behind
// it was gone. So every step down adds a diagnostic.
bool ViewMageApp::uploadTexture() {
    if (!renderer_ || !pixels_.ok() || pixels_.linear.empty()) return false;
    releaseTexture();

    const size_t texels = (size_t)pixels_.w * pixels_.h;

    // 1. Full float — what was asked for, and what loses nothing.
    texture_ = renderer_->create_texture(
        reinterpret_cast<const uint8_t*>(pixels_.linear.data()),
        pixels_.w, pixels_.h, /*mips=*/false, TextureFormat::RGBA32F);
    if (texture_ != kInvalidTexture) return true;
    VCE_LOGI("ViewMage", "RGBA32F upload failed; trying half float");

    // 2. Half float — still linear, still unclipped, still fully explorable.
    try {
        std::vector<uint16_t> half(texels * 4);
        for (size_t i = 0; i < half.size(); ++i) half[i] = to_half(pixels_.linear[i]);
        texture_ = renderer_->create_texture(
            reinterpret_cast<const uint8_t*>(half.data()),
            pixels_.w, pixels_.h, /*mips=*/false, TextureFormat::RGBA16F);
    } catch (const std::bad_alloc&) {
        texture_ = kInvalidTexture;
    }
    if (texture_ != kInvalidTexture) {
        pixels_.notes.push_back(Diagnostic{
            DiagCode::kPrecisionReduced, DiagLevel::kNotice,
            "Held at reduced precision",
            "There was not enough graphics memory to keep this photo at full "
            "precision, so it is held at half precision instead. The difference "
            "is far below what this screen can show, and the exposure slider "
            "still reaches the whole range."});
        return true;
    }
    VCE_LOGI("ViewMage", "RGBA16F upload failed; falling back to 8-bit");

    // 3. Eight bits, exposure and tone curve baked in on the CPU. The picture
    //    survives; the ability to explore beyond it does not.
    try {
        std::vector<uint8_t> bytes(texels * 4);
        const float gain = std::exp2(ev_);
        for (size_t i = 0; i < texels; ++i) {
            const float* p = &pixels_.linear[i * 4];
            for (int c = 0; c < 3; ++c)
                bytes[i * 4 + c] = (uint8_t)std::lround(srgb_encode(p[c] * gain) * 255.0f);
            bytes[i * 4 + 3] = (uint8_t)std::lround(std::clamp(p[3], 0.0f, 1.0f) * 255.0f);
        }
        texture_ = renderer_->create_texture(bytes.data(), pixels_.w, pixels_.h,
                                             /*mips=*/false, TextureFormat::RGBA8_UNORM);
    } catch (const std::bad_alloc&) {
        return false;
    }
    if (texture_ == kInvalidTexture) return false;

    // The texture is now display-referred, so the shader must NOT tone-map it
    // a second time.
    toneMode_ = ToneMode::kPassthrough;
    pixels_.notes.push_back(Diagnostic{
        DiagCode::kPrecisionReduced, DiagLevel::kWarning,
        "Extra range could not be kept",
        "This device did not have enough graphics memory to hold this photo's "
        "full range, so it has been flattened to what the screen shows. The "
        "exposure control cannot reveal anything beyond this view."});
    return true;
}

void ViewMageApp::releasePixels() {
    // swap-with-empty, not clear(): clear() keeps the capacity, which is the
    // entire several hundred megabytes we are trying to give back.
    std::vector<float>().swap(pixels_.linear);
}

void ViewMageApp::releaseTexture() {
    if (texture_ != kInvalidTexture && renderer_) renderer_->destroy_texture(texture_);
    texture_ = kInvalidTexture;
}

// The panel's headroom is a LIVE measurement, not a fixed capability: on API
// 34+ it is Display.getHdrSdrRatio(), and SDR white is whatever the system is
// currently driving the screen at. The same phone has room for a bright
// highlight in a dark room and almost none outdoors, so a value read once at
// startup goes stale the moment the user walks through a door.
//
// Re-read where it is cheap and where it can have changed: surface recreation
// is how Android hands the app back after a trip through the background, which
// is exactly when the brightness is likely to be different.
void ViewMageApp::refreshHeadroom() {
#ifdef __ANDROID__
    headroom_ = hdr_ ? activity::display_hdr_headroom() : 1.0f;
#else
    headroom_ = hdr_ ? 2.2f : 1.0f;
#endif
}

void ViewMageApp::onSurfaceLost() {
    // The window is going away. The device is not: the pipelines and the
    // uploaded photo stay valid, and onSurfaceRecreated() builds a swapchain
    // for the window Android hands back. Nulling the texture handle here leaks
    // that image and forces a second upload. During a denoise the ensemble
    // buffers already fill memory, that upload fails, and the new surface
    // stays black.
    surfaceOk_ = false;
}

bool ViewMageApp::onSurfaceRecreated() {
    if (!renderer_) return false;

    // New ANativeWindow, same device. Without a new swapchain, acquire returns
    // SURFACE_LOST and every frame is skipped, so the window stays black while
    // the denoise percentage was still advancing on the old one.
    const bool keptDevice = renderer_->recreate_surface();
    if (!keptDevice) {
        VCE_LOGI("ViewMage", "surface incompatible with existing pipelines; rebuilding the renderer");
        releaseTexture();
        renderer_.reset();
        renderer_ = std::make_unique<Renderer>(host_->surfaceProvider(), host_->assetReader(),
                                               /*images=*/4, OutputTarget::Hdr10PQ);
        hdr_ = renderer_->hdrActive();
    }

    surfaceOk_ = true;
    refreshHeadroom();
    syncViewport();

    // A worker finished while the surface was lost — upload the most
    // processed result available.  Enhanced takes priority over denoised
    // because it is further along the pipeline (denoise → enhance).
    auto uploadPending = [&](DecodedImage& pending, const char* label,
                             bool* flag) -> bool {
        if (!pending.ok() || state_ != State::kReady) return false;
        VCE_LOGI("ViewMage", "uploading %s result that landed off-screen", label);
        DecodedImage previous = std::move(pixels_);
        const TextureHandle old = texture_;
        texture_ = kInvalidTexture;
        pixels_  = std::move(pending);
        if (uploadTexture()) {
            if (old != kInvalidTexture) renderer_->destroy_texture(old);
            if (!keep_pixels_) releasePixels();
            ev_    = pixels_.autoEv;
            white_ = pixels_.white;
            *flag  = true;
            dirty_ = true;
            return true;
        }
        VCE_LOGE("ViewMage", "pending %s upload failed; re-decoding", label);
        releaseTexture();
        pixels_ = std::move(previous);
        texture_ = old;
        return false;
    };

    if (pendingScan_.ok() && state_ == State::kReady) {
        VCE_LOGI("ViewMage", "uploading scan result that landed off-screen");
        DecodedImage previous = std::move(pixels_);
        const TextureHandle old = texture_;
        texture_ = kInvalidTexture;
        pixels_ = std::move(pendingScan_);
        if (uploadTexture()) {
            if (old != kInvalidTexture) renderer_->destroy_texture(old);
            keep_pixels_ = true;
            view_.setImage((int)pixels_.w, (int)pixels_.h);
            syncViewport();
            view_.fit();
            scanned_ = true;
            enhanced_ = false;
            ev_ = pixels_.autoEv;
            white_ = pixels_.white;
            dirty_ = true;
            return true;
        }
        releaseTexture();
        pixels_ = std::move(previous);
        texture_ = old;
    }

    if (uploadPending(pendingEnhanced_, "enhanced", &enhanced_)) return true;
    if (uploadPending(pendingDenoised_, "denoised", &denoised_)) return true;

    // The photo is still on the GPU. A second upload, and a second develop,
    // are what run out of memory while a denoise holds the ensemble buffers.
    if (texture_ != kInvalidTexture) {
        dirty_ = true;
        return true;
    }

    // A denoise still running is already holding the ensemble buffers. A
    // second develop here is what runs the process out of memory and leaves
    // the recreated surface black. The photo is in pixels_; put it back.
    if (refine_thread_.joinable() && !refined_ready_.load()) {
        if (state_ == State::kReady && !pixels_.linear.empty() &&
            !uploadTexture()) {
            VCE_LOGE("ViewMage", "re-upload during denoise failed; "
                     "the develop stays in memory");
        }
        dirty_ = true;
        return true;
    }

    if (state_ == State::kReady && !pixels_.linear.empty()) {
        if (!uploadTexture())
            VCE_LOGE("ViewMage", "re-upload after surface recreate failed");
        dirty_ = true;
        return true;
    }

    if (state_ == State::kReady) {
        const float savedEv = ev_;
        loadFromSource();
        if (state_ == State::kReady) ev_ = savedEv;
    }
    dirty_ = true;
    return true;
}

// ── Gestures ────────────────────────────────────────────────────────────────

ViewMageApp::Pointer* ViewMageApp::find(int id) {
    for (auto& p : pointers_) if (p.id == id) return &p;
    return nullptr;
}

void ViewMageApp::forget(int id) {
    pointers_.erase(std::remove_if(pointers_.begin(), pointers_.end(),
                                   [id](const Pointer& p) { return p.id == id; }),
                    pointers_.end());
}

void ViewMageApp::onPointerDown(int pointerId, int x, int y) {
    forget(pointerId);   // a stale id, if an up was ever missed
    pointers_.push_back(Pointer{pointerId, (float)x, (float)y,
                                (float)x, (float)y, nowSeconds()});
    if (scan_adjust_ && pointers_.size() == 1 && state_ == State::kReady) {
        const int handle = handleAt((float)x, (float)y);
        const int knot = pointAt((float)x, (float)y);
        bool useHandle = handle >= 0 && knot < 0;
        if (handle >= 0 && knot >= 0) {
            float kx, ky, hx, hy, tx, ty;
            imageToScreen(adjust_outline_.points[knot].x,
                          adjust_outline_.points[knot].y, kx, ky);
            handleVector(handle, hx, hy);
            imageToScreen(adjust_outline_.points[handle].x + hx,
                          adjust_outline_.points[handle].y + hy, tx, ty);
            useHandle = nearerHandle((float)x, (float)y, kx, ky, tx, ty);
        }
        const int hit = useHandle ? handle : knot;
        if (hit >= 0) {
            drag_corner_ = hit;
            drag_handle_ = useHandle;
            drag_pointer_ = pointerId;
            finger_x_ = (float)x;
            finger_y_ = (float)y;
            dirty_ = true;
        } else if (!barContains((float)x, (float)y)) {
            int after = -1;
            float ix = 0, iy = 0;
            if (segmentAt((float)x, (float)y, after, ix, iy)) {
                ensureAdjustPoints();
                placing_ = true;
                place_pointer_ = pointerId;
                place_after_ = after;
                updatePlacement((float)x, (float)y);
                dirty_ = true;
            }
        }
    }
    if (pointers_.size() == 2) {
        drag_corner_ = -1;
        drag_handle_ = false;
        drag_pointer_ = -1;
        placing_ = false;
        lastPinchDist_ = distance(pointers_[0].x, pointers_[0].y,
                                  pointers_[1].x, pointers_[1].y);
        lastMidX_ = (pointers_[0].x + pointers_[1].x) * 0.5f;
        lastMidY_ = (pointers_[0].y + pointers_[1].y) * 0.5f;
    }
}

void ViewMageApp::onPointerMove(int pointerId, int x, int y) {
    Pointer* p = find(pointerId);
    if (!p) return;
    const float px = p->x, py = p->y;
    p->x = (float)x;
    p->y = (float)y;
    if (state_ != State::kReady) return;

    if (placing_ && pointerId == place_pointer_ && pointers_.size() == 1) {
        updatePlacement(p->x, p->y);
        dirty_ = true;
        return;
    }

    if (drag_corner_ >= 0 && pointerId == drag_pointer_ && pointers_.size() == 1) {
        float ix, iy;
        screenToImage(p->x, p->y, ix, iy);
        ix = std::clamp(ix, 0.0f, std::max(0.0f, (float)pixels_.w - 1.0f));
        iy = std::clamp(iy, 0.0f, std::max(0.0f, (float)pixels_.h - 1.0f));
        if (drag_corner_ < (int)adjust_outline_.points.size()) {
            PagePoint& pt = adjust_outline_.points[drag_corner_];
            if (drag_handle_) {
                float ksx, ksy;
                imageToScreen(pt.x, pt.y, ksx, ksy);
                // Screen distance, not source pixels: at fit, six source
                // pixels is a couple of screen pixels and the stick can
                // never be put back down.
                if (distance(p->x, p->y, ksx, ksy) < kOutlineStraightenPx) {
                    pt.kind = PagePointKind::Fold;
                    pt.hx = pt.hy = 0.0f;
                } else {
                    pt.kind = PagePointKind::Curve;
                    pt.hx = ix - pt.x;
                    pt.hy = iy - pt.y;
                }
            } else {
                pt.x = ix;
                pt.y = iy;
                if (adjust_outline_.points.size() == 4 && drag_corner_ < 4) {
                    corner_x_[drag_corner_] = ix;
                    corner_y_[drag_corner_] = iy;
                }
            }
            syncShapeFromPoints();
        } else if (drag_corner_ < 4) {
            corner_x_[drag_corner_] = ix;
            corner_y_[drag_corner_] = iy;
        }
        finger_x_ = p->x;
        finger_y_ = p->y;
        dirty_ = true;
        return;
    }

    if (pointers_.size() == 1) {
        view_.pan(p->x - px, p->y - py);
        dirty_ = true;
        return;
    }

    if (pointers_.size() >= 2) {
        const float dist = distance(pointers_[0].x, pointers_[0].y,
                                    pointers_[1].x, pointers_[1].y);
        const float midX = (pointers_[0].x + pointers_[1].x) * 0.5f;
        const float midY = (pointers_[0].y + pointers_[1].y) * 0.5f;

        // Pan by the midpoint's travel and zoom by the fingers' separation.
        // Both, in that order: a two-finger gesture that only zoomed would
        // fight a user trying to reframe while zooming, which is what people
        // actually do.
        view_.pan(midX - lastMidX_, midY - lastMidY_);
        if (lastPinchDist_ > 1.0f && dist > 1.0f) {
            float ratio = dist / lastPinchDist_;
            // Adjustment is for landing an edge, not for inspecting photosites.
            // 32x fit stays available once the outline edit is over.
            if (scan_adjust_) {
                const float fit = view_.fitScale();
                const float cap = fit * 6.0f;
                if (cap > 0.0f && view_.scale() > 0.0f && view_.scale() * ratio > cap)
                    ratio = cap / view_.scale();
            }
            view_.zoomAbout(midX, midY, ratio);
        }

        lastPinchDist_ = dist;
        lastMidX_ = midX;
        lastMidY_ = midY;
        dirty_ = true;
    }
}

void ViewMageApp::onPointerUp(int pointerId, int x, int y) {
    Pointer* p = find(pointerId);
    if (!p) return;

    const bool wasPlacing = placing_ && pointerId == place_pointer_;
    if (wasPlacing) {
        commitPlacement();
        placing_ = false;
        dirty_ = true;
    }

    const bool wasDragging = drag_corner_ >= 0 && pointerId == drag_pointer_;
    if (wasDragging) {
        const bool stationary =
            distance((float)x, (float)y, p->startX, p->startY) <= kTapSlopPx &&
            (nowSeconds() - p->downTime) <= kTapMaxSeconds;
        if (!drag_handle_ && drag_corner_ < (int)adjust_outline_.points.size()) {
            const PagePointKind kind = adjust_outline_.points[drag_corner_].kind;
            const bool removable = kind != PagePointKind::Corner &&
                                   adjust_outline_.points.size() > 3;
            if (removable && pointRemoveArmed()) {
                adjust_outline_.points.erase(
                    adjust_outline_.points.begin() + drag_corner_);
                syncShapeFromPoints();
                last_point_tap_ = -1;
            } else if (removable && stationary) {
                const double now = nowSeconds();
                if (last_point_tap_ == drag_corner_ &&
                    (now - last_point_tap_time_) <= kDoubleTapGap) {
                    adjust_outline_.points.erase(
                        adjust_outline_.points.begin() + drag_corner_);
                    syncShapeFromPoints();
                    last_point_tap_ = -1;
                } else {
                    last_point_tap_ = drag_corner_;
                    last_point_tap_time_ = now;
                }
            } else if (!stationary) {
                last_point_tap_ = -1;
            }
        }
        drag_corner_ = -1;
        drag_handle_ = false;
        drag_pointer_ = -1;
        dirty_ = true;
    }

    const bool wasTap = distance((float)x, (float)y, p->startX, p->startY) <= kTapSlopPx &&
                        (nowSeconds() - p->downTime) <= kTapMaxSeconds &&
                        pointers_.size() == 1;
    forget(pointerId);

    // Dropping from two fingers to one: re-seed the pinch baseline from the
    // finger that is left, or the next move jumps by the distance between them.
    if (pointers_.size() == 1) {
        lastPinchDist_ = 0.0f;
        lastMidX_ = pointers_[0].x;
        lastMidY_ = pointers_[0].y;
    }

    if (wasDragging || wasPlacing || !wasTap || state_ != State::kReady) return;

    // ── Two coordinate spaces, and they are NOT the same one ────────────────
    //
    // Canvas coordinates are SURFACE space: y=0 is the top of the surface, and
    // the safe insets are a margin inside it (Canvas::top() returns insetTop).
    // Pointer coordinates arrive in CONTENT space: y=0 is the top of the
    // content, i.e. already below the status bar.
    //
    // So a control drawn at surface y must be hit-tested at y minus the top
    // inset -- on this phone 125 px, about a fifth of the bar's height. Compare
    // the two spaces directly and a tap on the visible button lands outside its
    // rect and does nothing, which is exactly what shipped: the button could not
    // be pressed at all.
    //
    // Worth knowing how that got past a test: an INJECTED tap is in screen
    // space, so `input tap` at a y well BELOW the button arrives as a value
    // inside the rect and the export fires. The automated check passed for the
    // wrong reason while the real control was dead. Verify a control by tapping
    // where it is DRAWN, and convert the spaces here rather than aiming at the
    // offset.
    // Pointer coordinates and Canvas coordinates are the SAME space -- both are
    // surface-relative. Measured, not assumed: a tap the app received at y=2915
    // hit a button Canvas had placed at 2878..2951.
    //
    // The POST bar takes the tap before the view does, so pressing Save does not
    // also zoom the picture underneath it.
    if (denoise_rect_.contains((float)x, (float)y)) {
        lastTapTime_ = -1.0;
        export_note_.clear();
        if (refine_thread_.joinable() && !refined_ready_.load())
            refine_abort_.store(true);
        else
            startRefinement();
        dirty_ = true;
        return;
    }

    if (enhance_rect_.contains((float)x, (float)y)) {
        lastTapTime_ = -1.0;
        export_note_.clear();
        startEnhance();
        dirty_ = true;
        return;
    }

    if (scan_rect_.contains((float)x, (float)y)) {
        lastTapTime_ = -1.0;
        export_note_.clear();
        startScan(true);
        dirty_ = true;
        return;
    }

    if (shape_rect_.contains((float)x, (float)y)) {
        lastTapTime_ = -1.0;
        export_note_.clear();
        startScan(false);
        dirty_ = true;
        return;
    }

    if (next_rect_.contains((float)x, (float)y)) {
        lastTapTime_ = -1.0;
        export_note_.clear();
        confirmScan();
        dirty_ = true;
        return;
    }

    if (export_rect_.contains((float)x, (float)y)) {
        lastTapTime_ = -1.0;
        doExport(/*jpeg=*/false);
        return;
    }

    if (jpg_rect_.contains((float)x, (float)y)) {
        lastTapTime_ = -1.0;
        doExport(/*jpeg=*/true);
        return;
    }

    const double t = nowSeconds();
    if (lastTapTime_ >= 0.0 && (t - lastTapTime_) <= kDoubleTapGap &&
        distance((float)x, (float)y, lastTapX_, lastTapY_) <= kDoubleTapSlop) {
        view_.toggleFitOneToOne((float)x, (float)y);
        lastTapTime_ = -1.0;          // consumed; a third tap starts over
        dirty_ = true;
        return;
    }
    lastTapTime_ = t;
    lastTapX_ = (float)x;
    lastTapY_ = (float)y;
}

// ── Drawing ─────────────────────────────────────────────────────────────────

void ViewMageApp::draw() {
    curves_.clear();
    images_.clear();
    frontShapes_.clear();

    const SafeInsets in = host_->safeInsets();
    const Font* fontPtr = font_.ftFace ? &font_ : nullptr;
    Canvas canvas(curves_, renderer_->width(), renderer_->height(), fontPtr,
                  (float)in.top, (float)in.bottom, (float)in.left, (float)in.right);
    // FOREGROUND, not background. ImageLayer's background pass runs BEFORE the
    // vector overlay composites, which is right for album art sitting behind
    // UI chrome — and wrong here, because clear() above is itself an overlay
    // rect covering the whole screen, so a background image is painted over by
    // the very thing that draws the letterbox around it. The photograph is not
    // chrome's backdrop; it is the subject, and it belongs on top.
    canvas.useImagesFg(&images_);
    canvas.clear(kBackground);

    if (state_ == State::kReady && texture_ != kInvalidTexture) {
        // Exposure is a view parameter, not a property of the image: it rides
        // to the GPU as four floats in a push constant, so dragging the slider
        // never re-decodes and never re-uploads.
        canvas.setImageTone(std::exp2(ev_), toneMode_, white_, clipWarn_);
        // Ignored outright on an SDR swapchain, so this is unconditional. On an
        // HDR one it is what stops clipWarn striping every highlight the panel
        // can genuinely show -- and what lets the rolloff reach them at all.
        canvas.setImageHdr(kGraphicsWhiteNits, headroom_);
        const ViewTransform::Quad q = view_.quad();
        // Full-screen coordinates deliberately, NOT the inset content box: a
        // photo should run under a display cutout rather than be letterboxed
        // away from it. The cutout is glass over the picture, not a margin.
        canvas.imageFg(texture_, q.x, q.y, q.w, q.h);
        canvas.clearImageTone();   // the error text below is not an image
    }

    // Corners and the context window sit on the photograph. The POST bar is
    // drawn afterwards so Next stays pressable when the window would cover it.
    float barTopForLoupe = canvas.top() + canvas.h();
    if (state_ == State::kReady && scan_adjust_) {
        const float size = std::max(22.0f, canvas.w() * 0.028f);
        const float barH = size * 2.4f;
        const SafeInsets sys = host_->safeInsets();
        const float navGuard = std::max((float)sys.bottom, canvas.h() * 0.05f);
        barTopForLoupe = canvas.top() + canvas.h() - barH - navGuard;
        // The photograph is a foreground image, composited after the curve
        // overlay, so corner marks emitted as curves land underneath it.
        // The shape list is drawn after that image.
        canvas.useShapes(&frontShapes_);
        drawScanChrome(canvas, barTopForLoupe);
        canvas.useShapes(nullptr);
    } else if (state_ == State::kError) {
        const float size = std::max(28.0f, canvas.w() * 0.045f);
        canvas.textCentered(message_, canvas.left() + canvas.w() * 0.5f,
                            canvas.top() + canvas.h() * 0.5f - size * 0.65f,
                            size, kMessage);
    }

    // ── The POST bar ────────────────────────────────────────────────────────
    //
    // One line of status and one button, and that is the whole UI on purpose:
    // the render is automatic, so there is nothing to adjust and no editor to
    // open. What is left is knowing whether the good version has arrived, and
    // leaving with a file. Anything more would be a photo editor, which this
    // deliberately is not.
    if (state_ == State::kReady) {
        const float size = std::max(22.0f, canvas.w() * 0.028f);
        const float pad  = size * 0.7f;
        const float barH = size * 2.4f;

        // ── Keep clear of the system gesture strip ──────────────────────────
        //
        // A control flush against the bottom of the surface CANNOT BE PRESSED.
        // Measured on an S23 Ultra: the surface is 1440x2963 while the screen is
        // 1440x3088, so the window sits 125 px down and a bar at the surface's
        // bottom edge lands at screen y 3003..3076 -- inside the gesture-
        // navigation area, which swallows the touch before the app sees it.
        // The button was drawn, looked pressable, and was completely dead.
        //
        // `safeInsets()` reports 0 on all four edges here, so it cannot be used
        // for this; the real fix is for app_shell to report the navigation-bar
        // inset, and until it does this margin is the honest stand-in. It is a
        // fraction of the surface rather than a pixel count so it survives a
        // different density.
        const SafeInsets sys = host_->safeInsets();
        const float navGuard = std::max((float)sys.bottom, canvas.h() * 0.05f);
        const float barY = canvas.top() + canvas.h() - barH - navGuard;

        canvas.rect(canvas.left(), barY, canvas.w(), barH, kBar);

        // Laid out from the right edge: Save, then Denoise beside it.
        auto button = [&](const std::string& label, float rightOf, bool live,
                          Rect& hit) {
            const float tw = std::max(canvas.textWidth(label, size), size * 4.0f);
            const float bw = tw + pad * 2.0f;
            const float bx = rightOf - bw - pad;
            const float by = barY + (barH - size * 1.8f) * 0.5f;
            canvas.rect(bx, by, bw, size * 1.8f, live ? kButton : kButtonOff,
                        size * 0.4f);
            // textCentered's y is the text TOP (the baseline is y + size), so
            // the label sits at by + 0.25*size with its baseline at the button
            // centre + cap-height/2 — not a full size lower than intended.
            canvas.textCentered(label, bx + bw * 0.5f, by + size * 0.25f, size,
                                live ? kMessage : kMessageOff);
            hit = live ? Rect{bx, by, bw, size * 1.8f} : Rect{0, 0, 0, 0};
            return bx;
        };

        const bool refining = refine_thread_.joinable() && !refined_ready_.load();
        const bool enhancing = enhance_thread_.joinable() && !enhanced_ready_.load();
        const bool scanning = scan_thread_.joinable() && !scan_ready_.load();
        const bool busy = refining || enhancing || scanning;
        const float right = canvas.left() + canvas.w();

        // Save stays available on the plain develop — someone who does not want
        // to wait should still be able to leave with their photo. It goes inert
        // only while a worker is IN FLIGHT, because exporting then writes the
        // un-processed pixels. JPG is the explicit second choice: same pixels,
        // same render, chat-app sized. Two buttons, no hidden long-press.
        const float saveX = button("Save PNG", right, !exporting_ && !busy,
                                   export_rect_);

        // Denoise is offered until it has been done; afterwards the status reads
        // "Denoised" and there is nothing left to press.
        float cursorX = saveX;
        if (!model_bytes_.empty() && !denoised_) {
            // Cancel is the way back. While the worker is in flight every
            // other button is inert, so this one has to stay live.
            if (refining)
                cursorX = button("Cancel", cursorX, true, denoise_rect_);
            else
                cursorX = button("Denoise", cursorX, !busy && !exporting_, denoise_rect_);
        } else {
            denoise_rect_ = {0, 0, 0, 0};
        }

        // Enhance is offered until it has been done. Recommended order is
        // Denoise first, then Enhance — but both buttons are available from
        // the start so the user can enhance a JXL or skip denoise entirely.
        if (!enhanced_) {
            cursorX = button("Enhance", cursorX, !busy && !exporting_, enhance_rect_);
        } else {
            enhance_rect_ = {0, 0, 0, 0};
        }

        // Rect is a rectangular sheet: the four corners are the page, and
        // Next pulls bends flat into that rectangle. Shape is any other
        // outline, and the outside of it stays white. Next replaces both.
        if (scan_adjust_) {
            scan_rect_ = {0, 0, 0, 0};
            shape_rect_ = {0, 0, 0, 0};
            cursorX = button("Next", cursorX, !busy && !exporting_, next_rect_);
        } else {
            next_rect_ = {0, 0, 0, 0};
            const bool canScan = !busy && !exporting_ && !pixels_.linear.empty();
            cursorX = button("Rect", cursorX, canScan, scan_rect_);
            cursorX = button("Shape", cursorX, canScan, shape_rect_);
        }

        if (!busy && !exporting_) {
            // cursorX MUST take this return: Save JPG extends further left
            // than anything drawn so far, and the status zone below is cut at
            // cursorX. Dropping it leaves the status parked ON this button.
            cursorX = button("Save JPG", cursorX, true, jpg_rect_);
        } else {
            jpg_rect_ = {0, 0, 0, 0};
        }

        // Logged once: the only way to confirm where a control ACTUALLY is on an
        // HDR surface, whose screencap comes back all zeroes.
        static bool logged = false;
        if (!logged && export_rect_.w > 0.0f) {
            logged = true;
            VCE_LOGI("ViewMage", "Save at %.0f,%.0f %.0fx%.0f; JPG at "
                     "%.0f,%.0f %.0fx%.0f; Denoise at %.0f,%.0f %.0fx%.0f; "
                     "Enhance at %.0f,%.0f %.0fx%.0f; Rect at %.0f,%.0f %.0fx%.0f",
                     export_rect_.x, export_rect_.y, export_rect_.w,
                     export_rect_.h,
                     jpg_rect_.x, jpg_rect_.y, jpg_rect_.w, jpg_rect_.h,
                     denoise_rect_.x, denoise_rect_.y,
                     denoise_rect_.w, denoise_rect_.h,
                     enhance_rect_.x, enhance_rect_.y,
                     enhance_rect_.w, enhance_rect_.h,
                     scan_rect_.x, scan_rect_.y, scan_rect_.w, scan_rect_.h);
        }

        // Status: right-anchored to the button cluster's left edge and centred
        // in the zone that leaves, so it can never paint over a button — with
        // all five buttons up, the row spans most of the surface width. Long
        // payloads (the export path) ellipsize instead of colliding.
        const float statusRight = cursorX - pad;
        const float statusW = std::max(
            0.0f, std::min(canvas.w() * 0.32f, statusRight - canvas.left()));
        if (statusW > 0.0f) {
            const std::string status = truncateToWidth(
                canvas, statusLine(), statusW, size, FontStyle::Roman);
            canvas.textCentered(status, canvas.left() + statusW * 0.5f,
                                barY + size * 0.55f, size, kMessage);
        }
    }

    renderer_->draw(curves_, /*overlay_rotation_deg=*/0,
                    /*images=*/{}, /*foregroundImages=*/images_,
                    /*msdfQuads=*/{}, /*shapeVerts=*/{}, frontShapes_);
}
