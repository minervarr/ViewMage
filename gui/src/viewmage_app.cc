#include "viewmage_app.hh"

#include "png_export.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

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
        const bool busy = refining || enhancing;
        // Also force a draw when a result is ready to pick up — without this,
        // refined_ready_==true makes refining==false, busy==false, pump blocks,
        // and pollRefinement() is never reached.
        const bool hasResult = refined_ready_.load() || enhanced_ready_.load();
        if (busy || hasResult) dirty_ = true;
        host_->pump(/*haveWork=*/dirty_ || busy || hasResult);
        // Don't exit while a worker is in flight.
        if (host_->quitRequested() && !busy) break;
        if (!running_) break;

        pollRefinement();
        pollEnhancement();

        if (dirty_ && surfaceOk_ && renderer_) {
            draw();
            dirty_ = false;
        }
    }
}

void ViewMageApp::shutdown() {
    running_ = false;
    joinEnhancement();
    joinRefinement();
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
        // Building the session is the expensive half -- on an S23 Ultra it is
        // minutes, not seconds, which is exactly why it lives on this thread.
        if (!denoiser_) {
            const double t0 = nowSeconds();
            std::string err;
            denoiser_ = RawDenoiser::load(model_bytes_.data(), model_bytes_.size(), err);
            if (!denoiser_) {
                VCE_LOGE("ViewMage", "denoiser unavailable: %s", err.c_str());
                return;
            }
            VCE_LOGI("ViewMage", "denoiser ready (%s) in %.1f s",
                     denoiser_->provider().c_str(), nowSeconds() - t0);
        }
        if (refine_abort_.load()) return;
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
        if (refine_abort_.load()) return;
        VCE_LOGI("ViewMage", "neural denoise: %.1f s for %ux%u",
                 nowSeconds() - t1, out.w, out.h);
        {
            std::lock_guard<std::mutex> lock(refined_mutex_);
            refined_ = std::move(out);
        }
        refined_ready_.store(true);
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
        const int pct = (int)(refine_progress_.load() * 100.0f + 0.5f);
        return "Denoising " + std::to_string(pct) + "%";
    }
    if (enhance_thread_.joinable() && !enhanced_ready_.load())
        return "Enhancing...";
    if (enhanced_)             return "Enhanced";
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
        VCE_LOGE("ViewMage", "neural denoise failed: %s", got.error.c_str());
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
    // Sharpening is for the plain develop only.
    EnhanceParams effective = params;
    if (denoised_) effective.sharpen = 0.0f;

    enhance_thread_ = std::thread([this, snap = std::move(snapshot), effective]() mutable {
        const double t0 = nowSeconds();
        enhanceImage(snap.linear, snap.w, snap.h, effective,
                     snap.widePrimaries);
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
    // The handle belongs to a device that is going away. Drop it, keep the
    // pixels: pixels_ is the expensive thing and it is still perfectly good.
    texture_  = kInvalidTexture;
    surfaceOk_ = false;
}

bool ViewMageApp::onSurfaceRecreated() {
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

    if (uploadPending(pendingEnhanced_, "enhanced", &enhanced_)) return true;
    if (uploadPending(pendingDenoised_, "denoised", &denoised_)) return true;

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
    if (pointers_.size() == 2) {
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
        if (lastPinchDist_ > 1.0f && dist > 1.0f)
            view_.zoomAbout(midX, midY, dist / lastPinchDist_);

        lastPinchDist_ = dist;
        lastMidX_ = midX;
        lastMidY_ = midY;
        dirty_ = true;
    }
}

void ViewMageApp::onPointerUp(int pointerId, int x, int y) {
    Pointer* p = find(pointerId);
    if (!p) return;

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

    if (!wasTap || state_ != State::kReady) return;

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
    } else if (state_ == State::kError) {
        const float size = std::max(28.0f, canvas.w() * 0.045f);
        canvas.textCentered(message_, canvas.left() + canvas.w() * 0.5f,
                            canvas.top() + canvas.h() * 0.5f, size, kMessage);
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
            canvas.textCentered(label, bx + bw * 0.5f, by + size * 1.25f, size,
                                live ? kMessage : kMessageOff);
            hit = live ? Rect{bx, by, bw, size * 1.8f} : Rect{0, 0, 0, 0};
            return bx;
        };

        const bool refining = refine_thread_.joinable() && !refined_ready_.load();
        const bool enhancing = enhance_thread_.joinable() && !enhanced_ready_.load();
        const bool busy = refining || enhancing;
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

        if (!busy && !exporting_) {
            button("Save JPG", cursorX, true, jpg_rect_);
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
                     "Enhance at %.0f,%.0f %.0fx%.0f",
                     export_rect_.x, export_rect_.y, export_rect_.w,
                     export_rect_.h,
                     jpg_rect_.x, jpg_rect_.y, jpg_rect_.w, jpg_rect_.h,
                     denoise_rect_.x, denoise_rect_.y,
                     denoise_rect_.w, denoise_rect_.h,
                     enhance_rect_.x, enhance_rect_.y,
                     enhance_rect_.w, enhance_rect_.h);
        }

        canvas.textCentered(statusLine(), canvas.left() + canvas.w() * 0.32f,
                            barY + barH * 0.5f + size * 0.35f, size, kMessage);
    }

    renderer_->draw(curves_, /*overlay_rotation_deg=*/0,
                    /*images=*/{}, /*foregroundImages=*/images_);
}
