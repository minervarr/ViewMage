#include "viewmage_app.hh"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "log.hh"

namespace {

// The viewer's whole palette. A photo is the subject; anything else on screen
// competes with it, so there is nothing else on screen.
constexpr Color kBackground{0.06f, 0.06f, 0.07f, 1.0f};
constexpr Color kMessage   {0.85f, 0.85f, 0.88f, 1.0f};

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
    releaseTexture();
    renderer_.reset();
}

double ViewMageApp::nowSeconds() const {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

bool ViewMageApp::create() {
    if (!host_ || !host_->init(this)) return false;
    renderer_ = std::make_unique<Renderer>(host_->surfaceProvider(), host_->assetReader());
    syncViewport();
    return true;
}

void ViewMageApp::run() {
    while (running_) {
        // Nothing decided before pump() may be trusted after it: pump() is
        // where Android says "your surface is gone". So the draw decision is
        // made entirely below this line.
        host_->pump(/*haveWork=*/dirty_);
        if (host_->quitRequested()) break;
        if (!running_) break;

        if (dirty_ && surfaceOk_ && renderer_) {
            draw();
            dirty_ = false;
        }
    }
}

void ViewMageApp::shutdown() { running_ = false; }

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

    pixels_ = decode_jxl(source_.data(), source_.size(), maxDim);
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

    if (!uploadTexture()) {
        state_   = State::kError;
        message_ = "Not enough graphics memory";
        return;
    }
    releasePixels();

    view_.setImage((int)pixels_.w, (int)pixels_.h);
    syncViewport();
    view_.fit();
    state_ = State::kReady;
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

void ViewMageApp::onSurfaceLost() {
    // The handle belongs to a device that is going away. Drop it, keep the
    // pixels: pixels_ is the expensive thing and it is still perfectly good.
    texture_  = kInvalidTexture;
    surfaceOk_ = false;
}

bool ViewMageApp::onSurfaceRecreated() {
    surfaceOk_ = true;
    syncViewport();

    // The decoded pixels were handed back after the first upload, so there is
    // nothing in RAM to re-upload — decode again from the compressed source.
    //
    // That is the deliberate trade for this format: at 16 bytes a pixel,
    // keeping the buffer resident against the chance of a surface loss costs
    // hundreds of megabytes for the entire time the app is open, while
    // decoding again costs a fraction of a second on the rare occasion it
    // actually happens. The user's own exposure is preserved across it, which
    // is the part they would notice.
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
    // font = nullptr: the stroke-fallback glyphs from the font engine's
    // glyphs.cc. ViewMage's happy path draws no text at all, and the failure
    // path draws one short line — carrying an MSDF atlas and a font file in
    // the APK to set five words would be weight for nothing.
    Canvas canvas(curves_, renderer_->width(), renderer_->height(), nullptr,
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

    renderer_->draw(curves_, /*overlay_rotation_deg=*/0,
                    /*images=*/{}, /*foregroundImages=*/images_);
}
