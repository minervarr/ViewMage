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

    if (!uploadTexture()) {
        state_   = State::kError;
        message_ = "Not enough graphics memory";
        return;
    }

    view_.setImage((int)pixels_.w, (int)pixels_.h);
    syncViewport();
    view_.fit();
    state_ = State::kReady;
    VCE_LOGI("ViewMage", "showing %ux%u", pixels_.w, pixels_.h);
}

bool ViewMageApp::uploadTexture() {
    if (!renderer_ || !pixels_.ok()) return false;
    releaseTexture();
    texture_ = renderer_->create_texture(pixels_.rgba.data(), pixels_.w, pixels_.h,
                                         /*mips=*/false);
    return texture_ != kInvalidTexture;
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
    if (state_ == State::kReady || pixels_.ok()) {
        if (!uploadTexture()) {
            state_   = State::kError;
            message_ = "Not enough graphics memory";
        }
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
        const ViewTransform::Quad q = view_.quad();
        // Full-screen coordinates deliberately, NOT the inset content box: a
        // photo should run under a display cutout rather than be letterboxed
        // away from it. The cutout is glass over the picture, not a margin.
        canvas.imageFg(texture_, q.x, q.y, q.w, q.h);
    } else if (state_ == State::kError) {
        const float size = std::max(28.0f, canvas.w() * 0.045f);
        canvas.textCentered(message_, canvas.left() + canvas.w() * 0.5f,
                            canvas.top() + canvas.h() * 0.5f, size, kMessage);
    }

    renderer_->draw(curves_, /*overlay_rotation_deg=*/0,
                    /*images=*/{}, /*foregroundImages=*/images_);
}
