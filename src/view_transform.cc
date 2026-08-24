#include "view_transform.hh"

#include <algorithm>
#include <cmath>

float ViewTransform::fitScale() const {
    if (!valid()) return 1.0f;
    return std::min((float)vw_ / (float)iw_, (float)vh_ / (float)ih_);
}

void ViewTransform::setImage(int w, int h) {
    iw_ = w > 0 ? w : 0;
    ih_ = h > 0 ? h : 0;
    fit();
}

void ViewTransform::setViewport(int w, int h) {
    const bool wasFit = atFit();
    vw_ = w > 0 ? w : 0;
    vh_ = h > 0 ? h : 0;
    if (!valid()) return;
    // A rotation while fitted refits; a rotation while zoomed keeps the zoom.
    // Refitting unconditionally would throw away a crop the user chose, and
    // keeping the scale unconditionally would leave a fitted image no longer
    // filling the new shape.
    if (wasFit) fit(); else clampAll();
}

void ViewTransform::fit() {
    if (!valid()) { scale_ = 1.0f; ox_ = oy_ = 0; return; }
    scale_ = fitScale();
    ox_ = ((float)vw_ - (float)iw_ * scale_) * 0.5f;
    oy_ = ((float)vh_ - (float)ih_ * scale_) * 0.5f;
}

bool ViewTransform::atFit() const {
    if (!valid()) return true;
    const float f = fitScale();
    // Relative epsilon: fit scales span orders of magnitude between a thumbnail
    // and a 100-megapixel photo, so a fixed one is wrong at both ends.
    return std::fabs(scale_ - f) <= f * 1e-4f;
}

void ViewTransform::pan(float dx, float dy) {
    if (!valid()) return;
    ox_ += dx;
    oy_ += dy;
    clampAll();
}

void ViewTransform::zoomAbout(float cx, float cy, float ratio) {
    if (!valid() || !(ratio > 0.0f)) return;
    const float target = std::clamp(scale_ * ratio, minScale(), maxScale());
    if (target == scale_) return;
    // Keep the image point under (cx,cy) under it: the offset moves so that
    // (c - o) scales with the image.
    const float k = target / scale_;
    ox_ = cx - (cx - ox_) * k;
    oy_ = cy - (cy - oy_) * k;
    scale_ = target;
    clampAll();
}

void ViewTransform::toggleFitOneToOne(float cx, float cy) {
    if (!valid()) return;
    if (!atFit()) { fit(); return; }
    // 1:1 is only a zoom IN when the image was being shown shrunk. For an
    // image smaller than the viewport, fit already magnifies it and 1:1 would
    // be smaller — below the floor, and visibly wrong. Stay put.
    if (fitScale() >= 1.0f) return;
    zoomAbout(cx, cy, 1.0f / scale_);
}

void ViewTransform::clampAll() {
    if (!valid()) return;
    scale_ = std::clamp(scale_, minScale(), maxScale());

    const float dw = (float)iw_ * scale_;
    const float dh = (float)ih_ * scale_;

    // Bigger than the viewport on this axis: the edges may not come inside the
    // viewport's, so the offset lives in [v - d, 0]. Smaller: centred, because
    // there is no crop to preserve and a loose small image drifting around its
    // gap is noise, not freedom.
    ox_ = (dw <= (float)vw_) ? ((float)vw_ - dw) * 0.5f
                             : std::clamp(ox_, (float)vw_ - dw, 0.0f);
    oy_ = (dh <= (float)vh_) ? ((float)vh_ - dh) * 0.5f
                             : std::clamp(oy_, (float)vh_ - dh, 0.0f);
}

ViewTransform::Quad ViewTransform::quad() const {
    if (!valid()) return {};
    return Quad{ox_, oy_, (float)iw_ * scale_, (float)ih_ * scale_};
}
