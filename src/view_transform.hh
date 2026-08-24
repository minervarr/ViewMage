// view_transform.hh — where the image sits on screen, and nothing else.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// ---------------------------------------------------------------------------
// The whole interaction model of ViewMage, expressed as pure arithmetic. This
// file includes no Vulkan, no Android, no libjxl and nothing of app_shell's:
// it knows an image size, a viewport size, a scale and an offset, and it can
// be compiled and tested on a desktop with no device and no GPU. That is the
// point — the feel of a viewer is entirely in these rules, so this is the one
// part that must be provable rather than eyeballed.
// ---------------------------------------------------------------------------
#pragma once

class ViewTransform {
public:
    struct Quad { float x = 0, y = 0, w = 0, h = 0; };

    // Zoom ceiling, as a multiple of the fit scale rather than an absolute
    // one. An absolute cap means a 200x200 icon on a phone screen can be
    // magnified far less than a 6000px photo, since the photo starts scaled
    // DOWN and the icon starts scaled UP — the same number would mean two
    // different amounts of "closer than it opened".
    static constexpr float kMaxZoomOverFit = 32.0f;

    // Both are in pixels and both may be called at any time. setViewport() on
    // a rotation keeps the current scale and re-clamps, rather than refitting:
    // a user who zoomed in and turned the phone wants the same crop, not a
    // reset.
    void setImage(int w, int h);
    void setViewport(int w, int h);

    // Contain-fit and centre. The state on open, and after the image changes.
    void fit();

    // Translate by a screen-pixel delta, then clamp.
    void pan(float dx, float dy);

    // Multiply the scale by `ratio` about the screen point (cx, cy), keeping
    // whatever part of the image is under that point under it afterwards.
    // That fixed point is what makes a pinch feel attached to the fingers
    // rather than to the middle of the screen.
    void zoomAbout(float cx, float cy, float ratio);

    // Double-tap: fit becomes 1:1 about the tap, anything else becomes fit.
    // When the image is smaller than the viewport, 1:1 would be a zoom OUT
    // from fit and is below the floor, so this stays at fit — a double-tap
    // that visibly shrinks the picture reads as a bug.
    void toggleFitOneToOne(float cx, float cy);

    // The destination rectangle, in screen pixels, for this frame.
    Quad quad() const;

    float scale()    const { return scale_; }
    float fitScale() const;
    float minScale() const { return fitScale(); }
    float maxScale() const { return fitScale() * kMaxZoomOverFit; }
    bool  atFit()    const;
    bool  valid()    const { return iw_ > 0 && ih_ > 0 && vw_ > 0 && vh_ > 0; }

private:
    // Clamps the scale into [minScale, maxScale], then the offset so the image
    // can neither be flung off-screen nor leave a gap at an edge it is big
    // enough to cover. Centring on an axis the image does not fill is part of
    // the same rule, not a special case: it is what "no gap on either side"
    // means when both sides have gaps.
    void clampAll();

    int   iw_ = 0, ih_ = 0;     // image pixels
    int   vw_ = 0, vh_ = 0;     // viewport pixels
    float scale_ = 1.0f;
    float ox_ = 0, oy_ = 0;     // image top-left, in screen pixels
};
