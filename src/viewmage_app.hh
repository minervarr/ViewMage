// viewmage_app.hh — the AppView. Lifecycle, gestures, and one textured quad.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "app_view.hh"
#include "host.hh"
#include "canvas.hh"
#include "activity_bridge.hh"
#include "output_target.hh"
#include "renderer.hh"
#include "texture.hh"

#include "jxl_image.hh"
#include "view_transform.hh"

class ViewMageApp : public AppView {
public:
    // The host is injected rather than made here so android_main can hand us
    // one built from its `android_app*` — app_shell's rule that create()/run()
    // belong to the APP cuts both ways.
    explicit ViewMageApp(std::unique_ptr<Host> host);
    ~ViewMageApp() override;

    bool create();
    void run();

    // The compressed bytes to show, handed over before create(). Kept as the
    // app's own state because onHostReady() is where they may first be
    // decoded — the GPU limit that bounds the decode is not known until the
    // Renderer has picked a physical device.
    void setSource(std::vector<uint8_t> bytes) { source_ = std::move(bytes); }

    // ── AppView ─────────────────────────────────────────────────────────────
    void onHostResized() override;
    void shutdown() override;
    void onHostReady() override;

    // The CPU keeps the decoded pixels so that coming back to the app
    // re-uploads a texture instead of decoding again. That split — CPU state
    // survives, GPU state does not — is invisible until the second visit.
    void onSurfaceLost() override;
    bool onSurfaceRecreated() override;

    // Raw multi-pointer touch (the seam added to app_shell for this app).
    void onPointerDown(int pointerId, int x, int y) override;
    void onPointerMove(int pointerId, int x, int y) override;
    void onPointerUp(int pointerId, int x, int y) override;

private:
    enum class State { kLoading, kReady, kError };

    void loadFromSource();          // decode + upload; sets state_
    bool uploadTexture();           // pixels_ → GPU, degrading if it must
    void releaseTexture();
    // Drop the decoded pixels but keep everything we learned about them. At
    // 16 bytes a pixel this is the largest allocation in the app by two orders
    // of magnitude, and it is dead weight the moment the GPU has a copy.
    void releasePixels();
    void draw();
    void syncViewport();
    void refreshHeadroom();         // re-ask the display; it is a live value

    // Gesture state. Deliberately here and not in app_shell: a slop is a
    // property of a touch screen, but "two fingers mean zoom" is a property of
    // this application.
    struct Pointer { int id; float x, y; float startX, startY; double downTime; };
    Pointer* find(int id);
    void     forget(int id);
    double   nowSeconds() const;

    std::unique_ptr<Host> host_;
    std::unique_ptr<Renderer> renderer_;

    // Compressed bytes, kept for the whole session. They are a few megabytes
    // against the decoded buffer's few hundred, which is what makes "free the
    // pixels and decode again if the surface comes back" the cheap option.
    std::vector<uint8_t> source_;
    JxlImage             pixels_;   // .linear is freed after upload; the rest stays
    TextureHandle        texture_ = kInvalidTexture;

    // ── View state for the tone pipeline ────────────────────────────────────
    // Exposure in stops. Starts at the decoder's auto value and is what the
    // slider moves. The texture never changes; only a push constant does.
    float    ev_       = 0.0f;
    ToneMode toneMode_ = ToneMode::kPassthrough;
    float    white_    = 1.0f;

    // What the swapchain actually IS, resolved once in create(). Not what was
    // asked for: the request can be silently refused, and a viewer that assumed
    // otherwise would roll its highlights toward headroom that is not there.
    bool  hdr_      = false;
    float headroom_ = 1.0f;   // multiple of SDR white; 1.0 == none. LIVE — see
                              // refreshHeadroom(); it moves with the brightness.
    bool     clipWarn_ = false;

    ViewTransform view_;
    State         state_   = State::kLoading;
    std::string   message_;         // shown in kError

    std::vector<float>     curves_;
    std::vector<ImageDraw> images_;

    std::vector<Pointer> pointers_;
    float  lastPinchDist_ = 0.0f;
    float  lastMidX_ = 0.0f, lastMidY_ = 0.0f;
    double lastTapTime_ = -1.0;
    float  lastTapX_ = 0.0f, lastTapY_ = 0.0f;

    bool running_ = true;
    bool dirty_   = true;
    bool surfaceOk_ = true;
};
