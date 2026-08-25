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
#include "dng_image.hh"
#include "ai_denoise.hh"
#include "png_export.hh"
#include "view_transform.hh"

#include <atomic>
#include <mutex>
#include <thread>

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

    // ── Progressive refinement ──────────────────────────────────────────────
    //
    // The neural denoise is tens of seconds on a phone, which is far too long
    // to hold a blank screen for. So the ordinary develop is shown IMMEDIATELY
    // and the denoised one replaces it when it is ready. The user gets a
    // picture at once and a better picture shortly after, instead of a wait.
    //
    // Started only for DNGs, only when the model loaded. Everything it touches
    // is either its own or guarded by `refined_mutex_`; the swap itself happens
    // on the main thread in pollRefinement(), because uploading a texture is
    // not something a worker may do.
    // The POST bar: what it says, and what pressing it does.
    std::string statusLine() const;
    void        doExport();

    struct Rect { float x, y, w, h; bool contains(float px, float py) const {
        return w > 0.0f && px >= x && px <= x + w && py >= y && py <= y + h; } };
    Rect        export_rect_{0, 0, 0, 0};   // set by draw(), read by the touch
    bool        exporting_ = false;
    std::string export_note_;               // shown once a file has been written

    void startRefinement();
    void pollRefinement();
    void joinRefinement();          // safe to call twice; called from shutdown
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
    DecodedImage         pixels_;   // .linear is freed after upload; the rest stays

    // What the sensor said about its own noise, when the file was a DNG that
    // carried NoiseProfile. Kept rather than consumed: it is the input a
    // denoiser needs, and a denoiser that has to estimate the noise level
    // instead of being told it is the difference between cleaning an image and
    // inventing detail in it.
    NoiseModel           noise_;
    TextureHandle        texture_ = kInvalidTexture;

    std::vector<uint8_t>         model_bytes_;   // the ONNX file, read on the main
                                                 // thread, parsed on the worker
    std::unique_ptr<RawDenoiser> denoiser_;      // built ON the worker thread
    std::thread                  refine_thread_;
    std::mutex                   refined_mutex_;
    DecodedImage                 refined_;       // guarded by refined_mutex_
    std::atomic<bool>            refined_ready_{false};
    std::atomic<bool>            refine_abort_{false};
    std::atomic<float>           refine_progress_{0.0f};

    // The developed pixels are KEPT on the DNG path rather than freed after
    // upload. They are the largest allocation in the app, and they are also the
    // only thing "Save PNG" can write without spending another 26 seconds
    // re-denoising -- for a POST app whose whole purpose is leaving with a file,
    // that trade is the right way round. See releasePixels().
    bool                         keep_pixels_ = false;

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
