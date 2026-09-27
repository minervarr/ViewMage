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
#include "font.hh"
#ifdef __ANDROID__
#include "activity_bridge.hh"
#endif
#include "output_target.hh"
#include "renderer.hh"
#include "texture.hh"

#include "core/jxl_image.hh"
#include "core/dng_image.hh"
#include "core/ai_denoise.hh"
#include "core/enhance.hh"
#include "core/document_scan.hh"
#include "core/outline_place.hh"
#include "png_export.hh"
#include "core/view_transform.hh"

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

    // The window dies when the screen turns off and comes back as a new one.
    // The device and the uploaded photo stay; onSurfaceRecreated rebuilds the
    // swapchain. pixels_ is the fallback when that surface is incompatible.
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
    void        doExport(bool jpeg = false);

    struct Rect { float x, y, w, h; bool contains(float px, float py) const {
        return w > 0.0f && px >= x && px <= x + w && py >= y && py <= y + h; } };
    Rect        export_rect_{0, 0, 0, 0};   // set by draw(), read by the touch
    Rect        jpg_rect_{0, 0, 0, 0};
    Rect        denoise_rect_{0, 0, 0, 0};
    Rect        enhance_rect_{0, 0, 0, 0};
    Rect        scan_rect_{0, 0, 0, 0};
    Rect        shape_rect_{0, 0, 0, 0};
    Rect        next_rect_{0, 0, 0, 0};
    bool        scan_page_ = false;         // Rect rebuilds a rectangle; Shape does not
    bool        denoised_ = false;          // a refinement has landed
    bool        enhanced_ = false;          // an enhance has landed
    bool        scanned_  = false;          // a page has been flattened
    bool        exporting_ = false;
    std::string export_note_;               // shown once a file has been written

    void startRefinement();
    void pollRefinement();
    void joinRefinement();          // safe to call twice; called from shutdown

    // ── Enhance ───────────────────────────────────────────────────────────────
    void startEnhance();
    void pollEnhancement();
    void joinEnhancement();         // safe to call twice; called from shutdown

    // Scan finds the outline and waits. A tap on a line adds a straight
    // point. The point moves; the stick bends it, and the stick brought
    // back onto the point straightens it. Next warps the page.
    // The warp stays off the UI thread.
    void startScan(bool rectangle);
    void confirmScan();
    void pollScan();
    void joinScan();
    void drawScanChrome(Canvas& canvas, float barTop);
    int  cornerAt(float x, float y) const;
    int  pointAt(float x, float y) const;
    int  handleAt(float x, float y) const;
    bool segmentAt(float sx, float sy, int& after, float& ix, float& iy) const;
    bool segmentScreen(int after, float& x0, float& y0, float& x1, float& y1) const;
    bool barContains(float x, float y) const;
    void ensureAdjustPoints();
    void updatePlacement(float fx, float fy);
    void commitPlacement();
    bool pointRemoveArmed() const;
    void handleVector(int index, float& hx, float& hy) const;
    void syncShapeFromPoints();
    void insertOutlinePoint(int after, float ix, float iy,
                            PagePointKind kind, float hx, float hy);
    void screenToImage(float sx, float sy, float& ix, float& iy) const;
    void imageToScreen(float ix, float iy, float& sx, float& sy) const;
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
    Font                 font_;                // loaded from APK assets in create()

    std::vector<uint8_t>         model_bytes_;   // the ONNX file, read on the main
                                                 // thread, parsed on the worker
    std::unique_ptr<RawDenoiser> denoiser_;      // built ON the worker thread
    std::thread                  refine_thread_;
    std::mutex                   refined_mutex_;
    DecodedImage                 refined_;       // guarded by refined_mutex_
    std::atomic<bool>            refined_ready_{false};
    std::atomic<bool>            refine_abort_{false};
    std::atomic<float>           refine_progress_{0.0f};
    DecodedImage                 pendingDenoised_;  // set when refinement finishes while surface is lost

    // Enhance worker — same pattern as the denoise: runs on a thread, result
    // swapped in on the main thread in pollEnhancement().
    std::thread                  enhance_thread_;
    std::mutex                   enhanced_mutex_;
    DecodedImage                 enhance_result_;   // guarded by enhanced_mutex_
    std::atomic<bool>            enhanced_ready_{false};
    std::atomic<bool>            enhance_abort_{false};
    EnhanceParams                enhance_params_;   // tuning knobs
    DecodedImage                 pendingEnhanced_;   // enhance result stored while surface is lost

    std::thread                  scan_thread_;
    std::mutex                   scan_mutex_;
    DecodedImage                 scan_result_;      // guarded by scan_mutex_
    std::atomic<bool>            scan_ready_{false};
    std::atomic<bool>            scan_abort_{false};
    std::string                  scan_error_;       // guarded by scan_mutex_
    DocumentQuad                 scan_quad_{};      // guarded by scan_mutex_
    DocumentOutline              scan_outline_{};   // guarded by scan_mutex_
    bool                         scan_has_quad_ = false; // guarded by scan_mutex_
    bool                         scan_flattening_ = false; // job the worker was given
    DecodedImage                 pendingScan_;      // scan result while surface is lost

    // Outline edit, in source pixels. Shown after detection until Next.
    // Squares are corners, diamonds are fold points, circles are curve knots
    // with a tangent handle.
    bool  scan_adjust_ = false;
    DocumentShape   adjust_shape_ = DocumentShape::Quad;
    DocumentOutline adjust_outline_{};
    float corner_x_[4] = {};
    float corner_y_[4] = {};
    int   drag_corner_ = -1;
    int   drag_pointer_ = -1;
    bool  drag_handle_ = false;
    float finger_x_ = 0.0f;
    float finger_y_ = 0.0f;
    // A dot that does not exist yet. It is always a straight point. The stick
    // on it is what bends it into a curve, afterwards.
    bool  placing_ = false;
    int   place_pointer_ = -1;
    int   place_after_ = -1;
    float place_ghost_sx_ = 0.0f;
    float place_ghost_sy_ = 0.0f;
    float place_ix_ = 0.0f;
    float place_iy_ = 0.0f;
    int    last_point_tap_ = -1;
    double last_point_tap_time_ = -1.0;

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
    std::vector<float>     frontShapes_;   // corner marks, drawn above the photo
    std::vector<ImageDraw> images_;

    std::vector<Pointer> pointers_;
    float  lastPinchDist_ = 0.0f;
    float  lastMidX_ = 0.0f, lastMidY_ = 0.0f;
    double lastTapTime_ = -1.0;
    float  lastTapX_ = 0.0f, lastTapY_ = 0.0f;

    bool running_ = true;
    bool dirty_   = true;
    bool surfaceOk_ = true;
    bool keepAwake_ = false;   // host flag; true while a long worker is in flight
};
