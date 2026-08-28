// view_transform_test.cc — the whole feel of the viewer, checked without a GPU.
//
// Plain assert(), Debug-only, no framework: the convention this family of
// repositories uses. NDEBUG is undefined first so the asserts survive a
// Release build, exactly as vk_canvas's own tests do.
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>

#include "core/view_transform.hh"

static bool near(float a, float b, float eps = 0.01f) {
    return std::fabs(a - b) <= eps;
}

// ── fit ─────────────────────────────────────────────────────────────────────
static void test_fit_wide_image_in_tall_viewport() {
    ViewTransform t;
    t.setViewport(1000, 2000);
    t.setImage(2000, 1000);          // 2:1 image, 1:2 viewport → width-limited
    assert(near(t.scale(), 0.5f));
    auto q = t.quad();
    assert(near(q.w, 1000.0f) && near(q.h, 500.0f));
    assert(near(q.x, 0.0f));         // fills the width exactly
    assert(near(q.y, 750.0f));       // and is centred vertically
    assert(t.atFit());
}

static void test_fit_tall_image_in_wide_viewport() {
    ViewTransform t;
    t.setViewport(2000, 1000);
    t.setImage(1000, 2000);          // height-limited
    assert(near(t.scale(), 0.5f));
    auto q = t.quad();
    assert(near(q.w, 500.0f) && near(q.h, 1000.0f));
    assert(near(q.x, 750.0f) && near(q.y, 0.0f));
}

static void test_fit_square_in_square() {
    ViewTransform t;
    t.setViewport(800, 800);
    t.setImage(400, 400);            // smaller than the viewport → fit MAGNIFIES
    assert(near(t.scale(), 2.0f));
    auto q = t.quad();
    assert(near(q.w, 800.0f) && near(q.h, 800.0f));
    assert(near(q.x, 0.0f) && near(q.y, 0.0f));
}

// Aspect ratio is never distorted, whatever the two shapes are.
static void test_fit_never_stretches() {
    const int cases[][4] = {{1000,2000, 3000,1000}, {2000,1000, 1000,3000},
                            {640,480, 1080,1920},   {1080,1920, 640,480}};
    for (auto& c : cases) {
        ViewTransform t;
        t.setViewport(c[0], c[1]);
        t.setImage(c[2], c[3]);
        auto q = t.quad();
        assert(near(q.w / q.h, (float)c[2] / (float)c[3], 0.001f));
        // …and it is contained, never cropped.
        assert(q.w <= c[0] + 0.01f && q.h <= c[1] + 0.01f);
    }
}

// ── pan clamping, all four edges ────────────────────────────────────────────
static void test_pan_clamps_at_every_edge() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(1000, 1000);
    t.zoomAbout(500, 500, 4.0f);     // 4000x4000 on screen — larger on both axes
    assert(near(t.scale(), 4.0f));

    t.pan(100000, 100000);           // drag far past the top-left
    auto q = t.quad();
    assert(near(q.x, 0.0f) && near(q.y, 0.0f));      // edge stops AT the viewport edge

    t.pan(-100000, -100000);         // and far past the bottom-right
    q = t.quad();
    assert(near(q.x, 1000.0f - 4000.0f));
    assert(near(q.y, 1000.0f - 4000.0f));
}

// An image that does not fill an axis stays centred on it, however hard it is
// dragged — a small picture sliding around its own margin reads as broken.
static void test_pan_keeps_small_axis_centred() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(2000, 100);           // fit → 1000x50, short on the Y axis
    t.zoomAbout(500, 500, 4.0f);     // 4000x200: wide enough to pan, still short
    const float yBefore = t.quad().y;
    t.pan(0, 500);
    assert(near(t.quad().y, yBefore));
    t.pan(0, -500);
    assert(near(t.quad().y, yBefore));
}

// ── zoom about a focal point ────────────────────────────────────────────────
// The defining property of a pinch: whatever pixel was under the fingers is
// still under them afterwards.
static void test_zoom_keeps_focal_point_stationary() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(1000, 1000);

    const float cx = 300, cy = 700;
    auto before = t.quad();
    const float u = (cx - before.x) / before.w;      // image-relative coords
    const float v = (cy - before.y) / before.h;

    t.zoomAbout(cx, cy, 3.0f);

    auto after = t.quad();
    assert(near(after.x + u * after.w, cx, 0.5f));
    assert(near(after.y + v * after.h, cy, 0.5f));
}

// ── the floor and the ceiling ───────────────────────────────────────────────
static void test_zoom_floor_is_fit() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(4000, 4000);
    const float f = t.fitScale();
    t.zoomAbout(500, 500, 0.001f);   // try to pinch far below fit
    assert(near(t.scale(), f, f * 1e-3f));
    assert(t.atFit());
}

static void test_zoom_ceiling_is_32x_fit() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(4000, 4000);
    const float f = t.fitScale();
    t.zoomAbout(500, 500, 1e6f);
    assert(near(t.scale(), f * ViewTransform::kMaxZoomOverFit, f * 0.01f));
    // and it does not creep past it on a second attempt
    t.zoomAbout(500, 500, 2.0f);
    assert(near(t.scale(), f * ViewTransform::kMaxZoomOverFit, f * 0.01f));
}

// ── double tap ──────────────────────────────────────────────────────────────
static void test_double_tap_round_trips() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(4000, 4000);          // fit = 0.25, so 1:1 is a real zoom in
    assert(t.atFit());

    t.toggleFitOneToOne(500, 500);
    assert(near(t.scale(), 1.0f));
    assert(!t.atFit());

    t.toggleFitOneToOne(500, 500);
    assert(t.atFit());
    assert(near(t.scale(), 0.25f));
}

// For an image already magnified by fit, 1:1 would be SMALLER than fit and is
// below the floor. Double-tap must do nothing rather than shrink it.
static void test_double_tap_noop_when_image_smaller_than_viewport() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(200, 200);            // fit = 5.0
    t.toggleFitOneToOne(500, 500);
    assert(t.atFit());
    assert(near(t.scale(), 5.0f));
}

// Any zoomed state returns to fit, not to some intermediate.
static void test_double_tap_from_arbitrary_zoom_returns_to_fit() {
    ViewTransform t;
    t.setViewport(1000, 1000);
    t.setImage(4000, 2000);
    t.zoomAbout(100, 900, 7.3f);
    assert(!t.atFit());
    t.toggleFitOneToOne(100, 900);
    assert(t.atFit());
}

// ── viewport changes (rotation) ─────────────────────────────────────────────
static void test_rotation_refits_when_fitted() {
    ViewTransform t;
    t.setViewport(1080, 1920);
    t.setImage(4000, 3000);
    assert(t.atFit());
    t.setViewport(1920, 1080);       // turned the phone
    assert(t.atFit());               // still fitted, to the NEW shape
    auto q = t.quad();
    assert(q.w <= 1920.01f && q.h <= 1080.01f);
    assert(near(q.w, 1440.0f, 1.0f) && near(q.h, 1080.0f, 1.0f));
}

static void test_rotation_keeps_zoom_when_zoomed() {
    ViewTransform t;
    t.setViewport(1080, 1920);
    t.setImage(4000, 3000);
    t.zoomAbout(540, 960, 3.0f);
    const float s = t.scale();
    t.setViewport(1920, 1080);
    assert(near(t.scale(), s, s * 0.01f));   // the crop the user chose survives
}

// ── degenerate input never crashes and never divides by zero ────────────────
static void test_degenerate_states_are_safe() {
    ViewTransform t;                       // nothing set at all
    assert(!t.valid());
    t.pan(10, 10);
    t.zoomAbout(0, 0, 2.0f);
    t.toggleFitOneToOne(0, 0);
    auto q = t.quad();
    assert(near(q.w, 0.0f) && near(q.h, 0.0f));

    t.setViewport(1000, 1000);
    t.setImage(0, 0);                      // a decode that produced nothing
    assert(!t.valid());
    assert(near(t.quad().w, 0.0f));

    ViewTransform u;
    u.setImage(100, 100);                  // image before viewport
    assert(!u.valid());
    u.setViewport(1000, 500);
    assert(u.valid() && u.atFit());

    ViewTransform v;
    v.setViewport(1000, 1000);
    v.setImage(1000, 1000);
    v.zoomAbout(500, 500, 0.0f);           // a nonsense ratio is ignored
    assert(v.atFit());
    v.zoomAbout(500, 500, -1.0f);
    assert(v.atFit());
}

int main() {
    test_fit_wide_image_in_tall_viewport();
    test_fit_tall_image_in_wide_viewport();
    test_fit_square_in_square();
    test_fit_never_stretches();
    test_pan_clamps_at_every_edge();
    test_pan_keeps_small_axis_centred();
    test_zoom_keeps_focal_point_stationary();
    test_zoom_floor_is_fit();
    test_zoom_ceiling_is_32x_fit();
    test_double_tap_round_trips();
    test_double_tap_noop_when_image_smaller_than_viewport();
    test_double_tap_from_arbitrary_zoom_returns_to_fit();
    test_rotation_refits_when_fitted();
    test_rotation_keeps_zoom_when_zoomed();
    test_degenerate_states_are_safe();
    std::printf("view_transform_test: all passed\n");
    return 0;
}
