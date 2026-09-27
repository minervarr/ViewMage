// outline_place_test.cc — the dot leads the finger, and the kind is geometry.
//
// A hold is not part of this. Curve vs sharp is the perpendicular pull at
// the moment you ask, which is what makes a tap on the line a vertex and a
// pull off it a curve.
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>

#include "core/outline_place.hh"

static bool near(float a, float b, float eps = 0.05f) {
    return std::fabs(a - b) <= eps;
}

// Off the middle, the lead runs toward the roomier side — the middle —
// and the dot is not under the contact.
static void test_lead_toward_middle() {
    const OutlineAim a = aimOnSegment(100.0f, 0.0f, 0.0f, 0.0f, 400.0f, 0.0f);
    assert(a.ok);
    assert(near(a.ghostX, 164.0f));
    assert(near(a.ghostY, 0.0f));
    assert(!a.curve);
    assert(near(a.fingerDist, 0.0f));
    assert(std::fabs(a.ghostX - 100.0f) > 32.0f);
}

// A tap in the middle has no room toward the middle. The dot steps toward
// the next corner instead of sitting under the finger.
static void test_middle_steps_aside() {
    const OutlineAim a = aimOnSegment(200.0f, 0.0f, 0.0f, 0.0f, 400.0f, 0.0f);
    assert(a.ok);
    assert(near(a.ghostX, 264.0f));
    assert(near(a.ghostY, 0.0f));
    assert(!a.curve);
}

// A short segment cannot host a 64px lead. The dot stays on the segment.
static void test_short_segment_shrinks_lead() {
    const OutlineAim a = aimOnSegment(20.0f, 0.0f, 0.0f, 0.0f, 40.0f, 0.0f);
    assert(a.ok);
    assert(a.ghostX >= 12.0f && a.ghostX <= 28.0f);
    assert(near(a.ghostY, 0.0f));
    assert(std::fabs(a.ghostX - 20.0f) < kOutlineLeadPx);
    assert(!a.curve);
}

// Under the pull threshold the dot is a sharp vertex. Past it, a curve.
// The knot stays on the segment either way. Nothing here waits.
static void test_pull_picks_the_kind() {
    const OutlineAim stay = aimOnSegment(100.0f, 20.0f, 0.0f, 0.0f, 400.0f, 0.0f);
    assert(stay.ok);
    assert(!stay.curve);
    assert(near(stay.ghostX, 164.0f));
    assert(near(stay.ghostY, 0.0f));
    assert(near(stay.perp, 20.0f));

    const OutlineAim bow = aimOnSegment(100.0f, 40.0f, 0.0f, 0.0f, 400.0f, 0.0f);
    assert(bow.ok);
    assert(bow.curve);
    assert(near(bow.ghostX, 164.0f));
    assert(near(bow.ghostY, 0.0f));
    assert(near(bow.pullX, 0.0f));
    assert(near(bow.pullY, 40.0f));
}

// The pull points at the finger on a vertical side too.
static void test_vertical_pull_points_at_the_finger() {
    const OutlineAim a = aimOnSegment(40.0f, 100.0f, 0.0f, 0.0f, 0.0f, 400.0f);
    assert(a.ok);
    assert(a.curve);
    assert(near(a.ghostX, 0.0f));
    assert(near(a.ghostY, 164.0f));
    assert(near(a.pullX, 40.0f));
    assert(near(a.pullY, 0.0f));
}

static void test_degenerate_segment_is_not_a_hit() {
    const OutlineAim a = aimOnSegment(0.0f, 0.0f, 5.0f, 5.0f, 5.0f, 5.0f);
    assert(!a.ok);
}

// The stick tip sits 96px along the edge. A press on the point, including
// the side of it that faces the stick, moves the point. The tip bends.
static void test_point_wins_over_its_stick() {
    const float knotX = 100.0f, knotY = 100.0f;
    const float tipX = 196.0f, tipY = 100.0f;
    assert(!nearerHandle(100.0f, 100.0f, knotX, knotY, tipX, tipY));
    assert(!nearerHandle(120.0f, 100.0f, knotX, knotY, tipX, tipY));
    assert(nearerHandle(196.0f, 100.0f, knotX, knotY, tipX, tipY));
    assert(nearerHandle(160.0f, 100.0f, knotX, knotY, tipX, tipY));
}

int main() {
    test_lead_toward_middle();
    test_middle_steps_aside();
    test_short_segment_shrinks_lead();
    test_pull_picks_the_kind();
    test_vertical_pull_points_at_the_finger();
    test_degenerate_segment_is_not_a_hit();
    test_point_wins_over_its_stick();
    std::printf("outline_place_test: ok\n");
    return 0;
}
