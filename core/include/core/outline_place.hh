// outline_place.hh — where a new outline dot sits, relative to the finger.
//
// Pure screen-space arithmetic. The dot leads along the segment so the
// fingertip does not cover the edge it is placing, and a pull off the
// segment is a curve. Staying on the segment is a sharp vertex. There is
// no timer: the kind is the geometry at the moment the finger is down.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#pragma once

#include <algorithm>
#include <cmath>

// How far ahead of the contact the dot is drawn, along the segment.
inline constexpr float kOutlineLeadPx = 64.0f;
// A finger this close to a segment can start a dot.
inline constexpr float kOutlineHitPx  = 48.0f;
// Pulling this far off the segment, perpendicular, makes a curve.
inline constexpr float kOutlinePullPx = 32.0f;
// The stick straightens when the finger comes this close to the point.
inline constexpr float kOutlineStraightenPx = 64.0f;

// The stick is drawn a finger's width from its point, so both hit targets
// overlap. The nearer one is the grab: the point moves, the tip bends.
inline bool nearerHandle(float fx, float fy,
                         float knotX, float knotY,
                         float tipX, float tipY) {
    const float dkx = fx - knotX, dky = fy - knotY;
    const float dtx = fx - tipX, dty = fy - tipY;
    return dtx * dtx + dty * dty < dkx * dkx + dky * dky;
}

struct OutlineAim {
    bool  ok = false;
    float ghostX = 0.0f;     // screen position of the dot
    float ghostY = 0.0f;
    float t = 0.0f;          // dot's parameter along the segment, 0..1
    float perp = 0.0f;       // signed screen pixels, finger off the line
    float fingerDist = 0.0f; // contact to the nearest point of the segment
    float pullX = 0.0f;      // perpendicular vector, screen pixels
    float pullY = 0.0f;
    bool  curve = false;     // |perp| >= pullPx
};

// `leadPx` is taken in the direction that has more room, which on a long
// side is toward the middle, and at the middle is toward the next corner.
// A short segment shrinks the lead so the dot stays on the segment.
inline OutlineAim aimOnSegment(float fx, float fy,
                               float x0, float y0, float x1, float y1,
                               float leadPx = kOutlineLeadPx,
                               float pullPx = kOutlinePullPx) {
    OutlineAim a;
    const float dx = x1 - x0, dy = y1 - y0;
    const float len2 = dx * dx + dy * dy;
    const float len = std::sqrt(len2);
    if (!(len >= 1.0f) || !std::isfinite(len)) return a;

    const float tRaw = ((fx - x0) * dx + (fy - y0) * dy) / len2;
    const float inset = std::min(0.45f, 12.0f / len);
    const float tc = std::clamp(tRaw, inset, 1.0f - inset);
    const float px = x0 + dx * tc;
    const float py = y0 + dy * tc;
    const float ox = fx - px, oy = fy - py;
    a.fingerDist = std::sqrt(ox * ox + oy * oy);

    const float inv = 1.0f / len;
    const float nx = -dy * inv, ny = dx * inv;
    const float tClip = std::clamp(tRaw, 0.0f, 1.0f);
    const float rx = x0 + dx * tClip;
    const float ry = y0 + dy * tClip;
    a.perp = (fx - rx) * nx + (fy - ry) * ny;

    const float roomPos = (1.0f - inset - tc) * len;
    const float roomNeg = (tc - inset) * len;
    const float dir = roomPos >= roomNeg ? 1.0f : -1.0f;
    const float room = dir > 0.0f ? roomPos : roomNeg;
    const float lead = std::min(leadPx, std::max(0.0f, room));
    const float tg = std::clamp(tc + dir * (lead / len), inset, 1.0f - inset);
    a.ghostX = x0 + dx * tg;
    a.ghostY = y0 + dy * tg;
    a.t = tg;
    a.pullX = nx * a.perp;
    a.pullY = ny * a.perp;
    a.curve = std::fabs(a.perp) >= pullPx;
    a.ok = true;
    return a;
}
