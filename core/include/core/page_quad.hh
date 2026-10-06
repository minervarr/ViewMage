// page_quad.hh — a rectangle from oriented edges, not from the brightest blob.
//
// The page is a convex quad. Lines come from a Hough transform on two
// gradient maps (borders that run across the frame, and borders that run
// down it). A quad earns its place by edge support, by the page being
// brighter or a different colour just inside the border, and by its sides
// not continuing off into the table.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#pragma once

#include <cstdint>

// Source-pixel corners, top-left, top-right, bottom-right, bottom-left.
// `ok` is false when no quad cleared the geometry and the polarity gate.
// `score` is the ranking used to pick among quads that did.
struct PageQuad {
    float x[4] = {};
    float y[4] = {};
    float score = 0;
    bool ok = false;
};

// `rgba` is w*h*4 floats, straight alpha, top row first. The long edge of
// the working image is 480. A second pass repeats the search at 640 on a
// crop around that quad and keeps it only when the score holds and the
// corners barely move.
PageQuad findPageQuad(const float* rgba, uint32_t w, uint32_t h);

// Score a quad someone else proposed (a corner net). Same geometry and
// polarity gate as findPageQuad. `ok` is false when the quad is rejected,
// including when fewer than three sides are brighter or a different colour
// on the inside.
PageQuad scorePageQuad(const float* rgba, uint32_t w, uint32_t h,
                       const float x[4], const float y[4]);

// Constant-velocity filter, position and velocity of each corner.
// A measurement farther than 4% of the long edge from the prediction is
// held, not applied. Eight held frames in a row drop the lock.
// freeze() keeps the current quad until the next accepted measurement
// after unfreeze(); the caller decides that the phone is still.
class PageTrack {
public:
    PageTrack();

    void setLongEdge(float longEdge);
    bool update(const PageQuad& measured, float dt_s);
    PageQuad predict(float dt_s) const;
    void freeze();
    void unfreeze();
    bool locked() const { return locked_; }

private:
    struct Corner {
        double s[4] = {};     // x, y, vx, vy
        double P[16] = {};    // row-major
        bool inited = false;
    };

    void predictCorner(Corner& c, float dt) const;
    void correctCorner(Corner& c, float zx, float zy, float measVar);

    Corner corner_[4];
    float longEdge_ = 480.0f;
    int missed_ = 0;
    bool locked_ = false;
    bool frozen_ = false;
    float score_ = 0;
};
