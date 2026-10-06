// document_scan.hh — find a page in a photo and flatten it.
//
// A flat page in perspective is a quadrilateral: four corners, then a
// homography. A sheet in the hand or on the table is often not that. The
// outline is then a triangle, a polygon with a folded-off corner, or a bow,
// and the crop follows that shape instead of inventing a fourth corner.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct DocumentQuad {
    // Source-pixel corners, ordered top-left, top-right, bottom-right,
    // bottom-left. Valid when the outline is a quad.
    float x[4] = {};
    float y[4] = {};
};

enum class DocumentShape {
    Quad,       // four straight sides
    Triangle,   // three straight sides
    Fold,       // extra corner, an inward bite, or a crease to unfold
    Curve,      // an edge bows (a sheet curled in the hand)
};

// A point on the page boundary, in source pixels, in boundary order.
// Corner: a real corner of the sheet. The sides that meet there are straight.
// Fold: a sharp kink, the point of a V. It joins its two neighbours with
// straight segments. On a rectangular sheet it edits only the side between
// the two corners it sits on. On any other outline, paired with a mate
// across the page, it is a hinge.
// Curve: a smooth knot. (hx, hy) is the symmetric tangent handle, an offset
// from the point — the same arm an Inkscape smooth node drags. A handle
// pulled back to nothing is a fold, not a curve.
enum class PagePointKind {
    Corner,
    Fold,
    Curve,
};

struct PagePoint {
    PagePointKind kind = PagePointKind::Corner;
    float x = 0;
    float y = 0;
    float hx = 0;
    float hy = 0;
};

struct DocumentOutline {
    DocumentShape shape = DocumentShape::Quad;
    // The boundary the finger edits. Quads, triangles, folds and bows all
    // use this. x/y and the arc chains below stay filled for the same shape
    // so a caller that has not moved to points still sees the outline.
    std::vector<PagePoint> points;
    // Triangle and fold: the polygon in source pixels, in boundary order.
    std::vector<float> x;
    std::vector<float> y;
    // Curve: the two opposite boundaries, both running the same direction.
    // Horizontal bow: top then bottom, left to right. Vertical: left then
    // right, top to bottom.
    std::vector<float> ax, ay, bx, by;
    bool vertical = false;
};

struct DocumentDetect {
    bool found = false;
    DocumentQuad quad;
    DocumentOutline outline;
    std::string error;
};

struct DocumentScan {
    bool found = false;
    DocumentQuad quad;
    std::vector<float> linear;   // flattened RGBA, same layout as DecodedImage
    uint32_t w = 0;
    uint32_t h = 0;
    std::string error;           // set when found is false
};

// `linear` is w*h*4 floats, straight alpha, top row first. Safe off the UI
// thread. Does not modify the input.
//
// detectDocument finds the outline. flattenDocument warps a quad (after the
// user has moved its corners). flattenOutline crops a triangle, a fold, or a
// bow. scanDocument does both, for the command-line path.
//
// reconstructRectangle is the rectangular sheet. The quad keeps the table
// out and the page is one homography, so a straight line stays straight.
// A diamond or a circle on a side bends only that side.
//
// The outline overload keeps the four corner points as that same rectangle.
// Corners land on the corners, and each side runs straight from one corner
// to the next. A diamond or a circle edits only the side it sits on, and
// only as far in as that side bends. The rest of the page stays the
// straight corner mapping.
DocumentDetect detectDocument(const float* linear, uint32_t w, uint32_t h);

// Rectangular sheet. `model` is the DocAligner LC100 heatmap ONNX (Apache-2.0),
// or empty. A confident page replaces the luminance quad and the corners are
// then pulled onto the nearest paper edge. A missing model, a build without
// ONNX Runtime, or a weak detection returns detectDocument unchanged. Shape
// detection stays on detectDocument.
// `cornerPeak`, when not null, receives the four heatmap peaks (TL TR BR BL).
// They stay 0 when the model did not run. A rejected page still reports them.
DocumentDetect detectRectangle(const float* linear, uint32_t w, uint32_t h,
                               const uint8_t* model, size_t modelBytes,
                               float cornerPeak[4] = nullptr);
DocumentScan flattenDocument(const float* linear, uint32_t w, uint32_t h,
                             const DocumentQuad& quad);
DocumentScan reconstructRectangle(const float* linear, uint32_t w, uint32_t h,
                                  const DocumentQuad& quad);
DocumentScan reconstructRectangle(const float* linear, uint32_t w, uint32_t h,
                                  const DocumentOutline& outline);
DocumentScan flattenOutline(const float* linear, uint32_t w, uint32_t h,
                            const DocumentOutline& outline);
DocumentScan scanDocument(const float* linear, uint32_t w, uint32_t h);
