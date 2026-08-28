// enhance.cc — vibrance and luminance sharpening for share-ready output.
//
// All steps operate on linear float RGBA in-place. The pixel buffer is the
// same DecodedImage::linear that the develop produces — wide-gamut,
// unclamped float. Negative values (out-of-gamut speculars) are meaningful
// and survive; both steps handle them cleanly.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.

#include "core/enhance.hh"
#include "core/export_math.hh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

inline float smooth01(float t) {
    t = std::min(std::max(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ── Vibrance ────────────────────────────────────────────────────────────────
//
// Boost chrominance relative to luminance, with a soft knee that diminishes
// the effect as the colour approaches full saturation. A flat multiplier
// (rgb' = luma + (rgb - luma) * s) oversaturates already-vivid colours and
// clips them into flat patches — the opposite of natural, and the direct cause
// of out-of-gamut hue shifts at the sRGB export.
//
// The knee: effective factor = s * (1 - chroma^2), where chroma is the pixel's
// saturation in [0,1]. Already-saturated pixels get almost no boost; muted
// colours get nearly all of it.

void vibrance(std::vector<float>& px, float factor, const float luma_w[3]) {
    if (factor <= 1.001f) return;
    const float wr = luma_w[0], wg = luma_w[1], wb = luma_w[2];
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        const float luma = std::max(wr * px[i] + wg * px[i+1] + wb * px[i+2],
                                    0.0f);
        // Saturation of this pixel: 0 = grey, 1 = fully saturated.
        const float diff = std::max(std::max(px[i] - luma, px[i+1] - luma),
                                    std::max(px[i+2] - luma, 0.0f));
        const float chroma = std::min(diff / std::max(luma, 1e-4f), 1.0f);
        const float eff = factor - (factor - 1.0f) * chroma * chroma;
        px[i]   = luma + (px[i]   - luma) * eff;
        px[i+1] = luma + (px[i+1] - luma) * eff;
        px[i+2] = luma + (px[i+2] - luma) * eff;
    }
}

// ── Sharpening (luminance unsharp mask) ─────────────────────────────────────
//
// The detail signal is extracted from LUMINANCE and added EQUALLY to all three
// channels. A per-channel mask sharpens chroma edges into colour fringes that
// read exactly like lens CA — on top of whatever real CA the frame has.
//
// SOFT DEAD-ZONE GATE, not a binary threshold. The old hard gate was what
// produced "pixel art" around bright lights: on smooth highlight gradients the
// per-pixel detail flickers across a fixed floor, so neighbouring pixels got
// full strength or exactly zero — a staircase baked into the image. The gate
// now ramps smoothly across a band twice the noise-floor wide, so sub-floor
// detail is attenuated, never switched off.
//
// OVERSHOOT CAP: an unsharp mask on a strong edge overshoots into halos. What
// is added is capped at half the measured detail magnitude, so edges steepen
// without growing bright or dark rims.

void sharpen(std::vector<float>& px, uint32_t w, uint32_t h,
             float strength, int radius, const float luma_w[3]) {
    if (strength <= 0.001f || radius <= 0) return;

    const size_t n = (size_t)w * h;
    std::vector<float> blur(n);
    const int diam = radius * 2 + 1;
    const float invArea = 1.0f / (float)(diam * diam);
    const float wr = luma_w[0], wg = luma_w[1], wb = luma_w[2];

    // Box blur of luminance only — a quarter of a full RGBA pass, and the mask
    // is applied per pixel afterwards anyway. Borders clamp to edge.
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int sy = std::clamp((int)y + dy, 0, (int)h - 1);
                const float* row = &px[(size_t)sy * w];
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sx = std::clamp((int)x + dx, 0, (int)w - 1);
                    const size_t si = (size_t)sx * 4;
                    sum += wr * row[si] + wg * row[si + 1] + wb * row[si + 2];
                }
            }
            blur[(size_t)y * w + x] = float(sum * invArea);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const float* p = &px[i * 4];
        const float L = wr * p[0] + wg * p[1] + wb * p[2];
        const float detail = L - blur[i];

        // Soft dead-zone: zero below the noise floor, full above twice it,
        // smoothstep between — no per-pixel on/off decisions anywhere.
        const float floor_ = std::max(std::fabs(blur[i]) * 0.02f, 1e-3f);
        const float mag = std::fabs(detail);
        const float gate =
            (mag <= floor_) ? 0.0f
                            : smooth01((mag - floor_) / (2.0f * floor_));
        float added = strength * gate * detail;

        // Overshoot cap.
        const float cap = 0.5f * mag;
        added = std::min(std::max(added, -cap), cap);

        float* d = &px[i * 4];
        d[0] += added; d[1] += added; d[2] += added;
    }
}

}  // namespace

void enhanceImage(std::vector<float>& pixels, uint32_t w, uint32_t h,
                  const EnhanceParams& p, bool widePrimaries) {
    if (pixels.size() < (size_t)w * h * 4) return;

    // Order: vibrance first so boosted chroma is what gets (gently) sharpened;
    // sharpening last so its noise gate reads the final tones.
    const float* luma_w = widePrimaries ? export_math::kLumaBt2020
                                        : export_math::kLumaBt709;
    vibrance(pixels, p.vibrance, luma_w);
    sharpen(pixels, w, h, p.sharpen, p.blurRadius, luma_w);
}
