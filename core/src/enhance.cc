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

namespace {

float percentileAt(std::vector<float> v, double frac) {
    if (v.empty()) return 0.0f;
    frac = std::clamp(frac, 0.0, 1.0);
    const size_t k = (size_t)std::min((double)(v.size() - 1),
                                      std::max(0.0, (double)(v.size() - 1) * frac));
    std::nth_element(v.begin(), v.begin() + (long)k, v.end());
    return v[k];
}

void boxBlurLuma(const std::vector<float>& src, std::vector<float>& dst,
                 int w, int h, int r) {
    dst.resize(src.size());
    if (r < 1) { dst = src; return; }
    std::vector<float> tmp(src.size());
    std::vector<double> pref((size_t)std::max(w, h) + 1);
    for (int y = 0; y < h; ++y) {
        pref[0] = 0;
        for (int x = 0; x < w; ++x) pref[x + 1] = pref[x] + src[(size_t)y * w + x];
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - r), x1 = std::min(w - 1, x + r);
            tmp[(size_t)y * w + x] =
                (float)((pref[x1 + 1] - pref[x0]) / (double)(x1 - x0 + 1));
        }
    }
    for (int x = 0; x < w; ++x) {
        pref[0] = 0;
        for (int y = 0; y < h; ++y) pref[y + 1] = pref[y] + tmp[(size_t)y * w + x];
        for (int y = 0; y < h; ++y) {
            const int y0 = std::max(0, y - r), y1 = std::min(h - 1, y + r);
            dst[(size_t)y * w + x] =
                (float)((pref[y1 + 1] - pref[y0]) / (double)(y1 - y0 + 1));
        }
    }
}

float sampleBilinearGray(const std::vector<float>& img, int w, int h, float x, float y) {
    x = std::clamp(x, 0.0f, (float)w - 1.01f);
    y = std::clamp(y, 0.0f, (float)h - 1.01f);
    const int x0 = (int)x, y0 = (int)y;
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float tx = x - (float)x0, ty = y - (float)y0;
    const float a = img[(size_t)y0 * w + x0];
    const float b = img[(size_t)y0 * w + x1];
    const float c = img[(size_t)y1 * w + x0];
    const float d = img[(size_t)y1 * w + x1];
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

}  // namespace

void enhanceDocument(std::vector<float>& pixels, uint32_t w, uint32_t h,
                     bool widePrimaries) {
    if (w < 2 || h < 2 || pixels.size() < (size_t)w * h * 4) return;
    const float* luma_w = widePrimaries ? export_math::kLumaBt2020
                                        : export_math::kLumaBt709;
    const float wr = luma_w[0], wg = luma_w[1], wb = luma_w[2];
    const size_t n = (size_t)w * h;
    const uint32_t stride = (uint32_t)std::max(1.0, std::sqrt((double)n / 80000.0));

    auto lumaAt = [&](const float* p) { return wr * p[0] + wg * p[1] + wb * p[2]; };
    auto lowChroma = [&](const float* p, float L) {
        const float mx = std::max(p[0], std::max(p[1], p[2]));
        const float mn = std::min(p[0], std::min(p[1], p[2]));
        const float chroma = mx - mn;
        return chroma < 0.12f || chroma < 0.22f * std::max(L, 1e-4f);
    };

    // Paper white balance. Ink is high-chroma, so it is left out of the
    // estimate; one gain per channel removes the illuminant from everything,
    // which keeps a stamp's hue instead of painting the page grey.
    std::vector<float> paperLuma;
    paperLuma.reserve(n / stride + 1);
    for (uint32_t y = 0; y < h; y += stride) {
        for (uint32_t x = 0; x < w; x += stride) {
            const float* p = &pixels[((size_t)y * w + x) * 4];
            const float L = lumaAt(p);
            if (L > 0.05f && lowChroma(p, L)) paperLuma.push_back(L);
        }
    }
    if (paperLuma.size() >= 16) {
        const float cut = percentileAt(paperLuma, 0.60);
        double sr = 0, sg = 0, sb = 0;
        int cnt = 0;
        for (uint32_t y = 0; y < h; y += stride) {
            for (uint32_t x = 0; x < w; x += stride) {
                const float* p = &pixels[((size_t)y * w + x) * 4];
                const float L = lumaAt(p);
                if (!(L >= cut && lowChroma(p, L))) continue;
                sr += p[0]; sg += p[1]; sb += p[2];
                ++cnt;
            }
        }
        if (cnt > 0 && sr > 1e-6 && sg > 1e-6 && sb > 1e-6) {
            const double m = (sr + sg + sb) / (3.0 * cnt);
            const float gr = std::clamp((float)(m / (sr / cnt)), 0.5f, 2.0f);
            const float gg = std::clamp((float)(m / (sg / cnt)), 0.5f, 2.0f);
            const float gb = std::clamp((float)(m / (sb / cnt)), 0.5f, 2.0f);
            for (size_t i = 0; i < n; ++i) {
                float* p = &pixels[i * 4];
                p[0] *= gr; p[1] *= gg; p[2] *= gb;
            }
        }
    }

    // Shadow field. Blur tracks slow illumination and ignores thin strokes, so
    // dividing by it lifts a gradient and leaves the ink dark. Floor the
    // divisor so a solid black block cannot ask for an infinite gain.
    const int longSide = (int)std::max(w, h);
    const int dsLong = std::min(160, longSide);
    const float dsScale = (float)longSide / (float)dsLong;
    const int dw = std::max(8, (int)std::lround((float)w / dsScale));
    const int dh = std::max(8, (int)std::lround((float)h / dsScale));
    std::vector<float> small((size_t)dw * dh, 0.0f);
    for (int y = 0; y < dh; ++y) {
        const int y0 = std::min((int)((float)y * h / dh), (int)h - 1);
        const int y1 = std::min((int)((float)(y + 1) * h / dh), (int)h);
        for (int x = 0; x < dw; ++x) {
            const int x0 = std::min((int)((float)x * w / dw), (int)w - 1);
            const int x1 = std::min((int)((float)(x + 1) * w / dw), (int)w);
            double acc = 0;
            int cnt = 0;
            for (int py = y0; py < y1; ++py) {
                for (int px = x0; px < x1; ++px) {
                    acc += lumaAt(&pixels[((size_t)py * w + px) * 4]);
                    ++cnt;
                }
            }
            small[(size_t)y * dw + x] = cnt ? (float)(acc / cnt) : 0.0f;
        }
    }
    const int blurR = std::max(1, (int)std::lround(0.04f * (float)dsLong));
    std::vector<float> blurred;
    boxBlurLuma(small, blurred, dw, dh, blurR);

    std::vector<float> litPaper;
    for (uint32_t y = 0; y < h; y += stride) {
        for (uint32_t x = 0; x < w; x += stride) {
            const float* p = &pixels[((size_t)y * w + x) * 4];
            const float L = lumaAt(p);
            if (L > 0.05f && lowChroma(p, L)) litPaper.push_back(L);
        }
    }
    const float target = std::max(percentileAt(litPaper, 0.90), 1e-3f);
    for (uint32_t y = 0; y < h; ++y) {
        const float fy = ((float)y + 0.5f) * (float)dh / (float)h - 0.5f;
        for (uint32_t x = 0; x < w; ++x) {
            const float fx = ((float)x + 0.5f) * (float)dw / (float)w - 0.5f;
            const float b = sampleBilinearGray(blurred, dw, dh, fx, fy);
            const float div = std::max(b, 0.22f * target);
            const float g = std::min(target / div, 4.0f);
            float* p = &pixels[((size_t)y * w + x) * 4];
            p[0] *= g; p[1] *= g; p[2] *= g;
        }
    }

    // Paper near diffuse white, ink near black. One affine on RGB: adding the
    // same black to every channel does not rotate hue, so a red stamp stays red.
    std::vector<float> whites, allL;
    whites.reserve(litPaper.size());
    allL.reserve(n / stride + 1);
    for (uint32_t y = 0; y < h; y += stride) {
        for (uint32_t x = 0; x < w; x += stride) {
            const float* p = &pixels[((size_t)y * w + x) * 4];
            const float L = std::max(lumaAt(p), 0.0f);
            allL.push_back(L);
            if (lowChroma(p, L)) whites.push_back(L);
        }
    }
    const float white = percentileAt(whites, 0.85);
    const float black = percentileAt(allL, 0.02);
    if (white > black + 0.05f) {
        const float scale = (0.96f - 0.02f) / (white - black);
        for (size_t i = 0; i < n; ++i) {
            float* p = &pixels[i * 4];
            p[0] = (p[0] - black) * scale + 0.02f;
            p[1] = (p[1] - black) * scale + 0.02f;
            p[2] = (p[2] - black) * scale + 0.02f;
        }
    }

    const int radius = std::clamp((int)(std::min(w, h) / 900), 1, 3);
    sharpen(pixels, w, h, 0.75f, radius, luma_w);
}
