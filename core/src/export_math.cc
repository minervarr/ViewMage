// export_math.cc — see export_math.hh.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#include "core/export_math.hh"

#include <algorithm>
#include <cmath>

namespace export_math {

float rolloff(float x, float W) {
    const float k = 0.8f;
    if (x <= k) return x;
    const float P = std::min(1.0f, W);
    const float span = std::max(W - k, 1e-4f);
    const float a = (x - k) / span;
    const float d = span / std::max(P - k, 1e-4f);
    return k + (P - k) * (a * (1.0f + a / d) / (1.0f + a));
}

uint8_t to_srgb8(float x) {
    x = std::min(std::max(x, 0.0f), 1.0f);
    const float e = (x <= 0.0031308f) ? (12.92f * x)
                                       : (1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f);
    return uint8_t(std::lround(std::min(std::max(e, 0.0f), 1.0f) * 255.0f));
}

uint8_t to_srgb8_dithered(float x, float dither) {
    // Same curve as to_srgb8 — this must stay a mirror of the shader's encode
    // — but the round-to-nearest is replaced by adding `dither` (in [−0.5, 0.5)
    // code values) before truncation. A smooth sky crossing one code value
    // over many pixels then alternates levels instead of banding: the eye
    // integrates the pattern back into the missing steps.
    x = std::min(std::max(x, 0.0f), 1.0f);
    const float e = (x <= 0.0031308f) ? (12.92f * x)
                                       : (1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f);
    const float v = std::min(std::max(e, 0.0f), 1.0f) * 255.0f + dither;
    return uint8_t(std::min(std::max(v, 0.0f), 255.0f));
}

}  // namespace export_math
