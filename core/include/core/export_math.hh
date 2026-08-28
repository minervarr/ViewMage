// export_math.hh — pure math shared by the GUI export and the CLI export.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// The rolloff curve and the sRGB gamma encode live here, not duplicated
// between gui/src/png_export.cc and a future CLI tool. Both are verbatim
// mirrors of image_frag.slang — the export MUST be the same curve as the
// screen or it is a different picture.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace export_math {

// BT.2020 -> sRGB, D65. No chromatic adaptation needed (same white point).
// The export is sRGB because that is what a phone gallery, a chat app, and a
// browser all assume.
inline constexpr float kBt2020ToSrgb[9] = {
     1.6605f, -0.5876f, -0.0728f,
    -0.1246f,  1.1329f, -0.0083f,
    -0.0182f, -0.1006f,  1.1187f,
};

// Luminance weights per primaries.
inline constexpr float kLumaBt709[3]  = {0.2126f, 0.7152f, 0.0722f};
inline constexpr float kLumaBt2020[3] = {0.2627f, 0.6780f, 0.0593f};

// The shader's rolloff, in C++. MIRRORS rolloff() in image_frag.slang and
// rolloffCurve() in vk_canvas's output_target.cc — the export must be the
// same curve as the screen or it is a different picture.
float rolloff(float x, float W);

// Linear [0,1] -> sRGB [0,255] with proper gamma.
uint8_t to_srgb8(float x);

// The same encode with an ordered-dither offset added before quantisation.
// `dither` is a fraction of one code value in [−0.5, 0.5); use kBayerDither.
// Export-only: the shader's mirror stays to_srgb8, which is why the dither
// lives beside it rather than inside it.
uint8_t to_srgb8_dithered(float x, float dither);

// 4x4 ordered (Bayer) dither thresholds, centred on zero, in FRACTIONS OF A
// CODE VALUE (index/16 − 1/2). One definition for every exporter (PNG, JPEG,
// CLI): a smooth gradient crossing one code value alternates levels instead
// of banding.
struct BayerDither4 {
    float m[4][4];
    constexpr BayerDither4() : m{} {
        constexpr int b[16] = { 0, 8, 2,10,
                               12, 4,14, 6,
                                3,11, 1, 9,
                               15, 7,13, 5 };
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                m[y][x] = float(b[y * 4 + x]) / 16.0f - 0.5f;
    }
};
inline constexpr BayerDither4 kBayerDither{};

}  // namespace export_math
