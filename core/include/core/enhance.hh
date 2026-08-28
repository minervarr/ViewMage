// enhance.hh — post-processing to make a develop look like the scene did.
//
// The bare develop is faithful; this makes it share-ready: vibrance and mild
// luminance sharpening — the finishing the stock camera's ISP applies that
// ViewMage deliberately skips in its automatic render.
//
// WHY THERE IS NO CONTRAST CURVE HERE ANY MORE. Enhance used to stack an ACES
// approximation on top of the develop's own toe (decoded_image.cc) — two tone
// maps, then the shader/export rolloff made three. Measured, that re-shuffled
// the tonal balance the develop had already placed (ACES lifts 0.18 mid grey
// ~30%) and flattened highlights into patches. Contrast belongs to exactly one
// stage; here that stage is the develop.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#pragma once

#include <cstdint>
#include <vector>

struct EnhanceParams {
    float vibrance   = 1.15f;   // muted-colour boost (1.0 = none)
    float sharpen    = 0.40f;   // unsharp-mask strength (0.0 = none)
    int   blurRadius = 2;       // blur radius in pixels for sharpening
};

// Apply vibrance and luminance-only unsharp mask to linear float RGBA pixels
// in-place. `widePrimaries` selects BT.2020 luma weights for BT.2100-working
// data; using 709's weights there skews the chroma decision per hue, which is
// the same green-cast class of bug the export fixed at its own layer.
// Safe to call from a worker thread.
void enhanceImage(std::vector<float>& pixels, uint32_t w, uint32_t h,
                  const EnhanceParams& p, bool widePrimaries);
