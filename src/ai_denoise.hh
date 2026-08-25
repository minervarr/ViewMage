// ai_denoise.hh — neural raw denoise, on the Bayer mosaic, before demosaic.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// ---------------------------------------------------------------------------
// WHAT THIS IS
//
// RawNIND UtNet2 (Benoit Brummer, UCLouvain), the model darktable 5.6 ships as
// Neural Restore -> Raw Denoise. It is a JOINT DENOISE AND DEMOSAIC: it takes
// the packed Bayer mosaic and returns full-resolution camera RGB, so it does not
// sit next to our Malvar demosaic, it REPLACES it.
//
//   in   [1, 4, 512, 512]   packed Bayer, planes [R, G1, G2, B]
//   out  [1, 3, 1024, 1024] camera-native RGB (2x spatial, PixelShuffle head)
//
// Denoising here rather than after demosaic is the whole point. Once the mosaic
// is interpolated, each output pixel is a blend of neighbours and the
// per-channel noise statistics the sensor actually produced are gone; a denoiser
// downstream is guessing at what it can no longer see.
//
// The ColorMatrix is deliberately NOT baked into the graph, which is what makes
// this a clean graft: the model's output enters our develop exactly where the
// demosaic's did, and white balance, the CCM, the transfer and the tone curve
// all run downstream unchanged.
//
// Model:   GPL-3.0, https://github.com/darktable-org/darktable-ai
// Paper:   Brummer & De Vleeschouwer 2025, https://arxiv.org/abs/2501.08924
// Data:    RawNIND (real noisy/clean raw pairs), CC BY 4.0 / CC0
//
// This file is AGPLv3; the model weights it loads are GPL-3.0. Compatible, and
// the attribution above is a licence obligation, not a courtesy -- it is also
// surfaced to the user in the details panel.
//
// GUARDED by VIEWMAGE_WITH_ORT. Without ONNX Runtime the class still exists and
// `load()` fails with a readable message, so the desktop test binaries build and
// the develop's fallback path is exercised for free.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// The tile geometry is BAKED INTO THE ONNX EXPORT, not chosen by us: the model
// is exported with static 512x512 input, so these are facts about the file on
// disk. `kOverlap` is darktable's OVERLAP_DENOISE; the overlap is trimmed off
// every tile so seams cannot appear.
inline constexpr int kAiTile    = 512;
inline constexpr int kAiOverlap = 64;
inline constexpr int kAiStep    = kAiTile - 2 * kAiOverlap;   // 384
inline constexpr int kAiScale   = 2;                          // PixelShuffle head

class RawDenoiser {
public:
    // `onnx` is the model file's bytes. Returns null and fills `err` when ONNX
    // Runtime is absent, the model will not parse, or its shape is not the one
    // documented above -- never throws, never half-succeeds.
    static std::unique_ptr<RawDenoiser> load(const uint8_t* onnx, size_t bytes,
                                             std::string& err);

    ~RawDenoiser();

    // `packed`: 4 planes of pw*ph, order [R, G1, G2, B], each normalised to
    // [0,1] by the sensor's black and white levels. NOT white balanced -- the
    // model was trained on unbalanced mosaics and balancing first is a real,
    // easy-to-introduce bug.
    //
    // `out`: 3 INTERLEAVED camera-RGB floats at (pw*2, ph*2), gain-matched.
    //
    // `progress` is called with 0..1 between tiles; use it to drive a readout or
    // to abort by returning false.
    bool run(const float* packed, int pw, int ph,
             std::vector<float>& out, std::string& err,
             const std::function<bool(float)>& progress = {});

    // Which execution provider the session actually got ("CPU", "XNNPACK", ...).
    const std::string& provider() const { return provider_; }

private:
    RawDenoiser();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string provider_;
};

// Pack a Bayer mosaic into the 4 half-resolution planes the model expects.
//
// Split out of `run()` and exposed because it is the step that fails SILENTLY:
// get the plane order wrong and the picture still develops, just with the colour
// channels swapped, which reads as a colour-matrix bug. The desktop test pins it
// against the reference implementation.
//
// `cfa` is the DNG CFA code (0 RGGB, 1 GRBG, 2 GBRG, 3 BGGR). `black` is per CFA
// element in raster order, matching RawMeta. Output is clipped to [0,1]: the
// model is trained on that range.
void pack_bayer_rggb(const uint16_t* mosaic, int w, int h, int stride_px,
                     int cfa, const double black[4], double white,
                     std::vector<float>& packed, int& pw, int& ph);
