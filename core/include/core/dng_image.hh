// dng_image.hh — a camera RAW (DNG) file in, LINEAR floating-point light out.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// ---------------------------------------------------------------------------
// The DNG twin of jxl_image.hh, and it hands back the SAME struct on purpose:
// everything downstream — auto-exposure, the tone curve, the GPU upload, the
// details panel — cares about "linear light plus what the file said about
// itself", not about which container it arrived in. Adding a second decoder is
// therefore a new producer of DecodedImage, not a second pipeline.
//
// Why this exists at all: the camera is capture-only now. It writes DNGs and
// nothing else, so developing a RAW file is no longer an optional extra — it is
// the only way to see anything that was shot. A DNG is not a picture; it is the
// sensor's measurements plus the calibration needed to turn them into one, and
// that turning is what this file does.
//
// Scope: UNCOMPRESSED 16-bit Bayer CFA DNGs, which is what camera_without_blood
// writes. Compressed or tiled DNGs, linear (already-demosaiced) DNGs and
// non-Bayer CFAs are rejected with a sentence the user can read, rather than
// half-decoded into something misleading.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <functional>
#include <cstdint>
#include <string>
#include <vector>

#include "core/jxl_image.hh"   // DecodedImage, WorkingPrimaries, Diagnostic

// True when `data` starts with a TIFF/DNG signature ("II*\0" or "MM\0*").
// Cheap, and it exists so the app can route a file to the right decoder before
// committing to either.
bool looks_like_dng(const uint8_t* data, size_t size);

// A parsed-but-undeveloped RAW mosaic: the sensor's measurements and the
// calibration needed to normalise them, with no demosaic, white balance or
// tone curve applied. Exists so diagnostics can measure the MOSAIC itself --
// clipping census, per-channel behaviour around saturated blooms -- which is
// where highlight artifacts are born and where they are invisible to any
// downstream stage.
//
// `bayer` points INTO the caller's buffer and stays valid only while it lives.
struct RawMosaicView {
    const uint16_t* bayer = nullptr;   // width*height samples, row stride = width
    uint32_t width = 0, height = 0;
    int cfa = 0;                       // 0 RGGB, 1 GRBG, 2 GBRG, 3 BGGR
    double black[4] = {0, 0, 0, 0};    // per CFA element, raster order
    double white = 65535.0;
    bool has_neutral = false;
    double neutral[3] = {1, 1, 1};     // AsShotNeutral
    double baseline_exposure = 0.0;    // != 0 suggests a merged bracket
    bool ok = false;
    std::string error;
};

// Parse a DNG far enough to hand back its mosaic. Never throws; failures come
// back in `error` with ok == false.
RawMosaicView read_raw_mosaic(const uint8_t* data, size_t size,
                              struct NoiseModel* out_noise = nullptr);

// What the sensor said about its own noise: one (S, O) pair per CFA channel,
// from DNG tag 51041 (NoiseProfile). Variance at normalised signal x is
// S*x + O — the Poisson (shot) term scales with the signal, the Gaussian (read)
// term does not.
//
// This is carried out of the decode rather than consumed inside it because it
// is the input a denoiser needs, and a denoiser that has to ESTIMATE the noise
// level from the pixels is the difference between cleaning an image and
// inventing detail in it. Empty when the file did not say.
struct NoiseModel {
    bool   valid = false;
    int    count = 0;        // 2 * CFA channels (8 for Bayer)
    double so[8] = {0};      // S0,O0, S1,O1, ...
};

// Develop a DNG to linear float RGBA.
//
// `maxDimension` behaves exactly as in decode_jxl(): an image larger than the
// GPU's limit is box-downsampled by the smallest power of two that fits. Pass 0
// to disable. `primaries` is the gamut the returned floats are expressed in and
// must match the target the Renderer actually resolved to — see WorkingPrimaries.
//
// Never throws. Every failure comes back as a populated `error`.
//
// Convention of the returned floats: LINEAR, unclamped, with 1.0 == the
// reference exposure's clip point, and `intensityTarget` set to the nit level
// that clip stands for. That is deliberately the same convention the camera's
// PQ stills used (kPqScale: linear 1.0 = 1000 nits), so a developed DNG and a
// clip of the same scene land at the same absolute brightness.
// `denoiser`, when non-null, REPLACES the demosaic with the neural joint
// denoise+demosaic (see ai_denoise.hh). Everything after that stage -- white
// balance, highlight reconstruction, the CCM, the transfer and the tone curve --
// is identical either way, so the two paths differ in exactly one step. If the
// model fails at run time the develop falls back to Malvar and says so in the
// notes: a denoise that did not work must never mean a photo that did not open.
//
// `denoiseEnsemble` runs the model through all four phase-exact Bayer
// symmetries and averages. DEFAULT ON: it is the quality bar the POST render
// is judged against. (One flip to single-pass-by-default lasted half a
// session before the user read it as a resolution loss — the 4x speedup was
// real, and so was the regression.) Single-pass keeps ~71% of Malvar's fine
// structure, which the develop's detail layer (see dng_image.cc) then
// restores from sensor measurement; that layer is single-pass COMPENSATION
// and must not also run for the ensemble, or the two textures stack.
// Speed-critical callers pass false.
DecodedImage decode_dng(const uint8_t* data, size_t size, uint32_t maxDimension,
                        WorkingPrimaries primaries = WorkingPrimaries::Srgb,
                        NoiseModel* out_noise = nullptr,
                        class RawDenoiser* denoiser = nullptr,
                        std::function<bool(float)> progress = {},
                        bool denoiseEnsemble = true);
