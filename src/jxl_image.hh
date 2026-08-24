// jxl_image.hh — compressed JPEG XL bytes in, LINEAR floating-point light out.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// ---------------------------------------------------------------------------
// The ONLY place in ViewMage that includes anything of libjxl's, and the whole
// of our side of that boundary. libjxl lives in thirdparty/ and is never
// modified, so anything ViewMage needs that libjxl does not already do is
// written here rather than patched there.
//
// This used to hand back 8-bit sRGB and it was WRONG, in a way worth recording
// because the symptom pointed elsewhere. It asked for JXL_TYPE_UINT8, never
// subscribed to JXL_DEC_COLOR_ENCODING, and never set a CMS -- so libjxl
// handed over the file's own values with no colour transform at all and they
// were drawn as though they were sRGB. An HDR photo treated that way looks
// flat and washed out, which reads as "this screen cannot show it" when in
// fact nothing had tried to. Worse, the truncation to 8 bits happened at the
// door, destroying the extra range before any control could reach it.
//
// So the contract is now: colour-managed, linear, unclipped float. Values
// below 0 (out of gamut) and far above 1 (highlights) are MEANINGFUL and must
// survive to the GPU. Clamping happens once, at the very end, in the shader,
// after exposure and the tone curve.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ── What the viewer should tell the user ────────────────────────────────────
//
// A file can differ from what the screen can show in several independent ways,
// and silently papering over that is how a viewer becomes untrustworthy. Each
// difference we detect becomes one of these, shown behind the details button.
enum class DiagCode : uint8_t {
    kHdrTransfer,          // PQ or HLG: far more range than an SDR panel
    kWidePrimaries,        // BT.2100 / Display P3: more saturation than sRGB
    kHighBitDepth,         // more than 8 bits per channel in the file
    kDownsampledForGpu,    // larger than the GPU's maximum 2D image
    kPrecisionReduced,     // a memory fallback cost us precision
    kIccOnlyProfile,       // profile cannot say whether it is HDR
    kConversionRefused,    // libjxl would not convert to our working space
    kAnimationFirstFrame,  // multi-frame file, one frame shown
    kNoColorManagement,    // built without a CMS (should not happen)
};

enum class DiagLevel : uint8_t { kInfo, kNotice, kWarning };

struct Diagnostic {
    DiagCode  code  = DiagCode::kHdrTransfer;
    DiagLevel level = DiagLevel::kInfo;
    std::string title;    // a few words, for the list
    std::string detail;   // plain language, no jargon the user cannot act on
};

struct JxlImage {
    // w*h*4 floats. LINEAR light, sRGB/BT.709 primaries, D65, straight alpha,
    // top row first. NOT clamped to [0,1] and must not be.
    std::vector<float> linear;
    uint32_t w = 0, h = 0;

    // Empty on success, a sentence fit to show the user on failure. An error
    // is a VALUE rather than a status code or an exception: every caller has
    // to put something on the screen either way, and there is one caller.
    std::string error;

    // ── What the viewer needs to display it well ────────────────────────────

    // Exposure, in stops, that makes this image look right on an SDR screen.
    // The slider starts here and explores outward. See the derivation in the
    // .cc -- it is anchored on the file's declared peak brightness and only
    // nudged by the actual pixels.
    float autoEv = 0.0f;

    // Where the tone curve's knee should put "white", in linear units after
    // autoEv is applied. 1.0 makes the curve an exact identity, which is what
    // an ordinary SDR image wants.
    float white = 1.0f;

    // ── What the file said about itself, for the details panel ──────────────
    uint32_t bitsPerSample   = 0;
    uint32_t exponentBits    = 0;      // non-zero: the original was float
    float    intensityTarget = 0.0f;   // nits; 0 means the file did not say
    float    minNits         = 0.0f;
    bool     hdrTransfer     = false;  // PQ or HLG
    bool     widePrimaries   = false;  // anything beyond sRGB's
    bool     colorManaged    = false;  // did our conversion actually happen
    std::string transferName;          // "PQ", "HLG", "sRGB", ...
    std::string primariesName;         // "BT.2100", "Display P3", "sRGB", ...

    // How many times the image was halved to fit the GPU's limit. 1 for the
    // overwhelming majority; anything else is worth saying out loud, because
    // it is the difference between what the file holds and what is on screen.
    int downsampleFactor = 1;

    std::vector<Diagnostic> notes;   // most important first

    bool ok() const { return error.empty() && w > 0 && h > 0; }
};

// True when `data` starts with a JPEG XL signature -- bare codestream or the
// ISOBMFF container form. Cheap, and it exists so that opening a PNG by
// mistake says "Not a JPEG XL image" instead of surfacing a decoder error that
// means nothing to anyone.
bool looks_like_jxl(const uint8_t* data, size_t size);

// Decode to linear float RGBA.
//
// `maxDimension` is the largest 2D image the GPU will accept
// (VkPhysicalDeviceLimits::maxImageDimension2D). An image above it is
// box-downsampled by the smallest power of two that fits, so a picture too
// large for the hardware is shown slightly soft rather than not shown. Pass 0
// to disable the ceiling.
//
// Never throws. Every failure -- truncated input, wrong format, a libjxl
// error, an allocation that did not happen -- comes back as a populated
// `error`.
JxlImage decode_jxl(const uint8_t* data, size_t size, uint32_t maxDimension);
