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
    // w*h*4 floats. LINEAR light, D65, straight alpha, top row first, in the
    // primaries the caller asked decode_jxl() for (sRGB/BT.709 by default,
    // BT.2100 for the HDR10 PQ target). NOT clamped to [0,1] and must not be.
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

// The struct above is not JXL-specific and has not been for a while: it is
// "linear light plus what the file said about itself". A DNG develop produces
// exactly the same thing (see dng_image.hh), and everything downstream — auto
// exposure, tone curve, GPU upload, details panel — is written against this
// shape rather than against a format. The old name is kept so the existing
// call sites and tests still read naturally; new code should say DecodedImage.
using DecodedImage = JxlImage;

// True when `data` starts with a JPEG XL signature -- bare codestream or the
// ISOBMFF container form. Cheap, and it exists so that opening a PNG by
// mistake says "Not a JPEG XL image" instead of surfacing a decoder error that
// means nothing to anyone.
bool looks_like_jxl(const uint8_t* data, size_t size);

// Which primaries the decoded floats should be expressed in.
//
// This is NOT a preference, it is a contract with the swapchain. The shader
// hands its linear numbers to a surface whose colourspace the compositor
// already believes it knows: an HDR10/ST2084 surface is BT.2020 by definition,
// an sRGB or extended-linear-scRGB one is sRGB/BT.709. Decoding to the wrong
// set stretches every colour onto the other set's gamut -- most visibly toward
// green, because BT.2020's green primary is the one furthest from sRGB's.
//
// So the caller must pass whatever target the Renderer ACTUALLY resolved to
// (it can silently fall back to SDR), never the one it asked for.
enum class WorkingPrimaries {
    Srgb,     // sRGB / BT.709 -- the SDR and extended-linear-scRGB targets
    Bt2100,   // BT.2020 / BT.2100 -- the HDR10 PQ target
};

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
// `primaries` is the gamut the returned floats are expressed in; see
// WorkingPrimaries. It defaults to sRGB, which is what every pre-HDR10 caller
// wanted and keeps the decode self-consistent when nothing is passed.
JxlImage decode_jxl(const uint8_t* data, size_t size, uint32_t maxDimension,
                    WorkingPrimaries primaries = WorkingPrimaries::Srgb);

// The half of a decode that is NOT format-specific: downsample to the GPU's
// ceiling, compute the auto exposure, and add the notes that say how the file
// exceeds what the screen can show.
//
// Every producer of a DecodedImage must end by calling this. It is shared
// rather than duplicated because it reasons about linear light and about what
// the file claimed of itself — neither of which has anything to do with the
// container the pixels arrived in. `decode_jxl` calls it; so does `decode_dng`.
// A second copy would drift, and the drift would show up as "the same photo
// looks different depending on which format I saved it as".
//
// Expects `linear`, `w`, `h`, `intensityTarget`, `hdrTransfer`, `widePrimaries`
// and `bitsPerSample` to already be filled in. Does nothing to a failed decode.
// The display render: turns scene-referred linear light into a picture. Called
// ONLY from the RAW develop path -- a JXL arrives already rendered by whatever
// wrote it, and curving it a second time would double the contrast. See the
// long rationale in decoded_image.cc.
void apply_display_render(DecodedImage& im, WorkingPrimaries primaries);

void finalize_decoded(DecodedImage& out, uint32_t maxDimension, WorkingPrimaries primaries);

// Append one user-facing note. Shared by both decoders: the notes are about how
// the file exceeds the screen, which is not a per-format question.
void note(DecodedImage& im, DiagCode code, DiagLevel level,
          std::string title, std::string detail);
