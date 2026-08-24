// jxl_image.hh — compressed JPEG XL bytes in, RGBA8 pixels out.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// ---------------------------------------------------------------------------
// The ONLY place in ViewMage that includes anything of libjxl's, and the whole
// of our side of that boundary. libjxl lives in thirdparty/ and is never
// modified — so anything ViewMage needs that libjxl does not already do (a
// size ceiling, a signature check that does not need a decoder, an error as a
// value rather than a status code) is written here instead of patched there.
//
// Nothing in this header mentions Vulkan, and nothing in it mentions Android.
// It is a pure function from bytes to pixels, which is what lets
// tests/jxl_image_test.cc run it on a desktop.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct JxlImage {
    std::vector<uint8_t> rgba;      // w*h*4, straight alpha, top row first
    uint32_t w = 0, h = 0;

    // Empty on success, a message fit to show the user on failure. An error is
    // a VALUE here rather than a status code or an exception: every caller has
    // to put something on the screen either way, and there is exactly one
    // caller.
    std::string error;

    // How many times the decoded image was halved to fit maxDimension. Zero
    // for the overwhelming majority of images; non-zero is worth logging,
    // because it is the difference between what the file holds and what the
    // screen shows.
    int downsampleFactor = 1;

    bool ok() const { return error.empty() && w > 0 && h > 0; }
};

// True when `data` starts with a JPEG XL signature — either a bare codestream
// or the ISOBMFF container form. Cheap, and it exists so that opening a PNG by
// mistake says "not a JPEG XL image" instead of surfacing a decoder error that
// means nothing to anyone.
bool looks_like_jxl(const uint8_t* data, size_t size);

// Decode to RGBA8.
//
// `maxDimension` is the largest 2D image the GPU will accept
// (VkPhysicalDeviceLimits::maxImageDimension2D). An image above it is
// box-downsampled by the smallest power of two that fits, so a picture too
// large for the hardware is shown slightly soft rather than not shown. Pass 0
// to disable the ceiling entirely.
//
// Never throws. Every failure — truncated input, wrong format, a libjxl error,
// an allocation that did not happen — comes back as a populated `error`.
JxlImage decode_jxl(const uint8_t* data, size_t size, uint32_t maxDimension);
