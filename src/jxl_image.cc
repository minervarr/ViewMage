#include "jxl_image.hh"

#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/resizable_parallel_runner_cxx.h>

#include <algorithm>
#include <cstring>
#include <new>

namespace {

// Box-downsample by an integer factor. Averaging every source pixel that falls
// in a destination cell, rather than dropping all but one of them: a photo
// reduced by point-sampling aliases hard, and the images that reach this path
// are precisely the huge detailed ones where that is most visible.
//
// Done on our side because it is not libjxl's job — and because reaching for
// libjxl's progressive/reduced-resolution decode instead is the RIGHT fix but
// a much larger one (see the design doc's "known limitations"). This keeps the
// GPU limit from being a hard failure today.
std::vector<uint8_t> box_downsample(const std::vector<uint8_t>& src,
                                    uint32_t sw, uint32_t sh, int factor,
                                    uint32_t& dw, uint32_t& dh) {
    dw = std::max(1u, sw / (uint32_t)factor);
    dh = std::max(1u, sh / (uint32_t)factor);
    std::vector<uint8_t> dst((size_t)dw * dh * 4);

    for (uint32_t y = 0; y < dh; ++y) {
        const uint32_t y0 = y * (uint32_t)factor;
        const uint32_t y1 = std::min(y0 + (uint32_t)factor, sh);
        for (uint32_t x = 0; x < dw; ++x) {
            const uint32_t x0 = x * (uint32_t)factor;
            const uint32_t x1 = std::min(x0 + (uint32_t)factor, sw);
            uint32_t acc[4] = {0, 0, 0, 0};
            uint32_t n = 0;
            for (uint32_t sy = y0; sy < y1; ++sy) {
                const uint8_t* row = src.data() + ((size_t)sy * sw + x0) * 4;
                for (uint32_t sx = x0; sx < x1; ++sx, row += 4) {
                    acc[0] += row[0]; acc[1] += row[1];
                    acc[2] += row[2]; acc[3] += row[3];
                    ++n;
                }
            }
            uint8_t* out = dst.data() + ((size_t)y * dw + x) * 4;
            if (n == 0) { out[0] = out[1] = out[2] = 0; out[3] = 255; continue; }
            for (int c = 0; c < 4; ++c) out[c] = (uint8_t)(acc[c] / n);
        }
    }
    return dst;
}

}  // namespace

bool looks_like_jxl(const uint8_t* data, size_t size) {
    if (!data || size < 12) return false;
    const JxlSignature sig = JxlSignatureCheck(data, size);
    return sig == JXL_SIG_CODESTREAM || sig == JXL_SIG_CONTAINER;
}

JxlImage decode_jxl(const uint8_t* data, size_t size, uint32_t maxDimension) {
    JxlImage out;

    if (!data || size == 0) { out.error = "No image to show"; return out; }
    if (!looks_like_jxl(data, size)) { out.error = "Not a JPEG XL image"; return out; }

    auto runner = JxlResizableParallelRunnerMake(nullptr);
    auto dec    = JxlDecoderMake(nullptr);
    if (!dec || !runner) { out.error = "Could not decode this image"; return out; }

    if (JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE)
            != JXL_DEC_SUCCESS ||
        JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner,
                                    runner.get()) != JXL_DEC_SUCCESS) {
        out.error = "Could not decode this image";
        return out;
    }

    // Straight (unpremultiplied) alpha, top row first — what ImageLayer's
    // create_texture() documents it wants, so no conversion pass afterwards.
    const JxlPixelFormat fmt{4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

    JxlDecoderSetInput(dec.get(), data, size);
    JxlDecoderCloseInput(dec.get());

    JxlBasicInfo info{};
    bool haveInfo = false;

    for (;;) {
        const JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
        switch (status) {
            case JXL_DEC_ERROR:
                out.error = "Could not decode this image";
                return out;

            case JXL_DEC_NEED_MORE_INPUT:
                // The input was closed above, so this means the file simply
                // stops early — a partial download or a truncated copy.
                out.error = "This image file is incomplete";
                return out;

            case JXL_DEC_BASIC_INFO: {
                if (JxlDecoderGetBasicInfo(dec.get(), &info) != JXL_DEC_SUCCESS) {
                    out.error = "Could not decode this image";
                    return out;
                }
                if (info.xsize == 0 || info.ysize == 0) {
                    out.error = "This image is empty";
                    return out;
                }
                haveInfo = true;
                JxlResizableParallelRunnerSetThreads(
                    runner.get(),
                    JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
                break;
            }

            case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
                if (!haveInfo) { out.error = "Could not decode this image"; return out; }
                size_t needed = 0;
                if (JxlDecoderImageOutBufferSize(dec.get(), &fmt, &needed)
                        != JXL_DEC_SUCCESS) {
                    out.error = "Could not decode this image";
                    return out;
                }
                // A 100-megapixel image is 400 MB of RGBA. Asking for it can
                // genuinely fail, and that is a message rather than a
                // std::terminate.
                try {
                    out.rgba.resize(needed);
                } catch (const std::bad_alloc&) {
                    out.error = "This image is too large to open";
                    return out;
                }
                if (JxlDecoderSetImageOutBuffer(dec.get(), &fmt, out.rgba.data(),
                                                out.rgba.size()) != JXL_DEC_SUCCESS) {
                    out.error = "Could not decode this image";
                    return out;
                }
                break;
            }

            case JXL_DEC_FULL_IMAGE:
                // The first frame is the image. An animated JXL has more, and
                // ViewMage deliberately shows only this one — stated in the
                // design doc rather than silently looped.
                out.w = info.xsize;
                out.h = info.ysize;
                goto decoded;

            case JXL_DEC_SUCCESS:
                // Ran out of events without a frame: a valid file containing
                // no image (metadata only).
                if (out.w == 0) { out.error = "This file contains no image"; return out; }
                goto decoded;

            default:
                break;   // an event we did not subscribe to; keep going
        }
    }

decoded:
    if (out.rgba.size() < (size_t)out.w * out.h * 4) {
        out.error = "Could not decode this image";
        return out;
    }

    // ── The GPU's ceiling ───────────────────────────────────────────────────
    if (maxDimension > 0 && (out.w > maxDimension || out.h > maxDimension)) {
        int factor = 1;
        while (out.w / (uint32_t)factor > maxDimension ||
               out.h / (uint32_t)factor > maxDimension) {
            factor *= 2;
            if (factor > 1024) { out.error = "This image is too large to open"; return out; }
        }
        uint32_t dw = 0, dh = 0;
        try {
            out.rgba = box_downsample(out.rgba, out.w, out.h, factor, dw, dh);
        } catch (const std::bad_alloc&) {
            out.error = "This image is too large to open";
            return out;
        }
        out.w = dw;
        out.h = dh;
        out.downsampleFactor = factor;
    }

    return out;
}
