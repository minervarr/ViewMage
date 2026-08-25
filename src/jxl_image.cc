#include "jxl_image.hh"

#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/resizable_parallel_runner_cxx.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

namespace {

const char* transfer_name(JxlTransferFunction tf) {
    switch (tf) {
        case JXL_TRANSFER_FUNCTION_709:    return "BT.709";
        case JXL_TRANSFER_FUNCTION_LINEAR: return "Linear";
        case JXL_TRANSFER_FUNCTION_SRGB:   return "sRGB";
        case JXL_TRANSFER_FUNCTION_PQ:     return "PQ (HDR)";
        case JXL_TRANSFER_FUNCTION_DCI:    return "DCI";
        case JXL_TRANSFER_FUNCTION_HLG:    return "HLG (HDR)";
        case JXL_TRANSFER_FUNCTION_GAMMA:  return "Gamma";
        default:                           return "Unknown";
    }
}

const char* primaries_name(JxlPrimaries p) {
    switch (p) {
        case JXL_PRIMARIES_SRGB:   return "sRGB / BT.709";
        case JXL_PRIMARIES_2100:   return "BT.2100";
        case JXL_PRIMARIES_P3:     return "Display P3";
        case JXL_PRIMARIES_CUSTOM: return "Custom";
        default:                   return "Unknown";
    }
}

// Box-downsample by an integer factor, over FLOAT data.
//
// This accumulated into uint32_t when the buffer was 8-bit. Over linear floats
// that would have been silently catastrophic -- every HDR value truncated to an
// integer, every value below 1.0 flattened to zero -- and it would only have
// fired on images large enough to hit the GPU's size limit, which are exactly
// the ones nobody tests with. Hence its own test.
//
}  // namespace

JxlImage decode_jxl(const uint8_t* data, size_t size, uint32_t maxDimension,
                    WorkingPrimaries primaries) {
    JxlImage out;

    if (!data || size == 0) { out.error = "No image to show"; return out; }
    if (!looks_like_jxl(data, size)) { out.error = "Not a JPEG XL image"; return out; }

    auto runner = JxlResizableParallelRunnerMake(nullptr);
    auto dec    = JxlDecoderMake(nullptr);
    if (!dec || !runner) { out.error = "Could not decode this image"; return out; }

    // COLOR_ENCODING must be subscribed or the event never fires and there is
    // no legal window in which to request an output profile.
    if (JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
                                             JXL_DEC_COLOR_ENCODING |
                                             JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS ||
        JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner,
                                    runner.get()) != JXL_DEC_SUCCESS) {
        out.error = "Could not decode this image";
        return out;
    }

    // The CMS, and the whole reason colours were wrong before. Without one,
    // libjxl performs no gamut or white-point conversion at all and any ICC
    // request is a hard error. Note the asymmetry: the accessor returns a
    // POINTER, the setter takes the struct BY VALUE. Must precede both any
    // ProcessInput and any SetOutputColorProfile.
    if (JxlDecoderSetCms(dec.get(), *JxlGetDefaultCms()) != JXL_DEC_SUCCESS) {
        note(out, DiagCode::kNoColorManagement, DiagLevel::kWarning,
             "Colour management unavailable",
             "This build could not start its colour engine, so colours are shown "
             "as the file stores them rather than as this screen needs them.");
    } else {
        out.colorManaged = true;
    }

    // Linear light, unclipped. Exposure is only meaningful in linear light, and
    // keeping the range above 1.0 is the entire point of the exercise.
    const JxlPixelFormat fmt{4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

    JxlDecoderSetInput(dec.get(), data, size);
    JxlDecoderCloseInput(dec.get());

    JxlBasicInfo info{};
    bool haveInfo   = false;
    int  frameCount = 0;

    for (;;) {
        const JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
        switch (status) {
            case JXL_DEC_ERROR:
                out.error = "Could not decode this image";
                return out;

            case JXL_DEC_NEED_MORE_INPUT:
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
                out.bitsPerSample   = info.bits_per_sample;
                out.exponentBits    = info.exponent_bits_per_sample;
                out.intensityTarget = info.intensity_target;
                out.minNits         = info.min_nits;
                JxlResizableParallelRunnerSetThreads(
                    runner.get(),
                    JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
                break;
            }

            case JXL_DEC_COLOR_ENCODING: {
                // The ONLY legal window for SetOutputColorProfile: after this
                // event and before any other.
                JxlColorEncoding orig{};
                const bool haveEncoded =
                    JxlDecoderGetColorAsEncodedProfile(
                        dec.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &orig)
                    == JXL_DEC_SUCCESS;

                if (haveEncoded) {
                    out.transferName  = transfer_name(orig.transfer_function);
                    out.primariesName = primaries_name(orig.primaries);
                    out.hdrTransfer   = orig.transfer_function == JXL_TRANSFER_FUNCTION_PQ ||
                                        orig.transfer_function == JXL_TRANSFER_FUNCTION_HLG;
                    out.widePrimaries = orig.color_space == JXL_COLOR_SPACE_RGB &&
                                        orig.primaries != JXL_PRIMARIES_SRGB;
                } else {
                    // An ICC-only profile. libjxl's own header warns that ICC
                    // (before v4.4) cannot say "this is PQ", so such an image is
                    // read as SDR and comes out several stops too dim. We cannot
                    // fix that; we can refuse to hide it.
                    out.transferName  = "Embedded ICC profile";
                    out.primariesName = "Embedded ICC profile";
                    note(out, DiagCode::kIccOnlyProfile, DiagLevel::kNotice,
                         "Colour profile is ambiguous",
                         "This photo carries a colour profile that cannot say whether it "
                         "is HDR. It has been treated as ordinary range. If it looks flat "
                         "or too dim, that is why.");
                }

                // Ask for linear light in the CALLER'S primaries. Whichever
                // set that is, it is not lossy here, because the buffer is
                // float and unclipped: colours outside that gamut come back as
                // negative or greater-than-one components and survive intact.
                // Doing the gamut matrix in the CMS is both more correct and
                // cheaper than repeating it per pixel in a shader.
                //
                // It MUST agree with the swapchain the shader writes into. The
                // HDR10 target is BT.2020 by definition and the compositor
                // reads it that way, so handing it sRGB-primaries numbers
                // stretches every colour onto BT.2020's much wider gamut --
                // green-cast overall, because BT.2020's green primary is the
                // furthest from sRGB's, and with a per-hue error big enough to
                // make the tone curve's desaturate-toward-white step engage at
                // a different point per hue, which reads as abrupt colour
                // transitions rather than as a gamut error. This was latent and
                // harmless under the extended-linear-scRGB target, which really
                // is sRGB primaries.
                //
                // The transfer function stays LINEAR and the white point D65
                // either way: the shader works in linear light and
                // encodeImageLinear() re-applies the PQ OETF at the end.
                JxlColorEncoding want{};
                want.color_space       = (info.num_color_channels == 1)
                                       ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB;
                want.white_point       = JXL_WHITE_POINT_D65;
                want.primaries         = (primaries == WorkingPrimaries::Bt2100)
                                       ? JXL_PRIMARIES_2100 : JXL_PRIMARIES_SRGB;
                want.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
                want.rendering_intent  = JXL_RENDERING_INTENT_RELATIVE;
                want.gamma             = 0.0;

                if (JxlDecoderSetOutputColorProfile(dec.get(), &want, nullptr, 0)
                        != JXL_DEC_SUCCESS) {
                    // libjxl refuses some XYB conversions outright. Not fatal:
                    // fall back to whatever it decides on its own and say so,
                    // rather than turning a viewable photo into an error.
                    out.colorManaged = false;
                    note(out, DiagCode::kConversionRefused, DiagLevel::kWarning,
                         "Colours shown unconverted",
                         "The decoder would not convert this photo's colours into the form "
                         "this viewer works in, so it is shown using its own colours "
                         "unchanged. Brightness and saturation may be wrong.");
                }
                break;
            }

            case JXL_DEC_FRAME:
                ++frameCount;
                break;

            case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
                if (!haveInfo) { out.error = "Could not decode this image"; return out; }
                size_t needed = 0;
                if (JxlDecoderImageOutBufferSize(dec.get(), &fmt, &needed)
                        != JXL_DEC_SUCCESS) {
                    out.error = "Could not decode this image";
                    return out;
                }
                // 16 bytes a pixel: a 100-megapixel photo is 1.6 GB and asking
                // for it can genuinely fail. That is a message, not a terminate.
                try {
                    out.linear.resize(needed / sizeof(float));
                } catch (const std::bad_alloc&) {
                    out.error = "This image is too large to open";
                    return out;
                }
                if (JxlDecoderSetImageOutBuffer(dec.get(), &fmt, out.linear.data(),
                                                out.linear.size() * sizeof(float))
                        != JXL_DEC_SUCCESS) {
                    out.error = "Could not decode this image";
                    return out;
                }
                // Deliberately NOT calling JxlDecoderSetImageOutBitDepth: it is
                // meaningless for float output and its ordering rule is a
                // landmine for no benefit.
                break;
            }

            case JXL_DEC_FULL_IMAGE:
                out.w = info.xsize;
                out.h = info.ysize;
                goto decoded;

            case JXL_DEC_SUCCESS:
                if (out.w == 0) { out.error = "This file contains no image"; return out; }
                goto decoded;

            default:
                break;
        }
    }

decoded:
    if (out.linear.size() < (size_t)out.w * out.h * 4) {
        out.error = "Could not decode this image";
        return out;
    }

    // ── The GPU's ceiling ───────────────────────────────────────────────────
    finalize_decoded(out, maxDimension, primaries);

    if (frameCount > 1) {
        note(out, DiagCode::kAnimationFirstFrame, DiagLevel::kInfo,
             "Animation, first frame only",
             "This file contains more than one frame. ViewMage shows the first.");
    }

    return out;
}

bool looks_like_jxl(const uint8_t* data, size_t size) {
    if (!data || size < 12) return false;
    const JxlSignature sig = JxlSignatureCheck(data, size);
    return sig == JXL_SIG_CODESTREAM || sig == JXL_SIG_CONTAINER;
}
