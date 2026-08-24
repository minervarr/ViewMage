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

void note(JxlImage& im, DiagCode code, DiagLevel level,
          std::string title, std::string detail) {
    im.notes.push_back(Diagnostic{code, level, std::move(title), std::move(detail)});
}

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
// Averaging rather than dropping all but one source pixel: the images that
// reach this path are the huge detailed ones, where point-sampling aliases
// hardest.
std::vector<float> box_downsample(const std::vector<float>& src,
                                  uint32_t sw, uint32_t sh, int factor,
                                  uint32_t& dw, uint32_t& dh) {
    dw = std::max(1u, sw / (uint32_t)factor);
    dh = std::max(1u, sh / (uint32_t)factor);
    std::vector<float> dst((size_t)dw * dh * 4);

    for (uint32_t y = 0; y < dh; ++y) {
        const uint32_t y0 = y * (uint32_t)factor;
        const uint32_t y1 = std::min(y0 + (uint32_t)factor, sh);
        for (uint32_t x = 0; x < dw; ++x) {
            const uint32_t x0 = x * (uint32_t)factor;
            const uint32_t x1 = std::min(x0 + (uint32_t)factor, sw);
            double acc[4] = {0, 0, 0, 0};
            uint32_t n = 0;
            for (uint32_t sy = y0; sy < y1; ++sy) {
                const float* row = src.data() + ((size_t)sy * sw + x0) * 4;
                for (uint32_t sx = x0; sx < x1; ++sx, row += 4) {
                    acc[0] += row[0]; acc[1] += row[1];
                    acc[2] += row[2]; acc[3] += row[3];
                    ++n;
                }
            }
            float* out = dst.data() + ((size_t)y * dw + x) * 4;
            if (n == 0) { out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f; continue; }
            for (int c = 0; c < 4; ++c) out[c] = (float)(acc[c] / n);
        }
    }
    return dst;
}

// ── Auto-exposure ───────────────────────────────────────────────────────────
//
// Two stages, deliberately combined rather than one chosen over the other.
//
// The ANCHOR is principled and does not look at a single pixel: libjxl's linear
// output normalises 1.0 to the file's declared peak (`intensity_target`, in
// nits), and BT.2408 puts SDR diffuse white at 203 nits. So a 1000-nit PQ photo
// wants about -2.3 EV, and that alone removes most of the washout. It is
// data-independent, so it cannot be fooled by an unusual picture.
//
// The HISTOGRAM is the sanity check, because some files declare a nonsense peak
// and some are simply dark. It is applied at HALF weight and clamped to +/-2 EV
// from the anchor: an auto-exposure that swings between two frames of the same
// burst is worse than one that is slightly conservative.
struct AutoExposure { float ev; float white; };

AutoExposure compute_auto_exposure(const std::vector<float>& px,
                                   uint32_t w, uint32_t h,
                                   float intensityTarget, bool isHdr) {
    // An ordinary image is shown EXACTLY as authored. No anchor, no histogram,
    // no opinion -- 0 EV and a white point of 1.0, which makes the tone curve a
    // provable identity and leaves an SDR photo byte-for-byte what it always
    // was.
    //
    // This is not a shortcut, it is the correct behaviour, and getting it wrong
    // was visible immediately: libjxl reports intensity_target = 255 for SDR
    // content, so feeding that to the 203-nit rule below darkened every
    // ordinary photo by a third of a stop. Auto-exposure exists to cope with a
    // range the display cannot show. When there is no such range, there is
    // nothing to cope with.
    if (!isHdr && intensityTarget <= 255.0f) return AutoExposure{0.0f, 1.0f};

    const float anchorGain = (intensityTarget > 0.0f) ? (203.0f / intensityTarget) : 1.0f;
    const float evAnchor   = std::log2(std::max(anchorGain, 1e-6f));

    // ~200k samples whatever the image size: enough for a stable percentile,
    // cheap enough to not be noticed next to the decode.
    const double total = (double)w * h;
    uint32_t stride = (uint32_t)std::max(1.0, std::sqrt(total / 200000.0));

    constexpr int kBins = 256;          // -16..+16 EV in 1/8-EV steps
    uint64_t hist[kBins] = {0};
    uint64_t count = 0;

    for (uint32_t y = 0; y < h; y += stride) {
        const float* row = px.data() + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; x += stride) {
            const float* p = row + (size_t)x * 4;
            const float L = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
            if (!(L > 0.0f)) continue;   // also rejects NaN
            int bin = (int)((std::log2(L) + 16.0f) * 8.0f);
            bin = std::clamp(bin, 0, kBins - 1);
            ++hist[bin];
            ++count;
        }
    }

    AutoExposure out{evAnchor, 1.0f};
    if (count == 0) {
        out.ev = std::clamp(evAnchor, -8.0f, 8.0f);
        return out;
    }

    auto percentile = [&](double frac) {
        const uint64_t want = (uint64_t)(frac * (double)count);
        uint64_t seen = 0;
        for (int b = 0; b < kBins; ++b) {
            seen += hist[b];
            if (seen >= want) return std::exp2((float)b / 8.0f - 16.0f);
        }
        return 1.0f;
    };

    const float l995  = percentile(0.995);
    const float evPct = -std::log2(std::max(l995, 1e-4f));   // put p99.5 at 1.0

    float ev = evAnchor + std::clamp(evPct - evAnchor, -2.0f, 2.0f) * 0.5f;
    ev = std::clamp(ev, -8.0f, 8.0f);
    out.ev = ev;

    // Where the tone curve's knee should call "white": the extreme highlight,
    // after exposure. An SDR image lands at 1.0, which makes the curve an exact
    // identity, so an ordinary photo renders exactly as it did before.
    const float l9999 = percentile(0.9999);
    out.white = std::clamp(l9999 * std::exp2(ev), 1.0f, 16.0f);
    return out;
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

                // Ask for linear light with sRGB primaries. Choosing sRGB
                // primaries is NOT lossy here, because the buffer is float and
                // unclipped: colours outside that gamut come back as negative or
                // greater-than-one components and survive intact. Doing the
                // gamut matrix in the CMS is both more correct and cheaper than
                // repeating it per pixel in a shader.
                JxlColorEncoding want{};
                want.color_space       = (info.num_color_channels == 1)
                                       ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB;
                want.white_point       = JXL_WHITE_POINT_D65;
                want.primaries         = JXL_PRIMARIES_SRGB;
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
    if (maxDimension > 0 && (out.w > maxDimension || out.h > maxDimension)) {
        int factor = 1;
        while (out.w / (uint32_t)factor > maxDimension ||
               out.h / (uint32_t)factor > maxDimension) {
            factor *= 2;
            if (factor > 1024) { out.error = "This image is too large to open"; return out; }
        }
        uint32_t dw = 0, dh = 0;
        try {
            out.linear = box_downsample(out.linear, out.w, out.h, factor, dw, dh);
        } catch (const std::bad_alloc&) {
            out.error = "This image is too large to open";
            return out;
        }
        out.w = dw;
        out.h = dh;
        out.downsampleFactor = factor;
    }

    const AutoExposure ae = compute_auto_exposure(out.linear, out.w, out.h,
                                                  out.intensityTarget,
                                                  out.hdrTransfer || out.widePrimaries);
    out.autoEv = ae.ev;
    out.white  = ae.white;

    // ── What to tell the user ───────────────────────────────────────────────
    if (out.hdrTransfer) {
        note(out, DiagCode::kHdrTransfer, DiagLevel::kInfo,
             "HDR photo on a standard-range screen",
             std::string("This photo is stored in ") + out.transferName +
             ", which holds far brighter highlights than this screen can show at "
             "once. The bright parts have been compressed to fit. Move the "
             "exposure slider to look inside them -- the data is all still here.");
    }
    if (out.widePrimaries) {
        note(out, DiagCode::kWidePrimaries, DiagLevel::kNotice,
             "More colours than the screen has",
             std::string("The photo uses the ") + out.primariesName +
             " colour range, which is wider than this screen's. The most vivid "
             "colours are shown as close as the screen can get.");
    }
    if (out.bitsPerSample > 8 || out.exponentBits > 0) {
        note(out, DiagCode::kHighBitDepth, DiagLevel::kInfo,
             "More precision than the screen has",
             "Stored with " + std::to_string(out.bitsPerSample) +
             " bits per colour; the screen shows 8. Everything is kept at full "
             "precision in memory, so moving the exposure reveals detail that a "
             "plain 8-bit viewer would already have thrown away.");
    }
    if (out.downsampleFactor > 1) {
        note(out, DiagCode::kDownsampledForGpu, DiagLevel::kWarning,
             "Shown smaller than it is",
             "This photo is larger than this device's graphics hardware can hold, "
             "so it is displayed at 1/" + std::to_string(out.downsampleFactor) +
             " size. Detail visible here is not the full detail in the file.");
    }
    if (frameCount > 1) {
        note(out, DiagCode::kAnimationFirstFrame, DiagLevel::kInfo,
             "Animation, first frame only",
             "This file contains more than one frame. ViewMage shows the first.");
    }

    return out;
}
