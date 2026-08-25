// decoded_image.cc -- everything about a decoded image that is NOT about its
// format.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// ---------------------------------------------------------------------------
// Downsampling to the GPU's ceiling, the auto exposure, and the notes that tell
// the user how a file exceeds what the screen can show are all statements about
// LINEAR LIGHT and about what the file claimed of itself. None of them has
// anything to do with JPEG XL or with TIFF.
//
// They live here, in a file that depends on NEITHER decoder, for two reasons.
// The obvious one is that a second copy would drift, and the drift would show
// as "the same photo looks different depending which format I saved it as".
// The other is dependency direction: dng_image.cc must not have to link libjxl
// to find out how bright to show a picture, and its test must stay runnable
// with no libjxl on the machine at all.
// ---------------------------------------------------------------------------

#include "jxl_image.hh"

#include "log.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

void note(DecodedImage& im, DiagCode code, DiagLevel level,
          std::string title, std::string detail) {
    im.notes.push_back(Diagnostic{code, level, std::move(title), std::move(detail)});
}

namespace {


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
// nits) -- TF_PQ::DisplayFromEncoded returns d * (10000 / intensity_target), so
// a sample of 1.0 IS intensity_target nits -- while the extended-linear scRGB
// surface we present to normalises 1.0 to SDR diffuse white, 203 nits per
// BT.2408. Going from the first scale to the second is therefore a MULTIPLY by
// intensity_target/203: a 1000-nit PQ photo wants about +2.3 EV, and a
// 3439-nit one about +4.1 EV.
//
// This was inverted (203/intensity_target) and cost 2x the exposure error --
// -4.1 EV instead of +4.1, ~287x too dark. A 32-nit midtone reached the panel
// at 0.22 nits, so the whole picture read as black at full brightness. The
// symptom it was written against ("washout") predates JxlDecoderSetCms below:
// once the CMS ran, the compensation became a second darkening.
//
// It is data-independent, so it cannot be fooled by an unusual picture.
//
// The HISTOGRAM is the sanity check, because some files declare a nonsense peak
// and some are simply dark. It is applied at HALF weight and clamped to +/-2 EV
// from the anchor: an auto-exposure that swings between two frames of the same
// burst is worse than one that is slightly conservative.
struct AutoExposure { float ev; float white; };

constexpr float kLumaBt709[3]  = {0.2126f, 0.7152f, 0.0722f};
constexpr float kLumaBt2020[3] = {0.2627f, 0.6780f, 0.0593f};

AutoExposure compute_auto_exposure(const std::vector<float>& px,
                                   uint32_t w, uint32_t h,
                                   float intensityTarget, bool isHdr,
                                   WorkingPrimaries primaries) {
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

    const float anchorGain = (intensityTarget > 0.0f) ? (intensityTarget / 203.0f) : 1.0f;
    // CLAMPED, because intensity_target is a self-declaration and nothing
    // validates it. A file claiming a 100000-nit peak asks for +9 EV and blows
    // the picture to white before the half-weight histogram term below gets a
    // chance to argue. +4 EV is 203 * 16 = ~3250 nits, past the peak of any
    // consumer panel, so the clamp cannot bite on an honestly-tagged file --
    // it only refuses nonsense. Same spirit as the histogram's +/-2 EV clamp:
    // conservative beats a viewer that occasionally goes white.
    const float evAnchor   = std::clamp(std::log2(std::max(anchorGain, 1e-6f)),
                                        -4.0f, 4.0f);

    // ~200k samples whatever the image size: enough for a stable percentile,
    // cheap enough to not be noticed next to the decode.
    const double total = (double)w * h;
    uint32_t stride = (uint32_t)std::max(1.0, std::sqrt(total / 200000.0));

    constexpr int kBins = 256;          // -16..+16 EV in 1/8-EV steps
    uint64_t hist[kBins] = {0};
    uint64_t count = 0;

    // Luminance weights have to match the primaries the decode actually landed
    // in, the same way the tone curve's do in image_frag.slang. Feeding BT.2020
    // pixels the BT.709 weights over-counts green and mis-places the percentile.
    // The error is small here -- this term is half-weight and clamped to +/-2 EV,
    // and the p99.5 of a whole image is not very hue-sensitive -- but there is no
    // reason for the two luminance definitions in this program to disagree.
    const float* lumaW = (primaries == WorkingPrimaries::Bt2100)
                       ? kLumaBt2020 : kLumaBt709;

    for (uint32_t y = 0; y < h; y += stride) {
        const float* row = px.data() + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; x += stride) {
            const float* p = row + (size_t)x * 4;
            const float L = lumaW[0] * p[0] + lumaW[1] * p[1] + lumaW[2] * p[2];
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

// ── The display render ──────────────────────────────────────────────────────
//
// A developed RAW is SCENE-REFERRED: black subtract, white balance, CCM,
// transfer, and nothing else. That is the right thing to archive and the wrong
// thing to look at. Scene-linear light has no toe, so shadows sit exactly where
// the sensor put them and read as lifted and grey next to any camera JPEG,
// which crushes its own shadows hard. The complaint that started this ("the
// blacks don't look like one") is that missing curve, NOT a black-level error --
// the sensor's floor was measured and is correct (see the long comment in
// dng_image.cc).
//
// This is a TOE, not an S-curve, and the difference is the whole design:
//
//     log2 output = log2 input * (1 + (kContrast-1) * g(u))
//
// with u the input's distance above mid grey in stops and g a smoothstep that is
// 1 at mid grey and 0 by diffuse white. So the slope is kContrast in the shadows
// and returns to EXACTLY 1 at and above diffuse white. Highlights are therefore
// numerically untouched.
//
// That last property is not fastidiousness -- it is a bug fix. The first version
// here was a straight log-log line (slope kContrast everywhere), which amplifies
// anything above the pivot by (L/pivot)^0.25. Measured on a real night frame it
// pushed p99 from 1526 nits to 3878 and the peak past 8600. Highlights then get
// compressed a SECOND time by the shader's rolloff, which is the pass that
// actually knows the display's headroom, and stacking the two flattens speculars
// into grey mush. Compression belongs where the display limit is known; this
// pass only has business in the shadows.
//
// PIVOT: where mid grey ACTUALLY LANDS, i.e. 18% of diffuse white pulled back
// through the auto-exposure this picture was given. Anchoring to a fixed
// scene-referred pivot instead was the other half of that first bug -- a dark
// scene sits well below the fixed pivot, so nearly every pixel fell on the
// darkening side and the curve read as a big underexposure. Because the pivot is
// a fixed point of the curve, mid grey does not move: the picture gains
// contrast, not an exposure shift the auto-exposure would then fight.
//
// APPLIED ON LUMINANCE, with the channels scaled by the resulting ratio.
// Powering each channel independently is the usual shortcut and it distorts hue,
// pulling saturation up in the shadows. A luminance ratio holds chromaticity
// exactly, so only the tone changes.
namespace {

constexpr float kContrast = 1.25f;    // log-log slope at mid grey; 1.0 = no-op
constexpr float kSdrDiffuseWhiteNits = 203.0f;
constexpr float kMidGrey = 0.18f;

}  // namespace

void apply_display_render(DecodedImage& im, WorkingPrimaries primaries) {
    if (im.w == 0 || im.h == 0 || im.linear.empty()) return;
    if (!(im.intensityTarget > 0.0f)) return;

    // Scene value that the renderer's exposure will put at display mid grey.
    const float displayWhite = kSdrDiffuseWhiteNits / im.intensityTarget;
    const float pivot = kMidGrey * displayWhite * std::pow(2.0f, -im.autoEv);
    if (!(pivot > 0.0f) || !std::isfinite(pivot)) return;

    // Stops from mid grey up to diffuse white: where the toe has faded out.
    const float uWhite = -std::log2(kMidGrey);   // ~2.47

    const float wr = (primaries == WorkingPrimaries::Bt2100) ? 0.2627f : 0.2126f;
    const float wg = (primaries == WorkingPrimaries::Bt2100) ? 0.6780f : 0.7152f;
    const float wb = (primaries == WorkingPrimaries::Bt2100) ? 0.0593f : 0.0722f;

    const size_t n = size_t(im.w) * im.h;
    for (size_t i = 0; i < n; ++i) {
        float* p = &im.linear[i * 4];
        const float L = wr * p[0] + wg * p[1] + wb * p[2];
        // Below this the pixel is deep in the noise floor and the ratio would be
        // numerically meaningless; the curve sends it to ~0 anyway.
        if (!(L > 1e-8f)) { p[0] = p[1] = p[2] = 0.0f; continue; }

        const float u = std::log2(L / pivot);
        if (u >= uWhite) continue;              // at or above diffuse white: untouched

        const float t = std::max(u, 0.0f) / uWhite;
        const float g = 1.0f - t * t * (3.0f - 2.0f * t);   // 1 at grey, 0 at white
        const float s = std::exp2(u * (kContrast - 1.0f) * g);
        p[0] *= s; p[1] *= s; p[2] *= s;
    }

    VCE_LOGI("ViewMage", "display render: toe x%.2f, pivot %.1f nits (autoEV %.2f)",
             kContrast, pivot * im.intensityTarget, im.autoEv);

    im.notes.push_back(Diagnostic{
        DiagCode::kHighBitDepth, DiagLevel::kInfo,
        "Rendered for the screen",
        "A display contrast curve has been applied to the shadows so the picture "
        "reads the way the scene looked. The file itself is untouched."});
}

void finalize_decoded(DecodedImage& out, uint32_t maxDimension, WorkingPrimaries primaries) {
    if (out.w == 0 || out.h == 0 || !out.error.empty()) return;

    if (maxDimension > 0 && (out.w > maxDimension || out.h > maxDimension)) {
        int factor = 1;
        while (out.w / (uint32_t)factor > maxDimension ||
               out.h / (uint32_t)factor > maxDimension) {
            factor *= 2;
            if (factor > 1024) { out.error = "This image is too large to open"; return; }
        }
        uint32_t dw = 0, dh = 0;
        try {
            out.linear = box_downsample(out.linear, out.w, out.h, factor, dw, dh);
        } catch (const std::bad_alloc&) {
            out.error = "This image is too large to open";
            return;
        }
        out.w = dw;
        out.h = dh;
        out.downsampleFactor = factor;
    }

    const AutoExposure ae = compute_auto_exposure(out.linear, out.w, out.h,
                                                  out.intensityTarget,
                                                  out.hdrTransfer || out.widePrimaries,
                                                  primaries);
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
}
