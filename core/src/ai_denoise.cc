// ai_denoise.cc — see ai_denoise.hh for what this model is and why it sits
// before the demosaic rather than after it.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// Ported from the reference implementation in darktable-ai's
// models/rawdenoise-nind/demo.py (release-5.6.0). Three details in there are
// load-bearing and each one fails in a way that looks like a colour bug rather
// than a denoise bug, so each is called out at its site below:
//
//   1. plane order [R, G1, G2, B] and which green is "G1"
//   2. mirror-padded tiling with the overlap TRIMMED, not blended
//   3. the global gain match against the input mean, after stitching
#include "core/ai_denoise.hh"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "log.hh"

#ifdef VIEWMAGE_WITH_ORT
#include "onnxruntime_cxx_api.h"
#endif

// ── Packing ─────────────────────────────────────────────────────────────────
//
// The mosaic's 2x2 cell becomes four half-resolution planes. Which plane a cell
// position feeds is decided by the CFA pattern, and G1/G2 are simply the first
// and second green in raster order -- the model does not care which physical
// green is which, but it does care that they stay in that order.
void pack_bayer_rggb(const uint16_t* mosaic, int w, int h, int stride_px,
                     int cfa, const double black[4], double white,
                     std::vector<float>& packed, int& pw, int& ph,
                     int& ox, int& oy,
                     std::vector<uint8_t>* clip_mask) {
    // 2x2 colour codes in raster order: 0=R, 1=G, 2=B. Same table as the
    // camera's dng_writer.cc cfa_pattern(), kept in the same order on purpose.
    static const int kPat[4][4] = {
        {0, 1, 1, 2},   // RGGB
        {1, 0, 2, 1},   // GRBG
        {1, 2, 0, 1},   // GBRG
        {2, 1, 1, 0},   // BGGR
    };
    const int* pat0 = kPat[(cfa >= 0 && cfa < 4) ? cfa : 0];

    // Shift onto RGGB -- see the header. RGGB needs (0,0), GRBG (1,0),
    // GBRG (0,1), BGGR (1,1); solved rather than tabulated so a wrong CFA code
    // cannot silently pick a wrong offset.
    ox = oy = 0;
    for (int dy = 0; dy < 2 && !(ox || oy); ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int c00 = pat0[((0 + dy) & 1) * 2 + ((0 + dx) & 1)];
            const int c01 = pat0[((0 + dy) & 1) * 2 + ((1 + dx) & 1)];
            const int c10 = pat0[((1 + dy) & 1) * 2 + ((0 + dx) & 1)];
            const int c11 = pat0[((1 + dy) & 1) * 2 + ((1 + dx) & 1)];
            if (c00 == 0 && c01 == 1 && c10 == 1 && c11 == 2) { ox = dx; oy = dy; }
            if (ox || oy) break;
        }
    }

    // After the shift the pattern IS RGGB, by construction.
    static const int kRggb[4] = {0, 1, 1, 2};
    const int* pat = kRggb;

    mosaic += size_t(oy) * stride_px + ox;
    w -= ox;
    h -= oy;

    pw = w / 2;
    ph = h / 2;
    packed.assign(size_t(pw) * ph * 4, 0.0f);
    if (clip_mask) clip_mask->assign(size_t(pw) * ph, 0);

    // Destination plane for each of the four cell positions. Greens are assigned
    // in the order encountered, which is what "G1 then G2" means.
    int dst[4];
    int green_seen = 0;
    for (int i = 0; i < 4; ++i) {
        switch (pat[i]) {
            case 0:  dst[i] = 0; break;                       // R
            case 2:  dst[i] = 3; break;                       // B
            default: dst[i] = (green_seen++ == 0) ? 1 : 2; break;  // G1, G2
        }
    }

    const size_t plane = size_t(pw) * ph;
    for (int i = 0; i < 4; ++i) {
        const int dy = i / 2, dx = i % 2;
        // Black level is per CFA ELEMENT and normalisation is per channel: a
        // sensor whose channels sit at different blacks would otherwise get a
        // per-channel gain error, i.e. a tint, which is exactly the class of bug
        // this whole file is easy to introduce.
        const double bl = black[(((i / 2) + oy) & 1) * 2 + (((i % 2) + ox) & 1)];
        const double span = std::max(white - bl, 1.0);
        float* out = packed.data() + size_t(dst[i]) * plane;
        for (int y = 0; y < ph; ++y) {
            const uint16_t* src = mosaic + size_t(y * 2 + dy) * stride_px + dx;
            float* row = out + size_t(y) * pw;
            for (int x = 0; x < pw; ++x) {
                const double v = (double(src[size_t(x) * 2]) - bl) / span;
                // Clipped to [0,1]: the range the model was trained on. This is
                // why highlight reconstruction downstream is approximate on the
                // AI path -- see the comment at its call site in dng_image.cc.
                row[x] = float(v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
                if (clip_mask && v >= 1.0)
                    (*clip_mask)[size_t(y) * pw + x] |= uint8_t(1u << dst[i]);
            }
        }
    }
}

#ifndef VIEWMAGE_WITH_ORT

// ── No ONNX Runtime ─────────────────────────────────────────────────────────
// The class still exists so callers need no #ifdefs and the fallback path in
// the develop is the same code in both builds.
struct RawDenoiser::Impl {};
RawDenoiser::RawDenoiser() = default;
RawDenoiser::~RawDenoiser() = default;

std::unique_ptr<RawDenoiser> RawDenoiser::load(const uint8_t*, size_t,
                                               std::string& err) {
    err = "this build has no neural denoiser";
    return nullptr;
}

bool RawDenoiser::run(const float*, int, int, std::vector<float>&,
                      std::string& err, const std::function<bool(float)>&,
                      bool) {
    err = "this build has no neural denoiser";
    return false;
}

#else

struct RawDenoiser::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "viewmage"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::string in_name, out_name;
};

RawDenoiser::RawDenoiser() : impl_(new Impl) {}
RawDenoiser::~RawDenoiser() = default;

std::unique_ptr<RawDenoiser> RawDenoiser::load(const uint8_t* onnx, size_t bytes,
                                               std::string& err) {
    if (!onnx || bytes == 0) { err = "no model data"; return nullptr; }

    std::unique_ptr<RawDenoiser> d(new RawDenoiser);
    try {
        auto& im = *d->impl_;
        im.opts.SetIntraOpNumThreads(0);   // 0 = all cores
        im.opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        d->provider_ = "CPU";

        im.session.reset(new Ort::Session(im.env, onnx, bytes, im.opts));

        Ort::AllocatorWithDefaultOptions alloc;
        im.in_name  = im.session->GetInputNameAllocated(0, alloc).get();
        im.out_name = im.session->GetOutputNameAllocated(0, alloc).get();

        // The tile geometry above is only correct if the export really is the
        // static 512x512 one. Check rather than trust: a differently-exported
        // model would otherwise produce silently mis-stitched output.
        const auto shape = im.session->GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[1] != 4 ||
            shape[2] != kAiTile || shape[3] != kAiTile) {
            err = "model input is not [1,4,512,512]";
            return nullptr;
        }
        // The output must be the 3-plane NCHW camera-RGB the blit reads. Its
        // absence once hid the interleaved-vs-planar bug for weeks: the model
        // silently swapped and the image developed grey and mosaicked, so this
        // check is the tripwire, not a formality.
        const auto oshape = im.session->GetOutputTypeInfo(0)
                                .GetTensorTypeAndShapeInfo().GetShape();
        if (oshape.size() != 4 || oshape[1] != 3 ||
            oshape[2] != kAiTile * kAiScale ||
            oshape[3] != kAiTile * kAiScale) {
            err = "model output is not [1,3,1024,1024]";
            return nullptr;
        }
    } catch (const Ort::Exception& e) {
        err = std::string("could not load the denoiser: ") + e.what();
        return nullptr;
    }
    return d;
}

// ── The self-ensemble: all four phase-exact Bayer symmetries ────────────────
//
// The model reads channel k as "the sample at cell offset k", and expects those
// offsets to hold R, G, G, B. Any transform is usable only if the transformed
// mosaic STILL reads R,G,G,B at offsets 0..3 -- channel index and cell offset
// are bound together and permuting the planes breaks that binding rather than
// repairing it.
//
// The four symmetries that preserve the 2x2 cell's spatial phase:
//
//   identity       RGGB               planes [0,1,2,3]          usable directly
//   transpose      R G2 / G1 B        planes [0,2,1,3]          usable directly
//   180° rotation  BGGR               planes [3,2,1,0]          needs 1px crop
//   anti-transpose B G1 / G2 R        planes [3,1,2,0]          needs 1px crop
//
// 180° and anti-transpose produce BGGR; re-forcing RGGB requires cropping one
// packed pixel (one mosaic row and column) from the transformed array. The crop
// shifts the cell alignment back to RGGB. The lost border is one photosite wide
// on each side -- negligible against the 64-pixel overlap that already provides
// real context on every side of every tile.
//
// A previous version "fixed" 180 and anti-transpose by permuting the input
// planes and exchanging the model's R and B outputs. That is wrong twice over:
// permuting inputs decouples channel from offset (a one-pixel diagonal chroma
// shift), and the output swap then contributed R where B belonged. Averaged
// with the good passes it drove R and B toward their mean -- a washed, magenta
// picture with green fringes on every high-contrast edge. Measured on a real
// frame: R/G 1.165 -> 1.426 and B/G 1.063 -> 0.816.
namespace {
struct Symmetry {
    bool transpose;   // swap x/y before possible 180° rotation
    bool rotate180;   // 180° reversal + 1px crop to re-force RGGB
    int plane[4];     // plane[k] = source plane for model channel k
};
constexpr Symmetry kSyms[4] = {
    {false, false, {0, 1, 2, 3}},   // identity
    {true,  false, {0, 2, 1, 3}},   // transpose (greens exchanged)
    {false, true,  {3, 2, 1, 0}},   // 180° rotation + crop
    {true,  true,  {3, 1, 2, 0}},   // anti-transpose (transpose + 180°) + crop
};
}  // namespace

bool RawDenoiser::run(const float* packed, int pw, int ph,
                      std::vector<float>& out, std::string& err,
                      const std::function<bool(float)>& progress,
                      bool ensemble) {
    if (!packed || pw <= 0 || ph <= 0) { err = "nothing to denoise"; return false; }

    const int passes = ensemble ? (int)(sizeof(kSyms) / sizeof(kSyms[0])) : 1;
    if (passes > 1) {
        // Accumulate in place: each pass produces a full-size image, so holding
        // all four at once would be four times the largest allocation here.
        // acc2 accumulates SQUARES alongside: the cross-pass variance is the
        // ensemble's own per-pixel confidence, consumed by the regularizer
        // below — disagreement between symmetries is where the mean cannot be
        // trusted at fine scale.
        std::vector<float> acc, acc2, one, src;
        const size_t n = size_t(pw) * kAiScale * ph * kAiScale * 3;
        try { acc.assign(n, 0.0f); acc2.assign(n, 0.0f);
              src.resize(size_t(pw) * ph * 4); }
        catch (const std::bad_alloc&) { err = "not enough memory to denoise"; return false; }

        // Input channel means for gain matching (same for all passes).
        const size_t in_plane = size_t(pw) * ph;
        double in_ch[3] = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < in_plane; ++i) {
            in_ch[0] += packed[i];
            in_ch[1] += packed[in_plane + i] + packed[in_plane * 2 + i];
            in_ch[2] += packed[in_plane * 3 + i];
        }
        in_ch[0] /= double(in_plane);
        in_ch[1] /= double(in_plane) * 2.0;
        in_ch[2] /= double(in_plane);

        for (int p = 0; p < passes; ++p) {
            const Symmetry& sym = kSyms[p];
            // 180° and anti-transpose crop one packed pixel to re-force RGGB,
            // so the model sees (pw-1)×(ph-1) or (ph-1)×(pw-1).
            const int tw = sym.rotate180
                ? (sym.transpose ? ph - 1 : pw - 1)
                : (sym.transpose ? ph : pw);
            const int th = sym.rotate180
                ? (sym.transpose ? pw - 1 : ph - 1)
                : (sym.transpose ? pw : ph);

            // Transform the packed mosaic into the model's expected layout.
            if (sym.rotate180) {
                // 180° rotation (with optional prior transpose) and 1px crop.
                // 180° turns RGGB into BGGR; cropping one packed pixel from the
                // top-left realigns the cell to RGGB.
                for (int c = 0; c < 4; ++c) {
                    const float* sp = packed + size_t(sym.plane[c]) * pw * ph;
                    float* dp = src.data() + size_t(c) * size_t(tw) * th;
                    for (int iy = 0; iy < th; ++iy) {
                        for (int ix = 0; ix < tw; ++ix) {
                            int px, py;
                            if (sym.transpose) {
                                // Anti-transpose: (ix,iy) → original (pw-2-iy, ph-2-ix)
                                px = pw - 2 - iy;
                                py = ph - 2 - ix;
                            } else {
                                // 180°: (ix,iy) → original (pw-2-ix, ph-2-iy)
                                px = pw - 2 - ix;
                                py = ph - 2 - iy;
                            }
                            dp[size_t(iy) * tw + ix] = sp[size_t(py) * pw + px];
                        }
                    }
                }
            } else if (sym.transpose) {
                for (int c = 0; c < 4; ++c) {
                    const float* sp = packed + size_t(sym.plane[c]) * pw * ph;
                    float* dp = src.data() + size_t(c) * pw * ph;
                    for (int y = 0; y < th; ++y)
                        for (int x = 0; x < tw; ++x) {
                            dp[size_t(y) * tw + x] = sp[size_t(x) * pw + y];
                        }
                }
            } else {
                for (int c = 0; c < 4; ++c) {
                    const float* sp = packed + size_t(sym.plane[c]) * pw * ph;
                    float* dp = src.data() + size_t(c) * pw * ph;
                    std::memcpy(dp, sp, size_t(pw) * ph * sizeof(float));
                }
            }

            auto sub = progress ? std::function<bool(float)>(
                [&](float f) { return progress((p + f) / (float)passes); })
                                : std::function<bool(float)>();
            if (!run(src.data(), tw, th, one, err, sub, /*ensemble=*/false))
                return false;

            // Undo the symmetry while accumulating. For 180° and anti-transpose
            // the input planes were REVERSED ({3,2,1,0} / {3,1,2,0}), so the
            // model's channel-0 output is actually B and channel-2 is R --
            // swap them back during accumulation or the 4-pass average mixes
            // R and B toward their mean (a desaturated, near-grey image).
            // Identity and transpose leave channels in place, no swap needed.
            const int ow = tw * kAiScale, oh = th * kAiScale;
            const int fw = pw * kAiScale;
            if (sym.rotate180) {
                for (int y = 0; y < oh; ++y)
                    for (int x = 0; x < ow; ++x) {
                        int fx, fy;
                        if (sym.transpose) {
                            // Anti-transpose undo: x/y swapped in source.
                            fx = 2 * pw - 4 - 2 * (y >> 1) + (y & 1);
                            fy = 2 * ph - 4 - 2 * (x >> 1) + (x & 1);
                        } else {
                            // 180° undo: reverse both axes.
                            fx = 2 * pw - 4 - x + 2 * (x & 1);
                            fy = 2 * ph - 4 - y + 2 * (y & 1);
                        }
                        const size_t di = size_t(fy) * fw + fx;
                        const float* s = &one[(size_t(y) * ow + x) * 3];
                        float* d = &acc[di * 3];
                        d[0] += s[2]; d[1] += s[1]; d[2] += s[0];  // R<->B swap
                        acc2[di * 3]     += s[2] * s[2];
                        acc2[di * 3 + 1] += s[1] * s[1];
                        acc2[di * 3 + 2] += s[0] * s[0];
                    }
            } else {
                for (int y = 0; y < oh; ++y)
                    for (int x = 0; x < ow; ++x) {
                        const size_t di = sym.transpose
                            ? (size_t(x) * fw + y)
                            : (size_t(y) * fw + x);
                        const float* s = &one[(size_t(y) * ow + x) * 3];
                        float* d = &acc[di * 3];
                        d[0] += s[0]; d[1] += s[1]; d[2] += s[2];
                        acc2[di * 3]     += s[0] * s[0];
                        acc2[di * 3 + 1] += s[1] * s[1];
                        acc2[di * 3 + 2] += s[2] * s[2];
                    }
            }
        }
        const float inv = 1.0f / (float)passes;
        for (float& v : acc) v *= inv;
        for (float& v : acc2) v *= inv;

        // ── Disagreement-guided speckle suppression ─────────────────────────
        //
        // The four symmetry passes disagree per pixel; their mean carries that
        // disagreement as spurious fine-scale energy. Measured on a night
        // frame with the patch grid in --compare, its magnitude is ERRATIC at
        // scene scale — 0.13x..2.61x of Malvar's Laplacian on far/low-detail
        // patches, ~1.1x on textured ones — and spatially incoherent variance
        // is what reads as a blocky "pixel art" texture on distant subjects.
        //
        // Wiener-style shrink of each pixel's high band against its own 3x3
        // low band, gated by cross-pass std s: where s rivals local structure
        // (unreliable prediction, typically far/soft content), the high band
        // fades to zero; where structure dominates (near detail), untouched.
        // Scale-invariant: s and |hp| share the net's arbitrary output scale,
        // so this runs BEFORE the gain match.
        {
            const int W2 = pw * kAiScale, H2 = ph * kAiScale;
            constexpr float kSpeckleK = 1.2f;
            std::vector<float> low(size_t(W2) * H2 * 3);
            // 3x3 box low-pass per channel. O(9N), runs once per denoise.
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < H2; ++y)
                    for (int x = 0; x < W2; ++x) {
                        double sum = 0; int cnt = 0;
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int yy = std::min(std::max(y + dy, 0), H2 - 1);
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int xx = std::min(std::max(x + dx, 0), W2 - 1);
                                sum += acc[(size_t(yy) * W2 + xx) * 3 + c];
                                ++cnt;
                            }
                        }
                        low[(size_t(y) * W2 + x) * 3 + c] = float(sum / cnt);
                    }
            }
            // SOFT THRESHOLD, not gain shrink: a Wiener-style gain flattened
            // whole weak-texture patches toward their local mean (measured
            // 0.04x..0.5x of Malvar — trading speckle for mush). Subtracting
            // the disagreement magnitude from each |hp| removes exactly the
            // coefficients the passes could not agree on and leaves anything
            // stronger intact, so far-field structure survives at its own
            // amplitude while the disagreement sprinkle disappears.
            for (int y = 0; y < H2; ++y)
                for (int x = 0; x < W2; ++x) {
                    const size_t i3 = (size_t(y) * W2 + x) * 3;
                    for (int c = 0; c < 3; ++c) {
                        const float hp   = acc[i3 + c] - low[i3 + c];
                        const float varS =
                            std::max(acc2[i3 + c] - acc[i3 + c] * acc[i3 + c],
                                     0.0f);
                        const float s = std::sqrt(varS);   // pass std, net units
                        const float mag = std::fabs(hp);
                        const float kept =
                            std::max(mag - kSpeckleK * s, 0.0f);
                        acc[i3 + c] = low[i3 + c] +
                                      ((mag > 0.f) ? hp / mag : 0.f) * kept;
                    }
                }
        }

        // Per-channel gain match on the accumulated result. The network emits
        // an arbitrary learned scale; without this every photo is the wrong
        // brightness. Same trimmed-mean logic as the single-pass path.
        // Each pass ALREADY gain-matched itself against its own transformed
        // input, so these gains land near 1.0 by construction -- they exist to
        // absorb what averaging the four symmetries leaves behind, not to fix
        // the net's scale twice.
        const size_t n_acc = size_t(pw) * kAiScale * ph * kAiScale;
        const size_t acc_step = std::max<size_t>(1, n_acc / 40000);
        const size_t n_asamp = n_acc / acc_step;
        std::vector<float> aR, aG, aB;
        aR.reserve(n_asamp); aG.reserve(n_asamp); aB.reserve(n_asamp);
        for (size_t i = 0; i < n_acc; i += acc_step) {
            aR.push_back(acc[i * 3]);
            aG.push_back(acc[i * 3 + 1]);
            aB.push_back(acc[i * 3 + 2]);
        }
        std::sort(aR.begin(), aR.end());
        std::sort(aG.begin(), aG.end());
        std::sort(aB.begin(), aB.end());
        const double trim = 0.05;
        const size_t lo = (size_t)(trim * aR.size());
        const size_t hi = aR.size() - (size_t)(trim * aR.size());
        auto trmean = [&](const std::vector<float>& v) {
            double sum = 0; size_t n = 0;
            for (size_t i = lo; i < hi; ++i) { sum += v[i]; ++n; }
            return n ? sum / n : 0.0;
        };
        double out_ch[3] = { trmean(aR), trmean(aG), trmean(aB) };
        float gain[3] = {1.0f, 1.0f, 1.0f};
        for (int c = 0; c < 3; ++c)
            if (std::fabs(out_ch[c]) > 1e-12)
                gain[c] = float(in_ch[c] / out_ch[c]);
        for (size_t i = 0; i < acc.size(); i += 3) {
            acc[i]     *= gain[0];
            acc[i + 1] *= gain[1];
            acc[i + 2] *= gain[2];
        }
        VCE_LOGI("ViewMage", "self-ensemble: %d phase-exact symmetries, "
                 "gain R %.3e G %.3e B %.3e",
                 passes, gain[0], gain[1], gain[2]);
        out = std::move(acc);
        return true;
    }

    const int ow = pw * kAiScale, oh = ph * kAiScale;
    const size_t in_plane = size_t(pw) * ph;

    try {
        out.assign(size_t(ow) * oh * 3, 0.0f);
    } catch (const std::bad_alloc&) {
        err = "not enough memory to denoise this photo";
        return false;
    }

    const int nx = (pw + kAiStep - 1) / kAiStep;
    const int ny = (ph + kAiStep - 1) / kAiStep;

    std::vector<float> tile(size_t(kAiTile) * kAiTile * 4);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t in_shape[4]  = {1, 4, kAiTile, kAiTile};

    // Mirror the coordinate back inside [0, n) -- the reference pads the whole
    // array with np.pad(mode="reflect"); doing it per fetch avoids materialising
    // a padded copy of a 12 MP mosaic.
    auto reflect = [](int v, int n) {
        if (n == 1) return 0;
        while (v < 0 || v >= n) {
            if (v < 0)  v = -v;
            if (v >= n) v = 2 * (n - 1) - v;
        }
        return v;
    };

    // PER-CHANNEL gain match, against the mosaic's own R, G and B means.
    //
    // The reference matches one global scale, which fixes brightness and leaves
    // the balance between channels to the network. Measured, the network does
    // not preserve it: R and B came out ~14% and ~10% high against a linear
    // demosaic of the same frame, i.e. a magenta cast over the whole picture.
    //
    // Global channel balance is a property of the SENSOR -- it is what the CFA
    // measured -- and a denoiser has no business changing it. Matching each
    // output channel to the mean of the photosites that produced it pins the
    // colour to the measurement and leaves the network doing only what it is
    // for: deciding where the detail is. Green is the mean of BOTH green planes.
    double in_ch[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < in_plane; ++i) {
        in_ch[0] += packed[i];
        in_ch[1] += packed[in_plane + i] + packed[in_plane * 2 + i];
        in_ch[2] += packed[in_plane * 3 + i];
    }
    in_ch[0] /= double(in_plane);
    in_ch[1] /= double(in_plane) * 2.0;
    in_ch[2] /= double(in_plane);

    // ── Stitching: overlap TRIMMED, not blended ─────────────────────────────
    //
    // Each tile's model output is trustworthy only over its core: the 64-px
    // packed rim that provided context sits where the net's receptive support
    // runs off into MIRRORED data, and its predictions there differ measurably
    // from what an interior view of the same place yields. The reference
    // stitcher therefore blits only [overlap : overlap+step] and never touches
    // the rim.
    //
    // An earlier version here feathered each tile ~16 output px INTO the
    // overlap and cross-faded via a weight accumulator. That reintroduced the
    // least-trusted band, and neighbouring tiles' rims disagree slightly — a
    // subtle patchwork that reads as low-detail blocks on distant subjects.
    // `--compare`'s on/near ratio is the tripwire for any regression to it.
    const int gw = kAiTile * kAiScale;
    const int gh = kAiTile * kAiScale;
    const size_t gw_plane = size_t(gw) * gh;
    const int src0 = kAiOverlap * kAiScale;          // core start in tile px

    for (int ty = 0; ty < ny; ++ty) {
        for (int tx = 0; tx < nx; ++tx) {
            const int core_x = tx * kAiStep, core_y = ty * kAiStep;
            const int core_w = std::min(kAiStep, pw - core_x);
            const int core_h = std::min(kAiStep, ph - core_y);

            // Gather the tile: the core plus `kAiOverlap` of real context on
            // every side, mirrored where that runs off the sensor.
            for (int c = 0; c < 4; ++c) {
                const float* src = packed + size_t(c) * in_plane;
                float* dst = tile.data() + size_t(c) * kAiTile * kAiTile;
                for (int y = 0; y < kAiTile; ++y) {
                    const int sy = reflect(core_y - kAiOverlap + y, ph);
                    for (int x = 0; x < kAiTile; ++x) {
                        const int sx = reflect(core_x - kAiOverlap + x, pw);
                        dst[size_t(y) * kAiTile + x] = src[size_t(sy) * pw + sx];
                    }
                }
            }

            Ort::Value in = Ort::Value::CreateTensor<float>(
                mem, tile.data(), tile.size(), in_shape, 4);
            const char* in_names[]  = {impl_->in_name.c_str()};
            const char* out_names[] = {impl_->out_name.c_str()};
            auto res = impl_->session->Run(Ort::RunOptions{nullptr}, in_names,
                                           &in, 1, out_names, 1);
            const float* got = res[0].GetTensorData<float>();

            // The model's output tensor is [1, 3, gw, gh] in NCHW PLANAR
            // layout: three full planes of gw*gh floats, R then G then B.
            // Reading it interleaved (got[si*3+c]) grabs R,G,B samples from
            // three different spatial positions, which turns the picture
            // greyscale and mosaics it — the "almost B/W grid" bug.
            const float* got_r = got;
            const float* got_g = got + gw_plane;
            const float* got_b = got + 2 * gw_plane;

            // Blit ONLY the verified core. Output pixels map 1:1 onto mosaic
            // cells at twice the packing, so destination rows are contiguous:
            // a straight per-row plane copy.
            const int dw = core_w * kAiScale, dh = core_h * kAiScale;
            const size_t dst_row0 = size_t(core_y) * kAiScale * ow
                                  + size_t(core_x) * kAiScale;
            const size_t src_row0 = size_t(src0) * gw + src0;
            for (int fy = 0; fy < dh; ++fy) {
                float* d = &out[(dst_row0 + size_t(fy) * ow) * 3];
                const float* sr = got_r + src_row0 + size_t(fy) * gw;
                const float* sg = got_g + src_row0 + size_t(fy) * gw;
                const float* sb = got_b + src_row0 + size_t(fy) * gw;
                for (int fx = 0; fx < dw; ++fx) {
                    d[size_t(fx) * 3]     = sr[fx];
                    d[size_t(fx) * 3 + 1] = sg[fx];
                    d[size_t(fx) * 3 + 2] = sb[fx];
                }
            }

            if (progress) {
                const float done = float(ty * nx + tx + 1) / float(nx * ny);
                if (!progress(done)) { err = "cancelled"; return false; }
            }
        }
    }

    // ── The gain match ──────────────────────────────────────────────────────
    //
    // The network was trained with match_gain=output, so what it emits is at an
    // ARBITRARY learned scale. Rescaling is not a tweak, it is the missing half
    // of the model: skip it and every photo comes back at the wrong brightness,
    // which reads as a develop or colour bug and sends you looking in the wrong
    // file entirely.
    //
    // Done once over the whole image, after stitching -- per tile it would give
    // each tile its own exposure -- and PER CHANNEL, see above.
    //
    // TRIMMED MEAN (5% on each tail): the flat arithmetic mean is skewed by
    // deep shadows (noisy, unreliable) and clipped highlights (a few very bright
    // pixels dominate the sum).  Trimming the extremes makes the gain match
    // robust to both, at the cost of ignoring the most and least reliable
    // signal -- a good trade when the mid-tones carry the colour balance the
    // eye actually judges.
    const size_t n_out = size_t(ow) * oh;
    const size_t sample_step = std::max<size_t>(1, n_out / 40000);
    const size_t n_samples = n_out / sample_step;

    // Per-channel samples, then sort for percentile computation.
    std::vector<float> sR, sG, sB;
    sR.reserve(n_samples); sG.reserve(n_samples); sB.reserve(n_samples);
    for (size_t i = 0; i < n_out; i += sample_step) {
        sR.push_back(out[i * 3]);
        sG.push_back(out[i * 3 + 1]);
        sB.push_back(out[i * 3 + 2]);
    }
    std::sort(sR.begin(), sR.end());
    std::sort(sG.begin(), sG.end());
    std::sort(sB.begin(), sB.end());

    const double trim = 0.05;
    const size_t lo = (size_t)(trim * sR.size());
    const size_t hi = sR.size() - (size_t)(trim * sR.size());
    auto trimmed_mean = [&](const std::vector<float>& v) {
        double sum = 0; size_t n = 0;
        for (size_t i = lo; i < hi; ++i) { sum += v[i]; ++n; }
        return n ? sum / n : 0.0;
    };
    double out_ch[3] = { trimmed_mean(sR), trimmed_mean(sG), trimmed_mean(sB) };

    float gain[3] = {1.0f, 1.0f, 1.0f};
    for (int c = 0; c < 3; ++c)
        if (std::fabs(out_ch[c]) > 1e-12)
            gain[c] = float(in_ch[c] / out_ch[c]);
    for (size_t i = 0; i < out.size(); i += 3) {
        out[i]     *= gain[0];
        out[i + 1] *= gain[1];
        out[i + 2] *= gain[2];
    }
    VCE_LOGI("ViewMage", "gain match per channel (trimmed mean): R %.3e G %.3e B %.3e",
             gain[0], gain[1], gain[2]);
    return true;
}

#endif  // VIEWMAGE_WITH_ORT
