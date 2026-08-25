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
#include "ai_denoise.hh"

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
                     int& ox, int& oy) {
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
    } catch (const Ort::Exception& e) {
        err = std::string("could not load the denoiser: ") + e.what();
        return nullptr;
    }
    return d;
}

// One symmetry of the square that keeps a Bayer cell's phase recoverable.
//
// Each is applied to EVERY PLANE with the plane order untouched -- which is the
// same tensor as packing the transformed mosaic and permuting the planes back,
// worked through on paper and confirmed numerically. `swap_rb` is the half that
// is easy to forget: 180 and anti-transpose land R's corner of the cell on B's,
// so the model is told a blue plane is red and its output channels have to be
// exchanged back. Omitting that is what made a naive ensemble wrong.
namespace {
struct Symmetry { bool transpose; bool rot180; bool swap_rb; };
constexpr Symmetry kSyms[4] = {
    {false, false, false},   // identity
    {true,  false, false},   // transpose      (the two greens exchange; R,B stay)
    {false, true,  true },   // 180
    {true,  true,  true },   // anti-transpose
};

// Map a destination pixel back to its source under `sym`.
inline void map_src(const Symmetry& s, int x, int y, int w, int h,
                    int& sx, int& sy) {
    if (s.transpose) { int t = x; x = y; y = t; int tw = w; w = h; h = tw; }
    if (s.rot180)    { x = w - 1 - x; y = h - 1 - y; }
    sx = x; sy = y;
}
}  // namespace

bool RawDenoiser::run(const float* packed, int pw, int ph,
                      std::vector<float>& out, std::string& err,
                      const std::function<bool(float)>& progress,
                      bool ensemble) {
    if (!packed || pw <= 0 || ph <= 0) { err = "nothing to denoise"; return false; }

    const int passes = ensemble ? 4 : 1;
    if (passes > 1) {
        // Accumulate in place: each pass produces a full-size image, so holding
        // all four at once would be four times the largest allocation here.
        std::vector<float> acc, one, src;
        const size_t n = size_t(pw) * kAiScale * ph * kAiScale * 3;
        try { acc.assign(n, 0.0f); src.resize(size_t(pw) * ph * 4); }
        catch (const std::bad_alloc&) { err = "not enough memory to denoise"; return false; }

        for (int p = 0; p < passes; ++p) {
            const Symmetry& sym = kSyms[p];
            const int tw = sym.transpose ? ph : pw;   // transformed plane size
            const int th = sym.transpose ? pw : ph;

            for (int c = 0; c < 4; ++c) {
                const float* sp = packed + size_t(c) * pw * ph;
                float* dp = src.data() + size_t(c) * pw * ph;
                for (int y = 0; y < th; ++y)
                    for (int x = 0; x < tw; ++x) {
                        int sx, sy; map_src(sym, x, y, tw, th, sx, sy);
                        dp[size_t(y) * tw + x] = sp[size_t(sy) * pw + sx];
                    }
            }

            auto sub = progress ? std::function<bool(float)>(
                [&](float f) { return progress((p + f) / (float)passes); })
                                : std::function<bool(float)>();
            if (!run(src.data(), tw, th, one, err, sub, /*ensemble=*/false))
                return false;

            // Undo the symmetry while accumulating, and exchange R/B where the
            // transform moved them.
            const int ow = tw * kAiScale, oh = th * kAiScale;
            const int fw = pw * kAiScale;
            for (int y = 0; y < oh; ++y)
                for (int x = 0; x < ow; ++x) {
                    int dx, dy; map_src(sym, x, y, ow, oh, dx, dy);
                    const float* s = &one[(size_t(y) * ow + x) * 3];
                    float* d = &acc[(size_t(dy) * fw + dx) * 3];
                    d[0] += sym.swap_rb ? s[2] : s[0];
                    d[1] += s[1];
                    d[2] += sym.swap_rb ? s[0] : s[2];
                }
        }
        const float inv = 1.0f / (float)passes;
        for (float& v : acc) v *= inv;
        out = std::move(acc);
        VCE_LOGI("ViewMage", "self-ensemble: %d valid symmetries averaged", passes);
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

    double in_sum = 0.0, out_sum = 0.0;
    for (size_t i = 0; i < in_plane * 4; ++i) in_sum += packed[i];

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

            // Trim the overlap and blit only the core. The reference does NOT
            // feather the seams -- with this much real context on each side the
            // tiles already agree, and blending would soften the very detail the
            // denoiser is being trusted to keep.
            const int gw = kAiTile * kAiScale;   // 1024
            const size_t gp = size_t(gw) * gw;   // one output plane of a tile
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < core_h * kAiScale; ++y) {
                    const float* s = got + size_t(c) * gp +
                                     size_t(kAiOverlap * kAiScale + y) * gw +
                                     kAiOverlap * kAiScale;
                    float* d = out.data() +
                               (size_t(core_y * kAiScale + y) * ow +
                                size_t(core_x * kAiScale)) * 3 + c;
                    for (int x = 0; x < core_w * kAiScale; ++x) {
                        d[size_t(x) * 3] = s[x];
                        out_sum += s[x];
                    }
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
    // ARBITRARY learned scale. Rescaling its mean onto the input's mean is not a
    // tweak, it is the missing half of the model: skip it and every photo comes
    // back at the wrong brightness, which reads as a develop or colour bug and
    // sends you looking in entirely the wrong file.
    //
    // Done once over the whole image, after stitching -- per tile it would give
    // each tile its own exposure.
    const double in_mean  = in_sum  / double(in_plane * 4);
    const double out_mean = out_sum / double(size_t(ow) * oh * 3);
    if (std::fabs(out_mean) > 1e-12) {
        const float gain = float(in_mean / out_mean);
        for (float& v : out) v *= gain;
        VCE_LOGI("ViewMage", "denoise gain match: in %.5f out %.5f -> x%.3e",
                 in_mean, out_mean, gain);
    }
    return true;
}

#endif  // VIEWMAGE_WITH_ORT
