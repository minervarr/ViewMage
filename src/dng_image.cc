// dng_image.cc — see dng_image.hh for the contract and the scope.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.

#include "dng_image.hh"

#include "ai_denoise.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace {

// ── A deliberately small TIFF reader ────────────────────────────────────────
//
// Not a general TIFF library, and it should not become one. It reads the tags a
// Bayer DNG needs and refuses everything else clearly. Every read is bounds
// checked against the buffer: a truncated or hostile file must produce an
// error string, never a read past the end.
struct Reader {
    const uint8_t* d = nullptr;
    size_t         n = 0;
    bool           big_endian = false;

    bool in(size_t off, size_t len) const { return off <= n && len <= n - off; }

    uint16_t u16(size_t o, bool& ok) const {
        if (!in(o, 2)) { ok = false; return 0; }
        return big_endian ? uint16_t(d[o] << 8 | d[o + 1])
                          : uint16_t(d[o + 1] << 8 | d[o]);
    }
    uint32_t u32(size_t o, bool& ok) const {
        if (!in(o, 4)) { ok = false; return 0; }
        return big_endian
            ? (uint32_t(d[o]) << 24 | uint32_t(d[o+1]) << 16 | uint32_t(d[o+2]) << 8 | d[o+3])
            : (uint32_t(d[o+3]) << 24 | uint32_t(d[o+2]) << 16 | uint32_t(d[o+1]) << 8 | d[o]);
    }
    double f64(size_t o, bool& ok) const {
        if (!in(o, 8)) { ok = false; return 0.0; }
        uint8_t b[8];
        for (int i = 0; i < 8; ++i) b[i] = big_endian ? d[o + 7 - i] : d[o + i];
        double v; std::memcpy(&v, b, 8); return v;
    }
};

enum : uint16_t {
    T_BYTE = 1, T_SHORT = 3, T_LONG = 4, T_RATIONAL = 5,
    T_SLONG = 9, T_SRATIONAL = 10, T_DOUBLE = 12,
};

size_t type_size(uint16_t t) {
    switch (t) {
        case T_BYTE: return 1;
        case T_SHORT: return 2;
        case T_LONG: case T_SLONG: return 4;
        case T_RATIONAL: case T_SRATIONAL: case T_DOUBLE: return 8;
        default: return 0;
    }
}

struct Entry { uint16_t type = 0; uint32_t count = 0; size_t value_off = 0; };

// Every tag of every IFD we walked, flattened. A DNG puts the raw image in a
// SubIFD while the colour tags sit in IFD0, so a flat map is both simpler and
// closer to how the data is actually used than a faithful tree would be.
struct Tags {
    static constexpr int kMax = 64;
    uint16_t tag[kMax]{};
    Entry    ent[kMax]{};
    int      n = 0;

    void put(uint16_t t, const Entry& e) {
        for (int i = 0; i < n; ++i) if (tag[i] == t) { ent[i] = e; return; }
        if (n < kMax) { tag[n] = t; ent[n] = e; ++n; }
    }
    const Entry* get(uint16_t t) const {
        for (int i = 0; i < n; ++i) if (tag[i] == t) return &ent[i];
        return nullptr;
    }
};

// Scalar read of element `i`, whatever the integer/rational type.
double num_at(const Reader& r, const Entry& e, uint32_t i, bool& ok) {
    const size_t sz = type_size(e.type);
    const size_t o  = e.value_off + size_t(i) * sz;
    switch (e.type) {
        case T_BYTE:  if (!r.in(o, 1)) { ok = false; return 0; } return r.d[o];
        case T_SHORT: return r.u16(o, ok);
        case T_LONG:  return r.u32(o, ok);
        case T_SLONG: return int32_t(r.u32(o, ok));
        case T_RATIONAL: {
            const double a = r.u32(o, ok), b = r.u32(o + 4, ok);
            return b != 0.0 ? a / b : 0.0;
        }
        case T_SRATIONAL: {
            const double a = int32_t(r.u32(o, ok)), b = int32_t(r.u32(o + 4, ok));
            return b != 0.0 ? a / b : 0.0;
        }
        case T_DOUBLE: return r.f64(o, ok);
        default: ok = false; return 0.0;
    }
}

bool read_n(const Reader& r, const Tags& t, uint16_t tag, double* out, int n) {
    const Entry* e = t.get(tag);
    if (!e || e->count < uint32_t(n)) return false;
    bool ok = true;
    for (int i = 0; i < n; ++i) out[i] = num_at(r, *e, uint32_t(i), ok);
    return ok;
}

// Walks one IFD, recursing into SubIFDs (330). `depth` stops a malformed file
// from pointing an IFD at itself forever.
bool walk_ifd(const Reader& r, size_t off, Tags& out, int depth, size_t& next) {
    if (depth > 4) return true;
    bool ok = true;
    const uint16_t count = r.u16(off, ok);
    if (!ok || count > 512) return false;
    for (uint16_t i = 0; i < count; ++i) {
        const size_t e = off + 2 + size_t(i) * 12;
        const uint16_t tag  = r.u16(e, ok);
        const uint16_t type = r.u16(e + 2, ok);
        const uint32_t cnt  = r.u32(e + 4, ok);
        if (!ok) return false;
        const size_t sz = type_size(type);
        if (sz == 0) continue;                       // unknown type: skip, don't fail
        if (cnt > (1u << 20)) continue;              // absurd count: skip
        const size_t total = sz * size_t(cnt);
        const size_t vo = total <= 4 ? e + 8 : size_t(r.u32(e + 8, ok));
        if (!ok || !r.in(vo, total)) continue;       // out of bounds: skip this tag
        Entry ent{type, cnt, vo};
        out.put(tag, ent);
        if (tag == 330) {                            // SubIFDs
            for (uint32_t k = 0; k < cnt; ++k) {
                bool k_ok = true;
                const size_t sub = size_t(num_at(r, ent, k, k_ok));
                size_t dummy = 0;
                if (k_ok) walk_ifd(r, sub, out, depth + 1, dummy);
            }
        }
    }
    next = r.u32(off + 2 + size_t(count) * 12, ok);
    if (!ok) next = 0;
    return true;
}

// ── Colour maths, ported from the camera's raw_develop.cc ───────────────────
//
// Kept numerically identical to the capture side on purpose: if the two ever
// disagree, the same scene develops to two different colours depending on which
// program opened it, and there is no way to tell which one is lying.

bool mat_inv(const double m[9], double out[9]) {
    const double det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6])
                     + m[2]*(m[3]*m[7]-m[4]*m[6]);
    if (std::fabs(det) < 1e-12) return false;
    const double d = 1.0 / det;
    out[0] =  (m[4]*m[8]-m[5]*m[7])*d; out[1] = -(m[1]*m[8]-m[2]*m[7])*d;
    out[2] =  (m[1]*m[5]-m[2]*m[4])*d; out[3] = -(m[3]*m[8]-m[5]*m[6])*d;
    out[4] =  (m[0]*m[8]-m[2]*m[6])*d; out[5] = -(m[0]*m[5]-m[2]*m[3])*d;
    out[6] =  (m[3]*m[7]-m[4]*m[6])*d; out[7] = -(m[0]*m[7]-m[1]*m[6])*d;
    out[8] =  (m[0]*m[4]-m[1]*m[3])*d;
    return true;
}

void mat_mul(const double a[9], const double b[9], double o[9]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            o[r*3+c] = a[r*3]*b[c] + a[r*3+1]*b[3+c] + a[r*3+2]*b[6+c];
}

const double kBradfordD50ToD65[9] = {
     0.9555766, -0.0230393,  0.0631636,
    -0.0282895,  1.0099416,  0.0210077,
     0.0122982, -0.0204830,  1.3299098,
};
const double kXyzD65ToSrgb[9] = {
     3.2404542, -1.5371385, -0.4985314,
    -0.9692660,  1.8760108,  0.0415560,
     0.0556434, -0.2040259,  1.0572252,
};
const double kXyzD65ToBt2020[9] = {
     1.7166512, -0.3556708, -0.2533663,
    -0.6666844,  1.6164812,  0.0157685,
     0.0176399, -0.0427706,  0.9421031,
};

constexpr int32_t kIlluminantD65 = 21;   // EXIF LightSource code

// The camera's reference-clip nit level (raw_develop.hh kPqScale = 0.10 means
// linear 1.0 == 1000 nits). Named here rather than inlined because it is a
// shared convention between two programs, not a local choice.
constexpr float kReferenceClipNits = 1000.0f;

struct RawMeta {
    int    width = 0, height = 0;
    int    cfa = 0;                       // 0 RGGB, 1 GRBG, 2 GBRG, 3 BGGR
    double black[4] = {0,0,0,0};
    double white = 65535.0;
    bool   has_cm1 = false, has_fm1 = false, has_fm2 = false;
    double cm1[9]{}, fm1[9]{}, fm2[9]{};
    int    illuminant2 = 0;
    bool   has_neutral = false;
    double neutral[3] = {1,1,1};
    double baseline_exposure = 0.0;
    int    orientation = 1;               // TIFF orientation
};

void derive_ccm(const RawMeta& m, const double xyz_to_target[9], float ccm[9]) {
    double cam2xyz[9] = {1,0,0, 0,1,0, 0,0,1};
    if (m.has_fm2 && m.illuminant2 == kIlluminantD65) {
        std::memcpy(cam2xyz, m.fm2, sizeof(cam2xyz));
    } else if (m.has_fm1) {
        std::memcpy(cam2xyz, m.fm1, sizeof(cam2xyz));
    } else if (m.has_cm1) {
        double inv[9];
        if (mat_inv(m.cm1, inv)) std::memcpy(cam2xyz, inv, sizeof(cam2xyz));
    }
    double adapt[9], full[9];
    mat_mul(xyz_to_target, kBradfordD50ToD65, adapt);
    mat_mul(adapt, cam2xyz, full);
    // Row-normalise so camera neutral (1,1,1) lands exactly on white.
    for (int r = 0; r < 3; ++r) {
        double e = full[r*3] + full[r*3+1] + full[r*3+2];
        if (std::fabs(e) < 1e-9) e = 1.0;
        for (int c = 0; c < 3; ++c) ccm[r*3+c] = float(full[r*3+c] / e);
    }
}

inline int cfa_colour(int cfa, int x, int y) {
    const int p = (y & 1) * 2 + (x & 1);
    switch (cfa) {
        case 0: return (p == 0) ? 0 : (p == 3) ? 2 : 1;   // RGGB
        case 1: return (p == 1) ? 0 : (p == 2) ? 2 : 1;   // GRBG
        case 2: return (p == 2) ? 0 : (p == 1) ? 2 : 1;   // GBRG
        default: return (p == 3) ? 0 : (p == 0) ? 2 : 1;  // BGGR
    }
}

// Where highlight reconstruction starts, as a fraction of the clip level.
constexpr float kHighlightKnee = 0.95f;

// ── No auto black point, and the measurement that says so ──────────────────
//
// One was written here and then removed, because the diagnosis behind it was a
// measurement error worth recording so that it is not repeated.
//
// The claim was that the sensor has a pedestal: the darkest 32x32 BLOCKS of a
// frame average 0.6-1.2 counts ABOVE the declared BlackLevel of 64, so black
// never reached zero. That number is real, but it does not mean what it looked
// like. A block mean averages the noise away, so it reports the local SIGNAL
// level -- and the darkest region of a normally-lit scene simply is not pure
// black. It was measuring the picture, not the sensor.
//
// The statistic that actually answers the question is the whole-frame low
// percentile, and on two real S23 Ultra frames it lands at or BELOW the
// declared black:
//
//     frame              BlackLevel   whole-frame 0.1th pct   raw min
//     bright, 12 MP          64               63.0              43
//     night,  12 MP          64               57.0              42
//
// Noise straddles the declared black level in both, so the black level is
// correct and there is nothing to subtract. An auto black point is inert at
// best and crushes real shadow detail at worst.
//
// What actually makes blacks look lifted is the absence of a display TONE
// CURVE: linear scene-referred data has no toe, while the stock-camera JPEG it
// gets compared against crushes its shadows hard. That is a rendering problem
// and belongs downstream, not a black-level problem to be fixed by moving the
// floor.

} // namespace

bool looks_like_dng(const uint8_t* data, size_t size) {
    if (!data || size < 8) return false;
    const bool le = data[0] == 'I' && data[1] == 'I' && data[2] == 42 && data[3] == 0;
    const bool be = data[0] == 'M' && data[1] == 'M' && data[2] == 0  && data[3] == 42;
    return le || be;
}

DecodedImage decode_dng(const uint8_t* data, size_t size, uint32_t maxDimension,
                        WorkingPrimaries primaries, NoiseModel* out_noise,
                        RawDenoiser* denoiser) {
    DecodedImage img;
    auto fail = [&img](const char* why) { img.error = why; return img; };

    if (!looks_like_dng(data, size)) return fail("Not a DNG or TIFF file.");

    Reader r{data, size, data[0] == 'M'};
    bool ok = true;
    const size_t ifd0 = r.u32(4, ok);
    if (!ok) return fail("This DNG's header is truncated.");

    Tags t;
    size_t off = ifd0, guard = 0;
    while (off && guard++ < 8) {
        size_t next = 0;
        if (!walk_ifd(r, off, t, 0, next)) break;
        off = next;
    }

    // ── What kind of file is this, really ───────────────────────────────────
    double v[9];
    if (!read_n(r, t, 256, v, 1)) return fail("This DNG has no image width.");
    const int W = int(v[0]);
    if (!read_n(r, t, 257, v, 1)) return fail("This DNG has no image height.");
    const int H = int(v[0]);
    if (W < 4 || H < 4) return fail("This DNG's image is too small to develop.");

    if (read_n(r, t, 259, v, 1) && v[0] != 1.0)
        return fail("This DNG is compressed. Only uncompressed RAW is supported.");
    if (read_n(r, t, 258, v, 1) && v[0] != 16.0)
        return fail("This DNG is not 16-bit. Only 16-bit RAW is supported.");
    if (read_n(r, t, 322, v, 1))
        return fail("This DNG is tiled. Only strip RAW is supported.");

    // PhotometricInterpretation 32803 == CFA. 34892 is a linear (already
    // demosaiced) DNG, which is a picture rather than measurements and would
    // need a different path entirely.
    if (read_n(r, t, 262, v, 1) && v[0] != 32803.0)
        return fail("This DNG is already demosaiced. Only Bayer CFA RAW is supported.");

    const Entry* so = t.get(273);   // StripOffsets
    if (!so || so->count < 1) return fail("This DNG has no pixel data.");
    const size_t raw_off = size_t(num_at(r, *so, 0, ok));
    const size_t need    = size_t(W) * size_t(H) * 2;
    if (!ok || !r.in(raw_off, need))
        return fail("This DNG's pixel data is truncated.");
    if (so->count != 1)
        return fail("This DNG is split across strips. Only single-strip RAW is supported.");

    RawMeta m;
    m.width = W; m.height = H;

    // CFA pattern -> the same 0..3 enum the camera's develop uses.
    {
        double p[4] = {0, 1, 1, 2};
        read_n(r, t, 33422, p, 4);
        const int a = int(p[0]), b = int(p[1]), c = int(p[2]), d = int(p[3]);
        if      (a == 0 && b == 1 && c == 1 && d == 2) m.cfa = 0;   // RGGB
        else if (a == 1 && b == 0 && c == 2 && d == 1) m.cfa = 1;   // GRBG
        else if (a == 1 && b == 2 && c == 0 && d == 1) m.cfa = 2;   // GBRG
        else if (a == 2 && b == 1 && c == 1 && d == 0) m.cfa = 3;   // BGGR
        else return fail("This DNG uses a colour filter layout we do not handle.");
    }

    if (read_n(r, t, 50717, v, 1)) m.white = v[0];
    {
        // BlackLevel is per CFA element when BlackLevelRepeatDim is 2x2, but a
        // single value is legal and common.
        const Entry* e = t.get(50714);
        if (e) {
            if (e->count >= 4) read_n(r, t, 50714, m.black, 4);
            else if (read_n(r, t, 50714, v, 1))
                for (int i = 0; i < 4; ++i) m.black[i] = v[0];
        }
    }
    if (m.white - m.black[0] < 1.0) return fail("This DNG's black and white levels are unusable.");

    m.has_cm1 = read_n(r, t, 50721, m.cm1, 9);
    m.has_fm1 = read_n(r, t, 50964, m.fm1, 9);
    m.has_fm2 = read_n(r, t, 50965, m.fm2, 9);
    if (read_n(r, t, 50779, v, 1)) m.illuminant2 = int(v[0]);
    m.has_neutral = read_n(r, t, 50728, m.neutral, 3);
    if (read_n(r, t, 50730, v, 1)) m.baseline_exposure = v[0];
    if (read_n(r, t, 274, v, 1))   m.orientation = int(v[0]);

    if (out_noise) {
        const Entry* e = t.get(51041);
        if (e && e->count >= 2) {
            const int n = int(std::min<uint32_t>(e->count, 8));
            bool n_ok = true;
            for (int i = 0; i < n; ++i) out_noise->so[i] = num_at(r, *e, uint32_t(i), n_ok);
            out_noise->count = n;
            out_noise->valid = n_ok;
        }
    }

    // ── Develop ─────────────────────────────────────────────────────────────
    const uint16_t* bayer = reinterpret_cast<const uint16_t*>(data + raw_off);
    const int stride_px = W;

    const double* eff_black = m.black;   // see "No auto black point" above

    // Normalised, black-subtracted plane, exactly as the camera does it: per CFA
    // element, clamped to [0,1] because a single shot really is clipped at white
    // by physics.
    std::vector<float> lin(size_t(W) * H);
    for (int y = 0; y < H; ++y) {
        const uint16_t* row = bayer + size_t(y) * stride_px;
        for (int x = 0; x < W; ++x) {
            const int p = (y & 1) * 2 + (x & 1);
            const float bl    = float(eff_black[p]);
            const float range = std::max(float(m.white) - bl, 1.0f);
            lin[size_t(y) * W + x] =
                std::min(std::max((float(row[x]) - bl) / range, 0.0f), 1.0f);
        }
    }

    // White balance: reciprocal gains from the as-shot neutral, matching the
    // camera (and therefore the video ISP) so absolute brightness agrees.
    float g[3] = {1.0f, 1.0f, 1.0f};
    if (m.has_neutral && m.neutral[0] > 1e-6 && m.neutral[1] > 1e-6 && m.neutral[2] > 1e-6) {
        for (int i = 0; i < 3; ++i) g[i] = float(1.0 / m.neutral[i]);
    }

    // BaselineExposure, when the file carries one, says the stored data was
    // deliberately scaled down (a merged bracket reserving highlight headroom).
    // Honouring it is what keeps a STATIC merge and a FAST shot of the same
    // scene at the same brightness.
    const float baseline_gain = float(std::pow(2.0, m.baseline_exposure));

    float ccm[9];
    derive_ccm(m, primaries == WorkingPrimaries::Bt2100 ? kXyzD65ToBt2020 : kXyzD65ToSrgb, ccm);

    auto at = [&](int x, int y) -> float {
        x = std::min(std::max(x, 0), W - 1);
        y = std::min(std::max(y, 0), H - 1);
        return lin[size_t(y) * W + x];
    };

    std::vector<float> rgba(size_t(W) * H * 4);
    const float clip_level = 1.0f;

    // ── The neural demosaic, when we have one ───────────────────────────────
    //
    // RawNIND UtNet2 denoises the MOSAIC and demosaics in the same pass, so it
    // stands in for the Malvar block below rather than running alongside it.
    // Denoising before the demosaic is the point: afterwards every pixel is a
    // blend of its neighbours and the per-channel sensor noise is no longer
    // there to be told apart from detail.
    //
    // It is fed the mosaic UNBALANCED. The white-balance gains `g[]` are applied
    // after this, exactly as they are for Malvar -- the model was trained on
    // unbalanced data, and "tidying" the gains up in front of it would be a
    // silent quality regression rather than a visible bug.
    std::vector<float> ai_rgb;
    if (denoiser) {
        std::vector<float> packed;
        int pw = 0, ph = 0;
        pack_bayer_rggb(bayer, W, H, stride_px, m.cfa, eff_black, m.white,
                        packed, pw, ph);
        std::string derr;
        if (denoiser->run(packed.data(), pw, ph, ai_rgb, derr)) {
            note(img, DiagCode::kHighBitDepth, DiagLevel::kInfo,
                 "Denoised by RawNIND UtNet2",
                 "The sensor noise was removed on the raw mosaic, before "
                 "demosaicing, by a neural network trained on real noisy/clean "
                 "raw pairs (RawNIND, Brummer & De Vleeschouwer). Colour and "
                 "white balance are still the camera's own.");
        } else {
            ai_rgb.clear();
            note(img, DiagCode::kHighBitDepth, DiagLevel::kWarning,
                 "Shown without neural denoise",
                 std::string("The denoiser could not run (") + derr +
                 "), so this is the ordinary demosaic. The photo is unaffected "
                 "apart from the noise being left in.");
        }
    }
    // Odd sensor dimensions cannot be packed into 2x2 cells; the packer halves
    // them, so a mismatch here means the mosaic was not what we assumed.
    const float* ai = (ai_rgb.size() == size_t(W) * H * 3) ? ai_rgb.data() : nullptr;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float rgb[3];
            if (ai) {
                const float* s = &ai[(size_t(y) * W + x) * 3];
                rgb[0] = s[0]; rgb[1] = s[1]; rgb[2] = s[2];
            } else {
            const float c  = at(x, y);
            const float n1 = at(x, y-1), s1 = at(x, y+1);
            const float w1 = at(x-1, y), e1 = at(x+1, y);
            const float n2 = at(x, y-2), s2 = at(x, y+2);
            const float w2 = at(x-2, y), e2 = at(x+2, y);
            const float nw = at(x-1, y-1), ne = at(x+1, y-1);
            const float sw = at(x-1, y+1), se = at(x+1, y+1);

            // Malvar-He-Cutler, the same kernels the camera and the video ISP
            // use. Each set of weights sums to 8.
            const float g_at_rb = (4.0f*c + 2.0f*(n1+s1+w1+e1) - (n2+s2+w2+e2)) / 8.0f;
            const float rb_h = (5.0f*c + 4.0f*(w1+e1) - (nw+ne+sw+se) - (w2+e2) + 0.5f*(n2+s2)) / 8.0f;
            const float rb_v = (5.0f*c + 4.0f*(n1+s1) - (nw+ne+sw+se) - (n2+s2) + 0.5f*(w2+e2)) / 8.0f;
            const float rb_d = (6.0f*c + 2.0f*(nw+ne+sw+se) - 1.5f*(n2+s2+w2+e2)) / 8.0f;

            const int col = cfa_colour(m.cfa, x, y);
            if (col == 1) {
                const int row_col = cfa_colour(m.cfa, x, y - 1);
                rgb[1] = c;
                if (row_col == 0) { rgb[0] = rb_v; rgb[2] = rb_h; }
                else              { rgb[0] = rb_h; rgb[2] = rb_v; }
            } else if (col == 0) {
                rgb[0] = c; rgb[1] = g_at_rb; rgb[2] = rb_d;
            } else {
                rgb[2] = c; rgb[1] = g_at_rb; rgb[0] = rb_d;
            }
            }  // end of the Malvar branch

            float rr = rgb[0]*g[0], gg = rgb[1]*g[1], bb = rgb[2]*g[2];

            // Highlight reconstruction: a channel that clipped is raised to the
            // brightest white-balanced channel, ramped in over the last stretch
            // before the clip.
            //
            // APPROXIMATE on the AI path: the model's input is clipped to [0,1]
            // by the pack, so a blown channel arrives at the ceiling rather than
            // above it, and the gain match then moves it slightly. It still
            // fires where it should and still beats leaving a magenta core in
            // every highlight -- but the two paths are not identical here, and
            // assuming they are will mislead. Channels that did NOT clip are left alone, so a
            // genuinely saturated colour keeps its hue instead of bleaching.
            {
                const float knee = clip_level * kHighlightKnee;
                const float inv  = 1.0f / std::max(clip_level - knee, 1e-6f);
                const float peak = std::max(std::max(rr, gg), bb);
                auto ramp = [&](float pre) {
                    const float tt = std::min(std::max((pre - knee) * inv, 0.0f), 1.0f);
                    return tt * tt * (3.0f - 2.0f * tt);   // smoothstep
                };
                rr += (peak - rr) * ramp(rgb[0]);
                gg += (peak - gg) * ramp(rgb[1]);
                bb += (peak - bb) * ramp(rgb[2]);
            }

            float* dst = &rgba[(size_t(y) * W + x) * 4];
            for (int k = 0; k < 3; ++k) {
                const float o = ccm[k*3]*rr + ccm[k*3+1]*gg + ccm[k*3+2]*bb;
                // Clamp at 0 only. Chroma can undershoot near clipped edges, but
                // the top must stay open: this is linear scene light and the
                // tone curve downstream is what decides how to show it.
                dst[k] = std::max(o, 0.0f) * baseline_gain;
            }
            dst[3] = 1.0f;
        }
    }

    img.linear = std::move(rgba);
    img.w = uint32_t(W);
    img.h = uint32_t(H);

    img.bitsPerSample   = 16;
    img.intensityTarget = kReferenceClipNits;
    img.hdrTransfer     = true;
    img.widePrimaries   = (primaries == WorkingPrimaries::Bt2100);
    img.colorManaged    = true;
    img.transferName    = "linear (developed from RAW)";
    img.primariesName   = img.widePrimaries ? "BT.2100" : "sRGB";

    img.notes.push_back(Diagnostic{
        DiagCode::kHighBitDepth, DiagLevel::kInfo,
        "Developed from RAW",
        "This is sensor data developed here and now, not a rendered picture. "
        "White balance and colour come from the camera's own calibration in the file."});

    // The shared, format-agnostic tail: GPU-limit downsample, auto exposure and
    // the "this exceeds your screen" notes. Deliberately the SAME code the JXL
    // path runs — see finalize_decoded in jxl_image.hh. Skipping it was a real
    // bug during development: the develop was correct but autoEv stayed 0, so a
    // 1000-nit-referenced image was shown against a 203-nit white and every
    // photo came out about 2.3 stops dark.
    finalize_decoded(img, maxDimension, primaries);

    // Scene-referred light becomes a picture. AFTER the shared tail, because the
    // curve is anchored to where auto-exposure puts mid grey -- and the exposure
    // is decided from the scene histogram, which is the honest thing to measure.
    apply_display_render(img, primaries);
    return img;
}
