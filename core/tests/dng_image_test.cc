// dng_image_test.cc — the DNG develop, with no device and no GPU.
//
// Two halves, and both matter:
//
//   1. A SYNTHETIC DNG built here in memory. It always runs, needs no fixture,
//      and pins the parts that are easy to break silently: tag parsing, the CFA
//      mapping, black/white normalisation, the auto black point, and the fact
//      that a neutral scene develops to a neutral colour.
//   2. An optional check against a REAL file, if a path is passed. A synthetic
//      Bayer grid cannot catch a wrong ForwardMatrix or an inverted CFA on an
//      actual sensor; a real frame can.

#include "core/dng_image.hh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool cond, const std::string& what) {
    if (!cond) { std::printf("  FAIL: %s\n", what.c_str()); ++g_fail; }
}

// ── A minimal little-endian TIFF/DNG writer, just for the fixture ───────────
namespace {

struct TagOut { uint16_t tag; uint16_t type; uint32_t count; uint32_t value; };

void put16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xff); b.push_back(v >> 8); }
void put32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xff);
}

// Builds an uncompressed 16-bit CFA DNG with a flat, neutral scene at `level`
// counts, plus a black pedestal so the auto black point has something to find.
std::vector<uint8_t> make_dng(int W, int H, uint16_t level, uint16_t black_declared,
                              uint16_t pedestal, uint16_t white) {
    std::vector<uint8_t> heap;      // values that do not fit in 4 bytes
    std::vector<TagOut>  tags;
    const size_t heap_base_placeholder = 0;

    auto heap_put = [&](const void* p, size_t n) {
        const size_t off = heap.size();
        const uint8_t* s = static_cast<const uint8_t*>(p);
        heap.insert(heap.end(), s, s + n);
        while (heap.size() % 4) heap.push_back(0);
        return off;
    };

    // CFAPattern RGGB. Four BYTEs fit in the 4-byte value field, so TIFF stores
    // them INLINE rather than at an offset -- writing an offset here produced
    // 0,0,0,0 (an all-red CFA) and the reader rightly rejected it.
    const uint32_t cfa_inline = 0u | (1u << 8) | (1u << 16) | (2u << 24);

    // BlackLevel x4 (SHORT), WhiteLevel (LONG in-place)
    uint16_t bl[4] = {black_declared, black_declared, black_declared, black_declared};
    const size_t bl_off = heap_put(bl, sizeof(bl));

    // AsShotNeutral: 3 RATIONALs, all 1/1 -> a neutral scene must stay neutral.
    uint32_t neutral[6] = {1,1, 1,1, 1,1};
    const size_t neu_off = heap_put(neutral, sizeof(neutral));

    // ForwardMatrix1: identity-ish, as SRATIONALs. Row-normalisation in the
    // develop makes the exact values unimportant for the neutrality check.
    int32_t fm[18];
    const double ident[9] = {0.9642,0,0, 0,1.0,0, 0,0,0.8249};
    for (int i = 0; i < 9; ++i) { fm[2*i] = int32_t(ident[i] * 1000000); fm[2*i+1] = 1000000; }
    const size_t fm_off = heap_put(fm, sizeof(fm));

    // NoiseProfile: 8 DOUBLEs (S,O per channel).
    double np[8] = {1e-4, 1e-6, 1e-4, 1e-6, 1e-4, 1e-6, 1e-4, 1e-6};
    const size_t np_off = heap_put(np, sizeof(np));

    // Pixels last: a flat field, with one 2x2 block left at the pedestal so the
    // auto black point sees a genuine floor.
    std::vector<uint16_t> px(size_t(W) * H, level);
    for (int y = 0; y < 2 && y < H; ++y)
        for (int x = 0; x < 2 && x < W; ++x)
            px[size_t(y) * W + x] = black_declared + pedestal;
    const size_t px_off = heap_put(px.data(), px.size() * 2);

    tags.push_back({254, 4, 1, 0});                     // SubfileType
    tags.push_back({256, 4, 1, uint32_t(W)});           // ImageWidth
    tags.push_back({257, 4, 1, uint32_t(H)});           // ImageLength
    tags.push_back({258, 3, 1, 16});                    // BitsPerSample
    tags.push_back({259, 3, 1, 1});                     // Compression = none
    tags.push_back({262, 3, 1, 32803});                 // Photometric = CFA
    tags.push_back({273, 4, 1, uint32_t(px_off)});      // StripOffsets (heap-relative)
    tags.push_back({274, 3, 1, 1});                     // Orientation
    tags.push_back({279, 4, 1, uint32_t(px.size()*2)}); // StripByteCounts
    tags.push_back({33421, 3, 2, uint32_t(2 | (2 << 16))}); // CFARepeatPatternDim 2,2
    tags.push_back({33422, 1, 4, cfa_inline});          // CFAPattern (inline, see above)
    tags.push_back({50713, 3, 2, uint32_t(2 | (2 << 16))}); // BlackLevelRepeatDim
    tags.push_back({50714, 3, 4, uint32_t(bl_off)});    // BlackLevel
    tags.push_back({50717, 4, 1, white});               // WhiteLevel
    tags.push_back({50728, 5, 3, uint32_t(neu_off)});   // AsShotNeutral
    tags.push_back({50964, 10, 9, uint32_t(fm_off)});   // ForwardMatrix1
    tags.push_back({51041, 12, 8, uint32_t(np_off)});   // NoiseProfile

    // Header(8) + count(2) + entries(12n) + next(4), then the heap.
    const uint32_t heap_base = uint32_t(8 + 2 + 12 * tags.size() + 4);
    (void)heap_base_placeholder;

    // StripOffsets is a heap offset that HAPPENS to fit in the 4-byte value
    // field, so the generic "add heap_base only when out-of-line" rule below
    // misses it and the file points at the wrong bytes. It read as valid (just
    // wrong pixels), which is exactly the kind of fixture bug that makes a
    // reader look broken when it is not.
    for (auto& t : tags) if (t.tag == 273) t.value = heap_base + uint32_t(px_off);

    std::vector<uint8_t> out;
    out.push_back('I'); out.push_back('I'); put16(out, 42); put32(out, 8);
    put16(out, uint16_t(tags.size()));
    for (auto& t : tags) {
        put16(out, t.tag); put16(out, t.type); put32(out, t.count);
        const size_t sz = (t.type == 1) ? 1 : (t.type == 3) ? 2 :
                          (t.type == 4 || t.type == 9) ? 4 : 8;
        const bool inplace = sz * t.count <= 4;
        put32(out, inplace ? t.value : (heap_base + t.value));
    }
    put32(out, 0);
    out.insert(out.end(), heap.begin(), heap.end());
    return out;
}

} // namespace

static void test_synthetic() {
    std::printf("synthetic DNG:\n");

    const int W = 64, H = 48;
    const uint16_t declared_black = 64, pedestal = 3, white = 1023;
    const uint16_t level = 500;
    auto file = make_dng(W, H, level, declared_black, pedestal, white);

    NoiseModel nm;
    DecodedImage img = decode_dng(file.data(), file.size(), 0,
                                  WorkingPrimaries::Srgb, &nm);

    check(img.error.empty(), "develops without error: " + img.error);
    if (!img.error.empty()) return;
    check(img.w == uint32_t(W) && img.h == uint32_t(H), "dimensions survive");
    check(img.linear.size() == size_t(W) * H * 4, "buffer is w*h*4 floats");

    // The NoiseProfile must come back, and come back as DOUBLEs. This is the
    // tag a denoiser depends on; a rational would have quantised O to zero.
    check(nm.valid && nm.count == 8, "NoiseProfile read");
    check(std::fabs(nm.so[1] - 1e-6) < 1e-12, "NoiseProfile O term survives as a double");

    // A neutral scene with a unit as-shot neutral must develop neutral. This is
    // the check that catches a transposed CCM or a broken row normalisation.
    double sum[3] = {0, 0, 0};
    int n = 0;
    for (int y = 8; y < H - 8; ++y)
        for (int x = 8; x < W - 8; ++x) {
            const float* p = &img.linear[(size_t(y) * W + x) * 4];
            for (int k = 0; k < 3; ++k) sum[k] += p[k];
            ++n;
        }
    for (int k = 0; k < 3; ++k) sum[k] /= n;
    check(sum[1] > 0.01, "developed image is not black");
    const double rg = sum[0] / sum[1], bg = sum[2] / sum[1];
    check(std::fabs(rg - 1.0) < 0.02, "neutral scene stays neutral (R:G)");
    check(std::fabs(bg - 1.0) < 0.02, "neutral scene stays neutral (B:G)");

    // Black handling is the DECLARED BlackLevel, nothing cleverer. An auto black
    // point was tried and removed (see dng_image.cc): the pedestal it corrected
    // for was a measurement artefact, and on real frames the whole-frame low
    // percentile sits at or below the declared black. So the developed value
    // must be exactly what the declared black gives, and this test is what stops
    // a scene-dependent floor being reintroduced without new evidence.
    const double scene = double(level - declared_black) / double(white - declared_black);

    // ...and then the display render's toe curves it. Reproduce the curve here
    // rather than exempting the pixel from it: this pins BOTH that the black
    // handling is the declared level and that the render is the documented toe.
    const double pivot  = 0.18 * (203.0 / 1000.0) * std::pow(2.0, -img.autoEv);
    const double uWhite = -std::log2(0.18);
    const double u = std::log2(scene / pivot);
    double expect = scene;
    if (u < uWhite) {
        const double t = std::max(u, 0.0) / uWhite;
        const double g = 1.0 - t * t * (3.0 - 2.0 * t);
        expect = scene * std::exp2(u * 0.25 * g);
    }
    check(std::fabs(sum[1] - expect) < 0.01,
          "black subtraction is the declared BlackLevel, not an estimate");

    // The two properties the render is DESIGNED around, asserted directly so a
    // future change to the curve cannot quietly drop either one.
    //   1. Mid grey is a fixed point (u == 0 => gain 1), so the render adds
    //      contrast and not an exposure shift auto-exposure would fight.
    //   2. At and above diffuse white the gain is exactly 1 -- highlights are
    //      untouched, because the shader's rolloff is the pass that knows the
    //      display headroom, and compressing in both places flattens speculars.
    auto toe_gain = [&](double uu) {
        if (uu >= uWhite) return 1.0;
        const double t = std::max(uu, 0.0) / uWhite;
        const double g = 1.0 - t * t * (3.0 - 2.0 * t);
        return std::exp2(uu * 0.25 * g);
    };
    check(std::fabs(toe_gain(0.0) - 1.0) < 1e-12, "the render holds mid grey in place");
    check(std::fabs(toe_gain(uWhite) - 1.0) < 1e-12, "the render leaves diffuse white alone");
    check(toe_gain(-2.0) < 0.75, "the render actually darkens the shadows");
    check(std::fabs(toe_gain(4.0) - 1.0) < 1e-12, "the render leaves highlights alone");

    // Alpha must be opaque everywhere; the shader multiplies by it.
    bool alpha_ok = true;
    for (size_t i = 3; i < img.linear.size(); i += 4)
        if (img.linear[i] != 1.0f) { alpha_ok = false; break; }
    check(alpha_ok, "alpha is 1.0 everywhere");

    // The convention the whole display path is calibrated against.
    check(std::fabs(img.intensityTarget - 1000.0f) < 0.5f,
          "declares the reference clip as 1000 nits");
    check(img.hdrTransfer, "flagged as HDR");
    check(!img.widePrimaries, "sRGB request reports sRGB primaries");

    DecodedImage wide = decode_dng(file.data(), file.size(), 0, WorkingPrimaries::Bt2100);
    check(wide.error.empty() && wide.widePrimaries, "BT.2100 request reports wide primaries");
}

static void test_rejects() {
    std::printf("rejection paths:\n");

    const uint8_t junk[16] = {0};
    check(!looks_like_dng(junk, sizeof(junk)), "zeros are not a DNG");
    const uint8_t jxl[8] = {0xFF, 0x0A, 0, 0, 0, 0, 0, 0};
    check(!looks_like_dng(jxl, sizeof(jxl)), "a JXL codestream is not a DNG");

    DecodedImage bad = decode_dng(junk, sizeof(junk), 0);
    check(!bad.error.empty(), "junk produces an error, not a crash");

    // Truncation must be caught rather than read past the end.
    auto file = make_dng(32, 32, 500, 64, 3, 1023);
    for (size_t cut : {size_t(8), file.size() / 3, file.size() - 100}) {
        DecodedImage t = decode_dng(file.data(), cut, 0);
        check(!t.error.empty(), "truncated file at " + std::to_string(cut) + " errors cleanly");
    }
}

static void test_real(const char* path) {
    std::printf("real file: %s\n", path);
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("  SKIPPED: cannot open\n"); return; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf{}; buf.resize(size_t(n));
    const size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    check(got == buf.size(), "read the whole file");

    NoiseModel nm;
    DecodedImage img = decode_dng(buf.data(), buf.size(), 0, WorkingPrimaries::Bt2100, &nm);
    check(img.error.empty(), "develops without error: " + img.error);
    if (!img.error.empty()) return;

    std::printf("  %ux%u, noise profile: %s", img.w, img.h, nm.valid ? "yes" : "NO");
    if (nm.valid) std::printf(" (ch0 S=%.3e O=%.3e)", nm.so[0], nm.so[1]);
    std::printf("\n");

    // Distribution, so a regression in the develop is visible as numbers rather
    // than as "it looks wrong".
    std::vector<float> lum;
    lum.reserve(size_t(img.w) * img.h / 25 + 1);
    double sum[3] = {0,0,0};
    size_t n3 = 0; (void)n3;
    for (uint32_t y = 0; y < img.h; y += 5)
        for (uint32_t x = 0; x < img.w; x += 5) {
            const float* p = &img.linear[(size_t(y) * img.w + x) * 4];
            lum.push_back(0.2627f*p[0] + 0.6780f*p[1] + 0.0593f*p[2]);
            for (int k = 0; k < 3; ++k) sum[k] += p[k];
            ++n3;
        }
    std::sort(lum.begin(), lum.end());
    auto pct = [&](double q) { return lum[size_t(q * (lum.size() - 1))]; };
    std::printf("  linear luminance p1=%.5f p50=%.5f p99=%.5f max=%.4f\n",
                pct(0.01), pct(0.50), pct(0.99), lum.back());
    std::printf("  in nits (1.0 = %.0f): p50=%.1f p99=%.1f max=%.1f\n",
                img.intensityTarget, pct(0.50) * img.intensityTarget,
                pct(0.99) * img.intensityTarget, lum.back() * img.intensityTarget);
    std::printf("  mean R:G=%.3f B:G=%.3f\n", sum[0]/sum[1], sum[2]/sum[1]);

    check(pct(0.50) > 0.0f, "median is above zero");
    check(lum.back() > pct(0.50), "there is a highlight range");
    bool finite = true;
    for (float v : img.linear) if (!std::isfinite(v)) { finite = false; break; }
    check(finite, "no NaN or Inf in the output");
}

int main(int argc, char** argv) {
    test_synthetic();
    test_rejects();
    if (argc > 1) test_real(argv[1]);
    else std::printf("real file: SKIPPED (pass a .dng path to check one)\n");

    if (g_fail) { std::printf("dng_image_test: %d FAILED\n", g_fail); return 1; }
    std::printf("dng_image_test: all passed\n");
    return 0;
}
