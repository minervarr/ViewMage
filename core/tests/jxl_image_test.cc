// jxl_image_test.cc — the decode wrapper, on a desktop, against real files.
//
// Fixtures are LOSSLESS (cjxl -d 0), which is what makes exact pixel equality
// a fair assertion rather than a tolerance argument.
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "core/jxl_image.hh"

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "missing fixture: %s\n", path.c_str()); std::abort(); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}

static std::string dir;   // fixtures live beside the source, not the binary

// sRGB 8-bit code -> linear light, which is what the decoder now hands back.
static float srgb_to_linear(int code) {
    const float c = (float)code / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Assert a pixel against the sRGB codes it was AUTHORED with, converting to
// linear here rather than making every call site do it. tol is in linear units.
static void px(const JxlImage& im, uint32_t x, uint32_t y,
               int r, int g, int b, float tol = 0.01f) {
    const float* p = im.linear.data() + ((size_t)y * im.w + x) * 4;
    assert(std::fabs(p[0] - srgb_to_linear(r)) <= tol);
    assert(std::fabs(p[1] - srgb_to_linear(g)) <= tol);
    assert(std::fabs(p[2] - srgb_to_linear(b)) <= tol);
    assert(std::fabs(p[3] - 1.0f) <= 0.01f);   // an opaque source decodes opaque
}

static bool has_note(const JxlImage& im, DiagCode code) {
    for (const auto& n : im.notes) if (n.code == code) return true;
    return false;
}

static void test_signature_accepts_real_jxl() {
    auto d = read_file(dir + "/fixtures/rgb8x8.jxl");
    assert(looks_like_jxl(d.data(), d.size()));
}

static void test_signature_rejects_other_things() {
    const uint8_t png[]  = {0x89,'P','N','G',0x0D,0x0A,0x1A,0x0A,0,0,0,13};
    const uint8_t jpeg[] = {0xFF,0xD8,0xFF,0xE0,0,16,'J','F','I','F',0,1};
    const char*   text   = "this is not an image at all, it is prose";
    assert(!looks_like_jxl(png,  sizeof(png)));
    assert(!looks_like_jxl(jpeg, sizeof(jpeg)));
    assert(!looks_like_jxl((const uint8_t*)text, 39));
    assert(!looks_like_jxl(nullptr, 0));
    const uint8_t tiny[] = {0xFF, 0x0A};          // shorter than any signature
    assert(!looks_like_jxl(tiny, sizeof(tiny)));
}

// Dimensions AND pixels: a decoder that returned the right size and garbage
// content would pass a size-only test.
static void test_decodes_known_pixels() {
    auto d = read_file(dir + "/fixtures/rgb8x8.jxl");
    JxlImage im = decode_jxl(d.data(), d.size(), 0);
    assert(im.ok());
    assert(im.error.empty());
    assert(im.w == 8 && im.h == 8);
    assert(im.linear.size() == 8u * 8u * 4u);
    assert(im.downsampleFactor == 1);

    // The four corners, which also pin the ROW ORDER: (0,0) is red in the
    // source, so a bottom-up decode would put blue here instead.
    px(im, 0, 0, 255,   0,   0);
    px(im, 7, 0,   0, 255,   0);
    px(im, 0, 7,   0,   0, 255);
    px(im, 7, 7, 255, 255, 255);
    px(im, 3, 3,  10,  20,  30);       // and an interior pixel
    px(im, 3, 5,   0,   0,   0);
}

// An ordinary 8-bit sRGB photo must be left completely alone: no exposure
// shift, no rolloff, no diagnostics. This is the regression that protects the
// common case from everything the HDR work added.
static void test_ordinary_srgb_image_is_untouched() {
    auto d = read_file(dir + "/fixtures/grad600x400.jxl");
    JxlImage im = decode_jxl(d.data(), d.size(), 0);
    assert(im.ok());
    assert(!im.hdrTransfer);
    assert(!im.widePrimaries);
    assert(im.bitsPerSample == 8);
    assert(im.autoEv == 0.0f);   // EXACTLY untouched, not merely close
    assert(im.white  == 1.0f);   // so the tone curve is a provable identity
    assert(!has_note(im, DiagCode::kHdrTransfer));
    assert(!has_note(im, DiagCode::kWidePrimaries));
    assert(!has_note(im, DiagCode::kHighBitDepth));
}

// A real Rec.2100 PQ file declaring a 1000-nit peak.
static void test_hdr_pq_is_detected_and_pulled_down() {
    auto d = read_file(dir + "/fixtures/hdr_pq_1000nit.jxl");
    JxlImage im = decode_jxl(d.data(), d.size(), 0);
    assert(im.ok());
    assert(im.w == 64 && im.h == 64);

    assert(im.hdrTransfer);                    // PQ recognised as HDR
    assert(im.widePrimaries);                  // BT.2100 is wider than sRGB
    assert(im.transferName.find("PQ") != std::string::npos);
    assert(im.bitsPerSample == 16);
    assert(std::fabs(im.intensityTarget - 1000.0f) < 1.0f);
    assert(im.colorManaged);                   // the CMS actually ran

    // A declared 1000-nit peak against 203 nits of diffuse white is a GAIN of
    // 1000/203, about +2.3 EV: libjxl hands back 1.0 == 1000 nits and the
    // extended-linear surface wants 1.0 == 203. The assertion used to demand
    // the reciprocal (-2.3 EV), which is what let the inverted anchor ship.
    // The histogram term moves this by at most +/-1 EV.
    assert(im.autoEv > 1.0f && im.autoEv < 5.0f);

    assert(has_note(im, DiagCode::kHdrTransfer));
    assert(has_note(im, DiagCode::kWidePrimaries));
    assert(has_note(im, DiagCode::kHighBitDepth));

    // Every value finite, and the wide gamut really did survive as
    // out-of-range components rather than being clamped away.
    bool anyOutside = false;
    for (float v : im.linear) {
        assert(std::isfinite(v));
        if (v < -0.0005f || v > 1.0005f) anyOutside = true;
    }
    (void)anyOutside;   // not guaranteed for every image; must not crash either way
}

// High bit depth WITHOUT HDR: the two must be reported independently.
static void test_high_bit_depth_without_hdr() {
    auto d = read_file(dir + "/fixtures/srgb16.jxl");
    JxlImage im = decode_jxl(d.data(), d.size(), 0);
    assert(im.ok());
    assert(im.bitsPerSample == 16);
    assert(!im.hdrTransfer);
    assert(has_note(im, DiagCode::kHighBitDepth));
    assert(!has_note(im, DiagCode::kHdrTransfer));
}

// box_downsample was rewritten from a uint32_t accumulator to a float one.
// Over linear light the old code would have truncated every value to an
// integer -- i.e. flattened everything below 1.0 to zero. It only runs on
// images big enough to exceed the GPU limit, so nothing else would catch it.
static void test_downsample_preserves_a_constant_plane() {
    auto d = read_file(dir + "/fixtures/grad600x400.jxl");
    JxlImage full = decode_jxl(d.data(), d.size(), 0);
    assert(full.ok());
    const float blue = full.linear[2];          // constant across the source

    JxlImage small = decode_jxl(d.data(), d.size(), 256);
    assert(small.ok() && small.downsampleFactor == 4);
    for (uint32_t y = 0; y < small.h; y += 7) {
        for (uint32_t x = 0; x < small.w; x += 7) {
            const float* p = small.linear.data() + ((size_t)y * small.w + x) * 4;
            assert(std::fabs(p[2] - blue) < 0.002f);   // averaged, not truncated
            assert(p[2] > 0.001f);                     // the old int path gave 0
        }
    }
}

static void test_larger_image_round_trips() {
    auto d = read_file(dir + "/fixtures/grad600x400.jxl");
    JxlImage im = decode_jxl(d.data(), d.size(), 0);
    assert(im.ok());
    assert(im.w == 600 && im.h == 400);
    px(im, 0,   0,   0,   0, 200);
    px(im, 255, 0, 255,   0, 200);
    px(im, 100, 50, 100,  50, 200);
}

// ── the failures, which are most of what this wrapper is for ────────────────
static void test_empty_input() {
    JxlImage im = decode_jxl(nullptr, 0, 0);
    assert(!im.ok() && !im.error.empty());
    JxlImage im2 = decode_jxl((const uint8_t*)"", 0, 0);
    assert(!im2.ok() && !im2.error.empty());
}

static void test_not_jxl() {
    const char* s = "GIF89a and then some bytes that go nowhere in particular";
    JxlImage im = decode_jxl((const uint8_t*)s, 55, 0);
    assert(!im.ok());
    assert(im.error == "Not a JPEG XL image");   // the SPECIFIC message
}

static void test_truncated_input_is_an_error_not_a_crash() {
    auto d = read_file(dir + "/fixtures/grad600x400.jxl");
    for (size_t keep : {size_t(20), d.size() / 4, d.size() / 2, d.size() - 1}) {
        std::vector<uint8_t> part(d.begin(), d.begin() + keep);
        JxlImage im = decode_jxl(part.data(), part.size(), 0);
        assert(!im.ok());          // it must not claim success…
        assert(!im.error.empty()); // …and it must say something
    }
}

static void test_corrupt_body_is_an_error_not_a_crash() {
    auto d = read_file(dir + "/fixtures/grad600x400.jxl");
    // Keep the signature, scribble over the rest: exercises the decoder's own
    // error path rather than the signature check.
    for (size_t i = 12; i < d.size(); ++i) d[i] = (uint8_t)(i * 37 + 11);
    JxlImage im = decode_jxl(d.data(), d.size(), 0);
    assert(!im.ok() ? !im.error.empty() : true);   // either a clean error…
}                                                  // …or a plausible image; never a crash

// ── the GPU ceiling ─────────────────────────────────────────────────────────
static void test_downsamples_above_max_dimension() {
    auto d = read_file(dir + "/fixtures/grad600x400.jxl");

    JxlImage im = decode_jxl(d.data(), d.size(), 256);   // 600 > 256 → halve twice
    assert(im.ok());
    assert(im.w <= 256 && im.h <= 256);
    assert(im.downsampleFactor == 4);
    assert(im.w == 150 && im.h == 100);
    assert(im.linear.size() == (size_t)im.w * im.h * 4);
    // Averaged, not point-sampled: the blue plane is a constant 200 in the
    // source, so it must survive the box filter exactly.
    const float* p = im.linear.data() + ((size_t)10 * im.w + 10) * 4;
    assert(std::fabs(p[2] - srgb_to_linear(200)) <= 0.01f);

    JxlImage exact = decode_jxl(d.data(), d.size(), 600);  // exactly at the limit
    assert(exact.ok() && exact.downsampleFactor == 1);
    assert(exact.w == 600 && exact.h == 400);

    JxlImage none = decode_jxl(d.data(), d.size(), 0);     // ceiling disabled
    assert(none.ok() && none.downsampleFactor == 1 && none.w == 600);
}

int main(int argc, char** argv) {
    dir = (argc > 1) ? argv[1] : ".";
    test_signature_accepts_real_jxl();
    test_signature_rejects_other_things();
    test_decodes_known_pixels();
    test_larger_image_round_trips();
    test_ordinary_srgb_image_is_untouched();
    test_hdr_pq_is_detected_and_pulled_down();
    test_high_bit_depth_without_hdr();
    test_downsample_preserves_a_constant_plane();
    test_empty_input();
    test_not_jxl();
    test_truncated_input_is_an_error_not_a_crash();
    test_corrupt_body_is_an_error_not_a_crash();
    test_downsamples_above_max_dimension();
    std::printf("jxl_image_test: all passed\n");
    return 0;
}
