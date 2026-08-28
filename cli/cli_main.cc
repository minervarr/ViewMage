// cli_main.cc — headless DNG/JXL develop-to-PNG, no GPU, no Vulkan.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// Usage: viewmage_cli <input.{dng,jxl}> [output.png]
//
// This is the same pipeline the GUI runs — decode, auto-exposure, toe curve,
// rolloff, sRGB encode — but without a window, a GPU or a phone.  Useful for
// A/B testing render changes and for CI quality checks on desktop.
//
// Pass --enhance to also run the saturation/contrast/sharpen post-processing
// that the Enhance button triggers in the GUI.
//
// Exit codes: 0 = success, 1 = bad args, 2 = decode failed, 3 = write failed.
// ---------------------------------------------------------------------------
#include "core/jxl_image.hh"
#include "core/dng_image.hh"
#include "core/export_math.hh"
#include "core/enhance.hh"
#include "core/ai_denoise.hh"

#include "diagnose.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

// The canonical PNG export — mirrors gui/src/png_export.cc's export_png() but
// writes straight to a file instead of going through MediaStore.
bool write_png(const std::vector<float>& linear, uint32_t w, uint32_t h,
               float ev, float white, bool primariesAreWide,
               const char* path) {
    if (w == 0 || h == 0 || linear.size() < size_t(w) * h * 4) return false;

    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    const float exposure = std::exp2(ev);
    const float W = std::max(white, 1.0f);

    const float* lw = primariesAreWide
        ? export_math::kLumaBt2020 : export_math::kLumaBt709;
    const float lr = lw[0], lg = lw[1], lb = lw[2];

    for (uint32_t ry = 0; ry < h; ++ry) {
        const float* drow = export_math::kBayerDither.m[ry & 3];
        for (uint32_t x = 0; x < w; ++x) {
            const size_t i = size_t(ry) * w + x;
            const float* p = &linear[i * 4];
            float c[3] = {std::max(p[0] * exposure, 0.0f),
                          std::max(p[1] * exposure, 0.0f),
                          std::max(p[2] * exposure, 0.0f)};
            const float L = lr * c[0] + lg * c[1] + lb * c[2];
            if (L > 1e-6f) {
                const float s = export_math::rolloff(L, W) / L;
                c[0] *= s; c[1] *= s; c[2] *= s;
            }
            if (primariesAreWide) {
                const float r = c[0], g = c[1], b = c[2];
                for (int k = 0; k < 3; ++k)
                    c[k] = export_math::kBt2020ToSrgb[k*3]   * r +
                            export_math::kBt2020ToSrgb[k*3+1] * g +
                            export_math::kBt2020ToSrgb[k*3+2] * b;
            }
            uint8_t* d = &rgb[i * 3];
            const float dith = drow[x & 3];
            for (int k = 0; k < 3; ++k)
                d[k] = export_math::to_srgb8_dithered(c[k], dith);
        }
    }

    stbi_write_png_compression_level = 4;
    return stbi_write_png(path, int(w), int(h), 3, rgb.data(), int(w) * 3) != 0;
}

std::string default_output(const std::string& input) {
    std::string base = input;
    const size_t slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    const size_t dot = base.find_last_of('.');
    if (dot != std::string::npos && dot > 0) base = base.substr(0, dot);
    return base + "_developed.png";
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s <input.{dng,jxl}> [output.png]\n"
        "\n"
        "Develop a RAW file (or decode a JPEG XL) and write a PNG.\n"
        "The same pipeline as the GUI: auto-exposure, toe curve,\n"
        "highlight rolloff, sRGB encode.\n"
        "\n"
        "Options:\n"
        "  --enhance        Apply saturation/contrast/sharpen (like the Enhance button)\n"
        "  --denoise[=PATH] Run neural raw denoise (model defaults to models/model_bayer.onnx)\n"
        "  --no-ensemble    Single-pass + detail layer: 4x faster, softer base,\n"
        "                   measured micro-detail returned (default is the x4 ensemble)\n"
        "  --analyze        Highlight diagnostics: clip census, LED bloom rings,\n"
        "                   Malvar-vs-AI colour ratios, lateral-CA check\n"
        "  --compare [X Y W H]\n"
        "                   Detail-loss metrics: tile-seam strips and HF energy,\n"
        "                   Malvar vs AI single-pass vs AI ensemble\n"
        "  --stats          Print per-channel statistics to stdout\n"
        "  --help           This message\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_enhance = false;
    bool do_denoise = false;
    bool do_ensemble = true;
    bool do_stats   = false;
    bool do_analyze = false;
    bool do_compare = false;
    int  cmp_x = 0, cmp_y = 0, cmp_w = 0, cmp_h = 0;
    const char* model_path = nullptr;
    const char* input  = nullptr;
    const char* output = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
        else if (std::strcmp(argv[i], "--enhance") == 0) do_enhance = true;
        else if (std::strcmp(argv[i], "--no-ensemble") == 0) do_ensemble = false;
        else if (std::strcmp(argv[i], "--stats") == 0)   do_stats = true;
        else if (std::strcmp(argv[i], "--analyze") == 0) do_analyze = true;
        else if (std::strcmp(argv[i], "--compare") == 0) do_compare = true;
        // --compare swallows an optional crop: x y w h.
        if (do_compare && i + 4 < argc) {
            char* end1 = nullptr, *end2 = nullptr, *end3 = nullptr, *end4 = nullptr;
            long v1 = strtol(argv[i+1], &end1, 10);
            long v2 = strtol(argv[i+2], &end2, 10);
            long v3 = strtol(argv[i+3], &end3, 10);
            long v4 = strtol(argv[i+4], &end4, 10);
            if (end1 && *end1 == '\0' && end2 && *end2 == '\0' &&
                end3 && *end3 == '\0' && end4 && *end4 == '\0' &&
                v1 >= 0 && v2 >= 0 && v3 >= 0 && v4 >= 0) {
                cmp_x = int(v1); cmp_y = int(v2); cmp_w = int(v3); cmp_h = int(v4);
                i += 4;
            }
        }
        else if (std::strncmp(argv[i], "--denoise", 9) == 0) {
            do_denoise = true;
            if (argv[i][9] == '=') model_path = argv[i] + 10;
        }
        else if (!input)  input  = argv[i];
        else if (!output) output = argv[i];
        else { std::fprintf(stderr, "error: unexpected argument '%s'\n", argv[i]); return 1; }
    }
    if (!input) { print_usage(argv[0]); return 1; }

    // ── Highlight diagnostics ────────────────────────────────────────────────
    // Measurement before medicine: where the LED fringing comes from and how
    // large it is at each pipeline stage. Implies --denoise when a model is
    // available, so the report compares Malvar against the neural develop.
    if (do_analyze) {
        std::string mp = model_path ? model_path
            : "android/app/src/main/assets/models/model_bayer.onnx";
        return diagnose_dng_file(input, mp, do_ensemble) ? 0 : 2;
    }
    if (do_compare) {
        std::string mp = model_path ? model_path
            : "android/app/src/main/assets/models/model_bayer.onnx";
        return compare_variants(input, cmp_x, cmp_y, cmp_w, cmp_h, mp) ? 0 : 2;
    }

    // ── Read file ────────────────────────────────────────────────────────────
    FILE* f = std::fopen(input, "rb");
    if (!f) { std::fprintf(stderr, "error: cannot open '%s'\n", input); return 2; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); std::fprintf(stderr, "error: file is empty\n"); return 2; }
    std::vector<uint8_t> bytes(static_cast<size_t>(sz));
    const size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (got != bytes.size()) {
        std::fprintf(stderr, "error: short read (%zu of %zu bytes)\n", got, bytes.size());
        return 2;
    }

    // ── Decode ───────────────────────────────────────────────────────────────
    // maxDimension=0: the CLI has no GPU ceiling; show full resolution.
    // Primaries: sRGB for the CLI's PNG output.  BT.2100 would need a
    // BT.2020->sRGB gamut map on export, which the GUI does via the swapchain
    // being the right space; the CLI writes sRGB directly.
    DecodedImage img;
    std::unique_ptr<RawDenoiser> denoiser;
    if (do_denoise) {
        // Find the model: explicit path, or default relative to the executable.
        std::string mpath;
        if (model_path) {
            mpath = model_path;
        } else {
            // Try relative to executable, then CWD.
            mpath = "android/app/src/main/assets/models/model_bayer.onnx";
        }
        std::fprintf(stderr, "loading model: %s... ", mpath.c_str());
        FILE* mf = std::fopen(mpath.c_str(), "rb");
        if (!mf) { std::fprintf(stderr, "cannot open model\n"); return 2; }
        std::fseek(mf, 0, SEEK_END);
        const long msz = std::ftell(mf);
        std::fseek(mf, 0, SEEK_SET);
        std::vector<uint8_t> mbytes(static_cast<size_t>(msz));
        std::fread(mbytes.data(), 1, mbytes.size(), mf);
        std::fclose(mf);
        std::string merr;
        denoiser = RawDenoiser::load(mbytes.data(), mbytes.size(), merr);
        if (!denoiser) {
            std::fprintf(stderr, "failed: %s\n", merr.c_str());
            return 2;
        }
        std::fprintf(stderr, "ok (%s)\n", denoiser->provider().c_str());
    }
    if (looks_like_dng(bytes.data(), bytes.size())) {
        std::fprintf(stderr, "developing DNG... ");
        img = decode_dng(bytes.data(), bytes.size(), 0, WorkingPrimaries::Srgb,
                         nullptr, denoiser.get(), {}, do_ensemble);
    } else if (looks_like_jxl(bytes.data(), bytes.size())) {
        std::fprintf(stderr, "decoding JXL... ");
        img = decode_jxl(bytes.data(), bytes.size(), 0, WorkingPrimaries::Srgb);
    } else {
        std::fprintf(stderr, "error: not a DNG or JXL file\n");
        return 2;
    }
    if (!img.ok()) {
        std::fprintf(stderr, "failed: %s\n", img.error.c_str());
        return 2;
    }
    std::fprintf(stderr, "%ux%u, autoEV=%.2f, white=%.3f\n",
                 img.w, img.h, img.autoEv, img.white);

    // ── Enhance (optional) ───────────────────────────────────────────────────
    if (do_enhance) {
        std::fprintf(stderr, "enhancing... ");
        enhanceImage(img.linear, img.w, img.h, EnhanceParams{},
                     img.widePrimaries);
        std::fprintf(stderr, "done\n");
    }

    // ── Stats (optional) ─────────────────────────────────────────────────────
    if (do_stats) {
        double sum[3] = {0,0,0};
        float mn[3] = { 1e30f,  1e30f,  1e30f};
        float mx[3] = {-1e30f, -1e30f, -1e30f};
        // Saturation: mean |R-B| and |G-(R+B)/2| (how "colourful" the image is)
        double sat_rb = 0.0, sat_g = 0.0;
        size_t n = size_t(img.w) * img.h;
        for (size_t i = 0; i < n; ++i) {
            const float* p = &img.linear[i * 4];
            for (int k = 0; k < 3; ++k) {
                sum[k] += p[k];
                mn[k] = std::min(mn[k], p[k]);
                mx[k] = std::max(mx[k], p[k]);
            }
            sat_rb += std::fabs(p[0] - p[2]);
            sat_g  += std::fabs(p[1] - 0.5f*(p[0] + p[2]));
        }
        std::printf("channels   min       mean       max\n");
        const char* labels[] = {"R", "G", "B"};
        for (int k = 0; k < 3; ++k)
            std::printf("  %s    %8.4f  %8.4f  %8.4f\n",
                        labels[k], mn[k], sum[k]/n, mx[k]);
        std::printf("saturation: |R-B|=%.4f  |G-(R+B)/2|=%.4f\n",
                    sat_rb/n, sat_g/n);
    }

    // ── Export ────────────────────────────────────────────────────────────────
    const std::string out_path = output ? output : default_output(input);
    std::fprintf(stderr, "writing %s... ", out_path.c_str());
    if (!write_png(img.linear, img.w, img.h, img.autoEv, img.white,
                   img.widePrimaries, out_path.c_str())) {
        std::fprintf(stderr, "failed to write PNG\n");
        return 3;
    }
    std::fprintf(stderr, "done\n");
    std::printf("%s\n", out_path.c_str());
    return 0;
}
