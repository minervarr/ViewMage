// png_export.cc — see png_export.hh.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#include "png_export.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cerrno>
#include <ctime>
#include <sys/stat.h>

#include "core/export_math.hh"
#include "log.hh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef __ANDROID__
#include "activity_bridge.hh"
#include "app_paths.hh"
#endif

namespace {

using export_math::kBayerDither;
using export_math::kBt2020ToSrgb;
using export_math::kLumaBt709;
using export_math::kLumaBt2020;
using export_math::rolloff;
using export_math::to_srgb8_dithered;

bool make_dirs(const std::string& path) {
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (mkdir(part.c_str(), 0775) != 0 && errno != EEXIST) return false;
    }
    return mkdir(path.c_str(), 0775) == 0 || errno == EEXIST;
}

// ── The one render both formats share ───────────────────────────────────────
//
// Display-referred linear in (the same exposure/rolloff the shader runs),
// interleaved sRGB8 out. Dithered at the quantisation: a smooth gradient that
// crosses one code value every few pixels alternates levels instead of
// stepping into visible bands. JPEG benefits from the dither too — its own
// 8-bit DCT path bands the same way.
bool render_srgb8(const std::vector<float>& linear, uint32_t w, uint32_t h,
                  float ev, float white, bool primariesAreWide,
                  std::vector<uint8_t>& rgb, std::string& err) {
    if (w == 0 || h == 0 || linear.size() < size_t(w) * h * 4) {
        err = "there is nothing to export";
        return false;
    }
    try {
        rgb.resize(size_t(w) * h * 3);
    } catch (const std::bad_alloc&) {
        err = "not enough memory to export this photo";
        return false;
    }

    const float exposure = std::exp2(ev);
    const float W = std::max(white, 1.0f);
    // BT.2020 luma weights when the pixels are wide, matching the shader. Using
    // 709's on 2020 data skews the rolloff ratio per hue, which comes back as a
    // green cast -- the same bug, one layer down.
    const float* lw = primariesAreWide ? kLumaBt2020 : kLumaBt709;
    const float lr = lw[0], lg = lw[1], lb = lw[2];

    for (size_t y = 0, n = size_t(w) * h; y < n; y += w) {
        const uint32_t row = uint32_t(y / w);
        const float* drow = kBayerDither.m[row & 3];
        for (uint32_t x = 0; x < w; ++x) {
            const size_t i = y + x;
            const float* p = &linear[i * 4];
            // Clamp negatives FIRST. An out-of-gamut colour arrives with a negative
            // component; feeding that to the luminance dot product collapses L and
            // makes the ratio explode, which turns bright speculars magenta. Same
            // ordering, and the same reason, as image_frag.slang.
            float c[3] = {std::max(p[0] * exposure, 0.0f),
                          std::max(p[1] * exposure, 0.0f),
                          std::max(p[2] * exposure, 0.0f)};

            // Tone map on LUMINANCE and scale the channels, so a saturated highlight
            // keeps its hue instead of being desaturated toward white per channel.
            const float L = lr * c[0] + lg * c[1] + lb * c[2];
            if (L > 1e-6f) {
                const float s = rolloff(L, W) / L;
                c[0] *= s; c[1] *= s; c[2] *= s;
            }

            if (primariesAreWide) {
                const float r = c[0], g = c[1], b = c[2];
                for (int k = 0; k < 3; ++k)
                    c[k] = kBt2020ToSrgb[k*3] * r + kBt2020ToSrgb[k*3+1] * g +
                           kBt2020ToSrgb[k*3+2] * b;
            }

            const float dith = drow[x & 3];
            uint8_t* d = &rgb[i * 3];
            for (int k = 0; k < 3; ++k) d[k] = to_srgb8_dithered(c[k], dith);
        }
    }
    return true;
}

}  // namespace

// Shared tail of both formats: publish through MediaStore (the only route into
// the gallery from API 29), fall back to the app's own folder, and SAY which
// one happened -- naming the wrong place is how a photo gets lost.
static bool publish_bytes(const std::vector<uint8_t>& bytes, const char* mime,
                          const char* ext, const std::string& displayName,
                          uint32_t w, uint32_t h,
                          std::string& where, std::string& err) {
#ifdef __ANDROID__
    const std::string uri = activity::publish_image(displayName, mime,
                                                    "ViewMage", bytes.data(),
                                                    bytes.size());
    if (!uri.empty()) {
        where = "Pictures/ViewMage";
        VCE_LOGI("ViewMage", "published %ux%u %s (%zu KB) to %s",
                 w, h, ext + 1, bytes.size() / 1024, uri.c_str());
        return true;
    }
    VCE_LOGE("ViewMage", "MediaStore refused the image; falling back to app storage");

    std::string base = app_paths::stateDir();
    while (!base.empty() && base.back() == '/') base.pop_back();
    const std::string dir = base + "/exports";
    if (!make_dirs(dir)) { err = "nowhere to save to"; return false; }
    const std::string path = dir + "/" + displayName;
    where = "the app folder";
#else
    const std::string path = displayName;
    where = path;
#endif

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "could not create the file"; return false; }
    const size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (wrote != bytes.size()) {
        std::remove(path.c_str());
        err = "could not write the file";
        return false;
    }
    VCE_LOGI("ViewMage", "exported %ux%u %s to %s", w, h, ext + 1, path.c_str());
    return true;
}

// "IMG_20260825_012700.dng" -> "IMG_20260825_012700.png" / ".jpg", and
// anything without a usable name gets a timestamped one. Never empty.
std::string export_name_for(const std::string& sourceName, const char* ext) {
    std::string base = sourceName;
    const size_t slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    const size_t dot = base.find_last_of('.');
    if (dot != std::string::npos && dot > 0) base = base.substr(0, dot);

    if (base.empty()) {
        char buf[32];
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        std::strftime(buf, sizeof(buf), "IMG_%Y%m%d_%H%M%S", &tm);
        base = buf;
    }
    return base + ext;
}

bool export_png(const std::vector<float>& linear, uint32_t w, uint32_t h,
                float ev, float white, bool primariesAreWide,
                const std::string& displayName,
                std::string& where, std::string& err) {
    std::vector<uint8_t> rgb;
    if (!render_srgb8(linear, w, h, ev, white, primariesAreWide, rgb, err))
        return false;

    // 4, not the default 8: at 12 MP the highest level costs seconds of phone
    // CPU to save a few per cent, and the user is waiting on this.
    stbi_write_png_compression_level = 4;

    std::vector<uint8_t> png;
    stbi_write_png_to_func(
        [](void* ctx, void* d, int n) {
            auto* v = static_cast<std::vector<uint8_t>*>(ctx);
            const auto* b = static_cast<const uint8_t*>(d);
            v->insert(v->end(), b, b + n);
        },
        &png, int(w), int(h), 3, rgb.data(), int(w) * 3);

    if (png.empty()) {
        err = "could not encode the image";
        return false;
    }
    return publish_bytes(png, "image/png", ".png", displayName, w, h, where, err);
}

bool export_jpeg(const std::vector<float>& linear, uint32_t w, uint32_t h,
                 float ev, float white, bool primariesAreWide,
                 const std::string& displayName,
                 std::string& where, std::string& err) {
    std::vector<uint8_t> rgb;
    if (!render_srgb8(linear, w, h, ev, white, primariesAreWide, rgb, err))
        return false;

    // Quality 95: visually transparent on photographs, and roughly 5-8x
    // smaller than the PNG — the point of choosing JPEG for chat apps.
    std::vector<uint8_t> jpg;
    stbi_write_jpg_to_func(
        [](void* ctx, void* d, int n) {
            auto* v = static_cast<std::vector<uint8_t>*>(ctx);
            const auto* b = static_cast<const uint8_t*>(d);
            v->insert(v->end(), b, b + n);
        },
        &jpg, int(w), int(h), 3, rgb.data(), 95);

    if (jpg.empty()) {
        err = "could not encode the image";
        return false;
    }
    return publish_bytes(jpg, "image/jpeg", ".jpg", displayName, w, h, where, err);
}
