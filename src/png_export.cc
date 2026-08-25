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

#include "log.hh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef __ANDROID__
#include "activity_bridge.hh"
#include "app_paths.hh"
#endif

namespace {

// BT.2020 -> sRGB, D65, no chromatic adaptation needed (same white point).
//
// The export is sRGB because that is what a phone gallery, a chat app and a
// browser all assume. Sending BT.2020 numbers labelled as nothing is how the
// green cast at the very start of this work happened; the conversion has to be
// real, not a relabel.
const float kBt2020ToSrgb[9] = {
     1.6605f, -0.5876f, -0.0728f,
    -0.1246f,  1.1329f, -0.0083f,
    -0.0182f, -0.1006f,  1.1187f,
};

// The shader's rolloff, in C++. MIRRORS rolloff() in image_frag.slang and
// rolloffCurve() in vk_canvas's output_target.cc -- the export must be the same
// curve as the screen or it is a different picture. Ceiling is 1.0 here: a PNG
// has no headroom above display white, whatever the panel had.
float rolloff(float x, float W) {
    const float k = 0.8f;
    if (x <= k) return x;
    const float P = std::min(1.0f, W);
    const float span = std::max(W - k, 1e-4f);
    const float a = (x - k) / span;
    const float d = span / std::max(P - k, 1e-4f);
    return k + (P - k) * (a * (1.0f + a / d) / (1.0f + a));
}

uint8_t to_srgb8(float x) {
    x = std::min(std::max(x, 0.0f), 1.0f);
    const float e = (x <= 0.0031308f) ? (12.92f * x)
                                      : (1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f);
    return uint8_t(std::lround(std::min(std::max(e, 0.0f), 1.0f) * 255.0f));
}

bool make_dirs(const std::string& path) {
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (mkdir(part.c_str(), 0775) != 0 && errno != EEXIST) return false;
    }
    return mkdir(path.c_str(), 0775) == 0 || errno == EEXIST;
}

}  // namespace

std::string export_name_for(const std::string& sourceName) {
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
    return base + ".png";
}

bool export_png(const std::vector<float>& linear, uint32_t w, uint32_t h,
                float ev, float white, bool primariesAreWide,
                const std::string& displayName,
                std::string& where, std::string& err) {
    if (w == 0 || h == 0 || linear.size() < size_t(w) * h * 4) {
        err = "there is nothing to export";
        return false;
    }

    std::vector<uint8_t> rgb;
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
    const float lr = primariesAreWide ? 0.2627f : 0.2126f;
    const float lg = primariesAreWide ? 0.6780f : 0.7152f;
    const float lb = primariesAreWide ? 0.0593f : 0.0722f;

    for (size_t i = 0, n = size_t(w) * h; i < n; ++i) {
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

        uint8_t* d = &rgb[i * 3];
        for (int k = 0; k < 3; ++k) d[k] = to_srgb8(c[k]);
    }

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

#ifdef __ANDROID__
    // The supported route, and the only one that reaches the gallery.
    const std::string uri = activity::publish_image(displayName, "image/png",
                                                    "ViewMage", png.data(),
                                                    png.size());
    if (!uri.empty()) {
        where = "Pictures/ViewMage";
        VCE_LOGI("ViewMage", "published %ux%u PNG (%zu KB) to %s",
                 w, h, png.size() / 1024, uri.c_str());
        return true;
    }
    VCE_LOGE("ViewMage", "MediaStore refused the image; falling back to app storage");

    // Fallback: the app's own directory. Not where the user wants it, but a
    // real file they can still reach -- and the caller SAYS which one it was,
    // because naming the wrong place is how a photo gets lost.
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
    const size_t wrote = std::fwrite(png.data(), 1, png.size(), f);
    const bool ok = (wrote == png.size());
    std::fclose(f);
    if (!ok) { std::remove(path.c_str()); err = "could not write the file"; return false; }

    VCE_LOGI("ViewMage", "exported %ux%u PNG to %s", w, h, path.c_str());
    return true;
}
