// diagnose.cc — see diagnose.hh.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#include "diagnose.hh"

#include "core/dng_image.hh"
#include "core/ai_denoise.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Vec2 { double x = 0, y = 0; };

// Normalised, black-subtracted, UNclipped mosaic value in [0, ~1.x].
inline double plane_norm(const RawMosaicView& raw, int x, int y, int plane) {
    const size_t idx = size_t(y) * raw.width + size_t(x);
    const double v = raw.bayer[idx];
    return std::max((v - raw.black[plane]) /
                    std::max(raw.white - raw.black[plane], 1.0), 0.0);
}

// CFA position (0..3, raster order inside the 2x2 cell) -> colour
// 0=R 1=G(first in raster order) 2=G(second) 3=B, per CFA layout.
inline int cell_colour(int cfa, int x, int y) {
    const int p = (y & 1) * 2 + (x & 1);
    static const int kMap[4][4] = {
        {0, 1, 2, 3},   // RGGB: R G / G2 B
        {1, 0, 3, 2},   // GRBG: G R / B G2
        {1, 3, 0, 2},   // GBRG: G B / R G2   (this sensor)
        {3, 1, 2, 0},   // BGGR: B G / G2 R
    };
    return kMap[cfa][p];
}
inline int cell_plane(int cfa, int x, int y) {
    (void)cfa;
    return (y & 1) * 2 + (x & 1);
}

struct Bloom {
    Vec2 centroid;
    Vec2 hot;        // clipped pixel nearest the centroid: rings radiate here
    long count = 0;
};

// Connected components (8-neighbour BFS) over "any channel clipped".
std::vector<Bloom> find_blooms(const RawMosaicView& raw,
                               std::vector<uint8_t>& clipped) {
    const int W = int(raw.width), H = int(raw.height);
    std::vector<int> label(size_t(W) * H, -1);
    std::vector<Bloom> blooms;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = size_t(y) * W + x;
            if (!clipped[i] || label[i] >= 0) continue;

            Bloom b;
            double sx = 0, sy = 0;
            std::vector<size_t> stack{i};
            label[i] = int(blooms.size());
            while (!stack.empty()) {
                const size_t cur = stack.back();
                stack.pop_back();
                const int cx = int(cur % size_t(W)), cy = int(cur / size_t(W));
                sx += cx; sy += cy; ++b.count;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = cx + dx, ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                        const size_t ni = size_t(ny) * W + nx;
                        if (clipped[ni] && label[ni] < 0) {
                            label[ni] = label[i];
                            stack.push_back(ni);
                        }
                    }
            }
            b.centroid = {sx / double(b.count), sy / double(b.count)};
            if (b.count < 24) continue;   // noise specks need not apply

            // Light strings make centroids land in gaps between lamps; anchor
            // the ring analysis on the clipped member closest to the centroid.
            double best = 1e30;
            for (int yy = std::max(0, int(b.centroid.y) - 64);
                 yy <= std::min(H - 1, int(b.centroid.y) + 64); ++yy)
                for (int xx = std::max(0, int(b.centroid.x) - 64);
                     xx <= std::min(W - 1, int(b.centroid.x) + 64); ++xx) {
                    const size_t ci = size_t(yy) * W + xx;
                    if (!clipped[ci]) continue;
                    const double dxx = xx + 0.5 - b.centroid.x;
                    const double dyy = yy + 0.5 - b.centroid.y;
                    const double d = dxx * dxx + dyy * dyy;
                    if (d < best) { best = d; b.hot = {xx + 0.5, yy + 0.5}; }
                }
            blooms.push_back(b);
        }
    }
    // Re-filter kept blooms and sort by size.
    std::sort(blooms.begin(), blooms.end(),
              [](const Bloom& a, const Bloom& q) { return a.count > q.count; });
    return blooms;
}

// Mean per-pixel R/G and B/G of the developed image inside a ring. Ratios are
// computed per pixel BEFORE averaging so any per-pixel luminance scaling later
// in the pipeline cancels; only hue moves these numbers.
void develop_ring_stats(const std::vector<float>& rgba, uint32_t w, uint32_t h,
                        Vec2 centre, float r_lo, float r_hi,
                        double& out_rg, double& out_bg, long& out_n) {
    double srg = 0, sbg = 0;
    long n = 0;
    const int x0 = std::max(0, int(centre.x - r_hi) - 1);
    const int x1 = std::min(int(w) - 1, int(centre.x + r_hi) + 1);
    const int y0 = std::max(0, int(centre.y - r_hi) - 1);
    const int y1 = std::min(int(h) - 1, int(centre.y + r_hi) + 1);
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            const float dx = x + 0.5f - float(centre.x);
            const float dy = y + 0.5f - float(centre.y);
            const float d = std::sqrt(dx * dx + dy * dy);
            if (d < r_lo || d > r_hi) continue;
            const float* p = &rgba[(size_t(y) * w + size_t(x)) * 4];
            if (p[1] > 0.01f) {           // green as reference; skip blacks where
                srg += p[0] / p[1];       // the ratio is noise division
                sbg += p[2] / p[1];
                ++n;
            }
        }
    out_rg = n ? srg / n : 0.0;
    out_bg = n ? sbg / n : 0.0;
    out_n  = n;
}

}  // namespace

bool diagnose_dng_file(const std::string& path, const std::string& model_path,
                       bool ensemble) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(sz));
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        std::fclose(f);
        std::fprintf(stderr, "short read\n");
        return false;
    }
    std::fclose(f);

    const RawMosaicView raw = read_raw_mosaic(bytes.data(), bytes.size());
    if (!raw.ok) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), raw.error.c_str());
        return false;
    }
    const int W = int(raw.width), H = int(raw.height);

    // ── 1. Clip census ──────────────────────────────────────────────────────
    // A photosite is clipped when its RAW value reaches white: normalising
    // would clamp it to exactly 1.0 and destroy the fact that it kept going.
    static const char* kColourName[4] = {"R", "G", "G2", "B"};
    long clip_count[4] = {0, 0, 0, 0};
    long total[4] = {0, 0, 0, 0};
    std::vector<uint8_t> clipped(size_t(W) * H, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int co = cell_colour(raw.cfa, x, y);
            ++total[co];
            const bool is_clip =
                raw.bayer[size_t(y) * W + x] >= raw.white - 0.5;
            if (is_clip) { ++clip_count[co]; clipped[size_t(y) * W + x] = 1; }
        }
    std::printf("file            %s  (%ux%u, CFA %d, white %.0f)\n",
                path.c_str(), raw.width, raw.height, raw.cfa, raw.white);
    if (raw.baseline_exposure != 0.0)
        std::printf("baseline_expo   %.2f EV (suggests a merged bracket)\n",
                    raw.baseline_exposure);
    else
        std::printf("baseline_expo   0 (no merge headroom tag)\n");
    std::printf("clip census     ");
    for (int c = 0; c < 4; ++c)
        std::printf("%s %.3f%%   ", kColourName[c],
                    total[c] ? 100.0 * clip_count[c] / total[c] : 0.0);
    std::printf("\n");

    // ── 2. Blooms ───────────────────────────────────────────────────────────
    std::vector<Bloom> blooms = find_blooms(raw, clipped);
    if (blooms.empty()) {
        std::printf("blooms          none found — nothing saturated to analyse\n");
        return true;
    }
    std::printf("blooms          %zu found; analysing top %zu\n",
                blooms.size(), std::min<size_t>(blooms.size(), 3));

    // ── Develops (Malvar always; AI when a model is at hand) ────────────────
    DecodedImage malvar = decode_dng(bytes.data(), bytes.size(), 0,
                                     WorkingPrimaries::Srgb);
    if (!malvar.ok()) {
        std::fprintf(stderr, "develop failed: %s\n", malvar.error.c_str());
        return false;
    }
    std::unique_ptr<RawDenoiser> denoiser;
    DecodedImage ai;
    bool have_ai = false;
    if (!model_path.empty()) {
        FILE* mf = std::fopen(model_path.c_str(), "rb");
        if (mf) {
            std::fseek(mf, 0, SEEK_END);
            const long msz = std::ftell(mf);
            std::fseek(mf, 0, SEEK_SET);
            std::vector<uint8_t> mb(static_cast<size_t>(msz));
            if (std::fread(mb.data(), 1, mb.size(), mf) == mb.size()) {
                std::string err;
                denoiser = RawDenoiser::load(mb.data(), mb.size(), err);
                if (denoiser) {
                    ai = decode_dng(bytes.data(), bytes.size(), 0,
                                    WorkingPrimaries::Srgb, nullptr,
                                    denoiser.get());
                    have_ai = ai.ok();
                    if (!have_ai)
                        std::fprintf(stderr, "AI develop failed: %s\n",
                                     ai.error.c_str());
                } else {
                    std::fprintf(stderr, "model load failed: %s\n", err.c_str());
                }
            }
            std::fclose(mf);
        } else {
            std::fprintf(stderr, "model not found: %s (skipping AI stage)\n",
                         model_path.c_str());
        }
    }

    // Ring radii, in mosaic pixels. The bloom core is r=0; the fringe lives in
    // the first two rings; by r=30 a 12 MP frame is usually past the halo.
    static const float kRings[][2] = {{0, 1.5f}, {1.5f, 4}, {4, 10}, {10, 30}};
    static const int kNRings = 4;

    const size_t nb = std::min<size_t>(blooms.size(), 3);
    for (size_t bi = 0; bi < nb; ++bi) {
        const Bloom& bl = blooms[bi];
        std::printf("\n=== bloom %zu @ centroid (%.0f,%.0f), hot (%.0f,%.0f), clipped=%ld px ===\n",
                    bi, bl.centroid.x, bl.centroid.y, bl.hot.x, bl.hot.y, bl.count);

        // Mosaic-level rings: mean normalised value per COLOUR (G merges G1/G2).
        double ring_mean[kNRings][3] = {};
        long   ring_n[kNRings] = {};
        for (int y = std::max(0, int(bl.hot.y - 30));
             y <= std::min(H - 1, int(bl.hot.y + 30)); ++y)
            for (int x = std::max(0, int(bl.hot.x - 30));
                 x <= std::min(W - 1, int(bl.hot.x + 30)); ++x) {
                const float dx = x + 0.5f - float(bl.hot.x);
                const float dy = y + 0.5f - float(bl.hot.y);
                const float d = std::sqrt(dx * dx + dy * dy);
                int ri = -1;
                for (int r = 0; r < kNRings; ++r)
                    if (d >= kRings[r][0] && d < kRings[r][1]) { ri = r; break; }
                if (ri < 0) continue;
                const int co = cell_colour(raw.cfa, x, y);
                const double v = plane_norm(raw, x, y, cell_plane(raw.cfa, x, y));
                const int cc = (co == 2) ? 1 : co;      // G2 -> G
                ring_mean[ri][cc == 3 ? 2 : cc] += v;   // store as [R,G,B]
                ++ring_n[ri];
            }

        std::printf("ring       |  mosaic R     G     B");
        if (have_ai) std::printf("  |  malvar R/G   B/G  |  ai R/G   B/G");
        else         std::printf("  |  malvar R/G   B/G");
        std::printf("\n");
        std::printf("-----------+-----------------------");
        if (have_ai) std::printf("+---------------------+-------------------");
        else         std::printf("+------------------");
        std::printf("\n");
        for (int r = 0; r < kNRings; ++r) {
            if (!ring_n[r]) continue;
            std::printf("r %-8s |   %.3f  %.3f  %.3f",
                        (r == 0) ? "core" :
                        (r == 1) ? "1-4" : (r == 2) ? "4-10" : "10-30",
                        ring_mean[r][0] / ring_n[r],
                        ring_mean[r][1] / ring_n[r],
                        ring_mean[r][2] / ring_n[r]);
            double rg, bg; long n;
            develop_ring_stats(malvar.linear, malvar.w, malvar.h,
                               bl.hot, kRings[r][0], kRings[r][1], rg, bg, n);
            std::printf("  |   %6.3f  %6.3f", rg, bg);
            if (have_ai) {
                double arg, abg; long an;
                develop_ring_stats(ai.linear, ai.w, ai.h,
                                   bl.hot, kRings[r][0], kRings[r][1], arg, abg, an);
                std::printf("  |  %6.3f %6.3f", arg, abg);
            }
            std::printf("\n");
        }

        // ── Lateral-CA check: bright-mask centroids per colour ──────────────
        // CA displaces colour centroids radially; clipping distorts ratios but
        // leaves centroids put. Offset magnitudes of >=1 px are real CA.
        Vec2 cen[3]; long cn[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) { cen[c].x = cen[c].y = 0; }
        for (int y = std::max(0, int(bl.hot.y - 30));
             y <= std::min(H - 1, int(bl.hot.y + 30)); ++y)
            for (int x = std::max(0, int(bl.hot.x - 30));
                 x <= std::min(W - 1, int(bl.hot.x + 30)); ++x) {
                const int co = cell_colour(raw.cfa, x, y);
                const int cc = (co == 2) ? 1 : co;
                const double v = plane_norm(raw, x, y, cell_plane(raw.cfa, x, y));
                if (v >= 0.5) {
                    cen[cc == 3 ? 2 : cc].x += x + 0.5;
                    cen[cc == 3 ? 2 : cc].y += y + 0.5;
                    ++cn[cc == 3 ? 2 : cc];
                }
            }
        std::printf("CA check   ");
        const bool have_g = cn[1] > 0;
        const double gx = have_g ? cen[1].x / cn[1] : bl.hot.x;
        const double gy = have_g ? cen[1].y / cn[1] : bl.hot.y;
        for (int c = 0; c < 3; ++c) {
            if (!cn[c] || c == 1) continue;
            std::printf("  %s-G (%+.1f,%+.1f)px",
                        (c == 0) ? "R" : "B",
                        cen[c].x / cn[c] - gx,
                        cen[c].y / cn[c] - gy);
        }
        if (!cn[0] && !cn[2]) std::printf("  (no >=50%% pixels outside G)");
        std::printf("\n");
    }

    std::printf("\nnote: developed ratios use per-pixel mean(R/G), mean(B/G)\n"
                   "(toe/exposure invariant). A white LED develops to ≈1.0/≈1.0;\n"
                   "magenta = R/G high with B/G high; green fringe = both low;\n"
                   "cyan = B/G high with R/G low.\n");
    return true;
}

// ── compare_variants: where did the detail go? ──────────────────────────────

namespace {

struct Region { int x = 0, y = 0, w = 0, h = 0; };

// Mean |2c - l - r| on green inside the region (horizontal Laplacian).
double hf_energy(const std::vector<float>& rgba, uint32_t w,
                 const Region& r, bool vertical) {
    if (r.w < 16 || r.h < 16) return 0.0;
    double sum = 0;
    long n = 0;
    const int x0 = r.x + 8, x1 = r.x + r.w - 8;
    const int y0 = r.y + 8, y1 = r.y + r.h - 8;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            const size_t i = size_t(y) * w + x;
            const float c = rgba[i * 4 + 1];
            const float a = vertical ? rgba[(i - w) * 4 + 1] : rgba[(i - 1) * 4 + 1];
            const float b = vertical ? rgba[(i + w) * 4 + 1] : rgba[(i + 1) * 4 + 1];
            sum += std::fabs(2.0f * c - a - b);
            ++n;
        }
    return n ? sum / n : 0.0;
}

// Green-Laplacian samples pooled around tile-boundary lines versus control
// samples at distance 3..6 px from the same lines — close enough to share
// local content, outside the seam itself. A patchwork stitcher shows on/near
// diverging; a clean one shows ~1.0.
void strip_samples(const std::vector<float>& rgba, uint32_t w,
                   const Region& reg, bool verticalLines,
                   const std::vector<int>& lines,
                   double& sum_on, long& n_on,
                   double& sum_near, long& n_near) {
    for (int bd : lines) {
        const int lo = verticalLines ? reg.y + 8 : reg.x + 8;
        const int hi = verticalLines ? reg.y + reg.h - 8 : reg.x + reg.w - 8;
        for (int t = lo; t < hi; ++t) {
            for (int d = -6; d <= 6; ++d) {
                const int c = verticalLines ? bd + d : t;
                const int r = verticalLines ? t : bd + d;
                if (c < reg.x + 2 || c >= reg.x + reg.w - 2) continue;
                if (r < reg.y + 2 || r >= reg.y + reg.h - 2) continue;
                const size_t i = size_t(r) * w + c;
                const float g0 = rgba[i * 4 + 1];
                // Laplacian across the seam, not along it.
                const float gp = verticalLines ? rgba[(i + 1) * 4 + 1]
                                               : rgba[(i + w) * 4 + 1];
                const float gm = verticalLines ? rgba[(i - 1) * 4 + 1]
                                               : rgba[(i - w) * 4 + 1];
                const double lap = std::fabs(2.0f * g0 - gm - gp);
                if (std::abs(d) <= 2) { sum_on += lap; ++n_on; }
                else                  { sum_near += lap; ++n_near; }
            }
        }
    }
}

}  // namespace

bool compare_variants(const std::string& path, int x, int y, int w, int h,
                      const std::string& model_path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes(static_cast<size_t>(sz));
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    // Model first: both AI variants share one session.
    std::unique_ptr<RawDenoiser> denoiser;
    {
        FILE* mf = std::fopen(model_path.c_str(), "rb");
        if (!mf) { std::fprintf(stderr, "model not found: %s\n", model_path.c_str()); return false; }
        std::fseek(mf, 0, SEEK_END);
        const long msz = std::ftell(mf);
        std::fseek(mf, 0, SEEK_SET);
        std::vector<uint8_t> mb(static_cast<size_t>(msz));
        const bool got = std::fread(mb.data(), 1, mb.size(), mf) == mb.size();
        std::fclose(mf);
        if (!got) return false;
        std::string err;
        denoiser = RawDenoiser::load(mb.data(), mb.size(), err);
        if (!denoiser) { std::fprintf(stderr, "%s\n", err.c_str()); return false; }
    }

    DecodedImage malvar = decode_dng(bytes.data(), bytes.size(), 0,
                                     WorkingPrimaries::Srgb);
    if (!malvar.ok()) { std::fprintf(stderr, "%s\n", malvar.error.c_str()); return false; }

    // Single pass then ensemble; each replaces the previous buffer.
    auto logsz = [](const char* tag, const DecodedImage& d) {
        std::fprintf(stderr, "[cmp] %s: %ux%u floats=%zu (need %zu)\n", tag,
                     d.w, d.h, d.linear.size(), size_t(d.w) * d.h * 4);
    };
    DecodedImage single = decode_dng(bytes.data(), bytes.size(), 0,
                                     WorkingPrimaries::Srgb, nullptr,
                                     denoiser.get(), {}, false);
    if (!single.ok()) { std::fprintf(stderr, "%s\n", single.error.c_str()); return false; }
    logsz("single", single);
    DecodedImage ens = decode_dng(bytes.data(), bytes.size(), 0,
                                  WorkingPrimaries::Srgb, nullptr,
                                  denoiser.get(), {}, true);
    if (!ens.ok()) { std::fprintf(stderr, "%s\n", ens.error.c_str()); return false; }
    logsz("ens", ens);

    Region reg{x, y, w, h};
    if (reg.w <= 0 || reg.h <= 0) {
        reg.x = 64; reg.y = 64;
        reg.w = int(malvar.w) - 128;
        reg.h = int(malvar.h) - 128;
    }
    reg.w = std::min(reg.w, int(malvar.w) - reg.x);
    reg.h = std::min(reg.h, int(malvar.h) - reg.y);

    // Interior tile boundaries every kAiStep*kAiScale output px. The AI plane
    // may sit 1 px off image space (RGGB crop); the ±6 px windows absorb that.
    std::vector<int> vlines, hlines;
    for (int b = 768; b < reg.x + reg.w - 16; b += 768)
        if (b > reg.x + 16) vlines.push_back(b);
    for (int b = 768; b < reg.y + reg.h - 16; b += 768)
        if (b > reg.y + 16) hlines.push_back(b);

    struct V { const char* name; const DecodedImage& img; };
    const V variants[] = { {"malvar", malvar}, {"ai-single", single}, {"ai-x4", ens} };

    std::printf("compare: %s  region %dx%d @ (%d,%d)   boundaries v=%zu h=%zu\n",
                path.c_str(), reg.w, reg.h, reg.x, reg.y,
                vlines.size(), hlines.size());
    std::printf("%-10s | %7s %7s | %9s %9s | %7s\n",
                "variant", "hfH", "hfV", "grid-on", "grid-near", "on/near");
    for (const V& v : variants) {
        const double hfH = hf_energy(v.img.linear, v.img.w, reg, false);
        const double hfV = hf_energy(v.img.linear, v.img.w, reg, true);
        double so = 0, sn = 0;
        long no = 0, nn = 0;
        strip_samples(v.img.linear, v.img.w, reg, true,  vlines, so, no, sn, nn);
        strip_samples(v.img.linear, v.img.w, reg, false, hlines, so, no, sn, nn);
        const double ratio = (sn > 0 && nn > 0) ? (so / no) / (sn / nn) : 0.0;
        std::printf("%-10s | %7.5f %7.5f | %9.5f %9.5f | %7.3f\n",
                    v.name, hfH, hfV, so / std::max(no, 1L), sn / std::max(nn, 1L),
                    ratio);
    }
    std::printf("\nread:\n"
                "  hfH/hfV — sub-3px structure kept (Malvar is the ceiling).\n"
                "  on/near — seam-strip Laplacian vs same-content control;\n"
                "            ~1.0 clean, >>1 rim interference, <<1 rim smoothing.\n");

    // ── Patch grid: WHERE does AI diverge from Malvar? ──────────────────────
    // A 6x4 grid over the region; per cell, Malvar-vs-AI Laplacian ratio.
    // Distant/low-contrast subjects live in cells whose REAL fine structure is
    // small — if spurious ensemble speckle is the complaint, those cells show
    // ratio >> 1 while textured ones stay <=~1.2.
    const int gx_n = 6, gy_n = 4;
    struct PRow { double m, s, e; long n; };
    printf("\npatch grid (malvar | single | x4 Laplacian, ratio-to-malvar):\n");
    for (int gy = 0; gy < gy_n; ++gy) {
        for (int gx = 0; gx < gx_n; ++gx) {
            Region c{reg.x + gx * reg.w / gx_n,
                     reg.y + gy * reg.h / gy_n,
                     reg.w / gx_n, reg.h / gy_n};
            const double em = hf_energy(malvar.linear, malvar.w, c, false);
            const double es = hf_energy(single.linear, single.w, c, false);
            const double ee = hf_energy(ens.linear, ens.w, c, false);
            printf("  M%6.4f(%4.2f) S%6.4f(%4.2f) X%6.4f(%4.2f)",
                   em, es / std::max(em, 1e-9), es, es / std::max(em, 1e-9),
                   ee, ee / std::max(em, 1e-9));
        }
        printf("\n");
    }
    return true;
}
