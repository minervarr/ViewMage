// document_scan.cc — page detection and perspective flatten.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.

#include "core/document_scan.hh"
#include "core/enhance.hh"
#include "core/page_quad.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#ifdef VIEWMAGE_WITH_ORT
#include "onnxruntime_cxx_api.h"
#endif

namespace {

struct Pt {
    float x, y;
};

struct Line {
    double px = 0, py = 0, dx = 0, dy = 0;
    bool ok = false;
};

float dist(Pt a, Pt b) {
    const float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

long long cross(Pt o, Pt a, Pt b) {
    return (long long)(a.x - o.x) * (long long)(b.y - o.y) -
           (long long)(a.y - o.y) * (long long)(b.x - o.x);
}

// Andrew's monotone chain. `pts` is sorted by x then y.
std::vector<Pt> convexHull(std::vector<Pt> pts) {
    std::sort(pts.begin(), pts.end(), [](Pt a, Pt b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(std::unique(pts.begin(), pts.end(), [](Pt a, Pt b) {
        return a.x == b.x && a.y == b.y;
    }), pts.end());
    const int n = (int)pts.size();
    if (n < 3) return pts;
    std::vector<Pt> h(2 * n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    const int lower = k + 1;
    for (int i = n - 2; i >= 0; --i) {
        while (k >= lower && cross(h[k - 2], h[k - 1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    h.resize(k - 1);
    return h;
}

// TL, TR, BR, BL. Sum and difference of the coordinates separate the four
// corners of a convex page as long as it is not folded over itself.
void orderCorners(Pt p[4]) {
    auto sum = [](Pt q) { return q.x + q.y; };
    auto dif = [](Pt q) { return q.y - q.x; };
    Pt tl = p[0], br = p[0], tr = p[0], bl = p[0];
    for (int i = 1; i < 4; ++i) {
        if (sum(p[i]) < sum(tl)) tl = p[i];
        if (sum(p[i]) > sum(br)) br = p[i];
        if (dif(p[i]) < dif(tr)) tr = p[i];
        if (dif(p[i]) > dif(bl)) bl = p[i];
    }
    p[0] = tl; p[1] = tr; p[2] = br; p[3] = bl;
}

bool convexQuad(const Pt p[4]) {
    int sign = 0;
    for (int i = 0; i < 4; ++i) {
        const Pt& o = p[i];
        const Pt& a = p[(i + 1) % 4];
        const Pt& b = p[(i + 2) % 4];
        const double c = (double)(a.x - o.x) * (b.y - o.y) -
                         (double)(a.y - o.y) * (b.x - o.x);
        if (std::fabs(c) < 1.0) return false;
        const int s = c > 0 ? 1 : -1;
        if (sign == 0) sign = s;
        else if (s != sign) return false;
    }
    return true;
}

float quadArea(const Pt p[4]) {
    float a = 0;
    for (int i = 0; i < 4; ++i) {
        const Pt q = p[(i + 1) % 4];
        a += p[i].x * q.y - q.x * p[i].y;
    }
    return std::fabs(a) * 0.5f;
}

// Destination pixel (x, y) -> source. h is row-major 3x3 with h[8] == 1.
bool homography(const Pt dst[4], const Pt src[4], double h[9]) {
    double A[8][9];
    for (int i = 0; i < 4; ++i) {
        const double x = dst[i].x, y = dst[i].y;
        const double u = src[i].x, v = src[i].y;
        const int r = i * 2;
        A[r][0] = x; A[r][1] = y; A[r][2] = 1;
        A[r][3] = 0; A[r][4] = 0; A[r][5] = 0;
        A[r][6] = -x * u; A[r][7] = -y * u; A[r][8] = u;
        A[r + 1][0] = 0; A[r + 1][1] = 0; A[r + 1][2] = 0;
        A[r + 1][3] = x; A[r + 1][4] = y; A[r + 1][5] = 1;
        A[r + 1][6] = -x * v; A[r + 1][7] = -y * v; A[r + 1][8] = v;
    }
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        for (int r = col + 1; r < 8; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-12) return false;
        if (piv != col)
            for (int c = col; c < 9; ++c) std::swap(A[col][c], A[piv][c]);
        const double div = A[col][col];
        for (int c = col; c < 9; ++c) A[col][c] /= div;
        for (int r = 0; r < 8; ++r) if (r != col) {
            const double f = A[r][col];
            for (int c = col; c < 9; ++c) A[r][c] -= f * A[col][c];
        }
    }
    for (int i = 0; i < 8; ++i) h[i] = A[i][8];
    h[8] = 1.0;
    return true;
}

float cubicWeight(float t) {
    // Catmull-Rom, a = -0.5. `t` is the distance in pixels from the sample.
    const float a = -0.5f;
    t = std::fabs(t);
    if (t < 1.0f)
        return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
    if (t < 2.0f)
        return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
    return 0.0f;
}

// 4x4 Catmull-Rom. Flat colour is reproduced exactly, so a stamp stays a
// stamp; edges come back sharper than bilinear, which is what body text needs.
float sampleBicubic(const float* img, uint32_t w, uint32_t h, int c,
                    float x, float y) {
    const int x1 = (int)std::floor(x);
    const int y1 = (int)std::floor(y);
    const float tx = x - (float)x1;
    const float ty = y - (float)y1;
    float acc = 0.0f;
    for (int j = -1; j <= 2; ++j) {
        const float wy = cubicWeight(ty - (float)j);
        const int yy = std::clamp(y1 + j, 0, (int)h - 1);
        const float* row = img + (size_t)yy * w * 4 + c;
        for (int i = -1; i <= 2; ++i) {
            const int xx = std::clamp(x1 + i, 0, (int)w - 1);
            acc += wy * cubicWeight(tx - (float)i) * row[(size_t)xx * 4];
        }
    }
    return acc;
}

int otsu(const std::vector<uint8_t>& g) {
    int hist[256] = {};
    for (uint8_t v : g) ++hist[v];
    const int n = (int)g.size();
    double sum = 0;
    for (int i = 0; i < 256; ++i) sum += (double)i * hist[i];
    double sumB = 0;
    int wB = 0;
    double best = -1;
    int thresh = 128;
    for (int t = 0; t < 256; ++t) {
        wB += hist[t];
        if (wB == 0) continue;
        const int wF = n - wB;
        if (wF == 0) break;
        sumB += (double)t * hist[t];
        const double mB = sumB / wB;
        const double mF = (sum - sumB) / wF;
        const double var = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        if (var > best) { best = var; thresh = t; }
    }
    return thresh;
}

void boxBlur(const std::vector<float>& src, std::vector<float>& dst,
             int w, int h, int r) {
    dst.resize(src.size());
    if (r < 1) { dst = src; return; }
    std::vector<float> tmp(src.size());
    std::vector<double> pref((size_t)std::max(w, h) + 1);
    for (int y = 0; y < h; ++y) {
        pref[0] = 0;
        for (int x = 0; x < w; ++x)
            pref[x + 1] = pref[x] + src[(size_t)y * w + x];
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - r);
            const int x1 = std::min(w - 1, x + r);
            tmp[(size_t)y * w + x] =
                (float)((pref[x1 + 1] - pref[x0]) / (double)(x1 - x0 + 1));
        }
    }
    for (int x = 0; x < w; ++x) {
        pref[0] = 0;
        for (int y = 0; y < h; ++y)
            pref[y + 1] = pref[y] + tmp[(size_t)y * w + x];
        for (int y = 0; y < h; ++y) {
            const int y0 = std::max(0, y - r);
            const int y1 = std::min(h - 1, y + r);
            dst[(size_t)y * w + x] =
                (float)((pref[y1 + 1] - pref[y0]) / (double)(y1 - y0 + 1));
        }
    }
}

float percentile(std::vector<float> v, double frac) {
    if (v.empty()) return 0.0f;
    frac = std::clamp(frac, 0.0, 1.0);
    const size_t k = (size_t)std::clamp((double)(v.size() - 1) * frac, 0.0,
                                        (double)(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (long)k, v.end());
    return v[k];
}

void extremumFilter(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst,
                    int w, int h, int r, bool dilate) {
    dst.assign((size_t)w * h, 0);
    std::vector<uint8_t> tmp((size_t)w * h, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t m = dilate ? 0 : 1;
            const int x0 = std::max(0, x - r);
            const int x1 = std::min(w - 1, x + r);
            for (int xx = x0; xx <= x1; ++xx) {
                const uint8_t v = src[(size_t)y * w + xx];
                m = dilate ? std::max(m, v) : std::min(m, v);
            }
            tmp[(size_t)y * w + x] = m;
        }
    }
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - r);
        const int y1 = std::min(h - 1, y + r);
        for (int x = 0; x < w; ++x) {
            uint8_t m = dilate ? 0 : 1;
            for (int yy = y0; yy <= y1; ++yy) {
                const uint8_t v = tmp[(size_t)yy * w + x];
                m = dilate ? std::max(m, v) : std::min(m, v);
            }
            dst[(size_t)y * w + x] = m;
        }
    }
}

double pointLineDist(Pt p, const Line& L) {
    const double vx = (double)p.x - L.px;
    const double vy = (double)p.y - L.py;
    return std::fabs(vx * L.dy - vy * L.dx);
}

Line lineThrough(Pt a, Pt b) {
    Line L;
    double dx = (double)b.x - a.x, dy = (double)b.y - a.y;
    const double len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-6) return L;
    L.px = a.x; L.py = a.y;
    L.dx = dx / len; L.dy = dy / len;
    L.ok = true;
    return L;
}

Line fitLine(const std::vector<Pt>& pts) {
    Line none;
    const int n = (int)pts.size();
    if (n < 2) return none;
    const int step = std::max(1, n / 12);
    Line best;
    int bestCount = 0;
    for (int i = 0; i < n; i += step) {
        for (int j = i + step; j < n; j += step) {
            const Line cand = lineThrough(pts[i], pts[j]);
            if (!cand.ok) continue;
            int cnt = 0;
            for (const Pt& p : pts)
                if (pointLineDist(p, cand) <= 2.0) ++cnt;
            if (cnt > bestCount) { bestCount = cnt; best = cand; }
        }
    }
    // A two-point side never enters the loop above when step skips j.
    if (!best.ok) best = lineThrough(pts.front(), pts.back());
    if (!best.ok) return none;

    double mx = 0, my = 0;
    int used = 0;
    for (const Pt& p : pts) {
        if (pointLineDist(p, best) > 2.0) continue;
        mx += p.x; my += p.y; ++used;
    }
    const int need = n <= 6 ? 2 : std::max(2, (n * 3) / 10);
    if (used < need) return none;
    mx /= used; my /= used;
    double xx = 0, xy = 0, yy = 0;
    for (const Pt& p : pts) {
        if (pointLineDist(p, best) > 2.0) continue;
        const double dx = p.x - mx, dy = p.y - my;
        xx += dx * dx; xy += dx * dy; yy += dy * dy;
    }
    const double ang = 0.5 * std::atan2(2.0 * xy, xx - yy);
    Line out;
    out.px = mx; out.py = my;
    out.dx = std::cos(ang); out.dy = std::sin(ang);
    out.ok = true;
    return out;
}

bool intersectLines(const Line& a, const Line& b, Pt& o) {
    if (!a.ok || !b.ok) return false;
    const double cross = a.dx * b.dy - a.dy * b.dx;
    if (std::fabs(cross) < 1e-8) return false;
    const double qx = b.px - a.px, qy = b.py - a.py;
    const double t = (qx * b.dy - qy * b.dx) / cross;
    o.x = (float)(a.px + t * a.dx);
    o.y = (float)(a.py + t * a.dy);
    return std::isfinite(o.x) && std::isfinite(o.y);
}

bool inSweep(float a, float start, float end) {
    constexpr float kTwoPi = 6.2831853f;
    constexpr float kPad = 0.12f;
    auto norm = [](float x) {
        while (x < 0) x += kTwoPi;
        while (x >= kTwoPi) x -= kTwoPi;
        return x;
    };
    const float span = norm(end - start);
    if (span > 3.6f) return false;
    const float rel = norm(a - start);
    return rel <= span + kPad || rel >= kTwoPi - kPad;
}

// A finger becomes a hull vertex and the hull edge jumps straight to it, so
// the real page edge is no longer on the hull — it is still on the contour.
// Each side is the dominant straight run of contour points in that sector.
// The corner is where the two runs meet. A wild intersection keeps the
// extremal vertex.
bool cornersFromOutline(const std::vector<Pt>& contour, const std::vector<Pt>& hull,
                        int sw, int sh, Pt out[4]) {
    const int n = (int)hull.size();
    if (n < 4 || contour.size() < 8) return false;
    int iTL = 0, iTR = 0, iBR = 0, iBL = 0;
    for (int i = 1; i < n; ++i) {
        const float s = hull[i].x + hull[i].y;
        const float d = hull[i].y - hull[i].x;
        if (s < hull[iTL].x + hull[iTL].y) iTL = i;
        if (s > hull[iBR].x + hull[iBR].y) iBR = i;
        if (d < hull[iTR].y - hull[iTR].x) iTR = i;
        if (d > hull[iBL].y - hull[iBL].x) iBL = i;
    }
    Pt extremal[4] = {hull[iTL], hull[iTR], hull[iBR], hull[iBL]};
    if (extremal[0].x == extremal[1].x && extremal[0].y == extremal[1].y) return false;

    double cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) { cx += extremal[i].x; cy += extremal[i].y; }
    cx *= 0.25; cy *= 0.25;

    int order[4] = {0, 1, 2, 3};
    float ang[4];
    for (int i = 0; i < 4; ++i)
        ang[i] = std::atan2((float)(extremal[i].y - cy), (float)(extremal[i].x - cx));
    std::sort(order, order + 4, [&](int a, int b) { return ang[a] < ang[b]; });

    Line side[4];
    Pt sectorPt[4];
    for (int k = 0; k < 4; ++k) {
        const int a = order[k];
        const int b = order[(k + 1) % 4];
        sectorPt[k] = extremal[a];
        const float a0 = ang[a], a1 = ang[b];
        std::vector<Pt> pts;
        pts.reserve(contour.size() / 4);
        for (const Pt& p : contour) {
            const float pa = std::atan2(p.y - (float)cy, p.x - (float)cx);
            if (inSweep(pa, a0, a1)) pts.push_back(p);
        }
        side[k] = fitLine(pts);
        if (!side[k].ok) side[k] = lineThrough(extremal[a], extremal[b]);
    }

    Pt fitted[4];
    for (int k = 0; k < 4; ++k) {
        const int prev = (k + 3) % 4;
        const Pt& ext = sectorPt[k];
        Pt hit{};
        const bool got = intersectLines(side[prev], side[k], hit);
        const float sideLen = std::max(dist(sectorPt[prev], ext),
                                        dist(ext, sectorPt[(k + 1) % 4]));
        const bool inFrame = hit.x >= -0.08f * (float)sw && hit.x <= 1.08f * (float)sw &&
                             hit.y >= -0.08f * (float)sh && hit.y <= 1.08f * (float)sh;
        const bool outlier = pointLineDist(ext, side[prev]) > 2.5 ||
                             pointLineDist(ext, side[k]) > 2.5;
        const bool near = sideLen > 1.0f && dist(hit, ext) < 0.20f * sideLen;
        if (got && inFrame && (outlier || near)) fitted[k] = hit;
        else fitted[k] = ext;
    }
    if (!convexQuad(fitted)) {
        for (int k = 0; k < 4; ++k) fitted[k] = sectorPt[k];
        if (!convexQuad(fitted)) return false;
    }
    for (int k = 0; k < 4; ++k) out[k] = fitted[k];
    orderCorners(out);
    return convexQuad(out);
}

// Ordered outer boundary. The edge list used for the quad fit is only a set
// of pixels; a fold and a bow need the pixels in the order you walk them.
std::vector<Pt> traceBoundary(const std::vector<int>& labels, int sw, int sh, int label) {
    int sx = -1, sy = -1;
    for (int y = 0; y < sh && sx < 0; ++y) {
        for (int x = 0; x < sw; ++x) {
            if (labels[(size_t)y * sw + x] != label) continue;
            const bool edge = x == 0 || y == 0 ||
                              labels[(size_t)y * sw + (x - 1)] != label ||
                              labels[(size_t)(y - 1) * sw + x] != label;
            if (!edge && y > 0 && x > 0) continue;
            sx = x; sy = y;
            break;
        }
    }
    std::vector<Pt> loop;
    if (sx < 0) return loop;
    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    auto fg = [&](int x, int y) {
        return x >= 0 && y >= 0 && x < sw && y < sh &&
               labels[(size_t)y * sw + x] == label;
    };
    int x = sx, y = sy, dir = 6;
    const int limit = std::max(64, sw * sh);
    for (int step = 0; step < limit; ++step) {
        loop.push_back(Pt{(float)x, (float)y});
        bool moved = false;
        int used = dir;
        for (int k = 0; k < 8; ++k) {
            used = (dir + k) % 8;
            const int nx = x + dx[used], ny = y + dy[used];
            if (!fg(nx, ny)) continue;
            if (nx == sx && ny == sy && loop.size() > 8) return loop;
            dir = (used + 5) % 8;
            x = nx; y = ny;
            moved = true;
            break;
        }
        if (!moved) break;
    }
    return loop;
}

float perpDist(Pt p, Pt a, Pt b) {
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-3f) return dist(p, a);
    return std::fabs((p.x - a.x) * dy - (p.y - a.y) * dx) / len;
}

void douglasKeep(const std::vector<Pt>& pts, int i, int j, float eps, std::vector<char>& keep) {
    float best = 0;
    int at = -1;
    for (int t = i + 1; t < j; ++t) {
        const float d = perpDist(pts[t], pts[i], pts[j]);
        if (d > best) { best = d; at = t; }
    }
    if (at >= 0 && best > eps) {
        keep[at] = 1;
        douglasKeep(pts, i, at, eps, keep);
        douglasKeep(pts, at, j, eps, keep);
    }
}

std::vector<Pt> simplifyClosed(const std::vector<Pt>& loop, float eps) {
    const int n = (int)loop.size();
    if (n < 4) return loop;
    int far = 1;
    float best = 0;
    for (int i = 1; i < n; ++i) {
        const float d = dist(loop[i], loop[0]);
        if (d > best) { best = d; far = i; }
    }
    if (far < 1 || far >= n - 1) return {loop[0], loop[n / 3], loop[(2 * n) / 3]};
    std::vector<char> keep((size_t)n, 0);
    keep[0] = 1;
    keep[far] = 1;
    douglasKeep(loop, 0, far, eps, keep);
    std::vector<Pt> back;
    back.reserve((size_t)n - far + 1);
    for (int i = far; i < n; ++i) back.push_back(loop[i]);
    back.push_back(loop[0]);
    std::vector<char> keepB(back.size(), 0);
    keepB.front() = 1;
    keepB.back() = 1;
    douglasKeep(back, 0, (int)back.size() - 1, eps, keepB);
    std::vector<Pt> out;
    for (int i = 0; i <= far; ++i)
        if (keep[i]) out.push_back(loop[i]);
    for (int i = 1; i < (int)back.size() - 1; ++i)
        if (keepB[i]) out.push_back(back[i]);
    return out;
}

float turnFromStraight(Pt a, Pt b, Pt c) {
    const float ax = a.x - b.x, ay = a.y - b.y;
    const float cx = c.x - b.x, cy = c.y - b.y;
    const float la = std::sqrt(ax * ax + ay * ay);
    const float lc = std::sqrt(cx * cx + cy * cy);
    if (la < 1e-3f || lc < 1e-3f) return 0;
    float d = (ax * cx + ay * cy) / (la * lc);
    d = std::clamp(d, -1.0f, 1.0f);
    const float interior = std::acos(d) * (180.0f / 3.14159265f);
    return std::fabs(180.0f - interior);
}

bool significantReflex(const std::vector<Pt>& poly, float minBite) {
    const int n = (int)poly.size();
    if (n < 4) return false;
    double area2 = 0;
    for (int i = 0; i < n; ++i) {
        const Pt q = poly[(i + 1) % n];
        area2 += (double)poly[i].x * q.y - (double)q.x * poly[i].y;
    }
    const float sign = area2 >= 0 ? 1.0f : -1.0f;
    for (int i = 0; i < n; ++i) {
        const Pt& o = poly[i];
        const Pt& a = poly[(i + 1) % n];
        const Pt& b = poly[(i + 2) % n];
        const float c = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        if (c * sign < 0 && perpDist(a, o, b) > minBite &&
            turnFromStraight(o, a, b) > 50.0f)
            return true;
    }
    return false;
}

bool insideConvex(float x, float y, const Pt p[4]) {
    int pos = 0, neg = 0;
    for (int i = 0; i < 4; ++i) {
        const Pt& a = p[i];
        const Pt& b = p[(i + 1) % 4];
        const float c = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
        if (c > 0.5f) ++pos;
        else if (c < -0.5f) ++neg;
    }
    return pos == 0 || neg == 0;
}

// Fraction of the mask that sits inside the quad, and fraction of the quad
// that is mask. Contour inliers treat a thin wood spike as equal to the page
// edge, because a spike is all boundary. Area does not.
void maskQuadOverlap(const std::vector<int>& labels, int sw, int sh, int label,
                     const Pt quad[4], float& maskInside, float& quadFilled) {
    int maskN = 0, inside = 0, quadN = 0, filled = 0;
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            const bool inn = insideConvex((float)x + 0.5f, (float)y + 0.5f, quad);
            const bool fg = labels[(size_t)y * sw + x] == label;
            if (fg) {
                ++maskN;
                if (inn) ++inside;
            }
            if (inn) {
                ++quadN;
                if (fg) ++filled;
            }
        }
    }
    maskInside = maskN ? (float)inside / (float)maskN : 0;
    quadFilled = quadN ? (float)filled / (float)quadN : 0;
}

float chainSagitta(const std::vector<Pt>& c) {
    if (c.size() < 3) return 0;
    float m = 0;
    for (size_t i = 1; i + 1 < c.size(); ++i)
        m = std::max(m, perpDist(c[i], c.front(), c.back()));
    return m;
}

float edgeInlierFraction(const std::vector<Pt>& contour, const Pt quad[4], float tol) {
    if (contour.empty()) return 0;
    int hit = 0;
    for (const Pt& p : contour) {
        float best = 1e9f;
        for (int i = 0; i < 4; ++i) {
            const Pt& a = quad[i];
            const Pt& b = quad[(i + 1) % 4];
            const float dx = b.x - a.x, dy = b.y - a.y;
            const float len2 = dx * dx + dy * dy;
            float t = 0;
            if (len2 > 1e-6f)
                t = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / len2, 0.0f, 1.0f);
            const float px = a.x + t * dx, py = a.y + t * dy;
            const float ddx = p.x - px, ddy = p.y - py;
            best = std::min(best, std::sqrt(ddx * ddx + ddy * ddy));
        }
        if (best <= tol) ++hit;
    }
    return (float)hit / (float)contour.size();
}

// Top and bottom (or left and right) of the mask itself. A bowed sheet is a
// graph over that axis; the convex hull would cut the arc off.
void maskBoundaries(const std::vector<int>& labels, int sw, int sh, int label,
                    bool vertical, std::vector<Pt>& a, std::vector<Pt>& b) {
    a.clear();
    b.clear();
    if (!vertical) {
        for (int x = 0; x < sw; ++x) {
            int y0 = -1, y1 = -1;
            for (int y = 0; y < sh; ++y) {
                if (labels[(size_t)y * sw + x] != label) continue;
                if (y0 < 0) y0 = y;
                y1 = y;
            }
            if (y0 >= 0) {
                a.push_back(Pt{(float)x, (float)y0});
                b.push_back(Pt{(float)x, (float)y1});
            }
        }
    } else {
        for (int y = 0; y < sh; ++y) {
            int x0 = -1, x1 = -1;
            for (int x = 0; x < sw; ++x) {
                if (labels[(size_t)y * sw + x] != label) continue;
                if (x0 < 0) x0 = x;
                x1 = x;
            }
            if (x0 >= 0) {
                a.push_back(Pt{(float)x0, (float)y});
                b.push_back(Pt{(float)x1, (float)y});
            }
        }
    }
}

float boundarySagitta(const std::vector<Pt>& a) {
    return chainSagitta(a);
}

// A bowed band is one solid run along the curl and two runs across it.
int splitRuns(const std::vector<int>& labels, int sw, int sh, int label, bool rows) {
    int splits = 0;
    const int outer = rows ? sh : sw;
    const int inner = rows ? sw : sh;
    for (int o = 0; o < outer; ++o) {
        int runs = 0;
        bool on = false;
        for (int i = 0; i < inner; ++i) {
            const int x = rows ? i : o;
            const int y = rows ? o : i;
            const bool fg = labels[(size_t)y * sw + x] == label;
            if (fg && !on) ++runs;
            on = fg;
        }
        if (runs > 1) ++splits;
    }
    return splits;
}

Pt toSource(Pt p, float sx, float sy) {
    return {(p.x + 0.5f) * sx - 0.5f, (p.y + 0.5f) * sy - 0.5f};
}

PagePoint makePoint(PagePointKind kind, Pt s, float hx = 0, float hy = 0) {
    PagePoint p;
    p.kind = kind;
    p.x = s.x;
    p.y = s.y;
    p.hx = hx;
    p.hy = hy;
    return p;
}

void storePoly(DocumentOutline& o, const std::vector<Pt>& poly, float sx, float sy) {
    o.x.clear();
    o.y.clear();
    o.points.clear();
    double area2 = 0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Pt q = poly[(i + 1) % poly.size()];
        area2 += (double)poly[i].x * q.y - (double)q.x * poly[i].y;
    }
    const float sign = area2 >= 0 ? 1.0f : -1.0f;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Pt s = toSource(poly[i], sx, sy);
        const Pt& prev = poly[(i + poly.size() - 1) % poly.size()];
        const Pt& next = poly[(i + 1) % poly.size()];
        const float c = (poly[i].x - prev.x) * (next.y - prev.y) -
                        (poly[i].y - prev.y) * (next.x - prev.x);
        const bool reflex = poly.size() >= 4 && c * sign < 0 &&
                            turnFromStraight(prev, poly[i], next) > 50.0f &&
                            perpDist(poly[i], prev, next) > 2.5f;
        o.points.push_back(makePoint(reflex ? PagePointKind::Fold : PagePointKind::Corner, s));
        o.x.push_back(s.x);
        o.y.push_back(s.y);
    }
}

void storeCurve(DocumentOutline& o, const std::vector<Pt>& a, const std::vector<Pt>& b,
                float sx, float sy) {
    auto put = [&](const std::vector<Pt>& c, std::vector<float>& xs, std::vector<float>& ys) {
        const int step = std::max(1, (int)c.size() / 180);
        for (size_t i = 0; i < c.size(); i += (size_t)step) {
            const Pt s = toSource(c[i], sx, sy);
            xs.push_back(s.x);
            ys.push_back(s.y);
        }
        if (!c.empty()) {
            const Pt s = toSource(c.back(), sx, sy);
            if (xs.empty() || std::fabs(xs.back() - s.x) > 0.1f || std::fabs(ys.back() - s.y) > 0.1f) {
                xs.push_back(s.x);
                ys.push_back(s.y);
            }
        }
    };
    put(a, o.ax, o.ay);
    put(b, o.bx, o.by);
}

Pt bezAt(Pt p0, Pt p1, Pt p2, Pt p3, float t) {
    const float u = 1.0f - t;
    const float uu = u * u, tt = t * t;
    return {uu * u * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + tt * t * p3.x,
            uu * u * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + tt * t * p3.y};
}

// Symmetric handle at `knot`. The arm points along `tan` (unit, toward the
// next corner) with length L. The incoming arm is the opposite.
float cubicChainError(const std::vector<Pt>& chain, int at, Pt tan, float L) {
    if (chain.size() < 3 || at <= 0 || at >= (int)chain.size() - 1) return 1e9f;
    const Pt a = chain.front();
    const Pt m = chain[at];
    const Pt b = chain.back();
    const Pt h{tan.x * L, tan.y * L};
    const Pt in{m.x - h.x, m.y - h.y};
    const Pt out{m.x + h.x, m.y + h.y};
    float err = 0;
    int n = 0;
    for (int i = 1; i < (int)chain.size() - 1; ++i) {
        const bool left = i <= at;
        const float t = left ? (float)i / (float)at
                             : (float)(i - at) / (float)((int)chain.size() - 1 - at);
        const Pt q = left ? bezAt(a, a, in, m, t) : bezAt(m, out, b, b, t);
        err = std::max(err, dist(q, chain[i]));
        ++n;
    }
    return n ? err : 1e9f;
}

// One knot on `chain`, between the ends. A sharp kink is a fold. A bow keeps
// a handle even when a short arm fits best — a zero arm would become a hinge
// and unfold a page that is only curled.
void fitOnce(const std::vector<Pt>& chain, float tol, float vx, float vy,
             std::vector<PagePoint>& into) {
    if (chain.size() < 4) return;
    float sag = 0;
    int at = -1;
    for (int i = 1; i < (int)chain.size() - 1; ++i) {
        const float d = perpDist(chain[i], chain.front(), chain.back());
        if (d > sag) { sag = d; at = i; }
    }
    if (at < 0 || sag <= tol) return;
    const int w = std::max(1, (int)chain.size() / 12);
    const Pt t0 = chain[std::max(0, at - w)];
    const Pt t1 = chain[std::min((int)chain.size() - 1, at + w)];
    Pt tan{t1.x - t0.x, t1.y - t0.y};
    const float tl = std::sqrt(tan.x * tan.x + tan.y * tan.y);
    if (tl < 1e-3f) return;
    tan.x /= tl; tan.y /= tl;
    const Pt forward{chain.back().x - chain[at].x, chain.back().y - chain[at].y};
    if (tan.x * forward.x + tan.y * forward.y < 0) { tan.x = -tan.x; tan.y = -tan.y; }
    const float turn = turnFromStraight(t0, chain[at], t1);
    const float maxL = 0.45f * dist(chain.front(), chain.back());
    float bestL = 0, bestErr = 1e9f;
    if (turn <= 55.0f) {
        for (int k = 0; k <= 8; ++k) {
            const float L = maxL * (float)k / 8.0f;
            const float err = cubicChainError(chain, at, tan, L);
            if (err < bestErr) { bestErr = err; bestL = L; }
        }
        if (bestL < 1.5f) bestL = std::min(maxL, std::max(sag, 4.0f));
    }
    const bool fold = turn > 55.0f;
    const Pt s = toSource(chain[at], vx, vy);
    into.push_back(makePoint(fold ? PagePointKind::Fold : PagePointKind::Curve, s,
                             fold ? 0 : tan.x * bestL * vx,
                             fold ? 0 : tan.y * bestL * vy));
}

// The farthest knot, then at most one more on the half that still bows.
void fitSide(const std::vector<Pt>& chain, float tol, float vx, float vy,
             std::vector<PagePoint>& into) {
    if (chain.size() < 5) return;
    float sag = 0;
    int at = -1;
    for (int i = 1; i < (int)chain.size() - 1; ++i) {
        const float d = perpDist(chain[i], chain.front(), chain.back());
        if (d > sag) { sag = d; at = i; }
    }
    if (at < 0 || sag <= tol) return;
    std::vector<Pt> left(chain.begin(), chain.begin() + at + 1);
    std::vector<Pt> right(chain.begin() + at, chain.end());
    float ls = 0, rs = 0;
    for (int i = 1; i < (int)left.size() - 1; ++i)
        ls = std::max(ls, perpDist(left[i], left.front(), left.back()));
    for (int i = 1; i < (int)right.size() - 1; ++i)
        rs = std::max(rs, perpDist(right[i], right.front(), right.back()));
    if (ls >= rs && ls > tol) fitOnce(left, tol, vx, vy, into);
    fitOnce(chain, tol, vx, vy, into);
    if (rs > ls && rs > tol) fitOnce(right, tol, vx, vy, into);
}

bool pointInPoly(float x, float y, const std::vector<float>& px, const std::vector<float>& py) {
    bool in = false;
    const int n = (int)px.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const float yi = py[i], yj = py[j];
        const bool cross = (yi > y) != (yj > y);
        if (!cross) continue;
        const float xint = (px[j] - px[i]) * (y - yi) / (yj - yi + 1e-12f) + px[i];
        if (x < xint) in = !in;
    }
    return in;
}

struct ArcChain {
    std::vector<Pt> p;
    std::vector<float> acc;
    float total = 0;
};

ArcChain makeArc(const std::vector<float>& xs, const std::vector<float>& ys) {
    ArcChain c;
    const int n = (int)std::min(xs.size(), ys.size());
    c.p.resize((size_t)n);
    c.acc.resize((size_t)n, 0);
    for (int i = 0; i < n; ++i) c.p[i] = Pt{xs[i], ys[i]};
    for (int i = 1; i < n; ++i) {
        c.acc[i] = c.acc[i - 1] + dist(c.p[i - 1], c.p[i]);
    }
    c.total = n ? c.acc.back() : 0;
    return c;
}

Pt arcAt(const ArcChain& c, float t) {
    if (c.p.empty()) return {};
    if (c.p.size() == 1 || c.total < 1e-3f) return c.p.front();
    t = std::clamp(t, 0.0f, 1.0f);
    const float target = t * c.total;
    auto it = std::lower_bound(c.acc.begin(), c.acc.end(), target);
    int i = (int)std::distance(c.acc.begin(), it);
    if (i <= 0) return c.p.front();
    if (i >= (int)c.p.size()) return c.p.back();
    const float span = c.acc[i] - c.acc[i - 1];
    const float u = span > 1e-6f ? (target - c.acc[i - 1]) / span : 0;
    return {c.p[i - 1].x + (c.p[i].x - c.p[i - 1].x) * u,
            c.p[i - 1].y + (c.p[i].y - c.p[i - 1].y) * u};
}

// A photographed form is a grid of long ink. The table next to this one is
// as bright as the shaded face of the sheet, so a luminance threshold takes
// the wood and drops the page. A straight run of ink does not: grain is
// short, and the rulings stay on the paper through the fold.
bool maskFromRulings(const std::vector<float>& gray, int sw, int sh,
                     std::vector<uint8_t>& mask) {
    mask.assign((size_t)sw * sh, 0);
    const int rad = std::max(3, std::min(sw, sh) / 70);
    std::vector<float> mean;
    boxBlur(gray, mean, sw, sh, rad);
    // Ink sits on the paper: darker than its neighbourhood, with paper on
    // both sides. The drop off the edge of the sheet has paper on one side
    // only, and that halo must not become a ruling.
    const int side = std::max(2, rad);
    std::vector<uint8_t> ink((size_t)sw * sh, 0);
    int inkN = 0;
    for (int y = side; y < sh - side; ++y) {
        for (int x = side; x < sw - side; ++x) {
            const int i = y * sw + x;
            const float m = mean[i];
            const float g = gray[i];
            if (m < 1e-4f) continue;
            if (!(g < m * 0.78f && (m - g) > 0.10f * m)) continue;
            const float up = gray[(size_t)(y - side) * sw + x];
            const float dn = gray[(size_t)(y + side) * sw + x];
            const float lf = gray[(size_t)y * sw + (x - side)];
            const float rt = gray[(size_t)y * sw + (x + side)];
            const float need = std::max(g * 1.35f, g + 0.04f * m);
            const bool across = up > need && dn > need;
            const bool along = lf > need && rt > need;
            if (!across && !along) continue;
            ink[i] = 1;
            ++inkN;
        }
    }
    if (inkN < std::min(sw, sh)) return false;

    const int minH = std::max(12, sw / 28);
    const int minV = std::max(10, sh / 32);
    std::vector<uint8_t> lines((size_t)sw * sh, 0);
    int hRuns = 0, vRuns = 0;
    for (int y = 0; y < sh; ++y) {
        int x = 0;
        while (x < sw) {
            if (!ink[(size_t)y * sw + x]) { ++x; continue; }
            const int x0 = x;
            while (x < sw && ink[(size_t)y * sw + x]) ++x;
            if (x - x0 < minH) continue;
            ++hRuns;
            for (int xx = x0; xx < x; ++xx) lines[(size_t)y * sw + xx] = 1;
        }
    }
    for (int x = 0; x < sw; ++x) {
        int y = 0;
        while (y < sh) {
            if (!ink[(size_t)y * sw + x]) { ++y; continue; }
            const int y0 = y;
            while (y < sh && ink[(size_t)y * sw + x]) ++y;
            if (y - y0 < minV) continue;
            ++vRuns;
            for (int yy = y0; yy < y; ++yy) lines[(size_t)yy * sw + x] = 1;
        }
    }
    if (hRuns < 5 || vRuns < 3) return false;
    int minX = sw, maxX = 0, minY = sh, maxY = 0, lineN = 0;
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            if (!lines[(size_t)y * sw + x]) continue;
            ++lineN;
            minX = std::min(minX, x); maxX = std::max(maxX, x);
            minY = std::min(minY, y); maxY = std::max(maxY, y);
        }
    }
    if (lineN < 40) return false;
    if (maxX - minX < sw * 3 / 10 || maxY - minY < sh * 3 / 10) return false;

    // The outer rulings are the four sides. They do not have to connect
    // across a washed-out fold; each side is the strong line nearest the
    // outside of the grid.
    std::vector<int> col((size_t)sw, 0), row((size_t)sh, 0);
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            if (!lines[(size_t)y * sw + x]) continue;
            ++col[x];
            ++row[y];
        }
    }
    const int maxC = *std::max_element(col.begin(), col.end());
    const int maxR = *std::max_element(row.begin(), row.end());
    const int colCut = std::max(minV, (maxC * 2) / 5);
    const int rowCut = std::max(minH / 2, (maxR * 2) / 5);
    int left = -1, right = -1, top = -1, bot = -1;
    for (int x = 0; x < sw; ++x) if (col[x] >= colCut) { left = x; break; }
    for (int x = sw - 1; x >= 0; --x) if (col[x] >= colCut) { right = x; break; }
    for (int y = 0; y < sh; ++y) if (row[y] >= rowCut) { top = y; break; }
    for (int y = sh - 1; y >= 0; --y) if (row[y] >= rowCut) { bot = y; break; }
    if (left < 0 || right - left < sw / 5 || bot - top < sh / 5) return false;

    auto fitBand = [&](bool vertical, int at) {
        std::vector<Pt> pts;
        const int lo = std::max(0, at - 3);
        const int hi = std::min((vertical ? sw : sh) - 1, at + 4);
        if (vertical) {
            for (int x = lo; x <= hi; ++x)
                for (int y = top; y <= bot; ++y)
                    if (lines[(size_t)y * sw + x])
                        pts.push_back(Pt{(float)x, (float)y});
        } else {
            for (int y = lo; y <= hi; ++y)
                for (int x = left; x <= right; ++x)
                    if (lines[(size_t)y * sw + x])
                        pts.push_back(Pt{(float)x, (float)y});
        }
        Line L = fitLine(pts);
        if (L.ok) return L;
        if (vertical)
            return lineThrough(Pt{(float)at, (float)top}, Pt{(float)at, (float)bot});
        return lineThrough(Pt{(float)left, (float)at}, Pt{(float)right, (float)at});
    };
    const Line sideL[4] = {fitBand(false, top), fitBand(true, right),
                           fitBand(false, bot), fitBand(true, left)};
    Pt quad[4];
    for (int k = 0; k < 4; ++k)
        if (!intersectLines(sideL[k], sideL[(k + 1) % 4], quad[k])) return false;
    float cx = 0, cy = 0;
    for (int k = 0; k < 4; ++k) { cx += quad[k].x; cy += quad[k].y; }
    cx *= 0.25f; cy *= 0.25f;
    const float growPx = (float)std::max(4, std::min(sw, sh) / 70);
    for (int k = 0; k < 4; ++k) {
        float dx = quad[k].x - cx, dy = quad[k].y - cy;
        const float L = std::sqrt(dx * dx + dy * dy);
        if (L < 1.0f) continue;
        quad[k].x = std::clamp(quad[k].x + dx / L * growPx, 1.0f, (float)sw - 2);
        quad[k].y = std::clamp(quad[k].y + dy / L * growPx, 1.0f, (float)sh - 2);
    }
    orderCorners(quad);
    if (!convexQuad(quad) || quadArea(quad) < 0.08f * (float)sw * (float)sh) return false;

    mask.assign((size_t)sw * sh, 0);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x)
            if (insideConvex((float)x + 0.5f, (float)y + 0.5f, quad))
                mask[(size_t)y * sw + x] = 1;
    return true;
}

// One crease per direction: the strongest step that runs across the sheet.
// Printed rules are thinner than the blur, so they do not win.
bool creaseAt(const std::vector<float>& smooth, int sw, int sh,
              const Pt quad[4], bool horizontal, float& at) {
    float x0 = quad[0].x, x1 = quad[0].x, y0 = quad[0].y, y1 = quad[0].y;
    for (int i = 1; i < 4; ++i) {
        x0 = std::min(x0, quad[i].x); x1 = std::max(x1, quad[i].x);
        y0 = std::min(y0, quad[i].y); y1 = std::max(y1, quad[i].y);
    }
    const int xa = std::clamp((int)std::lround(x0), 1, sw - 2);
    const int xb = std::clamp((int)std::lround(x1), 1, sw - 2);
    const int ya = std::clamp((int)std::lround(y0), 1, sh - 2);
    const int yb = std::clamp((int)std::lround(y1), 1, sh - 2);
    if (xb - xa < 16 || yb - ya < 16) return false;

    const int n = horizontal ? sh : sw;
    std::vector<double> acc((size_t)n, 0);
    std::vector<int> cnt((size_t)n, 0);
    if (horizontal) {
        for (int y = ya + 1; y < yb; ++y) {
            for (int x = xa; x <= xb; ++x) {
                if (!insideConvex((float)x + 0.5f, (float)y + 0.5f, quad)) continue;
                const float gy = smooth[(size_t)(y + 1) * sw + x] -
                                 smooth[(size_t)(y - 1) * sw + x];
                acc[y] += gy;
                ++cnt[y];
            }
        }
    } else {
        for (int x = xa + 1; x < xb; ++x) {
            for (int y = ya; y <= yb; ++y) {
                if (!insideConvex((float)x + 0.5f, (float)y + 0.5f, quad)) continue;
                const float gx = smooth[(size_t)y * sw + (x + 1)] -
                                 smooth[(size_t)y * sw + (x - 1)];
                acc[x] += gx;
                ++cnt[x];
            }
        }
    }
    std::vector<float> mags;
    mags.reserve((size_t)n);
    int peak = -1;
    float peakMag = 0;
    const int i0 = horizontal ? ya + 2 : xa + 2;
    const int i1 = horizontal ? yb - 2 : xb - 2;
    // Stay off the border. The page edge itself is a step, and it is not a fold.
    const int inset = horizontal ? std::max(6, (yb - ya) / 8)
                                 : std::max(6, (xb - xa) / 8);
    for (int i = i0; i <= i1; ++i) {
        if (cnt[i] < 8) continue;
        if (horizontal && (i < ya + inset || i > yb - inset)) continue;
        if (!horizontal && (i < xa + inset || i > xb - inset)) continue;
        const float m = std::fabs((float)(acc[i] / cnt[i]));
        mags.push_back(m);
        if (m > peakMag) { peakMag = m; peak = i; }
    }
    if (peak < 0 || mags.size() < 8) return false;
    const float med = percentile(mags, 0.50);
    if (peakMag < std::max(0.002f, med * 2.2f)) return false;

    const float sign = (acc[peak] >= 0) ? 1.0f : -1.0f;
    int agree = 0, seen = 0;
    if (horizontal) {
        for (int x = xa; x <= xb; ++x) {
            if (!insideConvex((float)x + 0.5f, (float)peak + 0.5f, quad)) continue;
            ++seen;
            const float gy = smooth[(size_t)(peak + 1) * sw + x] -
                             smooth[(size_t)(peak - 1) * sw + x];
            if (gy * sign > peakMag * 0.35f) ++agree;
        }
    } else {
        for (int y = ya; y <= yb; ++y) {
            if (!insideConvex((float)peak + 0.5f, (float)y + 0.5f, quad)) continue;
            ++seen;
            const float gx = smooth[(size_t)y * sw + (peak + 1)] -
                             smooth[(size_t)y * sw + (peak - 1)];
            if (gx * sign > peakMag * 0.35f) ++agree;
        }
    }
    if (seen < 12 || agree < (seen * 55) / 100) return false;
    at = (float)peak;
    return true;
}

// The mask edge between two corners. `minSide` takes the lesser coordinate
// (top or left of the sheet).
std::vector<Pt> spanEdge(const std::vector<int>& labels, int sw, int sh, int label,
                         Pt a, Pt b, bool horizontal, bool minSide) {
    std::vector<Pt> chain;
    if (horizontal) {
        int x0 = (int)std::lround(std::min(a.x, b.x));
        int x1 = (int)std::lround(std::max(a.x, b.x));
        x0 = std::clamp(x0, 0, sw - 1);
        x1 = std::clamp(x1, 0, sw - 1);
        for (int x = x0; x <= x1; ++x) {
            int yHit = -1;
            if (minSide) {
                for (int y = 0; y < sh; ++y)
                    if (labels[(size_t)y * sw + x] == label) { yHit = y; break; }
            } else {
                for (int y = sh - 1; y >= 0; --y)
                    if (labels[(size_t)y * sw + x] == label) { yHit = y; break; }
            }
            if (yHit >= 0) chain.push_back(Pt{(float)x, (float)yHit});
        }
        if (a.x > b.x) std::reverse(chain.begin(), chain.end());
    } else {
        int y0 = (int)std::lround(std::min(a.y, b.y));
        int y1 = (int)std::lround(std::max(a.y, b.y));
        y0 = std::clamp(y0, 0, sh - 1);
        y1 = std::clamp(y1, 0, sh - 1);
        for (int y = y0; y <= y1; ++y) {
            int xHit = -1;
            if (minSide) {
                for (int x = 0; x < sw; ++x)
                    if (labels[(size_t)y * sw + x] == label) { xHit = x; break; }
            } else {
                for (int x = sw - 1; x >= 0; --x)
                    if (labels[(size_t)y * sw + x] == label) { xHit = x; break; }
            }
            if (xHit >= 0) chain.push_back(Pt{(float)xHit, (float)y});
        }
        if (a.y > b.y) std::reverse(chain.begin(), chain.end());
    }
    return chain;
}

float srgbEncode(float c) {
    c = std::max(0.0f, c);
    if (c <= 0.0031308f) return 12.92f * c;
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

float lumaNearest(const float* linear, int w, int h, float x, float y) {
    const int ix = std::clamp((int)std::lround(x), 0, w - 1);
    const int iy = std::clamp((int)std::lround(y), 0, h - 1);
    const float* p = linear + ((size_t)iy * w + ix) * 4;
    return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
}

// The network's corner is within a few percent of the edge. Walk in from the
// table and stop at the first step, which is the paper boundary. A printed
// rule further in is a stronger step and must not win.
bool refinePageCorners(const float* linear, uint32_t w, uint32_t h, Pt q[4]) {
    const Pt model[4] = {q[0], q[1], q[2], q[3]};
    float cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) { cx += q[i].x; cy += q[i].y; }
    cx *= 0.25f; cy *= 0.25f;
    float span = 0;
    for (int i = 0; i < 4; ++i)
        span = std::max(span, dist(q[i], q[(i + 1) % 4]));
    if (span < 16.0f) return false;
    const int nOff = std::max(4, (int)std::lround(span * 0.04f));
    Line sides[4];
    for (int i = 0; i < 4; ++i) {
        const Pt a = q[i], b = q[(i + 1) % 4];
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len < 8.0f) return false;
        float nx = dy / len, ny = -dx / len;
        const float mx = (a.x + b.x) * 0.5f, my = (a.y + b.y) * 0.5f;
        if ((mx - cx) * nx + (my - cy) * ny < 0) { nx = -nx; ny = -ny; }
        std::vector<float> prof((size_t)nOff * 2 + 1, 0);
        constexpr int kSamp = 16;
        for (int s = 1; s <= kSamp; ++s) {
            const float t = (float)s / (float)(kSamp + 1);
            const float px = a.x + dx * t, py = a.y + dy * t;
            for (int k = -nOff; k <= nOff; ++k)
                prof[(size_t)(k + nOff)] +=
                    lumaNearest(linear, (int)w, (int)h, px + nx * k, py + ny * k);
        }
        float maxG = 0;
        for (int k = 1; k <= nOff * 2; ++k)
            maxG = std::max(maxG, std::fabs(prof[(size_t)k] - prof[(size_t)k - 1]));
        float shift = 0;
        if (maxG > 1e-3f) {
            const float thr = maxG * 0.40f;
            int at = nOff;
            for (int k = nOff * 2; k >= 1; --k) {
                if (std::fabs(prof[(size_t)k] - prof[(size_t)k - 1]) >= thr) {
                    at = k;
                    break;
                }
            }
            shift = (at - 0.5f) - (float)nOff;
        }
        const Pt a2{a.x + nx * shift, a.y + ny * shift};
        const Pt b2{b.x + nx * shift, b.y + ny * shift};
        sides[i] = lineThrough(a2, b2);
        if (!sides[i].ok) return false;
    }
    Pt refined[4];
    for (int i = 0; i < 4; ++i) {
        if (!intersectLines(sides[(i + 3) % 4], sides[i], refined[i])) return false;
    }
    float jump = 0;
    for (int i = 0; i < 4; ++i)
        jump = std::max(jump, dist(refined[i], model[i]));
    if (jump > span * 0.08f) return true;
    if (!convexQuad(refined)) return true;
    for (int i = 0; i < 4; ++i) q[i] = refined[i];
    return true;
}

// The corner model often frames the sheet with a band of table still inside,
// or lands on the printed border and drops the margin past it. A side moves
// to where the sheet's own colour gives way to the table. Ink is dark, so
// the vote uses the bright part of a small patch: a rule or a line of text
// still counts as the sheet, and the words in the margin stay inside.
// Grey paper on grey wood shares whiteness, and that side is left alone.
void tightenPaperMargins(const float* linear, uint32_t w, uint32_t h, Pt q[4]) {
    float cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) { cx += q[i].x; cy += q[i].y; }
    cx *= 0.25f; cy *= 0.25f;
    auto bilerp = [&](float u, float v) {
        const float a = (1.0f - u) * (1.0f - v);
        const float b = u * (1.0f - v);
        const float c = u * v;
        const float d = (1.0f - u) * v;
        return Pt{a * q[0].x + b * q[1].x + c * q[2].x + d * q[3].x,
                  a * q[0].y + b * q[1].y + c * q[2].y + d * q[3].y};
    };
    auto take = [&](float x, float y, float& luma, float& white) {
        const int ix = (int)std::lround(x);
        const int iy = (int)std::lround(y);
        if (ix < 0 || iy < 0 || ix >= (int)w || iy >= (int)h) return false;
        const float* p = linear + ((size_t)iy * w + ix) * 4;
        const float r = std::max(0.0f, p[0]);
        const float g = std::max(0.0f, p[1]);
        const float b = std::max(0.0f, p[2]);
        const float mx = std::max(r, std::max(g, b));
        const float mn = std::min(r, std::min(g, b));
        luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        white = mx > 1e-4f ? mn / mx : 0.0f;
        return true;
    };
    std::vector<float> lumas, whites;
    for (int j = 0; j < 7; ++j) {
        for (int i = 0; i < 7; ++i) {
            const Pt p = bilerp(0.32f + 0.06f * (float)i, 0.32f + 0.06f * (float)j);
            float luma = 0, white = 0;
            if (!take(p.x, p.y, luma, white)) continue;
            lumas.push_back(luma);
            whites.push_back(white);
        }
    }
    if (lumas.size() < 16) return;
    const float refLuma = percentile(lumas, 0.75);
    const float refWhite = percentile(whites, 0.75);
    if (refLuma < 1e-4f) return;
    // 0 off the image, 1 table, 2 paper. Off the image is not table: a page
    // that already fills the frame must not be pulled in. The bright quarter
    // of the patch is the sheet showing between strokes; a sample that lands
    // on the ink itself must not turn the margin into table.
    auto classify = [&](float x, float y, float tx, float ty) {
        float ls[25], ws[25];
        int n = 0;
        for (int j = -2; j <= 2; ++j) {
            for (int i = -2; i <= 2; ++i) {
                float luma = 0, white = 0;
                if (!take(x + tx * (float)i * 7.0f + (-ty) * (float)j * 7.0f,
                          y + ty * (float)i * 7.0f + tx * (float)j * 7.0f,
                          luma, white))
                    continue;
                ls[n] = luma;
                ws[n] = white;
                ++n;
            }
        }
        if (n < 8) return 0;
        for (int a = 1; a < n; ++a) {
            const float key = ls[a];
            const float kw = ws[a];
            int b = a;
            while (b > 0 && ls[b - 1] > key) {
                ls[b] = ls[b - 1];
                ws[b] = ws[b - 1];
                --b;
            }
            ls[b] = key;
            ws[b] = kw;
        }
        const int at = (n * 3) / 4;
        if (ws[at] >= refWhite - 0.11f && ls[at] >= refLuma * 0.38f) return 2;
        return 1;
    };

    float span = 0;
    for (int i = 0; i < 4; ++i)
        span = std::max(span, dist(q[i], q[(i + 1) % 4]));
    Line sides[4];
    for (int i = 0; i < 4; ++i) {
        const Pt a = q[i], b = q[(i + 1) % 4];
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float len = std::sqrt(dx * dx + dy * dy);
        sides[i] = lineThrough(a, b);
        if (len < 8.0f || !sides[i].ok) return;
        float nx = dy / len, ny = -dx / len;
        const float mx = (a.x + b.x) * 0.5f, my = (a.y + b.y) * 0.5f;
        if ((mx - cx) * nx + (my - cy) * ny < 0) { nx = -nx; ny = -ny; }
        const float reach = std::min(len * 0.45f, span * 0.45f);
        // The printed border can sit well inside the sheet. Searching only a
        // few dozen pixels outside the model side stops on that rule and
        // drops the margin (and anything printed on it). The table is
        // further out; look far enough to reach it.
        const float out = std::min(len * 0.16f, 260.0f);
        const float step = std::max(2.0f, len / 400.0f);
        std::vector<float> hits;
        for (int s = 1; s <= 7; ++s) {
            const float t = 0.15f + 0.10f * (float)(s - 1);
            const float px = a.x + dx * t, py = a.y + dy * t;
            const int nOut = (int)std::ceil(out / step);
            const int nIn = (int)std::ceil(reach / step);
            const int n = nOut + nIn + 1;
            std::vector<char> kind((size_t)n);
            for (int k = 0; k < n; ++k) {
                const float off = out - (float)k * step;
                kind[(size_t)k] = (char)classify(px + nx * off, py + ny * off,
                                                dx / len, dy / len);
            }
            // The page edge is the outermost place where the inward window
            // is mostly paper and the outward window is on the table. A
            // printed rule fails the inward window; a bright knot in the
            // cloth fails it too.
            const int win = std::max(6, (int)std::lround(72.0f / step));
            for (int k = win; k + win < n; ++k) {
                int inOn = 0, inP = 0, outOn = 0, outP = 0;
                for (int t = 0; t < win; ++t) {
                    const int inn = kind[(size_t)(k + t)];
                    const int outk = kind[(size_t)(k - 1 - t)];
                    if (inn) { ++inOn; if (inn == 2) ++inP; }
                    if (outk) { ++outOn; if (outk == 2) ++outP; }
                }
                if (inOn * 5 >= win * 3 && inP * 5 >= inOn * 3 &&
                    outOn * 5 >= win * 3 && outP * 5 <= outOn * 2) {
                    int k2 = k;
                    while (k2 + win < n && kind[(size_t)k2] == 1 &&
                           (float)(k2 - k) * step < 64.0f)
                        ++k2;
                    hits.push_back(out - ((float)k2 - 0.5f) * step);
                    break;
                }
            }
        }
        if ((int)hits.size() < 4) continue;
        std::sort(hits.begin(), hits.end());
        const float shift = hits[hits.size() / 2];
        int agree = 0;
        const float tol = std::max(12.0f, len * 0.06f);
        for (float h : hits)
            if (std::fabs(h - shift) <= tol) ++agree;
        if (agree < 4 || std::fabs(shift) < 3.0f) continue;
        // A large inward step that crosses strokes is the printed page, not
        // the table. The sheet can be any colour; the ink is what marks it.
        if (shift < -len * 0.02f) {
            const float tx = dx / len, ty = dy / len;
            int ink = 0, seen = 0;
            for (int s = 1; s <= 5; ++s) {
                const float t = 0.15f * (float)s;
                const float sx = a.x + dx * t, sy = a.y + dy * t;
                for (float off = shift; off < -4.0f; off += 12.0f) {
                    float l0 = 0, w0 = 0, l1 = 0, w1 = 0;
                    if (!take(sx + nx * off, sy + ny * off, l0, w0)) continue;
                    if (!take(sx + nx * off + tx * 8.0f, sy + ny * off + ty * 8.0f, l1, w1))
                        continue;
                    ++seen;
                    const float hi = std::max(l0, l1);
                    const float lo = std::min(l0, l1);
                    if (hi > refLuma * 0.30f && lo < hi * 0.62f) ++ink;
                }
            }
            if (seen >= 8 && ink * 4 >= seen) continue;
        }
        const Pt a2{a.x + nx * shift, a.y + ny * shift};
        const Pt b2{b.x + nx * shift, b.y + ny * shift};
        const Line moved = lineThrough(a2, b2);
        if (moved.ok) sides[i] = moved;
    }
    Pt refined[4];
    for (int i = 0; i < 4; ++i) {
        if (!intersectLines(sides[(i + 3) % 4], sides[i], refined[i])) return;
    }
    if (!convexQuad(refined)) return;
    float jump = 0;
    for (int i = 0; i < 4; ++i)
        jump = std::max(jump, dist(refined[i], q[i]));
    if (jump > span * 0.45f) return;
    const float area0 = quadArea(q);
    const float area1 = quadArea(refined);
    // Growing past 8% used to be rejected outright, which kept a side that
    // had locked onto the printed rule. A real margin around that rule is
    // still the sheet. Shrinking stays capped so a misfire cannot eat the page.
    if (area0 < 1.0f || area1 < area0 * 0.35f || area1 > area0 * 1.40f) return;
    for (int i = 0; i < 4; ++i) q[i] = refined[i];
}

#ifdef VIEWMAGE_WITH_ORT
// DocAligner LC100 heatmap. Apache-2.0, DocsaidLab.
// img [1,3,256,256] BGR in 0..1, stretched (not letterboxed).
// `area` averages each source block. A point sample of a frame whose page
// runs to the edge drops the desk border and the heatmap falls apart; the
// average keeps it. A small ruled form prefers the point sample, so the
// caller tries the area first and the point sample when that peak is weak.
// heatmap [1,4,128,128], channels TL TR BR BL, peaks near 0.9 when it sees a page.
// `confOut` receives the four peak values even when the page is rejected.
bool pageModelCorners(const float* linear, uint32_t w, uint32_t h,
                      const uint8_t* model, size_t modelBytes, bool area,
                      Pt out[4], float confOut[4]) {
    if (!model || modelBytes < 64 || w < 16 || h < 16) return false;
    struct Session {
        Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "viewmage-page"};
        Ort::SessionOptions opts;
        std::unique_ptr<Ort::Session> session;
        const uint8_t* key = nullptr;
        size_t bytes = 0;
        Session() {
            opts.SetIntraOpNumThreads(2);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }
    };
    static Session held;
    try {
        if (!held.session || held.key != model || held.bytes != modelBytes) {
            held.session.reset(new Ort::Session(held.env, model, modelBytes, held.opts));
            held.key = model;
            held.bytes = modelBytes;
        }
        std::vector<float> input(1 * 3 * 256 * 256, 0.0f);
        if (area) {
            for (int y = 0; y < 256; ++y) {
                const int y0 = (int)((int64_t)y * h / 256);
                int y1 = (int)((int64_t)(y + 1) * h / 256);
                if (y1 <= y0) y1 = std::min(y0 + 1, (int)h);
                for (int x = 0; x < 256; ++x) {
                    const int x0 = (int)((int64_t)x * w / 256);
                    int x1 = (int)((int64_t)(x + 1) * w / 256);
                    if (x1 <= x0) x1 = std::min(x0 + 1, (int)w);
                    double acc[3] = {};
                    int n = 0;
                    for (int py = y0; py < y1; ++py) {
                        const float* row = linear + (size_t)py * w * 4;
                        for (int px = x0; px < x1; ++px) {
                            const float* p = row + (size_t)px * 4;
                            acc[0] += srgbEncode(p[0]);
                            acc[1] += srgbEncode(p[1]);
                            acc[2] += srgbEncode(p[2]);
                            ++n;
                        }
                    }
                    if (!n) continue;
                    const size_t i = (size_t)y * 256 + x;
                    input[0 * 256 * 256 + i] = (float)(acc[2] / n);
                    input[1 * 256 * 256 + i] = (float)(acc[1] / n);
                    input[2 * 256 * 256 + i] = (float)(acc[0] / n);
                }
            }
        } else {
            for (int y = 0; y < 256; ++y) {
                const float sy = ((float)y + 0.5f) * (float)h / 256.0f - 0.5f;
                const int y0 = std::clamp((int)std::floor(sy), 0, (int)h - 1);
                const int y1 = std::min(y0 + 1, (int)h - 1);
                const float fy = std::clamp(sy - (float)y0, 0.0f, 1.0f);
                for (int x = 0; x < 256; ++x) {
                    const float sx = ((float)x + 0.5f) * (float)w / 256.0f - 0.5f;
                    const int x0 = std::clamp((int)std::floor(sx), 0, (int)w - 1);
                    const int x1 = std::min(x0 + 1, (int)w - 1);
                    const float fx = std::clamp(sx - (float)x0, 0.0f, 1.0f);
                    float rgb[3] = {};
                    for (int c = 0; c < 3; ++c) {
                        const float* p00 = linear + ((size_t)y0 * w + x0) * 4;
                        const float* p10 = linear + ((size_t)y0 * w + x1) * 4;
                        const float* p01 = linear + ((size_t)y1 * w + x0) * 4;
                        const float* p11 = linear + ((size_t)y1 * w + x1) * 4;
                        const float v = p00[c] * (1 - fx) * (1 - fy) + p10[c] * fx * (1 - fy) +
                                        p01[c] * (1 - fx) * fy + p11[c] * fx * fy;
                        rgb[c] = srgbEncode(v);
                    }
                    const size_t px = (size_t)y * 256 + x;
                    input[0 * 256 * 256 + px] = rgb[2];
                    input[1 * 256 * 256 + px] = rgb[1];
                    input[2 * 256 * 256 + px] = rgb[0];
                }
            }
        }
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t shape[4] = {1, 3, 256, 256};
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape, 4);
        const char* inName = "img";
        const char* outName = "heatmap";
        auto res = held.session->Run(Ort::RunOptions{nullptr}, &inName, &in, 1, &outName, 1);
        auto info = res[0].GetTensorTypeAndShapeInfo();
        const auto dims = info.GetShape();
        if (dims.size() != 4 || dims[0] != 1 || dims[1] != 4 || dims[2] < 8 || dims[3] < 8)
            return false;
        const int hmH = (int)dims[2], hmW = (int)dims[3];
        const float* hm = res[0].GetTensorData<float>();
        float conf[4];
        for (int c = 0; c < 4; ++c) {
            const float* plane = hm + (size_t)c * hmH * hmW;
            int bestX = 0, bestY = 0;
            float best = plane[0];
            for (int y = 0; y < hmH; ++y) {
                for (int x = 0; x < hmW; ++x) {
                    const float v = plane[(size_t)y * hmW + x];
                    if (v > best) { best = v; bestX = x; bestY = y; }
                }
            }
            conf[c] = best;
            if (confOut) confOut[c] = best;
            const int x0 = std::max(0, bestX - 1), x1 = std::min(hmW - 1, bestX + 1);
            const int y0 = std::max(0, bestY - 1), y1 = std::min(hmH - 1, bestY + 1);
            float peak = plane[(size_t)bestY * hmW + bestX];
            double sumW = 0, sumX = 0, sumY = 0;
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const double wt = std::exp((double)plane[(size_t)y * hmW + x] - peak);
                    sumW += wt;
                    sumX += wt * (x + 0.5);
                    sumY += wt * (y + 0.5);
                }
            }
            const float cx256 = sumW > 0 ? (float)(sumX / sumW) : (bestX + 0.5f);
            const float cy256 = sumW > 0 ? (float)(sumY / sumW) : (bestY + 0.5f);
            out[c].x = cx256 * (float)w / (float)hmW;
            out[c].y = cy256 * (float)h / (float)hmH;
        }
        const float minConf = std::min(std::min(conf[0], conf[1]), std::min(conf[2], conf[3]));
        if (minConf < 0.55f) return false;
        if (!convexQuad(out) || quadArea(out) < 0.08f * (float)w * (float)h) return false;
        return true;
    } catch (const Ort::Exception&) {
        held.session.reset();
        held.key = nullptr;
        held.bytes = 0;
        return false;
    }
}
#endif

}  // namespace

void pairSingleFold(std::vector<PagePoint>& pts);

void spaceOutline(DocumentOutline& o, float minDist) {
    if (o.points.size() < 4 || minDist <= 0) return;
    std::vector<PagePoint> kept;
    kept.reserve(o.points.size());
    auto near = [&](const PagePoint& a, const PagePoint& b) {
        const float dx = a.x - b.x, dy = a.y - b.y;
        return dx * dx + dy * dy < minDist * minDist;
    };
    for (const PagePoint& p : o.points) {
        if (!kept.empty() && near(kept.back(), p)) {
            if (kept.back().kind != PagePointKind::Corner && p.kind == PagePointKind::Corner)
                kept.back() = p;
            continue;
        }
        kept.push_back(p);
    }
    if (kept.size() > 3 && near(kept.front(), kept.back())) {
        if (kept.front().kind != PagePointKind::Corner &&
            kept.back().kind == PagePointKind::Corner)
            kept.front() = kept.back();
        kept.pop_back();
    }
    if (kept.size() >= 3) o.points.swap(kept);
}

DocumentDetect quadFromCorners(const Pt c[4]) {
    DocumentDetect out;
    out.found = true;
    out.outline.shape = DocumentShape::Quad;
    for (int i = 0; i < 4; ++i) {
        out.quad.x[i] = c[i].x;
        out.quad.y[i] = c[i].y;
        out.outline.x.push_back(c[i].x);
        out.outline.y.push_back(c[i].y);
        out.outline.points.push_back(makePoint(PagePointKind::Corner, c[i]));
    }
    return out;
}

// A low-contrast frame is reported as the whole photo. That is right when
// there is no border. When the scored finder sees a page inside the frame,
// that quad is the page and the frame guess is the desk.
bool finderInsteadOfFrame(const float* linear, uint32_t w, uint32_t h, DocumentDetect& out) {
    const PageQuad pq = findPageQuad(linear, w, h);
    if (!pq.ok) return false;
    Pt c[4];
    for (int i = 0; i < 4; ++i) c[i] = Pt{pq.x[i], pq.y[i]};
    if (!convexQuad(c)) return false;
    const float area = quadArea(c);
    if (area < 0.08f * (float)w * (float)h) return false;
    if (area > 0.92f * (float)w * (float)h) return false;
    out = quadFromCorners(c);
    return true;
}

DocumentDetect detectDocument(const float* linear, uint32_t w, uint32_t h) {
    DocumentDetect out;
    if (!linear || w < 16 || h < 16) {
        out.error = "Image is too small to scan";
        return out;
    }

    const int longSide = 640;
    const int srcLong = (int)std::max(w, h);
    const int detLong = std::min(longSide, srcLong);
    const float scale = (float)srcLong / (float)detLong;
    const int sw = std::max(16, (int)std::lround((float)w / scale));
    const int sh = std::max(16, (int)std::lround((float)h / scale));
    const float sx = (float)w / (float)sw;
    const float sy = (float)h / (float)sh;

    std::vector<float> gray((size_t)sw * sh);
    std::vector<float> whiteScore((size_t)sw * sh, 1.0f);
    for (int y = 0; y < sh; ++y) {
        const int y0 = std::min((int)((float)y * sy), (int)h - 1);
        const int y1 = std::min((int)((float)(y + 1) * sy), (int)h);
        for (int x = 0; x < sw; ++x) {
            const int x0 = std::min((int)((float)x * sx), (int)w - 1);
            const int x1 = std::min((int)((float)(x + 1) * sx), (int)w);
            double acc = 0, ar = 0, ag = 0, ab = 0;
            int n = 0;
            for (int py = y0; py < y1; ++py) {
                for (int px = x0; px < x1; ++px) {
                    const float* p = linear + ((size_t)py * w + (size_t)px) * 4;
                    ar += p[0]; ag += p[1]; ab += p[2];
                    acc += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                    ++n;
                }
            }
            const size_t gi = (size_t)y * sw + x;
            gray[gi] = n ? (float)(acc / n) : 0.0f;
            if (!n) continue;
            const float r = (float)(ar / n), gch = (float)(ag / n), b = (float)(ab / n);
            const float mx = std::max(r, std::max(gch, b));
            const float mn = std::min(r, std::min(gch, b));
            whiteScore[gi] = mx > 1e-4f ? mn / mx : 0.0f;
        }
    }

    // A ruled form is found from its grid, before the whiteness gate, because
    // that gate is a brightness test and this sheet fails it.
    std::vector<uint8_t> ruledMask;
    const bool ruledForm = maskFromRulings(gray, sw, sh, ruledMask);

    // Wood sits next to a shadowed page in luminance and far from it in
    // colour. The brightest pixels are the lit paper; anything much less
    // neutral than that is the table, and it must not reach Otsu. A grey
    // desk has the same whiteness as the paper, so this gate leaves it for
    // the luminance test. The shading cap below stays — an uncapped divide
    // still lifts that grey desk.
    {
        const float p80 = percentile(gray, 0.80);
        std::vector<float> ww;
        ww.reserve(gray.size() / 5);
        for (size_t i = 0; i < gray.size(); ++i)
            if (gray[i] >= p80) ww.push_back(whiteScore[i]);
        const float paperWhite = ww.empty() ? 1.0f : percentile(ww, 0.50);
        const float gate = paperWhite - 0.07f;
        for (size_t i = 0; i < gray.size(); ++i)
            if (whiteScore[i] < gate) gray[i] = 0.0f;
    }

    // A shadow is a slow multiply. Dividing by a wide blur lifts the dark side
    // of the sheet. The gain is capped: an uncapped divide also lifts a grey
    // desk until it is as bright as the paper and the quad swallows the table.
    const int blurR = std::max(1, std::min(sw, sh) / 16);
    std::vector<float> blur;
    boxBlur(gray, blur, sw, sh, blurR);
    std::vector<float> sample = gray;
    const float p90 = percentile(sample, 0.90);
    std::vector<float> lifted((size_t)sw * sh);
    for (size_t i = 0; i < gray.size(); ++i) {
        const float g = std::clamp(p90 / std::max(blur[i], 0.04f), 1.0f, 1.35f);
        lifted[i] = gray[i] * g;
    }
    const float lo = percentile(lifted, 0.02);
    const float hi = percentile(lifted, 0.98);
    // A flat bright frame is the page. Stretching a zero range maps every
    // pixel to 0 and Otsu then reports that nothing is paper.
    if (hi - lo < 0.04f) {
        double mean = 0;
        for (float v : gray) mean += v;
        mean /= (double)gray.size();
        if (mean < 0.50) {
            out.error = "No document found";
            return out;
        }
        if (finderInsteadOfFrame(linear, w, h, out)) return out;
        out.found = true;
        out.outline.shape = DocumentShape::Quad;
        out.quad.x[0] = 0;              out.quad.y[0] = 0;
        out.quad.x[1] = (float)w - 1;   out.quad.y[1] = 0;
        out.quad.x[2] = (float)w - 1;   out.quad.y[2] = (float)h - 1;
        out.quad.x[3] = 0;              out.quad.y[3] = (float)h - 1;
        return out;
    }
    const float span = hi - lo;
    std::vector<uint8_t> g8((size_t)sw * sh);
    for (size_t i = 0; i < lifted.size(); ++i) {
        const float t = std::clamp((lifted[i] - lo) / span, 0.0f, 1.0f);
        g8[i] = (uint8_t)std::lround(t * 255.0f);
    }

    const int thresh = otsu(g8);
    std::vector<uint8_t> mask((size_t)sw * sh, 0);
    int bright = 0;
    bool touch[4] = {};
    if (ruledForm) {
        mask.swap(ruledMask);
        for (int y = 0; y < sh; ++y) {
            for (int x = 0; x < sw; ++x) {
                if (!mask[(size_t)y * sw + x]) continue;
                ++bright;
                if (x == 0) touch[0] = true;
                if (y == 0) touch[1] = true;
                if (x + 1 == sw) touch[2] = true;
                if (y + 1 == sh) touch[3] = true;
            }
        }
    } else {
        for (int y = 0; y < sh; ++y) {
            for (int x = 0; x < sw; ++x) {
                const size_t i = (size_t)y * sw + x;
                if (g8[i] <= thresh) continue;
                mask[i] = 1;
                ++bright;
                if (x == 0) touch[0] = true;
                if (y == 0) touch[1] = true;
                if (x + 1 == sw) touch[2] = true;
                if (y + 1 == sh) touch[3] = true;
            }
        }
    }
    if (bright < (sw * sh) / 25) {
        out.error = "No document found";
        return out;
    }
    // The page already fills the photo. Closing would eat the border, because
    // a dilate cannot grow past the frame and the matching erode then shrinks it.
    if (bright > (sw * sh * 85) / 100 && touch[0] && touch[1] && touch[2] && touch[3]) {
        if (finderInsteadOfFrame(linear, w, h, out)) return out;
        out.found = true;
        out.outline.shape = DocumentShape::Quad;
        out.quad.x[0] = 0;              out.quad.y[0] = 0;
        out.quad.x[1] = (float)w - 1;   out.quad.y[1] = 0;
        out.quad.x[2] = (float)w - 1;   out.quad.y[2] = (float)h - 1;
        out.quad.x[3] = 0;              out.quad.y[3] = (float)h - 1;
        return out;
    }

    // Close text holes without a net grow. Radius tracks the detection size:
    // a fixed 3 px kernel is a blot on a thumbnail and a no-op on a 640 px one.
    // A ruled form is already one solid sheet.
    const int rad = std::max(2, std::min(sw, sh) / 80);
    std::vector<uint8_t> dilated, closed;
    if (ruledForm) {
        closed = mask;
    } else {
        extremumFilter(mask, dilated, sw, sh, rad, true);
        extremumFilter(dilated, closed, sw, sh, rad, false);
    }

    std::vector<int> labels((size_t)sw * sh, 0);
    int bestLabel = 0, bestArea = 0, nextLabel = 0;
    std::queue<int> q;
    for (int i = 0; i < sw * sh; ++i) {
        if (!closed[i] || labels[i]) continue;
        ++nextLabel;
        int area = 0;
        labels[i] = nextLabel;
        q.push(i);
        while (!q.empty()) {
            const int cur = q.front(); q.pop();
            ++area;
            const int cx = cur % sw, cy = cur / sw;
            const int nbs[4] = {cx > 0 ? cur - 1 : -1,
                                cx + 1 < sw ? cur + 1 : -1,
                                cy > 0 ? cur - sw : -1,
                                cy + 1 < sh ? cur + sw : -1};
            for (int nb : nbs) {
                if (nb < 0 || !closed[nb] || labels[nb]) continue;
                labels[nb] = nextLabel;
                q.push(nb);
            }
        }
        if (area > bestArea) { bestArea = area; bestLabel = nextLabel; }
    }
    if (bestArea < (sw * sh) / 20) {
        out.error = "No document found";
        return out;
    }

    std::vector<Pt> contour;
    contour.reserve((size_t)bestArea / 4);
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            const int i = y * sw + x;
            if (labels[i] != bestLabel) continue;
            bool edge = x == 0 || y == 0 || x + 1 == sw || y + 1 == sh;
            if (!edge) {
                edge = labels[i - 1] != bestLabel || labels[i + 1] != bestLabel ||
                       labels[i - sw] != bestLabel || labels[i + sw] != bestLabel;
            }
            if (edge) contour.push_back(Pt{(float)x, (float)y});
        }
    }
    const std::vector<Pt> hull = convexHull(contour);
    const float longDet = (float)std::max(sw, sh);
    const float straightTol = std::max(2.5f, 0.02f * longDet);
    std::vector<Pt> loop = traceBoundary(labels, sw, sh, bestLabel);
    if (loop.size() < 8) loop = hull;
    std::vector<Pt> poly = simplifyClosed(loop, straightTol);

    Pt corners[4];
    const bool haveQuad = cornersFromOutline(contour, hull, sw, sh, corners);
    const float inliers = haveQuad ? edgeInlierFraction(contour, corners, straightTol) : 0;

    std::vector<Pt> top, bot, left, right;
    maskBoundaries(labels, sw, sh, bestLabel, false, top, bot);
    maskBoundaries(labels, sw, sh, bestLabel, true, left, right);
    const float sagH = std::max(boundarySagitta(top), boundarySagitta(bot));
    const float sagV = std::max(boundarySagitta(left), boundarySagitta(right));
    const float sag = std::max(sagH, sagV);
    const int splitRows = splitRuns(labels, sw, sh, bestLabel, true);
    const int splitCols = splitRuns(labels, sw, sh, bestLabel, false);
    const bool verticalBow = splitCols > splitRows ||
                             (splitCols == splitRows && sagV > sagH);

        // A reconstructed quad corner that lands in a bitten-off region is not a
    // corner of the paper. A finger fails this test: its four fitted corners
    // still sit on the page.
    int invented = 0;
    if (haveQuad && poly.size() >= 3) {
        for (int i = 0; i < 4; ++i) {
            const Pt c = corners[i];
            bool in = false;
            for (size_t k = 0, j = poly.size() - 1; k < poly.size(); j = k++) {
                const bool cross = (poly[k].y > c.y) != (poly[j].y > c.y);
                if (!cross) continue;
                const float xint = (poly[j].x - poly[k].x) * (c.y - poly[k].y) /
                                       (poly[j].y - poly[k].y + 1e-12f) + poly[k].x;
                if (c.x < xint) in = !in;
            }
            if (in) continue;
            float near = 1e9f;
            for (size_t k = 0; k < poly.size(); ++k)
                near = std::min(near, perpDist(c, poly[k], poly[(k + 1) % poly.size()]));
            if (near > straightTol) ++invented;
        }
    }

    // A wood-grain finger is all contour, so it wrecks an inlier fraction and
    // a sagitta, and a reflex on that finger is not a folded corner. Area is
    // the check that ignores it: the quad is full of paper, and almost all of
    // the mask sits inside the quad. A bowed sheet fails this — the chord
    // quad contains the background beside the concave edge. A bitten corner
    // fails it too, and `invented` catches that corner landing off the paper.
    float maskInside = 0, quadFilled = 0;
    if (haveQuad)
        maskQuadOverlap(labels, sw, sh, bestLabel, corners, maskInside, quadFilled);
    const bool areaQuad = haveQuad && invented == 0 &&
                          quadFilled >= 0.90f && maskInside >= 0.85f;

    // A contour with dozens of jags is the mask leaking onto the table, not
    // twenty corners of the sheet. A coarser pass keeps the real turns.
    if (poly.size() > 8) {
        poly = simplifyClosed(poly, straightTol * 3.0f);
        // A stair left by the coarser pass is two corners a few pixels apart.
        const float minGap = straightTol * 2.0f;
        std::vector<Pt> spaced;
        spaced.reserve(poly.size());
        for (const Pt& p : poly) {
            if (!spaced.empty() && dist(spaced.back(), p) < minGap) continue;
            spaced.push_back(p);
        }
        if (spaced.size() > 3 && dist(spaced.front(), spaced.back()) < minGap)
            spaced.pop_back();
        if (spaced.size() >= 3) poly = std::move(spaced);
    }

    // A quad that most of the boundary already agrees with wins first. A finger
    // is a sharp extra vertex, and the column bulge of that finger is huge, but
    // the page edges still lie on the four lines.
    // A chamfer is a sharp extra vertex that is not one of those four corners.
    // A bow's farthest point is a gentle turn, so it is not a chamfer, and it
    // must not be forced into a fold just because the simplifier kept five
    // vertices.
    bool sharpExtra = false;
    if (poly.size() > 4) {
        const int n = (int)poly.size();
        for (int i = 0; i < n; ++i) {
            const float turn = turnFromStraight(poly[(i + n - 1) % n], poly[i],
                                                poly[(i + 1) % n]);
            if (turn < 48.0f) continue;
            bool nearCorner = false;
            if (haveQuad) {
                for (int k = 0; k < 4; ++k)
                    if (dist(poly[i], corners[k]) <= straightTol * 4.0f) nearCorner = true;
            }
            if (!nearCorner) sharpExtra = true;
        }
    }

    // The grid already says this is one rectangle. Its four corners come
    // first; a crease that crosses the sheet is a fold on the two sides it
    // meets, and a side that actually bows gets one curve between its corners.
    bool ruledReady = false;
    if (ruledForm && haveQuad && invented == 0) {
        std::vector<float> blurred;
        boxBlur(gray, blurred, sw, sh, std::max(4, std::min(sw, sh) / 28));
        float creaseY = 0, creaseX = 0;
        const bool hCrease = creaseAt(blurred, sw, sh, corners, true, creaseY);
        const bool vCrease = creaseAt(blurred, sw, sh, corners, false, creaseX);
        auto knotsOn = [&](Pt a, Pt b, bool horizontal, bool minSide,
                           bool haveCrease, float creasePos, std::vector<PagePoint>& into) {
            const std::vector<Pt> chain =
                spanEdge(labels, sw, sh, bestLabel, a, b, horizontal, minSide);
            if (haveCrease && chain.size() >= 4) {
                // A brightness step is a hinge only where the side itself kinks.
                // The shaded face of a fold is not a second page.
                const float pos = creasePos;
                Pt hit = chain[chain.size() / 2];
                bool got = false;
                float sag = 0;
                for (const Pt& p : chain) {
                    sag = std::max(sag, perpDist(p, chain.front(), chain.back()));
                    const float d = horizontal ? std::fabs(p.x - pos) : std::fabs(p.y - pos);
                    if (d <= 2.0f) { hit = p; got = true; }
                }
                if (got && sag > straightTol)
                    into.push_back(makePoint(PagePointKind::Fold, toSource(hit, sx, sy)));
            } else if (!haveCrease) {
                fitOnce(chain, straightTol, sx, sy, into);
            }
        };
        std::vector<PagePoint> topK, rightK, botK, leftK;
        knotsOn(corners[0], corners[1], true, true, vCrease, creaseX, topK);
        knotsOn(corners[1], corners[2], false, false, hCrease, creaseY, rightK);
        knotsOn(corners[2], corners[3], true, false, vCrease, creaseX, botK);
        knotsOn(corners[3], corners[0], false, true, hCrease, creaseY, leftK);
        const bool any = !topK.empty() || !rightK.empty() || !botK.empty() || !leftK.empty();
        if (any) {
            auto addC = [&](int i) {
                out.outline.points.push_back(
                    makePoint(PagePointKind::Corner, toSource(corners[i], sx, sy)));
            };
            auto addAll = [&](const std::vector<PagePoint>& ks) {
                for (const PagePoint& p : ks) out.outline.points.push_back(p);
            };
            addC(0); addAll(topK); addC(1); addAll(rightK);
            addC(2); addAll(botK); addC(3); addAll(leftK);
            spaceOutline(out.outline, 0.02f * (float)std::max(w, h));
            bool fold = false, curve = false;
            int nCorner = 0;
            for (const PagePoint& p : out.outline.points) {
                if (p.kind == PagePointKind::Fold) fold = true;
                else if (p.kind == PagePointKind::Curve) curve = true;
                else ++nCorner;
            }
            out.outline.shape = fold ? DocumentShape::Fold
                               : curve ? DocumentShape::Curve
                                       : DocumentShape::Quad;
            if (nCorner == 4 && out.outline.shape == DocumentShape::Quad) {
                out.outline = DocumentOutline{};
                ruledReady = true;
            } else {
                for (int i = 0; i < 4; ++i) {
                    const Pt s = toSource(corners[i], sx, sy);
                    out.quad.x[i] = s.x;
                    out.quad.y[i] = s.y;
                }
                out.outline.x.clear();
                out.outline.y.clear();
                for (const PagePoint& p : out.outline.points) {
                    out.outline.x.push_back(p.x);
                    out.outline.y.push_back(p.y);
                }
                out.found = true;
                return out;
            }
        } else {
            ruledReady = true;
        }
    }

    DocumentShape shape = DocumentShape::Quad;
    if (ruledReady) {
        shape = DocumentShape::Quad;
    } else if (poly.size() == 3) shape = DocumentShape::Triangle;
    // Five vertices is a corner cut off a rectangle. It has to win before the
    // inlier shortcut: the cut is a short edge, so most of the contour still
    // lies on the four page lines and the quad is still mostly paper.
    else if (poly.size() == 5 && sharpExtra) shape = DocumentShape::Fold;
    else if (poly.size() == 5 && sag > straightTol) shape = DocumentShape::Curve;
    else if (invented > 0) shape = DocumentShape::Fold;
    else if (haveQuad && inliers >= 0.80f && invented == 0) shape = DocumentShape::Quad;
    else if (areaQuad) shape = DocumentShape::Quad;
    else if (significantReflex(poly, straightTol)) shape = DocumentShape::Fold;
    else if (sag > straightTol && poly.size() > 4) shape = DocumentShape::Curve;
    else if (haveQuad) shape = DocumentShape::Quad;
    else {
        out.error = "No document found";
        return out;
    }

    auto acceptArea = [&](const std::vector<Pt>& srcPoly) {
        if (srcPoly.size() < 3) return false;
        float a = 0;
        for (size_t i = 0; i < srcPoly.size(); ++i) {
            const Pt q = srcPoly[(i + 1) % srcPoly.size()];
            a += srcPoly[i].x * q.y - q.x * srcPoly[i].y;
        }
        if (std::fabs(a) * 0.5f < 0.03f * (float)w * (float)h) return false;
        float side = 1e9f;
        for (size_t i = 0; i < srcPoly.size(); ++i)
            side = std::min(side, dist(srcPoly[i], srcPoly[(i + 1) % srcPoly.size()]));
        return side >= 12.0f;
    };

    out.outline.shape = shape;
    if (shape == DocumentShape::Quad) {
        bool usedFinder = false;
        // The mask quad is already the page on a clean sheet. Replace it
        // only when it fails the same inside/outside test the line scorer
        // uses — that is the shadow, the seam, and the block of type.
        if (!ruledReady) {
            Pt maskSrc[4];
            float mx[4], my[4];
            for (int i = 0; i < 4; ++i) {
                maskSrc[i] = toSource(corners[i], sx, sy);
                mx[i] = maskSrc[i].x;
                my[i] = maskSrc[i].y;
            }
            if (!scorePageQuad(linear, w, h, mx, my).ok) {
                const PageQuad pq = findPageQuad(linear, w, h);
                if (pq.ok) {
                    Pt scored[4];
                    for (int i = 0; i < 4; ++i) scored[i] = Pt{pq.x[i], pq.y[i]};
                    const float sideMin = std::min({dist(scored[0], scored[1]),
                                                    dist(scored[1], scored[2]),
                                                    dist(scored[2], scored[3]),
                                                    dist(scored[3], scored[0])});
                    if (convexQuad(scored) &&
                        quadArea(scored) >= 0.03f * (float)w * (float)h &&
                        sideMin >= 16.0f) {
                        for (int i = 0; i < 4; ++i) corners[i] = scored[i];
                        usedFinder = true;
                    }
                }
            }
        }
        if (!usedFinder) {
            for (int i = 0; i < 4; ++i) corners[i] = toSource(corners[i], sx, sy);
        }
        if (!convexQuad(corners) || quadArea(corners) < 0.03f * (float)w * (float)h) {
            out.error = "No document found";
            return out;
        }
        const float sideMin = std::min({dist(corners[0], corners[1]),
                                        dist(corners[1], corners[2]),
                                        dist(corners[2], corners[3]),
                                        dist(corners[3], corners[0])});
        if (sideMin < 16.0f) {
            out.error = "No document found";
            return out;
        }
        out.found = true;
        for (int i = 0; i < 4; ++i) {
            out.quad.x[i] = corners[i].x;
            out.quad.y[i] = corners[i].y;
            out.outline.x.push_back(corners[i].x);
            out.outline.y.push_back(corners[i].y);
            out.outline.points.push_back(makePoint(PagePointKind::Corner, corners[i]));
        }
        return out;
    }
    if (shape == DocumentShape::Curve) {
        const std::vector<Pt>& a = verticalBow ? left : top;
        const std::vector<Pt>& b = verticalBow ? right : bot;
        if (a.size() < 8 || b.size() < 8) {
            out.error = "No document found";
            return out;
        }
        storeCurve(out.outline, a, b, sx, sy);
        out.outline.vertical = verticalBow;
        // Corners stay TL, TR, BR, BL. Knots sit on the side they belong to,
        // so a later edit of one side does not depend on which way the bow ran.
        auto addKnots = [&](const std::vector<Pt>& chain) {
            std::vector<PagePoint> knots;
            fitSide(chain, straightTol, sx, sy, knots);
            for (const PagePoint& p : knots) {
                out.outline.points.push_back(p);
                if (p.kind == PagePointKind::Fold) shape = DocumentShape::Fold;
            }
        };
        if (!verticalBow) {
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(a.front(), sx, sy)));
            addKnots(a);
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(a.back(), sx, sy)));
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(b.back(), sx, sy)));
            std::vector<Pt> rev(b.rbegin(), b.rend());
            addKnots(rev);
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(b.front(), sx, sy)));
        } else {
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(a.front(), sx, sy)));
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(b.front(), sx, sy)));
            addKnots(b);
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(b.back(), sx, sy)));
            out.outline.points.push_back(makePoint(PagePointKind::Corner, toSource(a.back(), sx, sy)));
            std::vector<Pt> rev(a.rbegin(), a.rend());
            addKnots(rev);
        }
        out.outline.shape = shape;
        spaceOutline(out.outline, 0.025f * (float)std::max(w, h));
        std::vector<Pt> ends = {toSource(a.front(), sx, sy), toSource(a.back(), sx, sy),
                                toSource(b.front(), sx, sy), toSource(b.back(), sx, sy)};
        if (!acceptArea(ends) && quadArea(corners) < 0.03f * (float)w * (float)h && haveQuad) {
            // ends of a thin arc can be close; the mask area already passed.
        }
        out.found = true;
        return out;
    }

    storePoly(out.outline, poly, sx, sy);
    pairSingleFold(out.outline.points);
    spaceOutline(out.outline, 0.025f * (float)std::max(w, h));
    out.outline.x.clear();
    out.outline.y.clear();
    for (const PagePoint& p : out.outline.points) {
        out.outline.x.push_back(p.x);
        out.outline.y.push_back(p.y);
    }
    std::vector<Pt> srcPoly(poly.size());
    for (size_t i = 0; i < poly.size(); ++i) srcPoly[i] = toSource(poly[i], sx, sy);
    if (!acceptArea(srcPoly)) {
        out.error = "No document found";
        return out;
    }
    out.found = true;
    return out;
}

DocumentDetect detectRectangle(const float* linear, uint32_t w, uint32_t h,
                               const uint8_t* model, size_t modelBytes,
                               float cornerPeak[4]) {
    if (cornerPeak)
        for (int i = 0; i < 4; ++i) cornerPeak[i] = 0.0f;
    DocumentDetect classical = detectDocument(linear, w, h);
#ifdef VIEWMAGE_WITH_ORT
    Pt corners[4];
    float peaks[4] = {};
    // Area first: it is what keeps a page that fills the frame. The point
    // sample still wins on a small sheet whose rules the average softens.
    bool got = pageModelCorners(linear, w, h, model, modelBytes, true, corners, peaks);
    if (!got) {
        float pointPeaks[4] = {};
        Pt pointCorners[4];
        if (pageModelCorners(linear, w, h, model, modelBytes, false, pointCorners, pointPeaks)) {
            got = true;
            for (int i = 0; i < 4; ++i) {
                corners[i] = pointCorners[i];
                peaks[i] = pointPeaks[i];
            }
        } else {
            const float areaMin = std::min(std::min(peaks[0], peaks[1]),
                                           std::min(peaks[2], peaks[3]));
            const float pointMin = std::min(std::min(pointPeaks[0], pointPeaks[1]),
                                            std::min(pointPeaks[2], pointPeaks[3]));
            if (pointMin > areaMin)
                for (int i = 0; i < 4; ++i) peaks[i] = pointPeaks[i];
        }
    }
    if (cornerPeak)
        for (int i = 0; i < 4; ++i) cornerPeak[i] = peaks[i];
    auto pullClassical = [&](DocumentDetect det) {
        if (!det.found || det.outline.shape != DocumentShape::Quad) return det;
        Pt q[4];
        for (int i = 0; i < 4; ++i) q[i] = {det.quad.x[i], det.quad.y[i]};
        tightenPaperMargins(linear, w, h, q);
        if (!convexQuad(q)) return det;
        for (int i = 0; i < 4; ++i) {
            det.quad.x[i] = q[i].x;
            det.quad.y[i] = q[i].y;
        }
        int n = 0;
        for (PagePoint& p : det.outline.points)
            if (p.kind == PagePointKind::Corner && n < 4) {
                p.x = q[n].x;
                p.y = q[n].y;
                ++n;
            }
        if (det.outline.x.size() == 4 && det.outline.y.size() == 4)
            for (int i = 0; i < 4; ++i) {
                det.outline.x[i] = q[i].x;
                det.outline.y[i] = q[i].y;
            }
        return det;
    };
    if (!got) return pullClassical(classical);
    if (!refinePageCorners(linear, w, h, corners)) return pullClassical(classical);
    tightenPaperMargins(linear, w, h, corners);
    orderCorners(corners);
    if (!convexQuad(corners) || quadArea(corners) < 0.03f * (float)w * (float)h)
        return pullClassical(classical);
    const float sideMin = std::min({dist(corners[0], corners[1]), dist(corners[1], corners[2]),
                                    dist(corners[2], corners[3]), dist(corners[3], corners[0])});
    if (sideMin < 16.0f) return pullClassical(classical);
    float gateX[4], gateY[4];
    for (int i = 0; i < 4; ++i) {
        gateX[i] = corners[i].x;
        gateY[i] = corners[i].y;
    }
    // A heatmap quad that fails the same polarity test as a classical one
    // is a shadow or a block of type. The classical result stays.
    if (!scorePageQuad(linear, w, h, gateX, gateY).ok) return pullClassical(classical);
    DocumentDetect out;
    out.found = true;
    out.outline.shape = DocumentShape::Quad;
    for (int i = 0; i < 4; ++i) {
        out.quad.x[i] = corners[i].x;
        out.quad.y[i] = corners[i].y;
        out.outline.x.push_back(corners[i].x);
        out.outline.y.push_back(corners[i].y);
        out.outline.points.push_back(makePoint(PagePointKind::Corner, corners[i]));
    }
    return out;
#else
    (void)model;
    (void)modelBytes;
    return classical;
#endif
}

DocumentScan flattenDocument(const float* linear, uint32_t w, uint32_t h,
                             const DocumentQuad& quad) {
    DocumentScan out;
    if (!linear || w < 2 || h < 2) {
        out.error = "Image is too small to scan";
        return out;
    }
    Pt corners[4];
    for (int i = 0; i < 4; ++i) {
        corners[i].x = quad.x[i];
        corners[i].y = quad.y[i];
    }
    if (!convexQuad(corners)) {
        out.error = "The corners cross — move them apart";
        return out;
    }

    int ow = (int)std::lround(std::max(dist(corners[0], corners[1]),
                                       dist(corners[3], corners[2])));
    int oh = (int)std::lround(std::max(dist(corners[0], corners[3]),
                                       dist(corners[1], corners[2])));
    ow = std::clamp(ow, 16, (int)w * 2);
    oh = std::clamp(oh, 16, (int)h * 2);
    const double budget = (double)w * (double)h * 1.25;
    if ((double)ow * oh > budget) {
        const double s = std::sqrt(budget / ((double)ow * oh));
        ow = std::max(16, (int)std::lround(ow * s));
        oh = std::max(16, (int)std::lround(oh * s));
    }

    Pt dst[4] = {{0, 0}, {(float)(ow - 1), 0},
                 {(float)(ow - 1), (float)(oh - 1)}, {0, (float)(oh - 1)}};
    double H[9];
    if (!homography(dst, corners, H)) {
        out.error = "Could not straighten the page";
        return out;
    }

    out.linear.assign((size_t)ow * oh * 4, 0.0f);
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            const double X = x, Y = y;
            const double den = H[6] * X + H[7] * Y + H[8];
            if (std::fabs(den) < 1e-9) continue;
            const float u = (float)((H[0] * X + H[1] * Y + H[2]) / den);
            const float v = (float)((H[3] * X + H[4] * Y + H[5]) / den);
            float* d = out.linear.data() + ((size_t)y * ow + x) * 4;
            for (int c = 0; c < 4; ++c)
                d[c] = sampleBicubic(linear, w, h, c, u, v);
        }
    }

    out.found = true;
    out.w = (uint32_t)ow;
    out.h = (uint32_t)oh;
    for (int i = 0; i < 4; ++i) {
        out.quad.x[i] = corners[i].x;
        out.quad.y[i] = corners[i].y;
    }
    return out;
}

DocumentScan reconstructRectangle(const float* linear, uint32_t w, uint32_t h,
                                  const DocumentQuad& quad) {
    DocumentScan flat = flattenDocument(linear, w, h, quad);
    // A creased sheet is still one homography, so a straight rule stays a
    // straight rule. The dark face is a slow multiply; lifting it does not
    // move those rules. A page that is already even is left as it is.
    if (flat.found)
        evenShadedPage(flat.linear, flat.w, flat.h, false);
    return flat;
}

Pt pp(const PagePoint& p) { return {p.x, p.y}; }

Pt segAt(const PagePoint& a, const PagePoint& b, float t) {
    const Pt p0 = pp(a);
    const Pt p3 = pp(b);
    Pt out{0, 0}, in{0, 0};
    if (a.kind == PagePointKind::Curve) out = {a.hx, a.hy};
    if (b.kind == PagePointKind::Curve) in = {-b.hx, -b.hy};
    return bezAt(p0, {p0.x + out.x, p0.y + out.y}, {p3.x + in.x, p3.y + in.y}, p3, t);
}

float segLenPts(const PagePoint& a, const PagePoint& b) {
    float len = 0;
    Pt prev = pp(a);
    for (int k = 1; k <= 8; ++k) {
        const Pt q = segAt(a, b, (float)k / 8.0f);
        len += dist(prev, q);
        prev = q;
    }
    return len;
}

void clampOut(int& ow, int& oh, uint32_t w, uint32_t h) {
    ow = std::clamp(ow, 16, (int)w * 2);
    oh = std::clamp(oh, 16, (int)h * 2);
    const double budget = (double)w * (double)h * 1.25;
    if ((double)ow * oh > budget) {
        const double s = std::sqrt(budget / ((double)ow * oh));
        ow = std::max(16, (int)std::lround(ow * s));
        oh = std::max(16, (int)std::lround(oh * s));
    }
}

void writeSrc(std::vector<float>& dst, int ow, int x, int y,
              const float* linear, uint32_t w, uint32_t h, float u, float v) {
    float* d = dst.data() + ((size_t)y * ow + x) * 4;
    for (int c = 0; c < 4; ++c)
        d[c] = sampleBicubic(linear, w, h, c, u, v);
}

std::vector<PagePoint> chainFwd(const std::vector<PagePoint>& pts, int i0, int i1) {
    std::vector<PagePoint> c;
    const int n = (int)pts.size();
    if (n == 0) return c;
    for (int i = i0, guard = 0; guard <= n; ++guard) {
        c.push_back(pts[i]);
        if (i == i1) break;
        i = (i + 1) % n;
    }
    return c;
}

void orientTo(std::vector<PagePoint>& c, Pt target) {
    if (c.size() < 2) return;
    if (dist(pp(c.front()), target) > dist(pp(c.back()), target))
        std::reverse(c.begin(), c.end());
}

float panelWidth(const std::vector<PagePoint>& c) {
    if (c.size() < 3) return 8.0f;
    const float a = segLenPts(c[0], c[1]);
    const float b = segLenPts(c[c.size() - 2], c.back());
    return std::max(8.0f, 0.5f * (a + b));
}

Pt rangeAt(const std::vector<PagePoint>& c, int i0, int i1, float s) {
    if (c.empty()) return {};
    i0 = std::clamp(i0, 0, (int)c.size() - 1);
    i1 = std::clamp(i1, 0, (int)c.size() - 1);
    if (i1 < i0) std::swap(i0, i1);
    if (i0 == i1) return pp(c[i0]);
    std::vector<float> acc(1, 0.0f);
    for (int i = i0; i < i1; ++i) acc.push_back(acc.back() + segLenPts(c[i], c[i + 1]));
    const float total = acc.back();
    if (total < 1e-3f) return pp(c[i0]);
    s = std::clamp(s, 0.0f, 1.0f);
    const float target = s * total;
    int seg = 0;
    while (seg + 1 < (int)acc.size() && acc[seg + 1] < target) ++seg;
    const float span = acc[seg + 1] - acc[seg];
    const float t = span > 1e-4f ? (target - acc[seg]) / span : 0;
    return segAt(c[i0 + seg], c[i0 + seg + 1], t);
}

Pt outerAt(const std::vector<PagePoint>& c, float s) {
    if (c.size() < 3) return c.empty() ? Pt{} : pp(c.front());
    if (c.size() == 3) return pp(c[1]);
    return rangeAt(c, 1, (int)c.size() - 2, s);
}

bool polyHas(const std::vector<PagePoint>& pts, float x, float y) {
    std::vector<float> xs, ys;
    xs.reserve(pts.size());
    ys.reserve(pts.size());
    for (const PagePoint& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
    return pointInPoly(x, y, xs, ys);
}

DocumentScan rasterHinge(const float* linear, uint32_t w, uint32_t h,
                         std::vector<PagePoint> chainA, std::vector<PagePoint> chainB) {
    DocumentScan out;
    if (chainA.size() < 3 || chainB.size() < 3) return out;
    const Pt end0 = pp(chainA.front());
    const Pt end1 = pp(chainA.back());
    const bool vertical = std::fabs(end1.y - end0.y) >= std::fabs(end1.x - end0.x);
    const Pt first = vertical ? (end0.y <= end1.y ? end0 : end1)
                              : (end0.x <= end1.x ? end0 : end1);
    orientTo(chainA, first);
    orientTo(chainB, first);
    const Pt hinge0 = pp(chainA.front());
    const Pt hinge1 = pp(chainA.back());

    auto centroidX = [](const std::vector<PagePoint>& c) {
        double s = 0;
        int n = 0;
        for (size_t i = 1; i + 1 < c.size(); ++i) { s += c[i].x; ++n; }
        if (!n) for (const PagePoint& p : c) { s += p.x; ++n; }
        return n ? s / n : 0;
    };
    auto centroidY = [](const std::vector<PagePoint>& c) {
        double s = 0;
        int n = 0;
        for (size_t i = 1; i + 1 < c.size(); ++i) { s += c[i].y; ++n; }
        if (!n) for (const PagePoint& p : c) { s += p.y; ++n; }
        return n ? s / n : 0;
    };
    if (vertical) {
        if (centroidX(chainA) > centroidX(chainB)) std::swap(chainA, chainB);
    } else if (centroidY(chainA) > centroidY(chainB)) {
        std::swap(chainA, chainB);
    }

    const float hingeLen = std::max(8.0f, dist(hinge0, hinge1));
    const float wA = panelWidth(chainA);
    const float wB = panelWidth(chainB);
    int ow, oh;
    if (vertical) {
        ow = (int)std::lround(wA + wB);
        oh = (int)std::lround(hingeLen);
    } else {
        ow = (int)std::lround(hingeLen);
        oh = (int)std::lround(wA + wB);
    }
    clampOut(ow, oh, w, h);
    out.linear.assign((size_t)ow * oh * 4, 1.0f);

    const int nA = vertical ? std::max(1, (int)std::lround(wA * (float)ow / std::max(1.0f, wA + wB)))
                            : std::max(1, (int)std::lround(wA * (float)oh / std::max(1.0f, wA + wB)));
    const int nB = vertical ? std::max(1, ow - nA) : std::max(1, oh - nA);

    auto paint = [&](int x, int y, Pt src) {
        if (x < 0 || y < 0 || x >= ow || y >= oh) return;
        writeSrc(out.linear, ow, x, y, linear, w, h, src.x, src.y);
    };

    if (vertical) {
        for (int y = 0; y < oh; ++y) {
            const float s = oh == 1 ? 0.0f : (float)y / (float)(oh - 1);
            const Pt hinge{hinge0.x + (hinge1.x - hinge0.x) * s,
                           hinge0.y + (hinge1.y - hinge0.y) * s};
            const Pt oA = outerAt(chainA, s);
            const Pt oB = outerAt(chainB, s);
            for (int x = 0; x < nA; ++x) {
                const float u = nA == 1 ? 1.0f : (float)x / (float)(nA - 1);
                paint(x, y, {oA.x + (hinge.x - oA.x) * u, oA.y + (hinge.y - oA.y) * u});
            }
            for (int x = 0; x < nB; ++x) {
                const float u = nB == 1 ? 1.0f : (float)x / (float)(nB - 1);
                paint(nA + x, y, {hinge.x + (oB.x - hinge.x) * u, hinge.y + (oB.y - hinge.y) * u});
            }
        }
    } else {
        for (int x = 0; x < ow; ++x) {
            const float s = ow == 1 ? 0.0f : (float)x / (float)(ow - 1);
            const Pt hinge{hinge0.x + (hinge1.x - hinge0.x) * s,
                           hinge0.y + (hinge1.y - hinge0.y) * s};
            const Pt oA = outerAt(chainA, s);
            const Pt oB = outerAt(chainB, s);
            for (int y = 0; y < nA; ++y) {
                const float u = nA == 1 ? 1.0f : (float)y / (float)(nA - 1);
                paint(x, y, {oA.x + (hinge.x - oA.x) * u, oA.y + (hinge.y - oA.y) * u});
            }
            for (int y = 0; y < nB; ++y) {
                const float u = nB == 1 ? 1.0f : (float)y / (float)(nB - 1);
                paint(x, nA + y, {hinge.x + (oB.x - hinge.x) * u, hinge.y + (oB.y - hinge.y) * u});
            }
        }
    }
    out.found = true;
    out.w = (uint32_t)ow;
    out.h = (uint32_t)oh;
    return out;
}

Pt sideAt(const std::vector<PagePoint>& side, float t) {
    if (side.size() == 1) return pp(side[0]);
    return rangeAt(side, 0, (int)side.size() - 1, t);
}

float sideLen(const std::vector<PagePoint>& side) {
    float len = 0;
    for (size_t i = 1; i < side.size(); ++i) len += segLenPts(side[i - 1], side[i]);
    return len;
}

void boundaryPoly(const std::vector<PagePoint>& pts,
                  std::vector<float>& xs, std::vector<float>& ys) {
    const int n = (int)pts.size();
    for (int i = 0; i < n; ++i) {
        const PagePoint& a = pts[i];
        const PagePoint& b = pts[(i + 1) % n];
        const bool curved = a.kind == PagePointKind::Curve || b.kind == PagePointKind::Curve;
        const int steps = curved ? 12 : 1;
        for (int k = 0; k < steps; ++k) {
            const Pt q = segAt(a, b, (float)k / (float)steps);
            xs.push_back(q.x);
            ys.push_back(q.y);
        }
    }
}

// A sample on the cubic itself is not reliably inside an even-odd test.
// A pixel and a half of slack keeps the edge and still rejects a bite.
bool onSheet(float u, float v, const std::vector<float>& xs, const std::vector<float>& ys) {
    if (pointInPoly(u, v, xs, ys)) return true;
    const float e = 1.5f;
    return pointInPoly(u + e, v, xs, ys) || pointInPoly(u - e, v, xs, ys) ||
           pointInPoly(u, v + e, xs, ys) || pointInPoly(u, v - e, xs, ys);
}

// Four corner-to-corner chains, every diamond a kink and every circle a cubic.
// A straight side cancels out of the patch, so a bow on one edge stays the
// ruled blend it was. A sample outside the outline is left white: a diamond
// pulled inside the quad cuts that bite, and a bulge keeps the paper past
// the straight chord.
DocumentScan rasterRuled(const float* linear, uint32_t w, uint32_t h,
                         const std::vector<PagePoint>& pts) {
    DocumentScan out;
    std::vector<int> corners;
    for (int i = 0; i < (int)pts.size(); ++i)
        if (pts[i].kind == PagePointKind::Corner) corners.push_back(i);
    if (corners.size() != 4) return out;
    std::vector<PagePoint> side[4];
    for (int k = 0; k < 4; ++k)
        side[k] = chainFwd(pts, corners[k], corners[(k + 1) % 4]);

    const Pt tl = pp(pts[corners[0]]);
    const Pt tr = pp(pts[corners[1]]);
    const Pt br = pp(pts[corners[2]]);
    const Pt bl = pp(pts[corners[3]]);
    // side 2 runs BR→BL and side 3 runs BL→TL. The patch walks both left to
    // right and top to bottom, so those two are sampled backwards.
    int ow = (int)std::lround(std::max(sideLen(side[0]), sideLen(side[2])));
    int oh = (int)std::lround(std::max(sideLen(side[1]), sideLen(side[3])));
    clampOut(ow, oh, w, h);

    std::vector<float> xs, ys;
    boundaryPoly(pts, xs, ys);
    if (xs.size() < 3) return out;

    out.linear.assign((size_t)ow * oh * 4, 1.0f);
    for (int y = 0; y < oh; ++y) {
        const float s = oh == 1 ? 0.0f : (float)y / (float)(oh - 1);
        const Pt left = sideAt(side[3], 1.0f - s);
        const Pt right = sideAt(side[1], s);
        for (int x = 0; x < ow; ++x) {
            const float t = ow == 1 ? 0.0f : (float)x / (float)(ow - 1);
            const Pt top = sideAt(side[0], t);
            const Pt bot = sideAt(side[2], 1.0f - t);
            const float u = 1.0f - t, v = 1.0f - s;
            const float sx = v * top.x + s * bot.x + u * left.x + t * right.x
                           - u * v * tl.x - t * v * tr.x - t * s * br.x - u * s * bl.x;
            const float sy = v * top.y + s * bot.y + u * left.y + t * right.y
                           - u * v * tl.y - t * v * tr.y - t * s * br.y - u * s * bl.y;
            if (!onSheet(sx, sy, xs, ys)) continue;
            writeSrc(out.linear, ow, x, y, linear, w, h, sx, sy);
        }
    }
    out.found = true;
    out.w = (uint32_t)ow;
    out.h = (uint32_t)oh;
    return out;
}

// Same pixel budget as flattenDocument. The four corners alone decide it.
void cornerPixels(const Pt c[4], uint32_t w, uint32_t h, int& ow, int& oh) {
    ow = (int)std::lround(std::max(dist(c[0], c[1]), dist(c[3], c[2])));
    oh = (int)std::lround(std::max(dist(c[0], c[3]), dist(c[1], c[2])));
    ow = std::clamp(ow, 16, (int)w * 2);
    oh = std::clamp(oh, 16, (int)h * 2);
    const double budget = (double)w * (double)h * 1.25;
    if ((double)ow * oh > budget) {
        const double s = std::sqrt(budget / ((double)ow * oh));
        ow = std::max(16, (int)std::lround(ow * s));
        oh = std::max(16, (int)std::lround(oh * s));
    }
}

bool sideDeparts(const std::vector<PagePoint>& side) {
    if (side.size() < 3) return false;
    const Pt a = pp(side.front()), b = pp(side.back());
    constexpr float kGuidePx = 2.5f;
    for (size_t i = 1; i + 1 < side.size(); ++i) {
        if (side[i].kind == PagePointKind::Curve &&
            std::hypot(side[i].hx, side[i].hy) > kGuidePx)
            return true;
        if (perpDist(pp(side[i]), a, b) > kGuidePx) return true;
    }
    return false;
}

float sideDeparture(const std::vector<PagePoint>& side) {
    if (side.size() < 3) return 0;
    const Pt a = pp(side.front()), b = pp(side.back());
    float d = 0;
    for (size_t i = 1; i + 1 < side.size(); ++i) {
        d = std::max(d, perpDist(pp(side[i]), a, b));
        if (side[i].kind == PagePointKind::Curve)
            d = std::max(d, std::hypot(side[i].hx, side[i].hy));
    }
    return d;
}

// The four yellow corners are a straight rectangle. A blue point edits only
// the side it sits on, and only as far into the page as that side actually
// bends. The opposite side, and the far half of the page, stay the straight
// corner mapping.
DocumentScan guideRectangle(const float* linear, uint32_t w, uint32_t h,
                            const std::vector<PagePoint> side[4], const Pt c[4],
                            int ow, int oh) {
    DocumentScan out;
    Pt dst[4] = {{0, 0}, {(float)(ow - 1), 0},
                 {(float)(ow - 1), (float)(oh - 1)}, {0, (float)(oh - 1)}};
    double H[9];
    if (!homography(dst, c, H)) {
        out.error = "Could not straighten the page";
        return out;
    }
    auto at = [&](float x, float y) {
        const double X = x, Y = y;
        const double den = H[6] * X + H[7] * Y + H[8];
        Pt p{0, 0};
        if (std::fabs(den) < 1e-9) return p;
        p.x = (float)((H[0] * X + H[1] * Y + H[2]) / den);
        p.y = (float)((H[3] * X + H[4] * Y + H[5]) / den);
        return p;
    };
    const float xR = (float)(ow - 1), yB = (float)(oh - 1);
    const bool hot[4] = {sideDeparts(side[0]), sideDeparts(side[1]),
                         sideDeparts(side[2]), sideDeparts(side[3])};
    float reach[4];
    for (int k = 0; k < 4; ++k) {
        const float span = (k % 2 == 0) ? (float)oh : (float)ow;
        reach[k] = std::clamp(sideDeparture(side[k]) / std::max(span, 1.0f) * 1.25f,
                              0.0f, 0.45f);
    }
    std::vector<Pt> edge[4], guide[4];
    edge[0].resize(ow); edge[2].resize(ow);
    edge[1].resize(oh); edge[3].resize(oh);
    for (int x = 0; x < ow; ++x) {
        edge[0][x] = at((float)x, 0);
        edge[2][x] = at((float)x, yB);
    }
    for (int y = 0; y < oh; ++y) {
        edge[1][y] = at(xR, (float)y);
        edge[3][y] = at(0, (float)y);
    }
    for (int k = 0; k < 4; ++k) if (hot[k]) guide[k].resize(edge[k].size());
    for (int x = 0; x < ow; ++x) {
        const float t = xR > 0 ? (float)x / xR : 0.0f;
        if (hot[0]) guide[0][x] = sideAt(side[0], t);
        if (hot[2]) guide[2][x] = sideAt(side[2], 1.0f - t);
    }
    for (int y = 0; y < oh; ++y) {
        const float s = yB > 0 ? (float)y / yB : 0.0f;
        if (hot[1]) guide[1][y] = sideAt(side[1], s);
        if (hot[3]) guide[3][y] = sideAt(side[3], 1.0f - s);
    }
    auto fall = [](float depth, float reachTo) {
        if (reachTo <= 1e-4f || depth <= 0) return depth <= 0 ? 1.0f : 0.0f;
        if (depth >= reachTo) return 0.0f;
        float u = depth / reachTo;
        u = u * u * (3.0f - 2.0f * u);
        return 1.0f - u;
    };
    out.linear.assign((size_t)ow * oh * 4, 0.0f);
    for (int y = 0; y < oh; ++y) {
        const float s = yB > 0 ? (float)y / yB : 0.0f;
        const float wTop = hot[0] ? fall(s, reach[0]) : 0.0f;
        const float wBot = hot[2] ? fall(1.0f - s, reach[2]) : 0.0f;
        for (int x = 0; x < ow; ++x) {
            const float t = xR > 0 ? (float)x / xR : 0.0f;
            Pt src = at((float)x, (float)y);
            if (wTop > 0) {
                src.x += wTop * (guide[0][x].x - edge[0][x].x);
                src.y += wTop * (guide[0][x].y - edge[0][x].y);
            }
            if (wBot > 0) {
                src.x += wBot * (guide[2][x].x - edge[2][x].x);
                src.y += wBot * (guide[2][x].y - edge[2][x].y);
            }
            const float wRight = hot[1] ? fall(1.0f - t, reach[1]) : 0.0f;
            const float wLeft = hot[3] ? fall(t, reach[3]) : 0.0f;
            if (wRight > 0) {
                src.x += wRight * (guide[1][y].x - edge[1][y].x);
                src.y += wRight * (guide[1][y].y - edge[1][y].y);
            }
            if (wLeft > 0) {
                src.x += wLeft * (guide[3][y].x - edge[3][y].x);
                src.y += wLeft * (guide[3][y].y - edge[3][y].y);
            }
            writeSrc(out.linear, ow, x, y, linear, w, h, src.x, src.y);
        }
    }
    out.found = true;
    out.w = (uint32_t)ow;
    out.h = (uint32_t)oh;
    for (int k = 0; k < 4; ++k) { out.quad.x[k] = c[k].x; out.quad.y[k] = c[k].y; }
    return out;
}

DocumentScan reconstructRectangle(const float* linear, uint32_t w, uint32_t h,
                                  const DocumentOutline& outline) {
    DocumentScan out;
    if (!linear || w < 2 || h < 2) {
        out.error = "Image is too small to scan";
        return out;
    }
    std::vector<int> id;
    for (int i = 0; i < (int)outline.points.size(); ++i)
        if (outline.points[i].kind == PagePointKind::Corner) id.push_back(i);
    if (id.size() != 4) {
        out.error = "No rectangle found";
        return out;
    }
    Pt c[4];
    for (int k = 0; k < 4; ++k) c[k] = pp(outline.points[id[k]]);
    if (!convexQuad(c)) {
        out.error = "The corners cross — move them apart";
        return out;
    }
    std::vector<PagePoint> side[4];
    for (int k = 0; k < 4; ++k)
        side[k] = chainFwd(outline.points, id[k], id[(k + 1) % 4]);

    bool departs = false;
    for (int k = 0; k < 4; ++k) departs = departs || sideDeparts(side[k]);
    if (!departs) {
        DocumentQuad q;
        for (int k = 0; k < 4; ++k) { q.x[k] = c[k].x; q.y[k] = c[k].y; }
        return reconstructRectangle(linear, w, h, q);
    }

    int ow = 0, oh = 0;
    cornerPixels(c, w, h, ow, oh);
    return guideRectangle(linear, w, h, side, c, ow, oh);
}

DocumentScan rasterPoly(const float* linear, uint32_t w, uint32_t h,
                        const std::vector<PagePoint>& pts) {
    DocumentScan out;
    std::vector<float> xs, ys;
    boundaryPoly(pts, xs, ys);
    if (xs.size() < 3) return out;
    float minx = xs[0], maxx = xs[0], miny = ys[0], maxy = ys[0];
    for (size_t i = 1; i < xs.size(); ++i) {
        minx = std::min(minx, xs[i]); maxx = std::max(maxx, xs[i]);
        miny = std::min(miny, ys[i]); maxy = std::max(maxy, ys[i]);
    }
    int ow = std::max(16, (int)std::lround(maxx - minx + 1));
    int oh = std::max(16, (int)std::lround(maxy - miny + 1));
    clampOut(ow, oh, w, h);
    out.linear.assign((size_t)ow * oh * 4, 1.0f);
    for (int y = 0; y < oh; ++y) {
        const float sy = miny + (oh == 1 ? 0.0f : (float)y / (float)(oh - 1) * (maxy - miny));
        for (int x = 0; x < ow; ++x) {
            const float sx = minx + (ow == 1 ? 0.0f : (float)x / (float)(ow - 1) * (maxx - minx));
            if (!pointInPoly(sx, sy, xs, ys)) continue;
            writeSrc(out.linear, ow, x, y, linear, w, h, sx, sy);
        }
    }
    out.found = true;
    out.w = (uint32_t)ow;
    out.h = (uint32_t)oh;
    return out;
}

bool segCross(Pt a, Pt b, Pt c, Pt d, Pt& hit) {
    const float d1x = b.x - a.x, d1y = b.y - a.y;
    const float d2x = d.x - c.x, d2y = d.y - c.y;
    const float den = d1x * d2y - d1y * d2x;
    if (std::fabs(den) < 1e-4f) return false;
    const float t = ((c.x - a.x) * d2y - (c.y - a.y) * d2x) / den;
    const float u = ((c.x - a.x) * d1y - (c.y - a.y) * d1x) / den;
    if (t <= 0.08f || t >= 0.92f || u <= 0.08f || u >= 0.92f) return false;
    hit = {a.x + d1x * t, a.y + d1y * t};
    return true;
}

// Four fold points whose hinges cross: one vertical line and one horizontal
// line in the output. Each quadrant is its own homography, so a corner stays
// a corner instead of being smeared toward the crossing.
DocumentScan rasterCross(const float* linear, uint32_t w, uint32_t h,
                         const std::vector<PagePoint>& pts, const std::vector<int>& folds) {
    DocumentScan skip;
    if (folds.size() != 4) return skip;
    int top = folds[0], bot = folds[0], left = folds[0], right = folds[0];
    for (int i : folds) {
        if (pts[i].y < pts[top].y) top = i;
        if (pts[i].y > pts[bot].y) bot = i;
        if (pts[i].x < pts[left].x) left = i;
        if (pts[i].x > pts[right].x) right = i;
    }
    if (top == bot || left == right || top == left || top == right ||
        bot == left || bot == right) return skip;
    Pt hit;
    if (!segCross(pp(pts[top]), pp(pts[bot]), pp(pts[left]), pp(pts[right]), hit))
        return skip;

    auto between = [&](int a, int b) {
        auto c1 = chainFwd(pts, a, b);
        auto c2 = chainFwd(pts, b, a);
        auto extraFold = [](const std::vector<PagePoint>& c) {
            int n = 0;
            for (size_t i = 1; i + 1 < c.size(); ++i)
                if (c[i].kind == PagePointKind::Fold) ++n;
            return n;
        };
        if (extraFold(c1) == 0 && c1.size() >= 3) return c1;
        if (extraFold(c2) == 0 && c2.size() >= 3) return c2;
        return std::vector<PagePoint>{};
    };
    auto tl = between(top, left);
    auto tr = between(top, right);
    auto br = between(bot, right);
    auto bl = between(bot, left);
    if (tl.size() < 3 || tr.size() < 3 || br.size() < 3 || bl.size() < 3) return skip;

    auto startAt = [](std::vector<PagePoint>& c, const PagePoint& fold) {
        if (dist(pp(c.front()), pp(fold)) > dist(pp(c.back()), pp(fold)))
            std::reverse(c.begin(), c.end());
    };
    startAt(tl, pts[top]);
    startAt(tr, pts[top]);
    startAt(br, pts[bot]);
    startAt(bl, pts[bot]);

    const float wL = 0.5f * (segLenPts(tl[0], tl[1]) + segLenPts(bl[0], bl[1]));
    const float wR = 0.5f * (segLenPts(tr[0], tr[1]) + segLenPts(br[0], br[1]));
    const float hT = 0.5f * (segLenPts(tl[tl.size() - 2], tl.back()) +
                             segLenPts(tr[tr.size() - 2], tr.back()));
    const float hB = 0.5f * (segLenPts(bl[bl.size() - 2], bl.back()) +
                             segLenPts(br[br.size() - 2], br.back()));
    int ow = (int)std::lround(wL + wR);
    int oh = (int)std::lround(hT + hB);
    clampOut(ow, oh, w, h);
    const int xH = std::clamp((int)std::lround(wL / std::max(1.0f, wL + wR) * (float)(ow - 1)), 1, ow - 2);
    const int yH = std::clamp((int)std::lround(hT / std::max(1.0f, hT + hB) * (float)(oh - 1)), 1, oh - 2);

    auto cornerOf = [](const std::vector<PagePoint>& c) {
        return pp(c[std::min((int)c.size() - 2, std::max(1, (int)c.size() / 2))]);
    };
    // Each chain starts at the horizontal fold and ends at the vertical one,
    // with the sheet corner in between.
    const Pt tP = pp(pts[top]), bP = pp(pts[bot]), lP = pp(pts[left]), rP = pp(pts[right]);
    const Pt cTL = cornerOf(tl), cTR = cornerOf(tr), cBR = cornerOf(br), cBL = cornerOf(bl);
    const float xHf = (float)xH, yHf = (float)yH, xR = (float)(ow - 1), yB = (float)(oh - 1);
    Pt srcs[4][4] = {
        {tP, cTL, lP, hit},
        {tP, cTR, rP, hit},
        {bP, cBL, lP, hit},
        {bP, cBR, rP, hit},
    };
    Pt dsts[4][4] = {
        {{xHf, 0}, {0, 0}, {0, yHf}, {xHf, yHf}},
        {{xHf, 0}, {xR, 0}, {xR, yHf}, {xHf, yHf}},
        {{xHf, yB}, {0, yB}, {0, yHf}, {xHf, yHf}},
        {{xHf, yB}, {xR, yB}, {xR, yHf}, {xHf, yHf}},
    };
    DocumentScan out;
    out.linear.assign((size_t)ow * oh * 4, 1.0f);
    for (int ci = 0; ci < 4; ++ci) {
        double H[9];
        if (!homography(dsts[ci], srcs[ci], H)) return skip;
        const Pt* d = dsts[ci];
        const int x0 = (int)std::lround(std::min(std::min(d[0].x, d[1].x), std::min(d[2].x, d[3].x)));
        const int x1 = (int)std::lround(std::max(std::max(d[0].x, d[1].x), std::max(d[2].x, d[3].x)));
        const int y0 = (int)std::lround(std::min(std::min(d[0].y, d[1].y), std::min(d[2].y, d[3].y)));
        const int y1 = (int)std::lround(std::max(std::max(d[0].y, d[1].y), std::max(d[2].y, d[3].y)));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (x < 0 || y < 0 || x >= ow || y >= oh) continue;
                const double X = x, Y = y;
                const double den = H[6] * X + H[7] * Y + H[8];
                if (std::fabs(den) < 1e-9) continue;
                const float u = (float)((H[0] * X + H[1] * Y + H[2]) / den);
                const float v = (float)((H[3] * X + H[4] * Y + H[5]) / den);
                writeSrc(out.linear, ow, x, y, linear, w, h, u, v);
            }
        }
    }
    out.found = true;
    out.w = (uint32_t)ow;
    out.h = (uint32_t)oh;
    return out;
}

DocumentScan flattenFromPoints(const float* linear, uint32_t w, uint32_t h,
                               const DocumentOutline& outline) {
    DocumentScan skip;
    const std::vector<PagePoint>& pts = outline.points;
    if (pts.size() < 3) return skip;
    int nCorner = 0, nFold = 0, nCurve = 0;
    std::vector<int> folds;
    for (int i = 0; i < (int)pts.size(); ++i) {
        if (pts[i].kind == PagePointKind::Corner) ++nCorner;
        else if (pts[i].kind == PagePointKind::Fold) { ++nFold; folds.push_back(i); }
        else ++nCurve;
    }
    if (nFold == 0 && nCurve == 0 && nCorner == 4 && pts.size() == 4) return skip;

    if (folds.size() >= 2) {
        int bestI = -1, bestJ = -1;
        float bestLen = 0;
        for (size_t a = 0; a < folds.size(); ++a) {
            for (size_t b = a + 1; b < folds.size(); ++b) {
                const int i = folds[a], j = folds[b];
                const auto ca = chainFwd(pts, i, j);
                const auto cb = chainFwd(pts, j, i);
                if (ca.size() < 3 || cb.size() < 3) continue;
                const float mx = 0.5f * (pts[i].x + pts[j].x);
                const float my = 0.5f * (pts[i].y + pts[j].y);
                if (!polyHas(pts, mx, my)) continue;
                const float len = dist(pp(pts[i]), pp(pts[j]));
                if (len > bestLen) { bestLen = len; bestI = i; bestJ = j; }
            }
        }
        auto simpleChain = [](const std::vector<PagePoint>& c) {
            return c.size() >= 3 && c.size() <= 6;
        };
        if (folds.size() == 4) {
            bool simple = true;
            for (int k = 0; k < 4 && simple; ++k) {
                const auto c = chainFwd(pts, folds[k], folds[(k + 1) % 4]);
                if (!simpleChain(c)) simple = false;
            }
            if (simple) {
                DocumentScan cross = rasterCross(linear, w, h, pts, folds);
                if (cross.found) return cross;
            }
        }
        if (bestI >= 0) {
            const auto ca = chainFwd(pts, bestI, bestJ);
            const auto cb = chainFwd(pts, bestJ, bestI);
            if (simpleChain(ca) && simpleChain(cb))
                return rasterHinge(linear, w, h, ca, cb);
        }
    }

    // A detected bow still carries the mask arcs. Those are the warp that
    // was measured against the sine sheet. Points are what the finger edits;
    // once it does, the arcs are cleared and the cubics below take over.
    if (nCurve > 0 && nFold == 0 && outline.ax.size() >= 2 && outline.bx.size() >= 2)
        return skip;

    // A hinge or a cross already returned. What remains still has four
    // corners plus a diamond, a circle, or both: the patch follows those
    // sides. The plain homography is only the four squares by themselves.
    if (nCorner == 4 && (nFold > 0 || nCurve > 0)) {
        DocumentScan ruled = rasterRuled(linear, w, h, pts);
        if (ruled.found) return ruled;
    }

    if (nFold > 0 || nCurve > 0 || pts.size() != 4)
        return rasterPoly(linear, w, h, pts);
    return skip;
}

void pairSingleFold(std::vector<PagePoint>& pts) {
    if (pts.size() < 4) return;
    int nFold = 0, fi = -1;
    for (int i = 0; i < (int)pts.size(); ++i) {
        if (pts[i].kind != PagePointKind::Fold) continue;
        ++nFold;
        fi = i;
    }
    if (nFold != 1) return;
    const int n = (int)pts.size();
    const Pt v = pp(pts[fi]);
    const Pt prev = pp(pts[(fi + n - 1) % n]);
    const Pt next = pp(pts[(fi + 1) % n]);
    auto norm2 = [](Pt d) {
        const float L = std::sqrt(d.x * d.x + d.y * d.y);
        if (L < 1e-3f) return Pt{0, 0};
        return Pt{d.x / L, d.y / L};
    };
    const Pt u1 = norm2({prev.x - v.x, prev.y - v.y});
    const Pt u2 = norm2({next.x - v.x, next.y - v.y});
    Pt bis = norm2({u1.x + u2.x, u1.y + u2.y});
    if (std::fabs(bis.x) + std::fabs(bis.y) < 1e-3f) return;
    auto insideStep = [&](Pt dir) {
        return polyHas(pts, v.x + dir.x * 6.0f, v.y + dir.y * 6.0f);
    };
    if (!insideStep(bis)) bis = {-bis.x, -bis.y};
    if (!insideStep(bis)) return;

    float best = 1e9f;
    int edge = -1;
    Pt hit{};
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        if (i == fi || j == fi) continue;
        const Pt a = pp(pts[i]), b = pp(pts[j]);
        const Pt e{b.x - a.x, b.y - a.y};
        const float den = bis.x * e.y - bis.y * e.x;
        if (std::fabs(den) < 1e-4f) continue;
        const float t = ((a.x - v.x) * e.y - (a.y - v.y) * e.x) / den;
        const float u = ((a.x - v.x) * bis.y - (a.y - v.y) * bis.x) / den;
        if (t < 8.0f || u <= 0.04f || u >= 0.96f || t >= best) continue;
        best = t;
        edge = i;
        hit = {v.x + bis.x * t, v.y + bis.y * t};
    }
    if (edge < 0) return;
    pts.insert(pts.begin() + edge + 1, makePoint(PagePointKind::Fold, hit));
}

DocumentScan flattenOutline(const float* linear, uint32_t w, uint32_t h,
                            const DocumentOutline& outline) {
    DocumentScan out;
    if (!linear || w < 2 || h < 2) {
        out.error = "Image is too small to scan";
        return out;
    }
    if (!outline.points.empty()) {
        DocumentScan via = flattenFromPoints(linear, w, h, outline);
        if (via.found || !via.error.empty()) return via;
    }
    if (outline.shape == DocumentShape::Quad && outline.x.size() == 4) {
        DocumentQuad q;
        for (int i = 0; i < 4; ++i) { q.x[i] = outline.x[i]; q.y[i] = outline.y[i]; }
        return flattenDocument(linear, w, h, q);
    }
    if (outline.shape == DocumentShape::Curve) {
        const ArcChain a = makeArc(outline.ax, outline.ay);
        const ArcChain b = makeArc(outline.bx, outline.by);
        if (a.p.size() < 2 || b.p.size() < 2) {
            out.error = "No document found";
            return out;
        }
        int ow = (int)std::lround(std::max(a.total, b.total));
        float span = 0;
        int ns = 0;
        for (int i = 0; i <= 8; ++i) {
            const float t = (float)i / 8.0f;
            span += dist(arcAt(a, t), arcAt(b, t));
            ++ns;
        }
        int oh = (int)std::lround(span / (float)ns);
        ow = std::clamp(ow, 16, (int)w * 2);
        oh = std::clamp(oh, 16, (int)h * 2);
        const double budget = (double)w * (double)h * 1.25;
        if ((double)ow * oh > budget) {
            const double s = std::sqrt(budget / ((double)ow * oh));
            ow = std::max(16, (int)std::lround(ow * s));
            oh = std::max(16, (int)std::lround(oh * s));
        }
        out.linear.assign((size_t)ow * oh * 4, 1.0f);
        for (int y = 0; y < oh; ++y) {
            const float s = oh == 1 ? 0.0f : (float)y / (float)(oh - 1);
            for (int x = 0; x < ow; ++x) {
                const float t = ow == 1 ? 0.0f : (float)x / (float)(ow - 1);
                const Pt pa = arcAt(a, t), pb = arcAt(b, t);
                const float u = pa.x + (pb.x - pa.x) * s;
                const float v = pa.y + (pb.y - pa.y) * s;
                float* d = out.linear.data() + ((size_t)y * ow + x) * 4;
                for (int c = 0; c < 3; ++c)
                    d[c] = sampleBicubic(linear, w, h, c, u, v);
                d[3] = 1.0f;
            }
        }
        out.found = true;
        out.w = (uint32_t)ow;
        out.h = (uint32_t)oh;
        return out;
    }

    if (outline.x.size() < 3 || outline.x.size() != outline.y.size()) {
        out.error = "No document found";
        return out;
    }
    float minx = outline.x[0], maxx = outline.x[0];
    float miny = outline.y[0], maxy = outline.y[0];
    for (size_t i = 1; i < outline.x.size(); ++i) {
        minx = std::min(minx, outline.x[i]); maxx = std::max(maxx, outline.x[i]);
        miny = std::min(miny, outline.y[i]); maxy = std::max(maxy, outline.y[i]);
    }
    int ow = std::max(16, (int)std::lround(maxx - minx + 1));
    int oh = std::max(16, (int)std::lround(maxy - miny + 1));
    ow = std::min(ow, (int)w * 2);
    oh = std::min(oh, (int)h * 2);
    const double budget = (double)w * (double)h * 1.25;
    if ((double)ow * oh > budget) {
        const double s = std::sqrt(budget / ((double)ow * oh));
        ow = std::max(16, (int)std::lround(ow * s));
        oh = std::max(16, (int)std::lround(oh * s));
    }
    out.linear.assign((size_t)ow * oh * 4, 1.0f);
    for (int y = 0; y < oh; ++y) {
        const float sy = miny + (oh == 1 ? 0.0f : (float)y / (float)(oh - 1) * (maxy - miny));
        for (int x = 0; x < ow; ++x) {
            const float sx = minx + (ow == 1 ? 0.0f : (float)x / (float)(ow - 1) * (maxx - minx));
            if (!pointInPoly(sx, sy, outline.x, outline.y)) continue;
            float* d = out.linear.data() + ((size_t)y * ow + x) * 4;
            for (int c = 0; c < 3; ++c)
                d[c] = sampleBicubic(linear, w, h, c, sx, sy);
            d[3] = 1.0f;
        }
    }
    out.found = true;
    out.w = (uint32_t)ow;
    out.h = (uint32_t)oh;
    return out;
}

DocumentScan scanDocument(const float* linear, uint32_t w, uint32_t h) {
    const DocumentDetect found = detectDocument(linear, w, h);
    if (!found.found) {
        DocumentScan out;
        out.error = found.error.empty() ? "No document found" : found.error;
        return out;
    }
    if (found.outline.shape == DocumentShape::Quad)
        return flattenDocument(linear, w, h, found.quad);
    DocumentScan scan = flattenOutline(linear, w, h, found.outline);
    scan.quad = found.quad;
    return scan;
}
