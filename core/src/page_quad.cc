// page_quad.cc — oriented edges, Hough lines, scored quad.
//
// A fixed-percentile edge map prices the cutoff off the strongest print in
// the frame, and a brightness mask follows a shadow. The cutoff here is
// clamped, the two directions are searched apart, and a quad whose border
// runs on into the table loses to one whose sides actually end.

#include "core/page_quad.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr float kPi = 3.14159265f;
constexpr float kMinMag = 8.0f / 255.0f;
constexpr float kMaxMag = 30.0f / 255.0f;

struct Pt {
    float x, y;
};

struct Line2 {
    float theta = 0;
    float rho = 0;
    float ct = 0;
    float st = 0;
    int votes = 0;
    int span = 0;
};

struct Maps {
    int sw = 0, sh = 0;
    float xScale = 1;   // source x = det x * xScale
    float yScale = 1;
    float cutoff = kMinMag;
    float ref = 0.1f;
    // `tone` is the unblurred luma. Polarity reads it; `y` is the blurred
    // copy Sobel runs on, so a one-pixel step is a ridge rather than a spike.
    std::vector<float> y, tone, r, g, b, mag;
    std::vector<uint8_t> horiz, vert;
};

float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

float percentile(std::vector<float> v, double frac) {
    if (v.empty()) return 0;
    const size_t i = (size_t)std::clamp(frac * (double)(v.size() - 1), 0.0, (double)(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)i, v.end());
    return v[i];
}

void boxDown(const float* rgba, uint32_t w, uint32_t h, int longEdge, Maps& m) {
    const int srcLong = (int)std::max(w, h);
    const int detLong = std::max(16, std::min(longEdge, srcLong));
    const float scale = (float)srcLong / (float)detLong;
    m.sw = std::max(16, (int)std::lround((float)w / scale));
    m.sh = std::max(16, (int)std::lround((float)h / scale));
    m.xScale = (float)w / (float)m.sw;
    m.yScale = (float)h / (float)m.sh;
    const size_t n = (size_t)m.sw * m.sh;
    m.y.assign(n, 0);
    m.r.assign(n, 0);
    m.g.assign(n, 0);
    m.b.assign(n, 0);
    for (int y = 0; y < m.sh; ++y) {
        const int y0 = std::min((int)((float)y * m.yScale), (int)h - 1);
        const int y1 = std::max(y0 + 1, std::min((int)((float)(y + 1) * m.yScale), (int)h));
        for (int x = 0; x < m.sw; ++x) {
            const int x0 = std::min((int)((float)x * m.xScale), (int)w - 1);
            const int x1 = std::max(x0 + 1, std::min((int)((float)(x + 1) * m.xScale), (int)w));
            double ar = 0, ag = 0, ab = 0;
            int cnt = 0;
            for (int py = y0; py < y1; ++py) {
                const float* row = rgba + (size_t)py * w * 4;
                for (int px = x0; px < x1; ++px) {
                    const float* p = row + (size_t)px * 4;
                    ar += p[0]; ag += p[1]; ab += p[2];
                    ++cnt;
                }
            }
            if (!cnt) continue;
            const size_t i = (size_t)y * m.sw + x;
            m.r[i] = (float)(ar / cnt);
            m.g[i] = (float)(ag / cnt);
            m.b[i] = (float)(ab / cnt);
            m.y[i] = 0.2126f * m.r[i] + 0.7152f * m.g[i] + 0.0722f * m.b[i];
        }
    }
}

void gaussian(std::vector<float>& img, int w, int h) {
    float k[7];
    float sum = 0;
    for (int i = -3; i <= 3; ++i) {
        k[i + 3] = std::exp(-0.5f * (float)(i * i));
        sum += k[i + 3];
    }
    for (float& v : k) v /= sum;
    std::vector<float> tmp(img.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float a = 0;
            for (int i = -3; i <= 3; ++i) {
                const int xx = std::clamp(x + i, 0, w - 1);
                a += k[i + 3] * img[(size_t)y * w + xx];
            }
            tmp[(size_t)y * w + x] = a;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float a = 0;
            for (int i = -3; i <= 3; ++i) {
                const int yy = std::clamp(y + i, 0, h - 1);
                a += k[i + 3] * tmp[(size_t)yy * w + x];
            }
            img[(size_t)y * w + x] = a;
        }
    }
}

float at(const std::vector<float>& im, int w, int h, int x, int y) {
    x = std::clamp(x, 0, w - 1);
    y = std::clamp(y, 0, h - 1);
    return im[(size_t)y * w + x];
}

// Mean of |Gx| / |Gy| over a 16 px window. A border lights one axis.
// Print lights both, and those pixels leave the maps.
void suppressText(const std::vector<float>& ax, const std::vector<float>& ay,
                  int w, int h, float cutoff, std::vector<uint8_t>& drop) {
    const int W = w + 1, H = h + 1;
    std::vector<double> ix((size_t)W * H), iy((size_t)W * H);
    for (int y = 0; y < h; ++y) {
        double rx = 0, ry = 0;
        for (int x = 0; x < w; ++x) {
            rx += std::fabs(ax[(size_t)y * w + x]);
            ry += std::fabs(ay[(size_t)y * w + x]);
            ix[(size_t)(y + 1) * W + (x + 1)] = ix[(size_t)y * W + (x + 1)] + rx;
            iy[(size_t)(y + 1) * W + (x + 1)] = iy[(size_t)y * W + (x + 1)] + ry;
        }
    }
    auto rect = [&](const std::vector<double>& ii, int x0, int y0, int x1, int y1) {
        return ii[(size_t)y1 * W + x1] - ii[(size_t)y0 * W + x1]
             - ii[(size_t)y1 * W + x0] + ii[(size_t)y0 * W + x0];
    };
    const int rad = 8;
    drop.assign((size_t)w * h, 0);
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - rad), y1 = std::min(h, y + rad);
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - rad), x1 = std::min(w, x + rad);
            const double area = (double)(x1 - x0) * (double)(y1 - y0);
            if (area < 1) continue;
            const float mx = (float)(rect(ix, x0, y0, x1, y1) / area);
            const float my = (float)(rect(iy, x0, y0, x1, y1) / area);
            if (mx > cutoff && my > cutoff) drop[(size_t)y * w + x] = 1;
        }
    }
}

bool buildMaps(const float* rgba, uint32_t w, uint32_t h, int longEdge, Maps& m) {
    if (!rgba || w < 16 || h < 16) return false;
    boxDown(rgba, w, h, longEdge, m);
    m.tone = m.y;
    gaussian(m.y, m.sw, m.sh);
    const int sw = m.sw, sh = m.sh;
    const size_t n = (size_t)sw * sh;
    std::vector<float> gx(n), gy(n);
    m.mag.assign(n, 0);
    for (int y = 1; y < sh - 1; ++y) {
        for (int x = 1; x < sw - 1; ++x) {
            const float gxv =
                -at(m.y, sw, sh, x - 1, y - 1) + at(m.y, sw, sh, x + 1, y - 1) +
                -2 * at(m.y, sw, sh, x - 1, y) + 2 * at(m.y, sw, sh, x + 1, y) +
                -at(m.y, sw, sh, x - 1, y + 1) + at(m.y, sw, sh, x + 1, y + 1);
            const float gyv =
                -at(m.y, sw, sh, x - 1, y - 1) - 2 * at(m.y, sw, sh, x, y - 1) - at(m.y, sw, sh, x + 1, y - 1) +
                 at(m.y, sw, sh, x - 1, y + 1) + 2 * at(m.y, sw, sh, x, y + 1) + at(m.y, sw, sh, x + 1, y + 1);
            const size_t i = (size_t)y * sw + x;
            gx[i] = gxv;
            gy[i] = gyv;
            m.mag[i] = std::sqrt(gxv * gxv + gyv * gyv);
        }
    }
    m.cutoff = clampf(percentile(m.mag, 0.90), kMinMag, kMaxMag);
    // A clean border peaks well above the cutoff. Dividing by the cutoff
    // alone makes every real edge score many times 1, and the weights stop
    // meaning anything. `ref` is that peak, floored so a faint border can
    // still reach 1.
    float peak = percentile(m.mag, 0.995f);
    m.ref = std::max(m.cutoff * 4.0f, std::min(peak, 0.80f));
    if (m.ref < 1e-3f) m.ref = m.cutoff * 4.0f;

    std::vector<uint8_t> drop;
    suppressText(gx, gy, sw, sh, m.cutoff, drop);
    m.horiz.assign(n, 0);
    m.vert.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (drop[i]) continue;
        const float ax = std::fabs(gx[i]), ay = std::fabs(gy[i]);
        if (ay > ax && ay > m.cutoff) m.horiz[i] = 1;
        else if (ax >= ay && ax > m.cutoff) m.vert[i] = 1;
    }
    return true;
}

// 1° bins tilt a long side by a couple of pixels. The inliers already know
// a better angle; keep the Hough line when the fit wanders off it.
void refineLine(Line2& L, const std::vector<uint8_t>& map, int sw, int sh) {
    double mx = 0, my = 0;
    int n = 0;
    const bool alongX = std::fabs(L.st) >= std::fabs(L.ct);
    const int steps = alongX ? sw : sh;
    for (int s = 0; s < steps; ++s) {
        const float a = alongX ? (float)s : 0;
        const float b = alongX ? 0 : (float)s;
        const float denom = alongX ? L.st : L.ct;
        if (std::fabs(denom) < 1e-6f) continue;
        const float o = (L.rho - (alongX ? a * L.ct : b * L.st)) / denom;
        const int o0 = (int)std::floor(o);
        for (int d = -1; d <= 1; ++d) {
            const int xx = alongX ? s : o0 + d;
            const int yy = alongX ? o0 + d : s;
            if (xx < 0 || yy < 0 || xx >= sw || yy >= sh) continue;
            if (!map[(size_t)yy * sw + xx]) continue;
            if (std::fabs((float)xx * L.ct + (float)yy * L.st - L.rho) > 1.4f) continue;
            mx += xx;
            my += yy;
            ++n;
            break;
        }
    }
    if (n < 12) return;
    mx /= n;
    my /= n;
    // Keep the Hough angle. A free fit on a short ridge tilts an axis-aligned
    // border by a degree, and that is several pixels at the corners. Sliding
    // ρ onto the inliers is the part that removes the bin error.
    L.rho = L.ct * (float)mx + L.st * (float)my;
}

int lineSupport(const std::vector<uint8_t>& map, int sw, int sh,
                float ct, float st, float rho, int gapLimit) {
    const bool alongX = std::fabs(st) >= std::fabs(ct);
    const int steps = alongX ? sw : sh;
    int run = 0, best = 0, gap = 0;
    for (int s = 0; s < steps; ++s) {
        bool hit = false;
        if (alongX) {
            if (std::fabs(st) < 1e-6f) continue;
            const float y = (rho - (float)s * ct) / st;
            const int y0 = (int)std::floor(y);
            for (int dy = -1; dy <= 1 && !hit; ++dy) {
                const int yy = y0 + dy;
                if (yy < 0 || yy >= sh) continue;
                if (map[(size_t)yy * sw + s] && std::fabs((float)s * ct + (float)yy * st - rho) <= 1.4f)
                    hit = true;
            }
        } else {
            if (std::fabs(ct) < 1e-6f) continue;
            const float x = (rho - (float)s * st) / ct;
            const int x0 = (int)std::floor(x);
            for (int dx = -1; dx <= 1 && !hit; ++dx) {
                const int xx = x0 + dx;
                if (xx < 0 || xx >= sw) continue;
                if (map[(size_t)s * sw + xx] && std::fabs((float)xx * ct + (float)s * st - rho) <= 1.4f)
                    hit = true;
            }
        }
        if (hit) {
            ++run;
            gap = 0;
        } else if (run > 0) {
            ++gap;
            if (gap > gapLimit) {
                best = std::max(best, run);
                run = 0;
                gap = 0;
            }
        }
    }
    return std::max(best, run);
}

std::vector<Line2> houghFamily(const std::vector<uint8_t>& map, int sw, int sh,
                               int theta0, int theta1, float longEdge) {
    std::vector<Line2> lines;
    if (theta1 <= theta0) return lines;
    const float diag = std::sqrt((float)sw * sw + (float)sh * sh);
    const int nRho = (int)std::ceil(diag) * 2 + 8;
    const int rho0 = nRho / 2;
    const int nTheta = theta1 - theta0;
    std::vector<int> acc((size_t)nTheta * nRho, 0);
    std::vector<float> ct(nTheta), st(nTheta);
    for (int t = 0; t < nTheta; ++t) {
        const float th = (float)(theta0 + t) * kPi / 180.0f;
        ct[t] = std::cos(th);
        st[t] = std::sin(th);
    }
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            if (!map[(size_t)y * sw + x]) continue;
            for (int t = 0; t < nTheta; ++t) {
                const float rho = (float)x * ct[t] + (float)y * st[t];
                const int ri = (int)std::lround(rho) + rho0;
                if (ri < 0 || ri >= nRho) continue;
                ++acc[(size_t)t * nRho + ri];
            }
        }
    }
    struct Peak { int t, r, v; };
    std::vector<Peak> peaks;
    const int minVote = std::max(8, (int)std::lround(0.025f * longEdge));
    for (int t = 0; t < nTheta; ++t) {
        for (int r = 2; r < nRho - 2; ++r) {
            const int v = acc[(size_t)t * nRho + r];
            if (v < minVote) continue;
            bool maxed = true;
            for (int dt = -1; dt <= 1 && maxed; ++dt) {
                const int tt = t + dt;
                if (tt < 0 || tt >= nTheta) continue;
                for (int dr = -2; dr <= 2; ++dr) {
                    if (dt == 0 && dr == 0) continue;
                    const int rr = r + dr;
                    if (rr < 0 || rr >= nRho) continue;
                    if (acc[(size_t)tt * nRho + rr] > v) maxed = false;
                }
            }
            if (maxed) peaks.push_back(Peak{t, r, v});
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](Peak a, Peak b) {
        if (a.v != b.v) return a.v > b.v;
        if (a.t != b.t) return a.t < b.t;
        return a.r < b.r;
    });
    const int need = std::max(8, (int)std::lround(0.08f * longEdge));
    const int gap = std::max(4, (int)std::lround(0.04f * longEdge));
    const float mergeRho = std::max(6.0f, 0.02f * longEdge);
    for (const Peak& p : peaks) {
        const float th = (float)(theta0 + p.t);
        const float rho = (float)(p.r - rho0);
        bool dup = false;
        for (const Line2& L : lines) {
            float dth = std::fabs(L.theta - th);
            if (dth > 180.0f) dth = 360.0f - dth;
            if (dth <= 5.0f && std::fabs(L.rho - rho) <= mergeRho) { dup = true; break; }
        }
        if (dup) continue;
        const float rad = th * kPi / 180.0f;
        const float c = std::cos(rad), s = std::sin(rad);
        const int span = lineSupport(map, sw, sh, c, s, rho, gap);
        if (span < need) continue;
        Line2 L;
        L.theta = th;
        L.rho = rho;
        L.ct = c;
        L.st = s;
        L.votes = p.v;
        L.span = span;
        refineLine(L, map, sw, sh);
        lines.push_back(L);
    }
    // Length, then votes. A page border is long; print is a shorter ridge
    // even when each of its pixels votes. Keeping the longest first stops
    // a page of rules from evicting the border.
    std::sort(lines.begin(), lines.end(), [](const Line2& a, const Line2& b) {
        if (a.span != b.span) return a.span > b.span;
        if (a.votes != b.votes) return a.votes > b.votes;
        if (a.theta != b.theta) return a.theta < b.theta;
        return a.rho < b.rho;
    });
    if (lines.size() > 12) lines.resize(12);
    return lines;
}

bool intersect(const Line2& a, const Line2& b, Pt& o) {
    const float det = a.ct * b.st - b.ct * a.st;
    if (std::fabs(det) < 0.34f) return false;   // closer than ~20° to parallel
    o.x = (a.rho * b.st - b.rho * a.st) / det;
    o.y = (a.ct * b.rho - b.ct * a.rho) / det;
    return std::isfinite(o.x) && std::isfinite(o.y);
}

float dist(Pt a, Pt b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

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

bool distinctCorners(const Pt p[4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (dist(p[i], p[j]) < 4.0f) return false;
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

float interiorDeg(Pt prev, Pt at, Pt next) {
    const float ax = prev.x - at.x, ay = prev.y - at.y;
    const float bx = next.x - at.x, by = next.y - at.y;
    const float la = std::hypot(ax, ay), lb = std::hypot(bx, by);
    if (la < 1e-3f || lb < 1e-3f) return 0;
    const float d = clampf((ax * bx + ay * by) / (la * lb), -1.0f, 1.0f);
    return std::acos(d) * (180.0f / kPi);
}

bool geometryOk(const Pt p[4], int sw, int sh, float& angleSum) {
    if (!distinctCorners(p) || !convexQuad(p)) return false;
    const float area = quadArea(p);
    if (area < 0.08f * (float)sw * (float)sh) return false;
    angleSum = 0;
    for (int i = 0; i < 4; ++i) {
        const float deg = interiorDeg(p[(i + 3) % 4], p[i], p[(i + 1) % 4]);
        if (deg < 60.0f || deg > 120.0f) return false;
        angleSum += std::fabs(deg - 90.0f);
        if (p[i].x < -0.10f * (float)sw || p[i].x > 1.10f * (float)sw) return false;
        if (p[i].y < -0.10f * (float)sh || p[i].y > 1.10f * (float)sh) return false;
    }
    return true;
}

float sampleBilinear(const std::vector<float>& im, int w, int h, float x, float y, bool& inside) {
    if (x < 0 || y < 0 || x > (float)(w - 1) || y > (float)(h - 1)) {
        inside = false;
        return 0;
    }
    inside = true;
    const int x0 = (int)std::floor(x);
    const int y0 = (int)std::floor(y);
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float tx = x - (float)x0, ty = y - (float)y0;
    const float a = im[(size_t)y0 * w + x0];
    const float b = im[(size_t)y0 * w + x1];
    const float c = im[(size_t)y1 * w + x0];
    const float d = im[(size_t)y1 * w + x1];
    return a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty;
}

float chromaAt(const Maps& m, float x, float y, bool& inside) {
    bool ir, ig, ib;
    const float r = sampleBilinear(m.r, m.sw, m.sh, x, y, ir);
    const float g = sampleBilinear(m.g, m.sw, m.sh, x, y, ig);
    const float b = sampleBilinear(m.b, m.sw, m.sh, x, y, ib);
    inside = ir && ig && ib;
    if (!inside) return 0;
    const float mx = std::max(r, std::max(g, b));
    const float mn = std::min(r, std::min(g, b));
    return mx > 1e-4f ? (mx - mn) / mx : 0;
}

// Returns false when polarity fails. On success, writes S.
bool scoreDet(const Maps& m, const Pt p[4], float angleSum, float& score) {
    const float sideLen[4] = {
        dist(p[0], p[1]), dist(p[1], p[2]), dist(p[2], p[3]), dist(p[3], p[0])
    };
    const float minSide = std::min(std::min(sideLen[0], sideLen[1]), std::min(sideLen[2], sideLen[3]));
    if (minSide < 8.0f) return false;
    float cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) { cx += p[i].x; cy += p[i].y; }
    cx *= 0.25f; cy *= 0.25f;
    // Median of the inner page, so a rule through the centroid is not the
    // colour the sides are compared with.
    float centerSamps[25];
    int centerN = 0;
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 5; ++i) {
            const float u = 0.30f + 0.10f * (float)i;
            const float v = 0.30f + 0.10f * (float)j;
            const float sx = (1 - u) * ((1 - v) * p[0].x + v * p[3].x) + u * ((1 - v) * p[1].x + v * p[2].x);
            const float sy = (1 - u) * ((1 - v) * p[0].y + v * p[3].y) + u * ((1 - v) * p[1].y + v * p[2].y);
            bool inside = false;
            const float l = sampleBilinear(m.tone, m.sw, m.sh, sx, sy, inside);
            if (inside) centerSamps[centerN++] = l;
        }
    }
    const bool centerInside = centerN >= 8;
    float centerL = 0;
    if (centerInside) {
        std::nth_element(centerSamps, centerSamps + centerN / 2, centerSamps + centerN);
        centerL = centerSamps[centerN / 2];
    }
    float support = 0;
    float extend = 0;
    int agreed = 0, measured = 0;
    int brighterSides = 0, chromaSides = 0;
    float diffs[4];
    int nDiff = 0;
    // Distances in detection pixels. A stroke is brighter on the inside
    // only while the probe sits on the ink. Further out, both sides are the
    // page again. A border stays different at every distance that is still
    // on the image.
    const float probe[6] = {3.0f, 6.0f, 10.0f, 14.0f, 20.0f, 28.0f};

    for (int i = 0; i < 4; ++i) {
        const Pt a = p[i], b = p[(i + 1) % 4];
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float len = sideLen[i];
        float nx = dy / len, ny = -dx / len;
        const float mx = (a.x + b.x) * 0.5f, my = (a.y + b.y) * 0.5f;
        if ((mx - cx) * nx + (my - cy) * ny < 0) { nx = -nx; ny = -ny; }

        float magSum = 0;
        int magN = 0;
        int contrastSteps = 0, flatSteps = 0, steps = 0;
        bool sideBrighter = false, sideChroma = false;
        float bestDiff = 0;
        float insideNear = 0, insideRaw = 0;
        int insideNearN = 0, insideRawN = 0;
        constexpr int kN = 16;
        for (int s = 0; s < kN; ++s) {
            const float t = 0.15f + 0.70f * ((float)s + 0.5f) / (float)kN;
            const float px = a.x + dx * t, py = a.y + dy * t;
            bool onEdge = false;
            const float mv = sampleBilinear(m.mag, m.sw, m.sh, px, py, onEdge);
            if (onEdge) { magSum += mv; ++magN; }
        }
        for (float dist : probe) {
            float inL = 0, outL = 0, inC = 0, outC = 0;
            int inN = 0, outN = 0;
            for (int s = 0; s < kN; ++s) {
                const float t = 0.15f + 0.70f * ((float)s + 0.5f) / (float)kN;
                const float px = a.x + dx * t, py = a.y + dy * t;
                auto take = [&](float sgn, float& lSum, float& cSum, int& cnt) {
                    bool okL = false, okC = false;
                    const float l = sampleBilinear(m.tone, m.sw, m.sh,
                                                   px + sgn * nx * dist, py + sgn * ny * dist, okL);
                    const float c = chromaAt(m, px + sgn * nx * dist, py + sgn * ny * dist, okC);
                    if (!okL || !okC) return;
                    lSum += l;
                    cSum += c;
                    ++cnt;
                    // A rule a few pixels in is not the sheet. Keep it out of
                    // the colour the side is compared with, and fall back to
                    // the raw mean when the strip is ink all the way across.
                    if (sgn < 0.0f && dist <= 6.0f) {
                        insideRaw += l;
                        ++insideRawN;
                        if (!centerInside || std::fabs(l - centerL) <= 0.18f) {
                            insideNear += l;
                            ++insideNearN;
                        }
                    }
                };
                take(-1.0f, inL, inC, inN);
                take(1.0f, outL, outC, outN);
            }
            if (inN < 8 || outN < 8) continue;
            ++steps;
            const float dL = inL / (float)inN - outL / (float)outN;
            const float dC = std::fabs(inC / (float)inN - outC / (float)outN);
            // Either sign counts as a step. A shaded sheet is darker than
            // the desk along one side. A rectangle that is darker on every
            // side, with no chroma step, is a shadow, not a page — that
            // check is on the quad, after every side has been measured.
            if (dL > 0.012f) sideBrighter = true;
            if (dC > 0.04f) sideChroma = true;
            if (std::fabs(dL) > 0.012f || dC > 0.04f) {
                ++contrastSteps;
                bestDiff = std::max(bestDiff, std::max(dL, 0.0f) + dC);
            } else if (centerInside &&
                       std::fabs(inL / (float)inN - centerL) > 0.18f) {
                // The probe landed on ink inside the page. That is not the
                // border giving way to the desk.
            } else {
                ++flatSteps;
            }
        }
        support += magN ? (magSum / (float)magN) / m.ref : 0;

        // Past both corners, half a side. A line that is really the table
        // keeps its gradient there; a side of the page does not.
        float ext = 0;
        int extN = 0;
        const float ux = dx / len, uy = dy / len;
        for (int end = 0; end < 2; ++end) {
            const Pt origin = end == 0 ? a : b;
            const float dir = end == 0 ? -1.0f : 1.0f;
            for (int s = 1; s <= 10; ++s) {
                const float d = dir * (0.05f + 0.45f * (float)s / 10.0f) * len;
                bool inside = false;
                const float mv = sampleBilinear(m.mag, m.sw, m.sh,
                                                origin.x + ux * d, origin.y + uy * d, inside);
                if (!inside) continue;
                ext += mv / m.ref;
                ++extN;
            }
        }
        extend += extN ? ext / (float)extN : 0;

        if (steps > 0) {
            ++measured;
            const float insideL = insideNearN >= 8 ? insideNear / (float)insideNearN
                                 : insideRawN ? insideRaw / (float)insideRawN
                                              : centerL;
            // A shadow bar is brighter just inside and is still the desk.
            const bool likePage = !centerInside || std::fabs(insideL - centerL) < 0.30f;
            // Print returns to the page within the probe. A border does not.
            if (likePage && contrastSteps >= 1 && flatSteps == 0) {
                ++agreed;
                if (sideBrighter) ++brighterSides;
                if (sideChroma) ++chromaSides;
                diffs[nDiff++] = bestDiff;
            }
        }
    }
    // Every side we could measure has to agree. Three-of-four lets a quad
    // keep three real borders and one line of type.
    // A shadow is a step on every side and none of them is the page
    // becoming brighter, or a different colour, than the desk.
    if (measured < 3 || agreed != measured || nDiff < 1) return false;
    if (brighterSides < 1 && chromaSides < 1) return false;
    const int nSort = std::min(nDiff, 4);
    std::sort(diffs, diffs + nSort);
    const float polarity = diffs[nSort / 2];
    support /= 4.0f;
    extend /= 4.0f;
    const float areaTerm = std::log(std::max(quadArea(p), 1.0f) / ((float)m.sw * (float)m.sh));
    score = support + 1.5f * polarity + 0.3f * areaTerm - 0.01f * angleSum - 0.8f * extend;
    return true;
}

PageQuad searchMaps(const Maps& m) {
    PageQuad best;
    const float longEdge = (float)std::max(m.sw, m.sh);
    // Horizontal borders: the normal is vertical, θ around 90°.
    // Vertical borders: the normal lies near the x axis.
    const std::vector<Line2> horiz = houghFamily(m.horiz, m.sw, m.sh, 45, 136, longEdge);
    std::vector<Line2> vert = houghFamily(m.vert, m.sw, m.sh, 0, 46, longEdge);
    const std::vector<Line2> vertB = houghFamily(m.vert, m.sw, m.sh, 135, 180, longEdge);
    vert.insert(vert.end(), vertB.begin(), vertB.end());
    std::sort(vert.begin(), vert.end(), [](const Line2& a, const Line2& b) {
        if (a.span != b.span) return a.span > b.span;
        if (a.votes != b.votes) return a.votes > b.votes;
        return a.theta < b.theta;
    });
    if (vert.size() > 12) vert.resize(12);
    if (horiz.size() < 2 || vert.size() < 2) return best;

    for (size_t h0 = 0; h0 < horiz.size(); ++h0) {
        for (size_t h1 = h0 + 1; h1 < horiz.size(); ++h1) {
            for (size_t v0 = 0; v0 < vert.size(); ++v0) {
                for (size_t v1 = v0 + 1; v1 < vert.size(); ++v1) {
                    Pt c[4];
                    if (!intersect(horiz[h0], vert[v0], c[0])) continue;
                    if (!intersect(horiz[h0], vert[v1], c[1])) continue;
                    if (!intersect(horiz[h1], vert[v0], c[2])) continue;
                    if (!intersect(horiz[h1], vert[v1], c[3])) continue;
                    orderCorners(c);
                    float angleSum = 0;
                    if (!geometryOk(c, m.sw, m.sh, angleSum)) continue;
                    float score = 0;
                    if (!scoreDet(m, c, angleSum, score)) continue;
                    if (!best.ok || score > best.score) {
                        best.ok = true;
                        best.score = score;
                        for (int i = 0; i < 4; ++i) {
                            best.x[i] = c[i].x * m.xScale;
                            best.y[i] = c[i].y * m.yScale;
                        }
                    }
                }
            }
        }
    }
    return best;
}

PageQuad searchImage(const float* rgba, uint32_t w, uint32_t h, int longEdge) {
    Maps m;
    if (!buildMaps(rgba, w, h, longEdge, m)) return {};
    return searchMaps(m);
}

float cornerShift(const PageQuad& a, const PageQuad& b) {
    float s = 0;
    for (int i = 0; i < 4; ++i)
        s = std::max(s, std::hypot(a.x[i] - b.x[i], a.y[i] - b.y[i]));
    return s;
}

}  // namespace

PageQuad findPageQuad(const float* rgba, uint32_t w, uint32_t h) {
    PageQuad coarse = searchImage(rgba, w, h, 480);
    if (!coarse.ok || w < 16 || h < 16) return coarse;

    float minx = coarse.x[0], maxx = coarse.x[0], miny = coarse.y[0], maxy = coarse.y[0];
    for (int i = 1; i < 4; ++i) {
        minx = std::min(minx, coarse.x[i]); maxx = std::max(maxx, coarse.x[i]);
        miny = std::min(miny, coarse.y[i]); maxy = std::max(maxy, coarse.y[i]);
    }
    const float mx = 0.15f * (maxx - minx);
    const float my = 0.15f * (maxy - miny);
    const int x0 = std::clamp((int)std::floor(minx - mx), 0, (int)w - 1);
    const int y0 = std::clamp((int)std::floor(miny - my), 0, (int)h - 1);
    const int x1 = std::clamp((int)std::ceil(maxx + mx), x0 + 1, (int)w);
    const int y1 = std::clamp((int)std::ceil(maxy + my), y0 + 1, (int)h);
    const int cw = x1 - x0, ch = y1 - y0;
    if (cw < 32 || ch < 32) return coarse;
    // The crop is already most of the frame. A second search sees the same
    // edges and cannot place the corners more finely.
    if (cw * ch > (int)w * (int)h * 9 / 10) return coarse;

    std::vector<float> crop((size_t)cw * ch * 4);
    for (int y = 0; y < ch; ++y) {
        const float* src = rgba + ((size_t)(y0 + y) * w + (size_t)x0) * 4;
        std::copy(src, src + (size_t)cw * 4, crop.data() + (size_t)y * cw * 4);
    }
    PageQuad fine = searchImage(crop.data(), (uint32_t)cw, (uint32_t)ch, 640);
    if (!fine.ok) return coarse;
    for (int i = 0; i < 4; ++i) {
        fine.x[i] += (float)x0;
        fine.y[i] += (float)y0;
    }
    const float limit = 0.04f * (float)std::max(w, h);
    if (fine.score >= coarse.score && cornerShift(fine, coarse) <= limit) return fine;
    return coarse;
}

PageQuad scorePageQuad(const float* rgba, uint32_t w, uint32_t h,
                       const float x[4], const float y[4]) {
    PageQuad out;
    Maps m;
    if (!buildMaps(rgba, w, h, 480, m)) return out;
    Pt p[4];
    for (int i = 0; i < 4; ++i) {
        p[i].x = x[i] / m.xScale;
        p[i].y = y[i] / m.yScale;
        out.x[i] = x[i];
        out.y[i] = y[i];
    }
    orderCorners(p);
    for (int i = 0; i < 4; ++i) {
        out.x[i] = p[i].x * m.xScale;
        out.y[i] = p[i].y * m.yScale;
    }
    float angleSum = 0;
    if (!geometryOk(p, m.sw, m.sh, angleSum)) return out;
    float score = 0;
    if (!scoreDet(m, p, angleSum, score)) return out;
    out.ok = true;
    out.score = score;
    return out;
}

namespace {

void identP(double P[16], double px, double py, double pv) {
    for (int i = 0; i < 16; ++i) P[i] = 0;
    P[0] = px; P[5] = py; P[10] = pv; P[15] = pv;
}

}  // namespace

PageTrack::PageTrack() = default;

void PageTrack::setLongEdge(float longEdge) {
    if (longEdge > 1.0f) longEdge_ = longEdge;
}

void PageTrack::freeze() { frozen_ = true; }
void PageTrack::unfreeze() { frozen_ = false; }

void PageTrack::predictCorner(Corner& c, float dt) const {
    if (!(dt > 0)) return;
    const double d = dt;
    const double x = c.s[0] + c.s[2] * d;
    const double y = c.s[1] + c.s[3] * d;
    c.s[0] = x;
    c.s[1] = y;

    double FP[16];
    for (int j = 0; j < 4; ++j) {
        FP[0 * 4 + j] = c.P[0 * 4 + j] + d * c.P[2 * 4 + j];
        FP[1 * 4 + j] = c.P[1 * 4 + j] + d * c.P[3 * 4 + j];
        FP[2 * 4 + j] = c.P[2 * 4 + j];
        FP[3 * 4 + j] = c.P[3 * 4 + j];
    }
    double N[16];
    for (int i = 0; i < 4; ++i) {
        N[i * 4 + 0] = FP[i * 4 + 0] + d * FP[i * 4 + 2];
        N[i * 4 + 1] = FP[i * 4 + 1] + d * FP[i * 4 + 3];
        N[i * 4 + 2] = FP[i * 4 + 2];
        N[i * 4 + 3] = FP[i * 4 + 3];
    }
    const double q = 25.0;
    const double d2 = d * d, d3 = d2 * d, d4 = d2 * d2;
    N[0] += d4 / 4.0 * q;
    N[2] += d3 / 2.0 * q;
    N[8] += d3 / 2.0 * q;
    N[10] += d2 * q;
    N[5] += d4 / 4.0 * q;
    N[7] += d3 / 2.0 * q;
    N[13] += d3 / 2.0 * q;
    N[15] += d2 * q;
    for (int i = 0; i < 16; ++i) c.P[i] = N[i];
}

void PageTrack::correctCorner(Corner& c, float zx, float zy, float measVar) {
    const double r = std::max(measVar, 1e-3f);
    const double S00 = c.P[0] + r, S01 = c.P[1];
    const double S10 = c.P[4], S11 = c.P[5] + r;
    const double det = S00 * S11 - S01 * S10;
    if (std::fabs(det) < 1e-12) return;
    const double i00 = S11 / det, i01 = -S01 / det;
    const double i10 = -S10 / det, i11 = S00 / det;
    double K[8];
    for (int i = 0; i < 4; ++i) {
        const double p0 = c.P[i * 4 + 0];
        const double p1 = c.P[i * 4 + 1];
        K[i * 2 + 0] = p0 * i00 + p1 * i10;
        K[i * 2 + 1] = p0 * i01 + p1 * i11;
    }
    const double ix = (double)zx - c.s[0];
    const double iy = (double)zy - c.s[1];
    for (int i = 0; i < 4; ++i) {
        c.s[i] += K[i * 2 + 0] * ix + K[i * 2 + 1] * iy;
    }
    double NP[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double a = 0;
            for (int k = 0; k < 4; ++k) {
                double ik = (i == k ? 1.0 : 0.0);
                if (k == 0) ik -= K[i * 2 + 0];
                if (k == 1) ik -= K[i * 2 + 1];
                a += ik * c.P[k * 4 + j];
            }
            NP[i * 4 + j] = a;
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            c.P[i * 4 + j] = 0.5 * (NP[i * 4 + j] + NP[j * 4 + i]);
}

bool PageTrack::update(const PageQuad& measured, float dt_s) {
    if (!measured.ok) {
        if (!locked_) return false;
        if (frozen_) return true;
        for (int i = 0; i < 4; ++i) predictCorner(corner_[i], dt_s);
        if (++missed_ >= 8) {
            locked_ = false;
            missed_ = 0;
            return false;
        }
        return false;
    }
    if (!locked_) {
        for (int i = 0; i < 4; ++i) {
            corner_[i] = Corner{};
            corner_[i].s[0] = measured.x[i];
            corner_[i].s[1] = measured.y[i];
            identP(corner_[i].P, 25.0, 25.0, 100.0);
            corner_[i].inited = true;
        }
        locked_ = true;
        missed_ = 0;
        score_ = measured.score;
        return true;
    }
    if (frozen_) return true;

    for (int i = 0; i < 4; ++i) predictCorner(corner_[i], dt_s);
    const float gate = 0.04f * longEdge_;
    bool near = true;
    for (int i = 0; i < 4; ++i) {
        const float d = std::hypot((float)corner_[i].s[0] - measured.x[i],
                                   (float)corner_[i].s[1] - measured.y[i]);
        if (d > gate) near = false;
    }
    if (!near) {
        if (++missed_ >= 8) {
            locked_ = false;
            missed_ = 0;
        }
        return false;
    }
    const float trust = std::max(measured.score, 0.25f);
    const float measVar = (6.0f * 6.0f) / trust;
    for (int i = 0; i < 4; ++i)
        correctCorner(corner_[i], measured.x[i], measured.y[i], measVar);
    missed_ = 0;
    score_ = measured.score;
    return true;
}

PageQuad PageTrack::predict(float dt_s) const {
    PageQuad q;
    q.ok = locked_;
    q.score = score_;
    if (!locked_) return q;
    for (int i = 0; i < 4; ++i) {
        const double d = frozen_ || !(dt_s > 0) ? 0.0 : (double)dt_s;
        q.x[i] = (float)(corner_[i].s[0] + corner_[i].s[2] * d);
        q.y[i] = (float)(corner_[i].s[1] + corner_[i].s[3] * d);
    }
    return q;
}
