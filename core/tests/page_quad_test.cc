// Scored quad: text, a seam that continues, a shadow bar, a faint border.
// The Kalman case is a synthetic track — nothing here opens a camera.
#include "core/page_quad.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    ++fails; } } while (0)

static bool inQuad(float x, float y, const float* px, const float* py) {
    int pos = 0, neg = 0;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) % 4;
        const float c = (px[j] - px[i]) * (y - py[i]) - (py[j] - py[i]) * (x - px[i]);
        if (c > 1.0f) ++pos;
        else if (c < -1.0f) ++neg;
    }
    return pos == 0 || neg == 0;
}

static void fill(std::vector<float>& img, uint32_t W, uint32_t H, float Y) {
    img.assign((size_t)W * H * 4, Y);
    for (size_t i = 0; i < (size_t)W * H; ++i) img[i * 4 + 3] = 1.0f;
}

static void paint(std::vector<float>& img, uint32_t W, uint32_t H,
                  const float* px, const float* py, float Y) {
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            if (!inQuad((float)x, (float)y, px, py)) continue;
            float* p = img.data() + ((size_t)y * W + x) * 4;
            p[0] = p[1] = p[2] = Y;
        }
    }
}

static float iouOf(const PageQuad& q, const float* px, const float* py,
                   uint32_t W, uint32_t H) {
    int inter = 0, uni = 0;
    for (uint32_t y = 0; y < H; y += 2) {
        for (uint32_t x = 0; x < W; x += 2) {
            const bool a = inQuad((float)x, (float)y, q.x, q.y);
            const bool b = inQuad((float)x, (float)y, px, py);
            if (a || b) ++uni;
            if (a && b) ++inter;
        }
    }
    return uni ? (float)inter / (float)uni : 0;
}

static float meanFrac(const PageQuad& q, const float* px, const float* py) {
    const float d0 = std::hypot(px[0] - px[2], py[0] - py[2]);
    const float d1 = std::hypot(px[1] - px[3], py[1] - py[3]);
    const float diag = std::max(0.5f * (d0 + d1), 1.0f);
    float e = 0;
    for (int i = 0; i < 4; ++i)
        e += std::hypot(q.x[i] - px[i], q.y[i] - py[i]);
    return (e / 4.0f) / diag;
}

static void report(const char* name, const PageQuad& q, const float* px, const float* py,
                   uint32_t W, uint32_t H) {
    const float iou = q.ok ? iouOf(q, px, py, W, H) : 0;
    const float frac = q.ok ? meanFrac(q, px, py) : 1;
    std::fprintf(stderr, "%s ok %d score %.3f iou %.3f corner %.3f diag\n",
                 name, (int)q.ok, q.score, iou, frac);
    if (q.ok) {
        std::fprintf(stderr, "  corners");
        for (int i = 0; i < 4; ++i)
            std::fprintf(stderr, " (%.1f,%.1f)", q.x[i], q.y[i]);
        std::fprintf(stderr, "\n");
    }
    CHECK(q.ok);
    CHECK(iou >= 0.95f);
    // A 1° line bin is a couple of pixels on these frames, about 2% of the
    // diagonal. Locking onto a rule or the shadow bar misses by far more.
    CHECK(frac < 0.021f);
}

static float stdev(const std::vector<float>& v) {
    double m = 0;
    for (float x : v) m += x;
    m /= (double)v.size();
    double s = 0;
    for (float x : v) s += ((double)x - m) * ((double)x - m);
    return (float)std::sqrt(s / (double)v.size());
}

int main() {
    const uint32_t W = 320, H = 240;

    // Rules across the page. The quad is the paper, not a line of type.
    {
        const float px[4] = {40, 280, 280, 40};
        const float py[4] = {30, 30, 210, 210};
        std::vector<float> img;
        fill(img, W, H, 0.25f);
        paint(img, W, H, px, py, 0.88f);
        for (int y = 56; y < 190; y += 16) {
            for (int x = 64; x < 256; ++x) {
                float* p = img.data() + ((size_t)y * W + x) * 4;
                p[0] = p[1] = p[2] = 0.05f;
                if (y + 1 < (int)H) {
                    float* q = img.data() + ((size_t)(y + 1) * W + x) * 4;
                    q[0] = q[1] = q[2] = 0.05f;
                }
            }
        }
        const PageQuad q = findPageQuad(img.data(), W, H);
        report("text", q, px, py, W, H);
    }

    // The page's bottom edge stops at the corners. A darker seam below it
    // runs off both sides of the frame.
    {
        const float px[4] = {70, 250, 250, 70};
        const float py[4] = {40, 40, 170, 170};
        std::vector<float> img;
        fill(img, W, H, 0.40f);
        paint(img, W, H, px, py, 0.82f);
        for (int y = 188; y <= 191; ++y)
            for (int x = 0; x < (int)W; ++x) {
                float* p = img.data() + ((size_t)y * W + x) * 4;
                p[0] = p[1] = p[2] = 0.05f;
            }
        const PageQuad q = findPageQuad(img.data(), W, H);
        report("seam", q, px, py, W, H);
    }

    // A dark bar left of the page, taller than the page, stronger than the
    // paper edge. The quad stays on the paper.
    {
        const float px[4] = {90, 250, 250, 90};
        const float py[4] = {40, 40, 200, 200};
        std::vector<float> img;
        fill(img, W, H, 0.42f);
        paint(img, W, H, px, py, 0.78f);
        for (int y = 0; y < (int)H; ++y)
            for (int x = 48; x <= 62; ++x) {
                float* p = img.data() + ((size_t)y * W + x) * 4;
                p[0] = p[1] = p[2] = 0.02f;
            }
        const PageQuad q = findPageQuad(img.data(), W, H);
        report("shadow", q, px, py, W, H);
    }

    // White page, white desk, step of 12/255. The floor on the cutoff has
    // to let that border in.
    {
        const float px[4] = {50, 270, 270, 50};
        const float py[4] = {36, 36, 204, 204};
        const float desk = 0.86f;
        const float page = desk + 12.0f / 255.0f;
        std::vector<float> img;
        fill(img, W, H, desk);
        paint(img, W, H, px, py, page);
        const PageQuad q = findPageQuad(img.data(), W, H);
        report("white", q, px, py, W, H);
    }

    // Page nearly fills the frame. A heavy heading must not become the top.
    {
        const float px[4] = {8, 311, 311, 8};
        const float py[4] = {8, 8, 231, 231};
        std::vector<float> img;
        fill(img, W, H, 0.30f);
        paint(img, W, H, px, py, 0.90f);
        for (int y = 28; y <= 46; ++y)
            for (int x = 24; x <= 290; ++x) {
                float* p = img.data() + ((size_t)y * W + x) * 4;
                p[0] = p[1] = p[2] = 0.04f;
            }
        for (int row = 0; row < 6; ++row) {
            const int y = 70 + row * 22;
            for (int x = 24; x <= 280; ++x) {
                float* p = img.data() + ((size_t)y * W + x) * 4;
                p[0] = p[1] = p[2] = 0.06f;
            }
        }
        const PageQuad q = findPageQuad(img.data(), W, H);
        report("heading", q, px, py, W, H);
    }

    // A step under the cutoff floor is not a page.
    {
        const float px[4] = {50, 270, 270, 50};
        const float py[4] = {36, 36, 204, 204};
        std::vector<float> img;
        fill(img, W, H, 0.90f);
        paint(img, W, H, px, py, 0.90f + 3.0f / 255.0f);
        const PageQuad q = findPageQuad(img.data(), W, H);
        std::fprintf(stderr, "below-floor ok %d score %.3f\n", (int)q.ok, q.score);
        CHECK(!q.ok);
    }

    // The shadow rectangle is darker inside than the desk around it.
    // The page beside it still scores.
    {
        const float px[4] = {40, 150, 150, 40};
        const float py[4] = {30, 30, 200, 200};
        const float sx[4] = {180, 300, 300, 180};
        const float sy[4] = {40, 40, 190, 190};
        std::vector<float> img;
        fill(img, W, H, 0.55f);
        paint(img, W, H, px, py, 0.88f);
        paint(img, W, H, sx, sy, 0.12f);
        const PageQuad page = scorePageQuad(img.data(), W, H, px, py);
        const PageQuad shadow = scorePageQuad(img.data(), W, H, sx, sy);
        std::fprintf(stderr, "polarity page %d %.3f shadow %d %.3f\n",
                     (int)page.ok, page.score, (int)shadow.ok, shadow.score);
        CHECK(page.ok);
        CHECK(!shadow.ok);
        const PageQuad found = findPageQuad(img.data(), W, H);
        report("two-rects", found, px, py, W, H);
    }

    // Still hold, then a jump the gate must refuse, then eight misses.
    {
        PageTrack track;
        track.setLongEdge(1000.0f);
        const float baseX[4] = {100, 400, 420, 80};
        const float baseY[4] = {80, 90, 500, 480};
        std::vector<float> raw, filt;
        PageQuad m;
        m.ok = true;
        m.score = 1.0f;
        for (int i = 0; i < 4; ++i) {
            m.x[i] = baseX[i];
            m.y[i] = baseY[i];
        }
        CHECK(track.update(m, 1.0f / 30.0f));
        for (int frame = 1; frame <= 30; ++frame) {
            const float n = 2.0f * std::sin(0.7f * (float)frame);
            const float n2 = 2.0f * std::cos(0.5f * (float)frame);
            for (int i = 0; i < 4; ++i) {
                m.x[i] = baseX[i] + n;
                m.y[i] = baseY[i] + n2;
            }
            raw.push_back(m.x[0]);
            CHECK(track.update(m, 1.0f / 30.0f));
            const PageQuad p = track.predict(0);
            filt.push_back(p.x[0]);
            CHECK(p.ok);
        }
        const float rs = stdev(raw), fs = stdev(filt);
        std::fprintf(stderr, "kalman std raw %.3f filtered %.3f\n", rs, fs);
        CHECK(fs < rs);

        for (int i = 0; i < 4; ++i) {
            m.x[i] = baseX[i] + 200.0f;
            m.y[i] = baseY[i];
        }
        CHECK(!track.update(m, 1.0f / 30.0f));
        CHECK(track.locked());
        const PageQuad held = track.predict(0);
        CHECK(std::fabs(held.x[0] - baseX[0]) < 15.0f);

        bool still = true;
        for (int k = 0; k < 7; ++k) {
            const bool took = track.update(m, 1.0f / 30.0f);
            still = still && !took;
        }
        CHECK(still);
        CHECK(!track.locked());

        PageTrack frozen;
        frozen.setLongEdge(1000.0f);
        for (int i = 0; i < 4; ++i) {
            m.x[i] = baseX[i];
            m.y[i] = baseY[i];
        }
        m.score = 1;
        CHECK(frozen.update(m, 1.0f / 30.0f));
        frozen.freeze();
        for (int i = 0; i < 4; ++i) m.x[i] = baseX[i] + 30.0f;
        CHECK(frozen.update(m, 1.0f / 30.0f));
        const PageQuad stay = frozen.predict(1.0f);
        CHECK(std::fabs(stay.x[0] - baseX[0]) < 1.0f);
        std::fprintf(stderr, "kalman gate and freeze ok\n");
    }

    if (fails) {
        std::fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    std::fprintf(stderr, "page_quad_test ok\n");
    return 0;
}
