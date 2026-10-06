// Page finding, the colour warp, and the document finish.
// Each case prints the measured error and fails on a fixed bound.
#include "core/document_scan.hh"
#include "core/enhance.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    ++fails; } } while (0)

static bool inside(float x, float y, const float* px, const float* py) {
    int pos = 0, neg = 0;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) % 4;
        const float c = (px[j] - px[i]) * (y - py[i]) - (py[j] - py[i]) * (x - px[i]);
        if (c > 0) ++pos;
        else if (c < 0) ++neg;
    }
    return pos == 0 || neg == 0;
}

static bool inTri(float x, float y, float ax, float ay, float bx, float by,
                  float cx, float cy) {
    const float c1 = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    const float c2 = (cx - bx) * (y - by) - (cy - by) * (x - bx);
    const float c3 = (ax - cx) * (y - cy) - (ay - cy) * (x - cx);
    const bool neg = c1 < 0 || c2 < 0 || c3 < 0;
    const bool pos = c1 > 0 || c2 > 0 || c3 > 0;
    return !(neg && pos);
}

static float cornerErr(const DocumentQuad& q, const float* px, const float* py) {
    float e = 0;
    for (int i = 0; i < 4; ++i) {
        e = std::max(e, std::fabs(q.x[i] - px[i]));
        e = std::max(e, std::fabs(q.y[i] - py[i]));
    }
    return e;
}

static void fillPage(std::vector<float>& img, uint32_t W, uint32_t H,
                     const float* px, const float* py,
                     float br, float bg, float bb, float fr, float fg, float fb) {
    img.assign((size_t)W * H * 4, 0.0f);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float* p = img.data() + ((size_t)y * W + x) * 4;
            if (inside((float)x, (float)y, px, py)) {
                p[0] = fr; p[1] = fg; p[2] = fb;
            } else {
                p[0] = br; p[1] = bg; p[2] = bb;
            }
            p[3] = 1.0f;
        }
    }
}

static float hueDeg(float r, float g, float b) {
    const float ang = std::atan2(std::sqrt(3.0f) * (g - b), 2.0f * r - g - b);
    return ang * (180.0f / 3.14159265f);
}

static float hueDelta(float a, float b) {
    float d = std::fabs(a - b);
    if (d > 180.0f) d = 360.0f - d;
    return d;
}

static float chromaOf(float r, float g, float b) {
    const float L = (r + g + b) / 3.0f;
    const float dr = r - L, dg = g - L, db = b - L;
    return std::sqrt(dr * dr + dg * dg + db * db);
}

int main() {
    const uint32_t W = 240, H = 180;
    const float px[4] = {40, 190, 200, 28};
    const float py[4] = {24, 30, 156, 148};
    std::vector<float> img((size_t)W * H * 4, 0.0f);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float* p = img.data() + ((size_t)y * W + x) * 4;
            const bool page = inside((float)x, (float)y, px, py);
            const bool ink = page && (y % 18 == 0) && x > 50 && x < 170;
            p[0] = p[1] = p[2] = page ? (ink ? 0.05f : 0.92f) : 0.06f;
            p[3] = 1.0f;
        }
    }

    const DocumentDetect cleanDet = detectDocument(img.data(), W, H);
    CHECK(cleanDet.found);
    CHECK(cleanDet.outline.shape == DocumentShape::Quad);
    const DocumentScan scan = scanDocument(img.data(), W, H);
    CHECK(scan.found);
    if (!scan.found) {
        std::fprintf(stderr, "  %s\n", scan.error.c_str());
        return 1;
    }
    const float cleanErr = cornerErr(scan.quad, px, py);
    std::fprintf(stderr, "clean max corner err %.2f px -> %ux%u\n",
                 cleanErr, scan.w, scan.h);
    CHECK(cleanErr < 4.0f);
    CHECK(scan.w > 120 && scan.w < 220);
    CHECK(scan.h > 90 && scan.h < 180);

    double mid = 0;
    int n = 0, dark = 0;
    for (uint32_t y = scan.h / 5; y < scan.h * 4 / 5; ++y) {
        for (uint32_t x = scan.w / 5; x < scan.w * 4 / 5; ++x) {
            const float g = scan.linear[((size_t)y * scan.w + x) * 4 + 1];
            mid += g;
            ++n;
            if (g < 0.3f) ++dark;
        }
    }
    CHECK(n > 0 && mid / n > 0.7);
    CHECK(dark > 10);

    // A speck of paper is not a page.
    std::vector<float> flat((size_t)64 * 64 * 4, 0.05f);
    for (int y = 4; y < 10; ++y)
        for (int x = 4; x < 10; ++x) {
            float* p = flat.data() + ((size_t)y * 64 + x) * 4;
            p[0] = p[1] = p[2] = 0.9f; p[3] = 1.0f;
        }
    const DocumentScan none = scanDocument(flat.data(), 64, 64);
    CHECK(!none.found);

    // Shadow across the sheet. The dark side has to stay inside the quad.
    {
        const float sx[4] = {30, 200, 210, 36};
        const float sy[4] = {28, 34, 160, 150};
        std::vector<float> sh(img.size(), 0.0f);
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                float* p = sh.data() + ((size_t)y * W + x) * 4;
                if (inside((float)x, (float)y, sx, sy)) {
                    const float Y = 0.95f + (0.40f - 0.95f) * ((float)x / (float)(W - 1));
                    p[0] = p[1] = p[2] = Y;
                } else {
                    p[0] = p[1] = p[2] = 0.08f;
                }
                p[3] = 1.0f;
            }
        }
        const DocumentScan s = scanDocument(sh.data(), W, H);
        CHECK(s.found);
        const float err = s.found ? cornerErr(s.quad, sx, sy) : 1e9f;
        std::fprintf(stderr, "shadow max corner err %.2f px\n", err);
        CHECK(err < 6.0f);
        if (s.found && s.w > 8 && s.h > 8) {
            double left = 0, right = 0, midv = 0;
            int nl = 0, nr = 0, nm = 0;
            for (uint32_t y = s.h / 5; y < s.h * 4 / 5; ++y) {
                for (uint32_t x = 2; x < s.w / 4; ++x) {
                    left += s.linear[((size_t)y * s.w + x) * 4 + 1];
                    ++nl;
                }
                for (uint32_t x = s.w * 3 / 4; x + 2 < s.w; ++x) {
                    right += s.linear[((size_t)y * s.w + x) * 4 + 1];
                    ++nr;
                }
                for (uint32_t x = s.w / 5; x < s.w * 4 / 5; ++x) {
                    midv += s.linear[((size_t)y * s.w + x) * 4 + 1];
                    ++nm;
                }
            }
            const float l = nl ? (float)(left / nl) : 0;
            const float r = nr ? (float)(right / nr) : 0;
            const float m = nm ? (float)(midv / nm) : 0;
            std::fprintf(stderr, "shadow luma left %.3f right %.3f mid %.3f\n", l, r, m);
            CHECK(m > 0.5f);
            CHECK(r > 0.28f);
            CHECK(l > r);
        }
    }

    // A bright spike on the top-right corner. Angle picking snaps to the tip.
    {
        const uint32_t FW = 320, FH = 240;
        const float fx[4] = {40, 250, 242, 48};
        const float fy[4] = {36, 40, 200, 190};
        const float tax = 210, tay = 38, tbx = 248, tby = 42, tcx = 300, tcy = 18;
        std::vector<float> fg;
        fillPage(fg, FW, FH, fx, fy, 0.07f, 0.07f, 0.07f, 0.90f, 0.90f, 0.90f);
        for (uint32_t y = 0; y < FH; ++y) {
            for (uint32_t x = 0; x < FW; ++x) {
                if (!inTri((float)x, (float)y, tax, tay, tbx, tby, tcx, tcy)) continue;
                float* p = fg.data() + ((size_t)y * FW + x) * 4;
                p[0] = p[1] = p[2] = 0.93f; p[3] = 1.0f;
            }
        }
        const DocumentScan s = scanDocument(fg.data(), FW, FH);
        CHECK(s.found);
        const float err = s.found ? cornerErr(s.quad, fx, fy) : 1e9f;
        std::fprintf(stderr, "spike max corner err %.2f px\n", err);
        CHECK(err < 6.0f);
        if (s.found) {
            for (int i = 0; i < 4; ++i) {
                const float dx = s.quad.x[i] - tcx, dy = s.quad.y[i] - tcy;
                CHECK(dx * dx + dy * dy > 15.0f * 15.0f);
            }
        }
    }

    // Desk around the page, split by a dark seam so the two must not merge.
    {
        const uint32_t DW = 320, DH = 240;
        std::vector<float> desk((size_t)DW * DH * 4, 0.0f);
        for (uint32_t y = 0; y < DH; ++y) {
            for (uint32_t x = 0; x < DW; ++x) {
                float* p = desk.data() + ((size_t)y * DW + x) * 4;
                const bool page = x >= 48 && x < 272 && y >= 40 && y < 200;
                const bool seam = !page && x >= 32 && x < 288 && y >= 24 && y < 216;
                float Y = 0.45f;
                if (page) Y = 0.82f;
                else if (seam) Y = 0.08f;
                p[0] = p[1] = p[2] = Y;
                p[3] = 1.0f;
            }
        }
        const float dx[4] = {48, 271, 271, 48};
        const float dy[4] = {40, 40, 199, 199};
        const DocumentScan s = scanDocument(desk.data(), DW, DH);
        CHECK(s.found);
        const float err = s.found ? cornerErr(s.quad, dx, dy) : 1e9f;
        std::fprintf(stderr, "desk max corner err %.2f px\n", err);
        CHECK(err < 6.0f);
    }

    // Shaded page on yellow wood. The dark side of the sheet is about as
    // bright as the table, and a couple of pale grain spikes leave the page.
    // Luminance alone cuts the shadow or swallows the wood; the outline has
    // to stay the paper's four corners.
    {
        const uint32_t WW = 320, WH = 240;
        const float wx[4] = {48, 268, 274, 42};
        const float wy[4] = {44, 38, 196, 204};
        std::vector<float> wood((size_t)WW * WH * 4, 0.0f);
        for (uint32_t y = 0; y < WH; ++y) {
            for (uint32_t x = 0; x < WW; ++x) {
                float* p = wood.data() + ((size_t)y * WW + x) * 4;
                if (inside((float)x, (float)y, wx, wy)) {
                    const float t = (float)x / (float)(WW - 1);
                    const float Y = 0.14f + (0.36f - 0.14f) * t;
                    p[0] = p[1] = p[2] = Y;
                } else {
                    p[0] = 0.16f; p[1] = 0.11f; p[2] = 0.075f;
                }
                p[3] = 1.0f;
            }
        }
        for (uint32_t y = 16; y < 44; ++y)
            for (uint32_t x = 150; x < 166; ++x) {
                float* p = wood.data() + ((size_t)y * WW + x) * 4;
                p[0] = p[1] = p[2] = 0.80f;
            }
        for (uint32_t y = 196; y < 228; ++y)
            for (uint32_t x = 150; x < 166; ++x) {
                float* p = wood.data() + ((size_t)y * WW + x) * 4;
                p[0] = p[1] = p[2] = 0.80f;
            }
        const DocumentDetect d = detectDocument(wood.data(), WW, WH);
        CHECK(d.found);
        const float err = d.found ? cornerErr(d.quad, wx, wy) : 1e9f;
        std::fprintf(stderr, "wood shape %d verts %zu corner err %.2f px\n",
                     (int)d.outline.shape, d.outline.x.size(), err);
        CHECK(d.outline.shape == DocumentShape::Quad);
        CHECK(err < 8.0f);
    }

    // The page already is the frame.
    {
        const uint32_t BW = 96, BH = 64;
        std::vector<float> full((size_t)BW * BH * 4, 0.88f);
        for (size_t i = 0; i < (size_t)BW * BH; ++i) full[i * 4 + 3] = 1.0f;
        const DocumentScan s = scanDocument(full.data(), BW, BH);
        CHECK(s.found);
        const float bx[4] = {0, (float)BW - 1, (float)BW - 1, 0};
        const float by[4] = {0, 0, (float)BH - 1, (float)BH - 1};
        const float err = s.found ? cornerErr(s.quad, bx, by) : 1e9f;
        std::fprintf(stderr, "full-bleed max corner err %.2f px\n", err);
        CHECK(err < 2.0f);
    }

    // Colour survives the warp: a red patch and a blue ruling.
    {
        const float cx[4] = {36, 200, 208, 32};
        const float cy[4] = {26, 32, 158, 150};
        std::vector<float> col;
        fillPage(col, W, H, cx, cy, 0.06f, 0.06f, 0.06f, 0.90f, 0.90f, 0.90f);
        for (uint32_t y = 70; y < 110; ++y) {
            for (uint32_t x = 90; x < 140; ++x) {
                if (!inside((float)x, (float)y, cx, cy)) continue;
                float* p = col.data() + ((size_t)y * W + x) * 4;
                p[0] = 1.0f; p[1] = 0.10f; p[2] = 0.10f;
            }
        }
        for (uint32_t y = 124; y < 132; ++y) {
            for (uint32_t x = 60; x < 180; ++x) {
                if (!inside((float)x, (float)y, cx, cy)) continue;
                float* p = col.data() + ((size_t)y * W + x) * 4;
                p[0] = 0.10f; p[1] = 0.20f; p[2] = 0.95f;
            }
        }
        const DocumentScan s = scanDocument(col.data(), W, H);
        CHECK(s.found);
        float bestRG = 0;
        int blue = 0;
        if (s.found) {
            for (size_t i = 0; i < (size_t)s.w * s.h; ++i) {
                const float r = s.linear[i * 4], g = s.linear[i * 4 + 1], b = s.linear[i * 4 + 2];
                if (g > 0.04f) bestRG = std::max(bestRG, r / g);
                if (b > 0.4f && b > r * 1.5f && b > g * 1.2f) ++blue;
            }
        }
        std::fprintf(stderr, "warp red R/G %.2f blue pixels %d\n", bestRG, blue);
        CHECK(bestRG > 3.0f);
        CHECK(blue > 10);
    }

    // Crossed corners are refused and the picture is not replaced.
    {
        DocumentQuad bad;
        bad.x[0] = 10; bad.y[0] = 10;
        bad.x[1] = 80; bad.y[1] = 70;
        bad.x[2] = 20; bad.y[2] = 20;
        bad.x[3] = 70; bad.y[3] = 60;
        const DocumentScan s = flattenDocument(img.data(), W, H, bad);
        CHECK(!s.found);
    }

    // Document finish: shading goes, paper is neutral and near white, ink
    // stays dark, red and blue keep their hue. Photo enhance does not flatten.
    {
        const uint32_t EW = 180, EH = 140;
        std::vector<float> page((size_t)EW * EH * 4, 0.0f);
        std::vector<char> inkMask((size_t)EW * EH, 0);
        const int rx0 = 70, ry0 = 50, rx1 = 110, ry1 = 90;
        const int bx0 = 30, by0 = 100, bx1 = 150, by1 = 118;
        const float redIn[3] = {0.82f, 0.10f, 0.08f};
        const float blueIn[3] = {0.10f, 0.16f, 0.78f};
        for (uint32_t y = 0; y < EH; ++y) {
            for (uint32_t x = 0; x < EW; ++x) {
                float* p = page.data() + ((size_t)y * EW + x) * 4;
                const float Y = 0.50f + 0.45f * ((float)x / (float)(EW - 1));
                p[0] = Y * 1.04f; p[1] = Y * 1.00f; p[2] = Y * 0.94f; p[3] = 1.0f;
                const bool red = x >= (uint32_t)rx0 && x < (uint32_t)rx1 &&
                                 y >= (uint32_t)ry0 && y < (uint32_t)ry1;
                const bool blue = x >= (uint32_t)bx0 && x < (uint32_t)bx1 &&
                                  y >= (uint32_t)by0 && y < (uint32_t)by1;
                const bool ink = (y % 16 == 8) && x > 16 && x + 16 < EW &&
                                 y > 8 && y + 8 < EH;
                if (red) { p[0] = redIn[0]; p[1] = redIn[1]; p[2] = redIn[2]; }
                else if (blue) { p[0] = blueIn[0]; p[1] = blueIn[1]; p[2] = blueIn[2]; }
                else if (ink) { p[0] = p[1] = p[2] = 0.04f; }
                if (ink) inkMask[(size_t)y * EW + x] = 1;
            }
        }
        const float redHue = hueDeg(redIn[0], redIn[1], redIn[2]);
        const float blueHue = hueDeg(blueIn[0], blueIn[1], blueIn[2]);
        const float redC = chromaOf(redIn[0], redIn[1], redIn[2]);
        const float blueC = chromaOf(blueIn[0], blueIn[1], blueIn[2]);

        // Paper is measured away from strokes and stamps. The pixels that
        // touch a letter pick up the sharpening halo and are not the sheet.
        auto region = [&](const std::vector<float>& px, int x0, int y0, int x1, int y1,
                          double& mean, double& cv, double& imb) {
            double sr = 0, sg = 0, sb = 0;
            int cnt = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const float* p = px.data() + ((size_t)y * EW + x) * 4;
                    sr += p[0]; sg += p[1]; sb += p[2];
                    ++cnt;
                }
            }
            mean = cv = imb = 0;
            if (!cnt) return;
            sr /= cnt; sg /= cnt; sb /= cnt;
            const double Lmean = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb;
            double var = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const float* p = px.data() + ((size_t)y * EW + x) * 4;
                    const double L = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                    const double d = L - Lmean;
                    var += d * d;
                }
            }
            var /= cnt;
            mean = Lmean;
            cv = Lmean > 1e-4 ? std::sqrt(var) / Lmean : 0;
            imb = std::max(std::fabs(sr - sg), std::max(std::fabs(sg - sb), std::fabs(sr - sb)));
        };
        auto paperStats = [&](const std::vector<float>& px, double& mean, double& cv, double& imb) {
            // Left sheet and right sheet, clear of the red patch and the rules.
            double m0, c0, i0, m1, c1, i1;
            region(px, 36, 32, 64, 33, m0, c0, i0);
            region(px, 130, 32, 158, 33, m1, c1, i1);
            mean = 0.5 * (m0 + m1);
            cv = std::max(c0, c1);
            // Spread between the two ends is the shading the finish has to kill.
            const double span = std::fabs(m0 - m1) / std::max(0.5 * (m0 + m1), 1e-4);
            cv = std::max(cv, span);
            imb = std::max(i0, i1);
        };

        std::vector<float> doc = page;
        enhanceDocument(doc, EW, EH, false);
        double mean = 0, cv = 0, imb = 0;
        paperStats(doc, mean, cv, imb);
        std::fprintf(stderr, "document paper mean %.3f cv %.4f imbalance %.4f\n",
                     mean, cv, imb);
        CHECK(mean > 0.85 && mean < 1.02);
        CHECK(cv < 0.04);
        CHECK(imb < 0.03);

        double inkSum = 0;
        int inkN = 0;
        for (uint32_t y = 0; y < EH; ++y) {
            if (y % 16 != 8 || y <= 8) continue;
            for (uint32_t x = 40; x < 120; ++x) {
                if (!inkMask[(size_t)y * EW + x]) continue;
                const float* p = doc.data() + ((size_t)y * EW + x) * 4;
                inkSum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                ++inkN;
            }
        }
        const double inkMean = inkN ? inkSum / inkN : 1;
        std::fprintf(stderr, "document ink mean %.3f\n", inkMean);
        CHECK(inkMean < 0.15);

        // Patch centres, inset so the blur's rim is not the sample.
        auto patchMean = [&](const std::vector<float>& px, int x0, int y0, int x1, int y1,
                             float& r, float& g, float& b) {
            double sr = 0, sg = 0, sb = 0;
            int cnt = 0;
            for (int y = y0 + 4; y < y1 - 4; ++y) {
                for (int x = x0 + 4; x < x1 - 4; ++x) {
                    const float* p = px.data() + ((size_t)y * EW + x) * 4;
                    sr += p[0]; sg += p[1]; sb += p[2];
                    ++cnt;
                }
            }
            r = cnt ? (float)(sr / cnt) : 0;
            g = cnt ? (float)(sg / cnt) : 0;
            b = cnt ? (float)(sb / cnt) : 0;
        };
        float rr, rg, rb, br, bgc, bb;
        patchMean(doc, rx0, ry0, rx1, ry1, rr, rg, rb);
        patchMean(doc, bx0, by0, bx1, by1, br, bgc, bb);
        const float redDH = hueDelta(hueDeg(rr, rg, rb), redHue);
        const float blueDH = hueDelta(hueDeg(br, bgc, bb), blueHue);
        const float redKeep = chromaOf(rr, rg, rb) / redC;
        const float blueKeep = chromaOf(br, bgc, bb) / blueC;
        std::fprintf(stderr, "document red hue %.2f deg keep %.2f; blue hue %.2f deg keep %.2f\n",
                     redDH, redKeep, blueDH, blueKeep);
        CHECK(redDH < 10.0f);
        CHECK(blueDH < 10.0f);
        CHECK(redKeep > 0.70f);
        CHECK(blueKeep > 0.70f);

        std::vector<float> photo = page;
        enhanceImage(photo, EW, EH, EnhanceParams{}, false);
        double pMean = 0, pCv = 0, pImb = 0;
        paperStats(photo, pMean, pCv, pImb);
        std::fprintf(stderr, "photo paper cv %.4f\n", pCv);
        CHECK(pCv > 0.08);
    }

    // Photo enhance stays what it was: grey stays grey, muted colour moves
    // out, a saturated colour barely moves.
    {
        std::vector<float> grey(8 * 8 * 4, 0.4f);
        for (size_t i = 0; i < 64; ++i) grey[i * 4 + 3] = 1.0f;
        enhanceImage(grey, 8, 8, EnhanceParams{}, false);
        float greyErr = 0;
        for (size_t i = 0; i < 64; ++i)
            for (int c = 0; c < 3; ++c)
                greyErr = std::max(greyErr, std::fabs(grey[i * 4 + c] - 0.4f));
        std::fprintf(stderr, "photo grey err %.6f\n", greyErr);
        CHECK(greyErr < 1e-5f);

        std::vector<float> muted(8 * 8 * 4);
        const float mr = 0.42f, mg = 0.30f, mb = 0.28f;
        for (size_t i = 0; i < 64; ++i) {
            muted[i * 4] = mr; muted[i * 4 + 1] = mg; muted[i * 4 + 2] = mb;
            muted[i * 4 + 3] = 1.0f;
        }
        const float c0 = chromaOf(mr, mg, mb);
        enhanceImage(muted, 8, 8, EnhanceParams{}, false);
        const float c1 = chromaOf(muted[0], muted[1], muted[2]);
        std::fprintf(stderr, "photo muted chroma %.4f -> %.4f\n", c0, c1);
        CHECK(c1 > c0 * 1.05f);

        std::vector<float> sat(8 * 8 * 4);
        for (size_t i = 0; i < 64; ++i) {
            sat[i * 4] = 0.85f; sat[i * 4 + 1] = 0.04f; sat[i * 4 + 2] = 0.04f;
            sat[i * 4 + 3] = 1.0f;
        }
        enhanceImage(sat, 8, 8, EnhanceParams{}, false);
        const float satMove = std::max(std::fabs(sat[0] - 0.85f),
                                        std::max(std::fabs(sat[1] - 0.04f),
                                                 std::fabs(sat[2] - 0.04f)));
        std::fprintf(stderr, "photo saturated move %.4f\n", satMove);
        CHECK(satMove < 0.02f);
    }

    auto nearest = [](const DocumentOutline& o, float x, float y) {
        float e = 1e9f;
        for (size_t i = 0; i < o.x.size(); ++i) {
            const float dx = o.x[i] - x, dy = o.y[i] - y;
            e = std::min(e, std::sqrt(dx * dx + dy * dy));
        }
        return e;
    };

    // A triangular sheet. The fourth corner must not be invented, and the
    // desk outside the long side stays out of the crop.
    {
        const uint32_t TW = 260, TH = 200;
        const float tx[3] = {48, 210, 86};
        const float ty[3] = {36, 52, 168};
        std::vector<float> tri((size_t)TW * TH * 4, 0.0f);
        for (uint32_t y = 0; y < TH; ++y) {
            for (uint32_t x = 0; x < TW; ++x) {
                float* p = tri.data() + ((size_t)y * TW + x) * 4;
                float px3[4] = {tx[0], tx[1], tx[2], tx[0]};
                float py3[4] = {ty[0], ty[1], ty[2], ty[0]};
                const bool page = inside((float)x, (float)y, px3, py3);
                p[0] = p[1] = p[2] = page ? 0.90f : 0.07f;
                p[3] = 1.0f;
            }
        }
        const DocumentDetect d = detectDocument(tri.data(), TW, TH);
        CHECK(d.found);
        CHECK(d.outline.shape == DocumentShape::Triangle);
        float err = 1e9f;
        if (d.outline.x.size() == 3) {
            err = 0;
            for (int i = 0; i < 3; ++i)
                err = std::max(err, nearest(d.outline, tx[i], ty[i]));
        }
        std::fprintf(stderr, "triangle shape %d corner err %.2f px\n",
                     (int)d.outline.shape, err);
        CHECK(err < 6.0f);
        const DocumentScan s = flattenOutline(tri.data(), TW, TH, d.outline);
        CHECK(s.found);
        if (s.found) {
            const float cx = (tx[0] + tx[1] + tx[2]) / 3.0f;
            const float cy = (ty[0] + ty[1] + ty[2]) / 3.0f;
            float minx = tx[0], maxx = tx[0], miny = ty[0], maxy = ty[0];
            for (int i = 1; i < 3; ++i) {
                minx = std::min(minx, tx[i]); maxx = std::max(maxx, tx[i]);
                miny = std::min(miny, ty[i]); maxy = std::max(maxy, ty[i]);
            }
            auto sample = [&](float sx, float sy) {
                const float fx = (s.w == 1) ? 0 : (sx - minx) / (maxx - minx) * (float)(s.w - 1);
                const float fy = (s.h == 1) ? 0 : (sy - miny) / (maxy - miny) * (float)(s.h - 1);
                const int ix = std::clamp((int)std::lround(fx), 0, (int)s.w - 1);
                const int iy = std::clamp((int)std::lround(fy), 0, (int)s.h - 1);
                return s.linear[((size_t)iy * s.w + ix) * 4 + 1];
            };
            const float mid = sample(cx, cy);
            const float outside = sample(maxx, maxy);
            std::fprintf(stderr, "triangle mid %.3f outside %.3f\n", mid, outside);
            CHECK(mid > 0.7f);
            CHECK(outside > 0.9f);
        }
    }

    // One corner bitten off. The table in that bite must not be copied.
    {
        const uint32_t DW = 280, DH = 220;
        std::vector<float> page((size_t)DW * DH * 4, 0.0f);
        for (uint32_t y = 0; y < DH; ++y) {
            for (uint32_t x = 0; x < DW; ++x) {
                float* p = page.data() + ((size_t)y * DW + x) * 4;
                const bool rect = x >= 40 && x <= 230 && y >= 36 && y <= 180;
                const float c1 = (230.0f - 150.0f) * ((float)y - 36.0f) - (100.0f - 36.0f) * ((float)x - 150.0f);
                const bool bite = x >= 150 && y <= 100 && c1 < 0;
                const bool paper = rect && !bite;
                p[0] = p[1] = p[2] = paper ? 0.88f : 0.08f;
                p[3] = 1.0f;
            }
        }
        const DocumentDetect d = detectDocument(page.data(), DW, DH);
        CHECK(d.found);
        std::fprintf(stderr, "fold shape %d verts %zu\n",
                     (int)d.outline.shape, d.outline.x.size());
        CHECK(d.outline.shape == DocumentShape::Fold);
        const DocumentScan s = flattenOutline(page.data(), DW, DH, d.outline);
        CHECK(s.found);
        if (s.found && d.outline.x.size() >= 3) {
            float minx = d.outline.x[0], maxx = minx, miny = d.outline.y[0], maxy = miny;
            for (size_t i = 1; i < d.outline.x.size(); ++i) {
                minx = std::min(minx, d.outline.x[i]); maxx = std::max(maxx, d.outline.x[i]);
                miny = std::min(miny, d.outline.y[i]); maxy = std::max(maxy, d.outline.y[i]);
            }
            auto sample = [&](float sx, float sy) {
                const float fx = (sx - minx) / std::max(maxx - minx, 1.0f) * (float)(s.w - 1);
                const float fy = (sy - miny) / std::max(maxy - miny, 1.0f) * (float)(s.h - 1);
                const int ix = std::clamp((int)std::lround(fx), 0, (int)s.w - 1);
                const int iy = std::clamp((int)std::lround(fy), 0, (int)s.h - 1);
                return s.linear[((size_t)iy * s.w + ix) * 4 + 1];
            };
            const float bite = sample(210, 50);
            const float paperG = sample(60, 150);
            std::fprintf(stderr, "fold bite %.3f paper %.3f\n", bite, paperG);
            CHECK(bite > 0.9f);
            CHECK(paperG > 0.7f);
        }
    }

    // A sheet bowed into an arc. The ink follows the bow and must come out flat.
    {
        const uint32_t BW = 240, BH = 200;
        const float x0 = 24, x1 = 216;
        const float amp = 22;
        auto bow = [&](float x, float base) {
            const float u = (x - x0) / (x1 - x0);
            return base + amp * std::sin(3.14159265f * u);
        };
        std::vector<float> bowed((size_t)BW * BH * 4, 0.0f);
        for (uint32_t y = 0; y < BH; ++y) {
            for (uint32_t x = 0; x < BW; ++x) {
                float* p = bowed.data() + ((size_t)y * BW + x) * 4;
                const float top = bow((float)x, 30);
                const float bot = bow((float)x, 150);
                const float mid = 0.5f * (top + bot);
                const bool page = (float)x >= x0 && (float)x <= x1 && (float)y >= top && (float)y <= bot;
                const bool ink = page && std::fabs((float)y - mid) <= 1.2f && x > 40 && x < 200;
                p[0] = p[1] = p[2] = page ? (ink ? 0.04f : 0.90f) : 0.06f;
                p[3] = 1.0f;
            }
        }
        const DocumentDetect d = detectDocument(bowed.data(), BW, BH);
        CHECK(d.found);
        std::fprintf(stderr, "curve shape %d\n", (int)d.outline.shape);
        CHECK(d.outline.shape == DocumentShape::Curve);
        int knots = 0;
        float knotErr = 1e9f;
        for (const PagePoint& p : d.outline.points) {
            if (p.kind != PagePointKind::Curve) continue;
            ++knots;
            const float dx = p.x - 120.0f, dy = p.y - 52.0f;
            knotErr = std::min(knotErr, std::sqrt(dx * dx + dy * dy));
            const float dx2 = p.x - 120.0f, dy2 = p.y - (150.0f + 22.0f);
            knotErr = std::min(knotErr, std::sqrt(dx2 * dx2 + dy2 * dy2));
        }
        std::fprintf(stderr, "curve knots %d nearest peak %.2f px\n", knots, knotErr);
        CHECK(knots >= 1);
        CHECK(knotErr < 20.0f);
        const DocumentScan s = flattenOutline(bowed.data(), BW, BH, d.outline);
        CHECK(s.found);
        if (s.found) {
            double sy = 0, sy2 = 0;
            int n = 0;
            for (uint32_t y = 0; y < s.h; ++y) {
                for (uint32_t x = s.w / 8; x < s.w * 7 / 8; ++x) {
                    const float g = s.linear[((size_t)y * s.w + x) * 4 + 1];
                    if (g > 0.25f) continue;
                    sy += y; sy2 += (double)y * y; ++n;
                }
            }
            const double mean = n ? sy / n : 0;
            const double var = n ? sy2 / n - mean * mean : 1e9;
            const double sd = std::sqrt(std::max(0.0, var));
            std::fprintf(stderr, "curve ink rows %d sd %.2f px\n", n, sd);
            CHECK(n > 30);
            CHECK(sd < 2.0);
        }
    }

    // A page folded along a vertical hinge. The right half is foreshortened
    // in the photo; its strip in the output follows the slanted edge, and an
    // ink line that sits halfway down both panels comes out on one row.
    {
        auto P = [](PagePointKind k, float x, float y) {
            PagePoint p;
            p.kind = k;
            p.x = x;
            p.y = y;
            return p;
        };
        DocumentOutline o;
        o.shape = DocumentShape::Fold;
        o.points = {
            P(PagePointKind::Corner, 10, 20),
            P(PagePointKind::Fold, 100, 40),
            P(PagePointKind::Corner, 112, 90),
            P(PagePointKind::Corner, 112, 110),
            P(PagePointKind::Fold, 100, 160),
            P(PagePointKind::Corner, 10, 180),
        };
        const uint32_t IW = 160, IH = 200;
        std::vector<float> img((size_t)IW * IH * 4, 0.2f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        auto paint = [&](float x0, float y0, float x1, float y1, float x2, float y2,
                         float x3, float y3, float yInk0, float yInk1) {
            for (int y = 0; y < (int)IH; ++y) {
                for (int x = 0; x < (int)IW; ++x) {
                    const float px[4] = {x0, x1, x2, x3};
                    const float py[4] = {y0, y1, y2, y3};
                    if (!inside((float)x, (float)y, px, py)) continue;
                    float* p = img.data() + ((size_t)y * IW + x) * 4;
                    p[0] = p[1] = p[2] = 0.9f;
                }
            }
            // Ink along the midline of this panel, in the panel's own
            // top-to-bottom parameter, so the unfold should stack the two
            // strokes on one output row.
            for (int step = 0; step <= 40; ++step) {
                const float t = (float)step / 40.0f;
                const float xa = x0 + (x3 - x0) * 0.5f;
                const float ya = y0 + (y3 - y0) * 0.5f;
                const float xb = x1 + (x2 - x1) * 0.5f;
                const float yb = y1 + (y2 - y1) * 0.5f;
                const int ix = (int)std::lround(xa + (xb - xa) * t);
                const int iy = (int)std::lround(ya + (yb - ya) * t);
                if (ix < 1 || iy < 1 || ix + 1 >= (int)IW || iy + 1 >= (int)IH) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    float* p = img.data() + ((size_t)(iy + dy) * IW + ix) * 4;
                    p[0] = p[1] = p[2] = 0.02f;
                }
                (void)yInk0; (void)yInk1;
            }
        };
        paint(10, 20, 100, 40, 100, 160, 10, 180, 0, 0);
        paint(100, 40, 112, 90, 112, 110, 100, 160, 0, 0);
        const DocumentScan s = flattenOutline(img.data(), IW, IH, o);
        CHECK(s.found);
        const float leftW = std::sqrt(90.0f * 90.0f + 20.0f * 20.0f);
        const float rightW = std::sqrt(12.0f * 12.0f + 50.0f * 50.0f);
        std::fprintf(stderr, "hinge page %ux%u (edge widths %.1f+%.1f)\n",
                     s.w, s.h, leftW, rightW);
        CHECK(s.found);
        if (s.found) {
            CHECK(s.w > 125);
            double sy = 0, sy2 = 0;
            int n = 0;
            int ymin = 10000, ymax = -1;
            for (uint32_t y = 0; y < s.h; ++y) {
                for (uint32_t x = 0; x < s.w; ++x) {
                    const float g = s.linear[((size_t)y * s.w + x) * 4 + 1];
                    if (g > 0.10f) continue;
                    sy += y; sy2 += (double)y * y; ++n;
                    ymin = std::min(ymin, (int)y);
                    ymax = std::max(ymax, (int)y);
                }
            }
            const double mean = n ? sy / n : 0;
            const double var = n ? sy2 / n - mean * mean : 1e9;
            const double sd = std::sqrt(std::max(0.0, var));
            std::fprintf(stderr, "hinge ink rows %d sd %.2f y %d..%d\n", n, sd, ymin, ymax);
            CHECK(n > 20);
            CHECK(sd < 4.0);
        }
    }

    // A long handle is a curve and stays one sheet. A zero handle is a fold,
    // and the narrow side keeps the length of its slanted edge.
    {
        auto P = [](PagePointKind k, float x, float y, float hx, float hy) {
            PagePoint p;
            p.kind = k;
            p.x = x; p.y = y; p.hx = hx; p.hy = hy;
            return p;
        };
        const uint32_t IW = 180, IH = 80;
        std::vector<float> img((size_t)IW * IH * 4, 0.15f);
        for (uint32_t y = 0; y < IH; ++y)
            for (uint32_t x = 0; x < IW; ++x) {
                float* p = img.data() + ((size_t)y * IW + x) * 4;
                if (x >= 10 && x <= 160 && y >= 10 && y <= 60) {
                    p[0] = p[1] = p[2] = 0.9f;
                }
                p[3] = 1.0f;
            }
        DocumentOutline curve;
        curve.shape = DocumentShape::Curve;
        curve.points = {
            P(PagePointKind::Corner, 10, 10, 0, 0),
            P(PagePointKind::Curve, 85, 28, 30, 0),
            P(PagePointKind::Corner, 160, 10, 0, 0),
            P(PagePointKind::Corner, 160, 60, 0, 0),
            P(PagePointKind::Corner, 10, 60, 0, 0),
        };
        const DocumentScan bowed = flattenOutline(img.data(), IW, IH, curve);
        CHECK(bowed.found);
        std::fprintf(stderr, "handle curve page %ux%u\n", bowed.w, bowed.h);
        CHECK(bowed.found && bowed.w > 100 && bowed.h > 30 && bowed.h < 120);

        DocumentOutline fold;
        fold.shape = DocumentShape::Fold;
        fold.points = {
            P(PagePointKind::Corner, 10, 20, 0, 0),
            P(PagePointKind::Fold, 100, 30, 0, 0),
            P(PagePointKind::Corner, 130, 70, 0, 0),
            P(PagePointKind::Fold, 100, 120, 0, 0),
            P(PagePointKind::Corner, 10, 130, 0, 0),
        };
        // The right chain is a triangle (one outer point). Its width is the
        // average of the two slanted sides, not the 30px the photo shows.
        const DocumentScan hinged = flattenOutline(img.data(), IW, IH, fold);
        const float slant = 0.5f * (std::sqrt(30.0f * 30.0f + 40.0f * 40.0f) +
                                    std::sqrt(30.0f * 30.0f + 50.0f * 50.0f));
        std::fprintf(stderr, "zero-handle fold page %ux%u slant %.1f\n",
                     hinged.w, hinged.h, slant);
        CHECK(hinged.found);
        CHECK(hinged.w > 90.0f + slant - 8.0f);
    }

    // Bent both ways: a vertical hinge and a horizontal one. Each corner of
    // the sheet lands in its own quadrant of the output.
    {
        auto P = [](PagePointKind k, float x, float y) {
            PagePoint p;
            p.kind = k;
            p.x = x;
            p.y = y;
            return p;
        };
        DocumentOutline o;
        o.shape = DocumentShape::Fold;
        o.points = {
            P(PagePointKind::Corner, 30, 30),
            P(PagePointKind::Fold, 100, 20),
            P(PagePointKind::Corner, 170, 30),
            P(PagePointKind::Fold, 180, 80),
            P(PagePointKind::Corner, 170, 130),
            P(PagePointKind::Fold, 100, 140),
            P(PagePointKind::Corner, 30, 130),
            P(PagePointKind::Fold, 20, 80),
        };
        const uint32_t IW = 200, IH = 160;
        std::vector<float> img((size_t)IW * IH * 4, 0.85f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        auto mark = [&](int x, int y, float r, float g, float b) {
            float* p = img.data() + ((size_t)y * IW + x) * 4;
            p[0] = r; p[1] = g; p[2] = b;
        };
        mark(30, 30, 1, 0, 0);
        mark(170, 30, 0, 1, 0);
        mark(170, 130, 0, 0, 1);
        mark(30, 130, 1, 1, 0);
        const DocumentScan s = flattenOutline(img.data(), IW, IH, o);
        CHECK(s.found);
        int rx, ry, gx, gy, bx, by, yx, yy;
        rx = ry = -1;
        float bestR = 0.4f;
        for (uint32_t y = 0; y < s.h; ++y)
            for (uint32_t x = 0; x < s.w; ++x) {
                const float* p = s.linear.data() + ((size_t)y * s.w + x) * 4;
                if (p[0] > bestR && p[1] < 0.3f && p[2] < 0.3f) {
                    bestR = p[0]; rx = (int)x; ry = (int)y;
                }
            }
        gx = gy = -1;
        float bestG = 0.4f;
        for (uint32_t y = 0; y < s.h; ++y)
            for (uint32_t x = 0; x < s.w; ++x) {
                const float* p = s.linear.data() + ((size_t)y * s.w + x) * 4;
                if (p[1] > bestG && p[0] < 0.3f && p[2] < 0.3f) {
                    bestG = p[1]; gx = (int)x; gy = (int)y;
                }
            }
        bx = by = -1;
        float bestB = 0.4f;
        for (uint32_t y = 0; y < s.h; ++y)
            for (uint32_t x = 0; x < s.w; ++x) {
                const float* p = s.linear.data() + ((size_t)y * s.w + x) * 4;
                if (p[2] > bestB && p[0] < 0.3f && p[1] < 0.3f) {
                    bestB = p[2]; bx = (int)x; by = (int)y;
                }
            }
        yx = yy = -1;
        float bestY = 0.4f;
        for (uint32_t y = 0; y < s.h; ++y)
            for (uint32_t x = 0; x < s.w; ++x) {
                const float* p = s.linear.data() + ((size_t)y * s.w + x) * 4;
                const float yv = std::min(p[0], p[1]);
                if (yv > bestY && p[2] < 0.3f) {
                    bestY = yv; yx = (int)x; yy = (int)y;
                }
            }
        std::fprintf(stderr, "cross page %ux%u red %d,%d green %d,%d blue %d,%d yellow %d,%d\n",
                     s.w, s.h, rx, ry, gx, gy, bx, by, yx, yy);
        const int mx = (int)s.w / 2, my = (int)s.h / 2;
        CHECK(rx >= 0 && rx < mx && ry < my);
        CHECK(gx > mx && gy < my);
        CHECK(bx > mx && by > my);
        CHECK(yx >= 0 && yx < mx && yy > my);
    }

    // A ruled form on warm wood. The table is as bright as the shaded side
    // of the sheet, so the corners have to come from the outer rulings.
    // The shade across the page is not a fold.
    {
        const uint32_t RW = 360, RH = 260;
        const float rx[4] = {40, 316, 316, 40};
        const float ry[4] = {32, 32, 222, 222};
        std::vector<float> form((size_t)RW * RH * 4, 0.0f);
        for (uint32_t y = 0; y < RH; ++y) {
            for (uint32_t x = 0; x < RW; ++x) {
                float* p = form.data() + ((size_t)y * RW + x) * 4;
                if (inside((float)x, (float)y, rx, ry)) {
                    const float shade = 0.45f + 0.40f * (1.0f - (float)x / (float)(RW - 1));
                    const bool hRule = y >= 38 && y <= 217 && x >= 46 && x <= 311 &&
                                       ((y == 38 || y == 39 || y == 216 || y == 217) ||
                                        ((y - 38) % 16 == 0 || (y - 38) % 16 == 1));
                    const bool vRule = x >= 46 && x <= 311 && y >= 38 && y <= 217 &&
                                       (x == 46 || x == 47 || x == 120 || x == 121 ||
                                        x == 200 || x == 201 || x == 260 || x == 261 ||
                                        x == 310 || x == 311);
                    if (hRule || vRule) {
                        p[0] = p[1] = p[2] = 0.06f;
                    } else {
                        p[0] = shade * 1.05f;
                        p[1] = shade * 0.92f;
                        p[2] = shade * 0.78f;
                    }
                } else {
                    const float g = 0.03f * std::sin(0.35f * (float)x) * std::sin(0.22f * (float)y);
                    p[0] = 0.40f + g;
                    p[1] = 0.28f + g;
                    p[2] = 0.14f;
                }
                p[3] = 1.0f;
            }
        }
        const DocumentDetect d = detectDocument(form.data(), RW, RH);
        CHECK(d.found);
        const float err = d.found ? cornerErr(d.quad, rx, ry) : 1e9f;
        int folds = 0, curves = 0;
        for (const PagePoint& pt : d.outline.points) {
            if (pt.kind == PagePointKind::Fold) ++folds;
            if (pt.kind == PagePointKind::Curve) ++curves;
        }
        std::fprintf(stderr, "ruled form shape %d points %zu folds %d curves %d corner err %.2f\n",
                     (int)d.outline.shape, d.outline.points.size(), folds, curves, err);
        CHECK(d.outline.shape == DocumentShape::Quad);
        CHECK(d.outline.points.size() == 4);
        CHECK(folds == 0 && curves == 0);
        CHECK(err < 12.0f);
        if (d.found) {
            const DocumentScan homo = flattenDocument(form.data(), RW, RH, d.quad);
            const DocumentScan rebuilt = reconstructRectangle(form.data(), RW, RH, d.quad);
            CHECK(homo.found && rebuilt.found);
            CHECK(homo.w == rebuilt.w && homo.h == rebuilt.h);
            float diff = 0;
            if (homo.found && rebuilt.found && homo.w == rebuilt.w && homo.h == rebuilt.h) {
                for (size_t i = 0; i < (size_t)homo.w * homo.h; ++i)
                    diff = std::max(diff, std::fabs(homo.linear[i * 4 + 1] -
                                                    rebuilt.linear[i * 4 + 1]));
            }
            std::fprintf(stderr, "ruled rebuild vs homography %.5f\n", diff);
            CHECK(diff < 1e-4f);
        }
        const DocumentScan s = scanDocument(form.data(), RW, RH);
        CHECK(s.found);
        if (s.found) {
            CHECK(s.w > 200 && s.h > 140);
            double mid = 0;
            int n = 0;
            for (uint32_t y = s.h / 5; y < s.h * 4 / 5; ++y)
                for (uint32_t x = s.w / 5; x < s.w * 4 / 5; ++x) {
                    mid += s.linear[((size_t)y * s.w + x) * 4 + 1];
                    ++n;
                }
            const float m = n ? (float)(mid / n) : 0;
            std::fprintf(stderr, "ruled form page %ux%u mid %.3f\n", s.w, s.h, m);
            CHECK(m > 0.25f);
        }
    }

    auto lineDev = [](const DocumentScan& s) {
        if (!s.found || s.w < 8 || s.h < 8) return 1e9f;
        const int x0 = (int)s.w / 8, x1 = (int)s.w * 7 / 8;
        float worst = 0;
        int lines = 0;
        // Each dark ridge across the width, grouped by which ridge it continues.
        struct Ridge { std::vector<float> y; float last = -1; };
        std::vector<Ridge> ridges;
        for (int x = x0; x < x1; ++x) {
            std::vector<int> ys;
            for (uint32_t y = 2; y + 2 < s.h; ++y) {
                const float g = s.linear[((size_t)y * s.w + x) * 4 + 1];
                const float up = s.linear[((size_t)(y - 2) * s.w + x) * 4 + 1];
                const float dn = s.linear[((size_t)(y + 2) * s.w + x) * 4 + 1];
                if (g < 0.25f && up > g + 0.2f && dn > g + 0.2f) ys.push_back((int)y);
            }
            std::vector<char> taken(ridges.size(), 0);
            for (int y : ys) {
                int best = -1;
                float bd = 5;
                for (int i = 0; i < (int)ridges.size(); ++i) {
                    if (taken[i] || ridges[i].last < 0) continue;
                    const float d = std::fabs(ridges[i].last - (float)y);
                    if (d < bd) { bd = d; best = i; }
                }
                if (best < 0) {
                    ridges.push_back(Ridge{});
                    best = (int)ridges.size() - 1;
                }
                taken.resize(ridges.size(), 0);
                taken[best] = 1;
                ridges[best].y.push_back((float)y);
                ridges[best].last = (float)y;
            }
        }
        for (const Ridge& r : ridges) {
            if ((int)r.y.size() < (x1 - x0) / 2) continue;
            float lo = r.y[0], hi = r.y[0];
            for (float y : r.y) { lo = std::min(lo, y); hi = std::max(hi, y); }
            worst = std::max(worst, hi - lo);
            ++lines;
        }
        std::fprintf(stderr, "  ridges %d worst span %.2f px\n", lines, worst);
        return lines ? worst : 1e9f;
    };

    // A rectangular sheet with rulings that bow. Wood outside the quad stays
    // out. The bow is ink on a flat quad, so the page is the homography and
    // the rulings keep their curve.
    {
        const uint32_t BW = 320, BH = 240;
        DocumentQuad q;
        q.x[0] = 36; q.y[0] = 28;
        q.x[1] = 284; q.y[1] = 28;
        q.x[2] = 284; q.y[2] = 210;
        q.x[3] = 36; q.y[3] = 210;
        std::vector<float> mountain((size_t)BW * BH * 4, 0.0f);
        auto bow = [](float x, float base) {
            const float u = (x - 36.0f) / (284.0f - 36.0f);
            return base + 14.0f * std::sin(3.14159265f * u);
        };
        for (uint32_t y = 0; y < BH; ++y) {
            for (uint32_t x = 0; x < BW; ++x) {
                float* p = mountain.data() + ((size_t)y * BW + x) * 4;
                const float px[4] = {q.x[0], q.x[1], q.x[2], q.x[3]};
                const float py[4] = {q.y[0], q.y[1], q.y[2], q.y[3]};
                if (!inside((float)x, (float)y, px, py)) {
                    p[0] = 0.95f; p[1] = 0.05f; p[2] = 0.05f; p[3] = 1;
                    continue;
                }
                bool ink = false;
                for (float base : {56.0f, 98.0f, 140.0f, 182.0f}) {
                    if (std::fabs((float)y - bow((float)x, base)) <= 1.0f) ink = true;
                }
                const float Y = ink ? 0.05f : 0.88f;
                p[0] = p[1] = p[2] = Y;
                p[3] = 1;
            }
        }
        const DocumentScan homo = flattenDocument(mountain.data(), BW, BH, q);
        const DocumentScan rebuilt = reconstructRectangle(mountain.data(), BW, BH, q);
        CHECK(homo.found && rebuilt.found);
        const float homoDev = lineDev(homo);
        const float pageDev = lineDev(rebuilt);
        std::fprintf(stderr, "mountain homography span %.2f rebuilt span %.2f\n",
                     homoDev, pageDev);
        CHECK(homoDev >= 8.0f);
        CHECK(pageDev >= 8.0f);
        float diff = 1e9f;
        if (homo.w == rebuilt.w && homo.h == rebuilt.h) {
            diff = 0;
            for (size_t i = 0; i < (size_t)homo.w * homo.h; ++i)
                diff = std::max(diff, std::fabs(homo.linear[i * 4 + 1] -
                                                rebuilt.linear[i * 4 + 1]));
        }
        std::fprintf(stderr, "mountain rebuild vs homography %.5f\n", diff);
        CHECK(diff < 1e-4f);
        int red = 0;
        if (rebuilt.found) {
            for (size_t i = 0; i < (size_t)rebuilt.w * rebuilt.h; ++i) {
                const float r = rebuilt.linear[i * 4];
                const float g = rebuilt.linear[i * 4 + 1];
                if (r > 0.6f && g < 0.25f) ++red;
            }
        }
        std::fprintf(stderr, "mountain wood pixels %d\n", red);
        CHECK(red < (int)(rebuilt.w * rebuilt.h) / 200);
    }

    // A brightness step with straight rulings is not a kink in the page.
    // The rules stay on the homography. The dark face is lifted.
    {
        const uint32_t SW = 300, SH = 200;
        DocumentQuad q;
        q.x[0] = 24; q.y[0] = 20;
        q.x[1] = 276; q.y[1] = 20;
        q.x[2] = 276; q.y[2] = 180;
        q.x[3] = 24; q.y[3] = 180;
        std::vector<float> shade((size_t)SW * SH * 4, 0.15f);
        for (uint32_t y = 0; y < SH; ++y) {
            for (uint32_t x = 0; x < SW; ++x) {
                float* p = shade.data() + ((size_t)y * SW + x) * 4;
                const float px[4] = {q.x[0], q.x[1], q.x[2], q.x[3]};
                const float py[4] = {q.y[0], q.y[1], q.y[2], q.y[3]};
                if (!inside((float)x, (float)y, px, py)) { p[3] = 1; continue; }
                const float face = x < SW / 2 ? 0.90f : 0.42f;
                const bool ink = y >= 36 && y <= 164 && (y % 18 == 0 || y % 18 == 1) &&
                                 x > 30 && x < 270;
                p[0] = p[1] = p[2] = ink ? 0.05f : face;
                p[3] = 1;
            }
        }
        const DocumentScan homo = flattenDocument(shade.data(), SW, SH, q);
        const DocumentScan rebuilt = reconstructRectangle(shade.data(), SW, SH, q);
        CHECK(homo.found && rebuilt.found);
        const float pageDev = lineDev(rebuilt);
        auto face = [](const DocumentScan& s, bool left) {
            if (!s.found) return 0.0;
            double acc = 0;
            int n = 0;
            const uint32_t x0 = left ? s.w / 8 : s.w / 2;
            const uint32_t x1 = left ? s.w / 2 : s.w * 7 / 8;
            for (uint32_t y = s.h / 5; y < s.h * 4 / 5; ++y)
                for (uint32_t x = x0; x < x1; ++x) {
                    const float g = s.linear[((size_t)y * s.w + x) * 4 + 1];
                    if (g < 0.2f) continue;
                    acc += g;
                    ++n;
                }
            return n ? acc / n : 0.0;
        };
        const double homoGap = std::fabs(face(homo, true) - face(homo, false));
        const double pageGap = std::fabs(face(rebuilt, true) - face(rebuilt, false));
        std::fprintf(stderr, "shade rule span %.2f face gap %.3f -> %.3f\n",
                     pageDev, homoGap, pageGap);
        CHECK(pageDev < 3.0f);
        CHECK(pageGap < homoGap * 0.55);
    }

    auto stamp = [](std::vector<float>& img, uint32_t W, uint32_t H,
                    int cx, int cy, int rad, float r, float g, float b) {
        for (int y = cy - rad; y <= cy + rad; ++y) {
            for (int x = cx - rad; x <= cx + rad; ++x) {
                if (x < 0 || y < 0 || x >= (int)W || y >= (int)H) continue;
                float* p = img.data() + ((size_t)y * W + (uint32_t)x) * 4;
                p[0] = r; p[1] = g; p[2] = b; p[3] = 1.0f;
            }
        }
    };
    auto isRed = [](const float* p) { return p[0] > 0.7f && p[1] < 0.3f && p[2] < 0.3f; };
    auto isGreen = [](const float* p) { return p[1] > 0.7f && p[0] < 0.3f && p[2] < 0.3f; };
    auto isBlue = [](const float* p) { return p[2] > 0.7f && p[0] < 0.3f && p[1] < 0.3f; };

    // One diamond pulled in from the top. The straight chord between the
    // corners is outside the sheet; the diamond itself is the top edge.
    {
        auto P = [](PagePointKind k, float x, float y) {
            PagePoint p;
            p.kind = k;
            p.x = x;
            p.y = y;
            return p;
        };
        const uint32_t IW = 200, IH = 160;
        std::vector<float> img((size_t)IW * IH * 4, 0.85f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        stamp(img, IW, IH, 100, 22, 4, 1, 0, 0);
        stamp(img, IW, IH, 100, 50, 4, 0, 1, 0);
        DocumentOutline o;
        o.shape = DocumentShape::Fold;
        o.points = {
            P(PagePointKind::Corner, 20, 20),
            P(PagePointKind::Fold, 100, 50),
            P(PagePointKind::Corner, 180, 20),
            P(PagePointKind::Corner, 180, 140),
            P(PagePointKind::Corner, 20, 140),
        };
        const DocumentScan s = flattenOutline(img.data(), IW, IH, o);
        CHECK(s.found);
        int reds = 0, topGreen = 0;
        if (s.found) {
            for (uint32_t y = 0; y < s.h; ++y)
                for (uint32_t x = 0; x < s.w; ++x)
                    if (isRed(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++reds;
            const int x0 = std::max(0, (int)s.w / 2 - 3);
            const int x1 = std::min((int)s.w - 1, (int)s.w / 2 + 3);
            const int y1 = std::min(2, (int)s.h - 1);
            for (int y = 0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                    if (isGreen(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++topGreen;
        }
        std::fprintf(stderr, "one fold page %ux%u chord %d top edge %d\n",
                     s.w, s.h, reds, topGreen);
        CHECK(reds == 0);
        CHECK(topGreen > 0);
    }

    // A circle on the left side only. The left edge of the page is the bulge,
    // not the straight chord between the corners.
    {
        auto P = [](PagePointKind k, float x, float y, float hx, float hy) {
            PagePoint p;
            p.kind = k;
            p.x = x; p.y = y; p.hx = hx; p.hy = hy;
            return p;
        };
        const uint32_t IW = 200, IH = 180;
        std::vector<float> img((size_t)IW * IH * 4, 0.85f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        stamp(img, IW, IH, 16, 90, 4, 0, 1, 0);
        stamp(img, IW, IH, 40, 90, 4, 1, 0, 0);
        DocumentOutline o;
        o.shape = DocumentShape::Curve;
        o.points = {
            P(PagePointKind::Corner, 40, 20, 0, 0),
            P(PagePointKind::Corner, 160, 20, 0, 0),
            P(PagePointKind::Corner, 160, 160, 0, 0),
            P(PagePointKind::Corner, 40, 160, 0, 0),
            P(PagePointKind::Curve, 16, 90, -22, 0),
        };
        const DocumentScan s = flattenOutline(img.data(), IW, IH, o);
        CHECK(s.found);
        int edgeGreen = 0, edgeRed = 0;
        if (s.found) {
            const int x1 = std::min(1, (int)s.w - 1);
            for (uint32_t y = 0; y < s.h; ++y) {
                for (int x = 0; x <= x1; ++x) {
                    const float* p = s.linear.data() + ((size_t)y * s.w + x) * 4;
                    if (isGreen(p)) ++edgeGreen;
                    if (isRed(p)) ++edgeRed;
                }
            }
        }
        std::fprintf(stderr, "left curve page %ux%u edge green %d red %d\n",
                     s.w, s.h, edgeGreen, edgeRed);
        CHECK(edgeGreen > 0);
        CHECK(edgeRed == 0);
    }

    // A diamond and a circle on the same four-corner sheet. Both sit on the
    // page boundary. A homography of the squares uses neither.
    {
        auto P = [](PagePointKind k, float x, float y, float hx, float hy) {
            PagePoint p;
            p.kind = k;
            p.x = x; p.y = y; p.hx = hx; p.hy = hy;
            return p;
        };
        const uint32_t IW = 230, IH = 190;
        std::vector<float> img((size_t)IW * IH * 4, 0.85f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        stamp(img, IW, IH, 100, 62, 4, 0, 1, 0);
        stamp(img, IW, IH, 196, 90, 4, 0, 0, 1);
        DocumentOutline o;
        o.shape = DocumentShape::Curve;
        o.points = {
            P(PagePointKind::Corner, 30, 30, 0, 0),
            P(PagePointKind::Fold, 100, 62, 0, 0),
            P(PagePointKind::Corner, 170, 30, 0, 0),
            P(PagePointKind::Curve, 196, 90, 26, 0),
            P(PagePointKind::Corner, 170, 150, 0, 0),
            P(PagePointKind::Corner, 30, 150, 0, 0),
        };
        const DocumentScan s = flattenOutline(img.data(), IW, IH, o);
        CHECK(s.found);
        int topGreen = 0, rightBlue = 0;
        if (s.found) {
            const int x0 = std::max(0, (int)s.w / 2 - 3);
            const int x1 = std::min((int)s.w - 1, (int)s.w / 2 + 3);
            const int y1 = std::min(2, (int)s.h - 1);
            for (int y = 0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                    if (isGreen(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++topGreen;
            const int rx = std::max(0, (int)s.w - 2);
            for (uint32_t y = 0; y < s.h; ++y)
                for (int x = rx; x < (int)s.w; ++x)
                    if (isBlue(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++rightBlue;
        }
        std::fprintf(stderr, "fold+curve page %ux%u top %d right %d\n",
                     s.w, s.h, topGreen, rightBlue);
        CHECK(topGreen > 0);
        CHECK(rightBlue > 0);
    }

    auto blank = [](const DocumentScan& s) {
        int white = 0, black = 0;
        if (!s.found) return std::pair<int, int>{1, 1};
        for (uint32_t i = 0; i < s.w * s.h; ++i) {
            const float* p = s.linear.data() + (size_t)i * 4;
            if (p[0] > 0.99f && p[1] > 0.99f && p[2] > 0.99f) ++white;
            if (p[0] < 0.01f && p[1] < 0.01f && p[2] < 0.01f) ++black;
        }
        return std::pair<int, int>{white, black};
    };

    // A diamond a couple of pixels off an already straight side is not a
    // bend. Rect keeps the four-corner reconstruction exactly.
    {
        auto P = [](PagePointKind k, float x, float y) {
            PagePoint p;
            p.kind = k;
            p.x = x;
            p.y = y;
            return p;
        };
        const uint32_t IW = 140, IH = 110;
        std::vector<float> img((size_t)IW * IH * 4, 0.0f);
        for (uint32_t y = 0; y < IH; ++y) {
            for (uint32_t x = 0; x < IW; ++x) {
                float* p = img.data() + ((size_t)y * IW + x) * 4;
                const float g = 0.25f + 0.5f * (float)x / (float)(IW - 1);
                p[0] = p[1] = p[2] = g;
                p[3] = 1.0f;
            }
        }
        DocumentOutline o;
        o.shape = DocumentShape::Fold;
        o.points = {
            P(PagePointKind::Corner, 10, 10),
            P(PagePointKind::Fold, 60, 12),
            P(PagePointKind::Corner, 110, 10),
            P(PagePointKind::Corner, 110, 80),
            P(PagePointKind::Corner, 10, 80),
        };
        DocumentQuad q;
        q.x[0] = 10; q.y[0] = 10;
        q.x[1] = 110; q.y[1] = 10;
        q.x[2] = 110; q.y[2] = 80;
        q.x[3] = 10; q.y[3] = 80;
        const DocumentScan guided = reconstructRectangle(img.data(), IW, IH, o);
        const DocumentScan plain = reconstructRectangle(img.data(), IW, IH, q);
        float diff = 1e9f;
        if (guided.found && plain.found && guided.w == plain.w && guided.h == plain.h &&
            guided.linear.size() == plain.linear.size()) {
            diff = 0;
            for (size_t i = 0; i < guided.linear.size(); ++i)
                diff = std::max(diff, std::fabs(guided.linear[i] - plain.linear[i]));
        }
        std::fprintf(stderr, "near fold page %ux%u diff %.5f\n",
                     guided.w, guided.h, diff);
        CHECK(guided.found);
        CHECK(guided.w == 100 && guided.h == 70);
        CHECK(diff == 0.0f);
    }

    // One diamond inset from the top. Rect keeps the corner rectangle, puts
    // the kink on that edge, and fills the page. The chord beyond the kink
    // stays out.
    {
        auto P = [](PagePointKind k, float x, float y) {
            PagePoint p;
            p.kind = k;
            p.x = x;
            p.y = y;
            return p;
        };
        const uint32_t IW = 200, IH = 160;
        std::vector<float> img((size_t)IW * IH * 4, 0.85f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        stamp(img, IW, IH, 100, 22, 4, 1, 0, 0);
        stamp(img, IW, IH, 100, 50, 4, 0, 1, 0);
        DocumentOutline o;
        o.shape = DocumentShape::Fold;
        o.points = {
            P(PagePointKind::Corner, 20, 20),
            P(PagePointKind::Fold, 100, 50),
            P(PagePointKind::Corner, 180, 20),
            P(PagePointKind::Corner, 180, 140),
            P(PagePointKind::Corner, 20, 140),
        };
        // The corner pixels are the yellow vertices. Anything past the bend
        // stays the straight four-corner page.
        float* tl = img.data() + ((size_t)20 * IW + 20) * 4;
        tl[0] = 1; tl[1] = 0; tl[2] = 1;
        float* tr = img.data() + ((size_t)20 * IW + 180) * 4;
        tr[0] = 0; tr[1] = 1; tr[2] = 1;
        const DocumentScan s = reconstructRectangle(img.data(), IW, IH, o);
        DocumentQuad q;
        q.x[0] = 20; q.y[0] = 20; q.x[1] = 180; q.y[1] = 20;
        q.x[2] = 180; q.y[2] = 140; q.x[3] = 20; q.y[3] = 140;
        const DocumentScan plain = flattenDocument(img.data(), IW, IH, q);
        int topGreen = 0, topRed = 0;
        if (s.found) {
            const int x0 = std::max(0, (int)s.w / 2 - 3);
            const int x1 = std::min((int)s.w - 1, (int)s.w / 2 + 3);
            const int y1 = std::min(2, (int)s.h - 1);
            for (int y = 0; y <= y1; ++y) {
                for (uint32_t x = 0; x < s.w; ++x)
                    if (isRed(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++topRed;
                for (int x = x0; x <= x1; ++x)
                    if (isGreen(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++topGreen;
            }
        }
        auto pixDiff = [](const DocumentScan& a, const DocumentScan& b, int x, int y) {
            if (!a.found || !b.found || x < 0 || y < 0 ||
                x >= (int)a.w || y >= (int)a.h || a.w != b.w || a.h != b.h)
                return 1e9f;
            const float* pa = a.linear.data() + ((size_t)y * a.w + x) * 4;
            const float* pb = b.linear.data() + ((size_t)y * b.w + x) * 4;
            float d = 0;
            for (int c = 0; c < 4; ++c) d = std::max(d, std::fabs(pa[c] - pb[c]));
            return d;
        };
        float far = 0, bottom = 0;
        if (s.found && plain.found && s.w == plain.w && s.h == plain.h) {
            for (uint32_t y = s.h / 2; y < s.h; ++y)
                for (uint32_t x = 0; x < s.w; ++x)
                    far = std::max(far, pixDiff(s, plain, (int)x, (int)y));
            for (uint32_t x = 0; x < s.w; ++x)
                bottom = std::max(bottom, pixDiff(s, plain, (int)x, (int)s.h - 1));
        } else far = bottom = 1e9f;
        const float c0 = pixDiff(s, plain, 0, 0);
        const float c1 = pixDiff(s, plain, (int)s.w - 1, 0);
        const auto holes = blank(s);
        std::fprintf(stderr,
                     "rect one fold %ux%u chord %d top %d far %.4f bottom %.4f corners %.4f %.4f\n",
                     s.w, s.h, topRed, topGreen, far, bottom, c0, c1);
        CHECK(s.found);
        CHECK(s.w == 160 && s.h == 120);
        CHECK(topRed == 0);
        CHECK(topGreen > 0);
        CHECK(holes.first == 0 && holes.second == 0);
        CHECK(far == 0.0f && bottom == 0.0f);
        CHECK(c0 == 0.0f && c1 == 0.0f);
        const float* outTL = s.linear.data();
        CHECK(outTL[0] > 0.5f && outTL[2] > 0.5f && outTL[1] < 0.3f);
    }

    // Blues on opposite sides edit those sides only. The middle of the page
    // stays the straight rectangle between the yellow corners.
    {
        auto P = [](PagePointKind k, float x, float y) {
            PagePoint p;
            p.kind = k;
            p.x = x;
            p.y = y;
            return p;
        };
        const uint32_t IW = 260, IH = 220;
        std::vector<float> img((size_t)IW * IH * 4, 0.85f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        stamp(img, IW, IH, 70, 55, 4, 0, 1, 0);
        stamp(img, IW, IH, 170, 145, 4, 0, 0, 1);
        stamp(img, IW, IH, 120, 100, 4, 1, 0, 0);
        DocumentOutline o;
        o.shape = DocumentShape::Fold;
        o.points = {
            P(PagePointKind::Corner, 20, 20),
            P(PagePointKind::Fold, 70, 55),
            P(PagePointKind::Corner, 220, 20),
            P(PagePointKind::Corner, 220, 180),
            P(PagePointKind::Fold, 170, 145),
            P(PagePointKind::Corner, 20, 180),
        };
        const DocumentScan s = reconstructRectangle(img.data(), IW, IH, o);
        DocumentQuad q;
        q.x[0] = 20; q.y[0] = 20; q.x[1] = 220; q.y[1] = 20;
        q.x[2] = 220; q.y[2] = 180; q.x[3] = 20; q.y[3] = 180;
        const DocumentScan plain = flattenDocument(img.data(), IW, IH, q);
        int topGreen = 0, botBlue = 0;
        float mid = 1e9f;
        if (s.found) {
            const int yTop = std::min(2, (int)s.h - 1);
            for (int y = 0; y <= yTop; ++y)
                for (uint32_t x = 0; x < s.w; ++x)
                    if (isGreen(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++topGreen;
            const int y0 = std::max(0, (int)s.h - 3);
            for (int y = y0; y < (int)s.h; ++y)
                for (uint32_t x = 0; x < s.w; ++x)
                    if (isBlue(s.linear.data() + ((size_t)y * s.w + x) * 4)) ++botBlue;
        }
        if (s.found && plain.found && s.w == plain.w && s.h == plain.h) {
            mid = 0;
            const uint32_t y0 = s.h * 4 / 10, y1 = s.h * 6 / 10;
            for (uint32_t y = y0; y <= y1; ++y)
                for (uint32_t x = 0; x < s.w; ++x)
                    for (int c = 0; c < 4; ++c)
                        mid = std::max(mid, std::fabs(
                            s.linear[((size_t)y * s.w + x) * 4 + c] -
                            plain.linear[((size_t)y * plain.w + x) * 4 + c]));
        }
        std::fprintf(stderr, "rect two sides %ux%u top %d bottom %d mid %.4f\n",
                     s.w, s.h, topGreen, botBlue, mid);
        CHECK(s.found);
        CHECK(s.w == 200 && s.h == 160);
        CHECK(topGreen > 0);
        CHECK(botBlue > 0);
        CHECK(mid == 0.0f);
    }

    // A circle on one side is pulled onto the straight edge of the corner
    // rectangle. The bulge is that edge. The page does not grow to hold it.
    {
        auto P = [](PagePointKind k, float x, float y, float hx, float hy) {
            PagePoint p;
            p.kind = k;
            p.x = x; p.y = y; p.hx = hx; p.hy = hy;
            return p;
        };
        const uint32_t IW = 200, IH = 180;
        std::vector<float> img((size_t)IW * IH * 4, 0.85f);
        for (size_t i = 0; i < (size_t)IW * IH; ++i) img[i * 4 + 3] = 1.0f;
        stamp(img, IW, IH, 16, 90, 4, 0, 1, 0);
        stamp(img, IW, IH, 40, 90, 4, 1, 0, 0);
        DocumentOutline o;
        o.shape = DocumentShape::Curve;
        o.points = {
            P(PagePointKind::Corner, 40, 20, 0, 0),
            P(PagePointKind::Corner, 160, 20, 0, 0),
            P(PagePointKind::Corner, 160, 160, 0, 0),
            P(PagePointKind::Corner, 40, 160, 0, 0),
            P(PagePointKind::Curve, 16, 90, -22, 0),
        };
        const DocumentScan s = reconstructRectangle(img.data(), IW, IH, o);
        int edgeGreen = 0, edgeRed = 0;
        if (s.found) {
            const int x1 = std::min(1, (int)s.w - 1);
            for (uint32_t y = 0; y < s.h; ++y) {
                for (int x = 0; x <= x1; ++x) {
                    const float* p = s.linear.data() + ((size_t)y * s.w + x) * 4;
                    if (isGreen(p)) ++edgeGreen;
                    if (isRed(p)) ++edgeRed;
                }
            }
        }
        const auto holes = blank(s);
        std::fprintf(stderr, "rect left curve %ux%u edge green %d red %d white %d black %d\n",
                     s.w, s.h, edgeGreen, edgeRed, holes.first, holes.second);
        CHECK(s.found);
        CHECK(s.w == 120 && s.h == 140);
        CHECK(edgeGreen > 0);
        CHECK(edgeRed == 0);
        CHECK(holes.first == 0 && holes.second == 0);
    }

#ifdef VIEWMAGE_WITH_ORT
    // DocAligner LC100. A shaded sheet on pale neutral wood shares luminance
    // with the table, which is where the luminance quad leaks. The model has
    // to land closer than that quad, and a ruled form must stay under 12 px.
    {
        std::ifstream in("../android/app/src/main/assets/models/page_quad.onnx",
                         std::ios::binary);
        std::vector<uint8_t> model((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
        if (model.empty()) {
            std::fprintf(stderr, "page model absent, rectangle checks skipped\n");
        } else {
            const uint32_t W = 960, H = 720;
            const float px[4] = {120, 820, 860, 80};
            const float py[4] = {90, 110, 620, 600};
            std::vector<float> wood((size_t)W * H * 4, 0.0f);
            for (uint32_t y = 0; y < H; ++y) {
                for (uint32_t x = 0; x < W; ++x) {
                    float* p = wood.data() + ((size_t)y * W + x) * 4;
                    const float n = 0.015f * std::sin(0.15f * (float)x) * std::cos(0.07f * (float)y);
                    const float base = 0.34f + n;
                    if (inside((float)x, (float)y, px, py)) {
                        const float t = ((float)x - 80.0f) / 780.0f;
                        const float Y = 0.18f + 0.42f * t;
                        p[0] = p[1] = p[2] = Y;
                    } else {
                        p[0] = p[1] = p[2] = base;
                    }
                    p[3] = 1.0f;
                }
            }
            for (uint32_t y = 30; y < 55; ++y)
                for (uint32_t x = 400; x < 430; ++x) {
                    float* p = wood.data() + ((size_t)y * W + x) * 4;
                    p[0] = p[1] = p[2] = 0.62f;
                }
            for (uint32_t y = 650; y < 690; ++y)
                for (uint32_t x = 400; x < 440; ++x) {
                    float* p = wood.data() + ((size_t)y * W + x) * 4;
                    p[0] = p[1] = p[2] = 0.70f;
                }
            const DocumentDetect luma = detectDocument(wood.data(), W, H);
            const DocumentDetect page = detectRectangle(wood.data(), W, H,
                                                       model.data(), model.size());
            const float lumaErr = luma.found ? cornerErr(luma.quad, px, py) : 1e9f;
            const float pageErr = page.found ? cornerErr(page.quad, px, py) : 1e9f;
            std::fprintf(stderr, "neutral wood luma %.1f px, model %.1f px\n",
                         lumaErr, pageErr);
            CHECK(page.found);
            CHECK(page.outline.shape == DocumentShape::Quad);
            CHECK(page.outline.points.size() == 4);
            CHECK(pageErr < lumaErr);
            CHECK(pageErr < 16.0f);

            const uint32_t RW = 360, RH = 260;
            const float rx[4] = {40, 316, 316, 40};
            const float ry[4] = {32, 32, 222, 222};
            std::vector<float> form((size_t)RW * RH * 4, 0.0f);
            for (uint32_t y = 0; y < RH; ++y) {
                for (uint32_t x = 0; x < RW; ++x) {
                    float* p = form.data() + ((size_t)y * RW + x) * 4;
                    if (inside((float)x, (float)y, rx, ry)) {
                        const float shade = 0.45f + 0.40f * (1.0f - (float)x / (float)(RW - 1));
                        const bool hRule = y >= 38 && y <= 217 && x >= 46 && x <= 311 &&
                                           ((y == 38 || y == 39 || y == 216 || y == 217) ||
                                            ((y - 38) % 16 == 0 || (y - 38) % 16 == 1));
                        const bool vRule = x >= 46 && x <= 311 && y >= 38 && y <= 217 &&
                                           (x == 46 || x == 47 || x == 120 || x == 121 ||
                                            x == 200 || x == 201 || x == 260 || x == 261 ||
                                            x == 310 || x == 311);
                        if (hRule || vRule) p[0] = p[1] = p[2] = 0.06f;
                        else {
                            p[0] = shade * 1.05f;
                            p[1] = shade * 0.92f;
                            p[2] = shade * 0.78f;
                        }
                    } else {
                        const float g = 0.03f * std::sin(0.35f * (float)x) * std::sin(0.22f * (float)y);
                        p[0] = 0.40f + g;
                        p[1] = 0.28f + g;
                        p[2] = 0.14f;
                    }
                    p[3] = 1.0f;
                }
            }
            float ruledPeak[4] = {};
            const DocumentDetect ruledLuma = detectDocument(form.data(), RW, RH);
            const DocumentDetect ruled = detectRectangle(form.data(), RW, RH,
                                                        model.data(), model.size(),
                                                        ruledPeak);
            const float ruledLumaErr = ruledLuma.found ? cornerErr(ruledLuma.quad, rx, ry) : 1e9f;
            const float ruledErr = ruled.found ? cornerErr(ruled.quad, rx, ry) : 1e9f;
            std::fprintf(stderr, "ruled form luma %.1f px, model %.1f px, peaks %.2f %.2f %.2f %.2f\n",
                         ruledLumaErr, ruledErr, ruledPeak[0], ruledPeak[1], ruledPeak[2],
                         ruledPeak[3]);
            CHECK(ruled.found);
            CHECK(ruled.outline.shape == DocumentShape::Quad);
            CHECK(ruled.outline.points.size() == 4);
            CHECK(ruledErr < 12.0f);
            CHECK(ruledErr < 2.0f);

            // A page that fills the frame, with ink. A point sample of this
            // stretch drops the desk border and the four peaks stay near 0.1
            // in the wrong place. The area average holds the border. A blank
            // sheet, below, stays under the gate: this model does not invent
            // a page that has no ink.
            const uint32_t TW = 2400, TH = 1800;
            const float tx[4] = {30, 2290, 2370, 70};
            const float ty[4] = {30, 80, 1770, 1740};
            std::vector<float> tight((size_t)TW * TH * 4, 0.0f);
            for (uint32_t y = 0; y < TH; ++y) {
                for (uint32_t x = 0; x < TW; ++x) {
                    float* p = tight.data() + ((size_t)y * TW + x) * 4;
                    if (inside((float)x, (float)y, tx, ty)) {
                        const bool ink = y > 140 && y < 1660 && x > 160 && x < 2200 &&
                                         (y % 28) < 2;
                        if (ink) p[0] = p[1] = p[2] = 0.05f;
                        else { p[0] = 0.82f; p[1] = 0.80f; p[2] = 0.76f; }
                    } else {
                        p[0] = 0.40f; p[1] = 0.28f; p[2] = 0.14f;
                    }
                    p[3] = 1.0f;
                }
            }
            float tightPeak[4] = {};
            const DocumentDetect tightLuma = detectDocument(tight.data(), TW, TH);
            const DocumentDetect tightPage = detectRectangle(tight.data(), TW, TH,
                                                            model.data(), model.size(),
                                                            tightPeak);
            const float tightLumaErr = tightLuma.found ? cornerErr(tightLuma.quad, tx, ty) : 1e9f;
            const float tightErr = tightPage.found ? cornerErr(tightPage.quad, tx, ty) : 1e9f;
            const float tightMin = std::min(std::min(tightPeak[0], tightPeak[1]),
                                            std::min(tightPeak[2], tightPeak[3]));
            std::fprintf(stderr,
                         "tight page luma %.1f px, model %.1f px, peaks %.2f %.2f %.2f %.2f\n",
                         tightLumaErr, tightErr, tightPeak[0], tightPeak[1], tightPeak[2],
                         tightPeak[3]);
            CHECK(tightPage.found);
            CHECK(tightPage.outline.shape == DocumentShape::Quad);
            CHECK(tightPage.outline.points.size() == 4);
            CHECK(tightMin >= 0.55f);
            CHECK(tightErr < 48.0f);

            std::vector<float> blank((size_t)W * H * 4, 0.55f);
            for (size_t i = 0; i < (size_t)W * H; ++i) blank[i * 4 + 3] = 1.0f;
            float blankPeak[4] = {};
            const DocumentDetect blankLuma = detectDocument(blank.data(), W, H);
            const DocumentDetect blankPage = detectRectangle(blank.data(), W, H,
                                                            model.data(), model.size(),
                                                            blankPeak);
            const float blankMin = std::min(std::min(blankPeak[0], blankPeak[1]),
                                            std::min(blankPeak[2], blankPeak[3]));
            std::fprintf(stderr, "blank peaks %.3f %.3f %.3f %.3f\n",
                         blankPeak[0], blankPeak[1], blankPeak[2], blankPeak[3]);
            CHECK(blankMin < 0.2f);
            CHECK(blankPage.found == blankLuma.found);
            if (blankPage.found && blankLuma.found)
                CHECK(cornerErr(blankPage.quad, blankLuma.quad.x, blankLuma.quad.y) < 1.0f);
        }
    }
#endif

    if (fails) {
        std::fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    std::fprintf(stderr, "document_scan_test ok\n");
    return 0;
}
