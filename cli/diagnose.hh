// diagnose.hh — Phase 1 measurement: where do LED highlight artifacts come
// from, and how large are they at each stage of the pipeline?
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
#pragma once

#include <cstdint>
#include <string>

// Analyses one DNG and prints a text report to stdout:
//
//   1. Clip census — the fraction of photosites at or above white, per CFA
//      colour. Green clipping first is what turns blown light green at the
//      fringes; R/B overshooting under their WB gains is what turns cores
//      magenta.
//   2. Bloom detection — connected components of clipped photosites, largest
//      first. These are the LED sources.
//   3. Per-bloom rings — mean normalised value per colour in concentric rings
//      around each bloom centre (mosaic level), plus per-pixel R/G and B/G
//      ratios on the developed image (Malvar, and AI when a model is given).
//      Per-pixel ratio MEANS are used because they are invariant to any later
//      per-pixel luminance scaling (toe curve, exposure): only chroma moves
//      them, which is exactly what we are measuring.
//   4. Lateral-CA check — per colour, the centroid of the bright mask around
//      each bloom. A lens with lateral CA puts R and B centroids at slightly
//      different radii/directions from G; clipping does not displace centroids,
//      it distorts ratios. The two causes separate cleanly this way.
//
// `model_path` empty skips the AI stage (still useful); `ensemble` mirrors the
// denoiser's flag. Returns false when the file could not be read as RAW.
bool diagnose_dng_file(const std::string& path, const std::string& model_path,
                       bool ensemble);

// Detail-loss / tiling investigation on an optional crop [x y w h] (0,0,0,0 =
// whole frame). Develops the SAME region three ways — Malvar, AI single-pass,
// AI x4 ensemble — and prints two metric families:
//
//   GRID  high-frequency energy in 12-px-wide strips centred on every interior
//         tile boundary (multiples of kAiStep*kAiScale = 768 px) against
//         mid-tile control strips. A stitcher that lets unreliable rim
//         predictions through shows grid/mid ratios far from 1.0.
//   HF    mean |2c - l - r| and mean |4c - 2l - 2r - up - down| /4 style
//         Laplacians on green: how much sub-3px structure each variant kept,
//         with Malvar — which only interpolates — as the reference ceiling.
//
// Returns false when anything could not be run.
bool compare_variants(const std::string& path, int x, int y, int w, int h,
                      const std::string& model_path);
