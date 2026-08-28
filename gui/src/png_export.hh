// png_export.hh — write the picture the screen is showing, as a PNG.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// ---------------------------------------------------------------------------
// This is the END of the POST workflow: shoot a DNG, develop and denoise it
// here, and leave with one ordinary file anything can open. A DNG is the
// archive; this is the copy you send to someone.
//
// It is deliberately 8-BIT sRGB, which is a real loss of range and the correct
// choice anyway. The alternatives were considered and rejected:
//
//   * 16-bit PQ/BT.2100 keeps everything, and almost nothing outside a colour-
//     managed viewer renders it as anything but a washed, too-dark picture --
//     which is exactly the complaint that started this work.
//   * JXL is smaller and better in every technical respect, and Android cannot
//     decode it, so the phone that shot the photo cannot show the export.
//
// What is written is what was on screen: the same exposure, the same tone
// curve, the same rolloff to the display's range. "Ready to share" means the
// receiver sees what the sender saw, on a screen that knows nothing about PQ.
// The scene-referred data is never lost -- it stays in the DNG.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// `linear` is RGBA float scene-referred light, `w`*`h`, in `primariesAreWide`'s
// gamut. `ev` and `white` are the view's own tone parameters, so the file
// matches the screen rather than being a second, differently-rendered picture.
//
// The PNG is encoded IN MEMORY and then published, rather than written to a
// path. That is not a style choice: from API 29 an app may not create a file in
// a shared collection by path at all, so "write it to Pictures/" is not a thing
// that can succeed. MediaStore takes bytes, so bytes are what we produce.
//
// On success `where` is filled with something short to show the user -- the
// gallery album it landed in, or the folder if it had to fall back. Returns
// false with a user-readable `err`; never throws.
bool export_png(const std::vector<float>& linear, uint32_t w, uint32_t h,
                float ev, float white, bool primariesAreWide,
                const std::string& displayName,
                std::string& where, std::string& err);

// The JPEG sibling, for chat-app sized copies. Same pixels, same render, same
// dither — the only difference is the container and its ~5-8x smaller size.
// Quality 95.
bool export_jpeg(const std::vector<float>& linear, uint32_t w, uint32_t h,
                 float ev, float white, bool primariesAreWide,
                 const std::string& displayName,
                 std::string& where, std::string& err);

// "IMG_20260825_012700.dng" -> "IMG_20260825_012700.<ext>", and anything
// without a usable name gets a timestamped one. Never returns empty.
std::string export_name_for(const std::string& sourceName, const char* ext = ".png");
