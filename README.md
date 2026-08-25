# ViewMage

A JPEG XL image viewer for Android, written in C++ against the
[`app_shell`](https://github.com/minervarr/App_shell) +
[`vk_canvas`](https://github.com/minervarr/Vk_Canvas_Lb_LAW) framework and
decoding with [libjxl](https://github.com/libjxl/libjxl).

Open a `.jxl` from any file manager and look at it: fit to screen, pinch to
zoom, drag to pan, double-tap to toggle 1:1. That is the whole application.

There is no gallery, no folder browser and no second format. The name is the
scope.

## What it is made of

| | |
|---|---|
| `framework/app_shell` | submodule — the window, the message pump, the input, the Android host |
| `framework/vk_canvas` | submodule — the Vulkan renderer, canvas, MSDF text, textured quads |
| `thirdparty/libjxl` | submodule — **pinned, never modified**; decode only |
| `src/` | ViewMage's own code: the `AppView`, the decode wrapper, the pan/zoom math |

`vk_canvas` draws it; `app_shell` makes it a program; libjxl turns the bytes
into pixels. ViewMage is the small amount of code between them.

## Building

```bash
git submodule update --init --recursive     # libjxl brings its own nested deps
./gradlew assembleDebug
```

Requires the Android NDK and CMake 3.22.1+; see `CLAUDE.md` for versions and
for the desktop test build.

## Committing

Use `./git_wrapper` — never plain `git commit`/`git push`. It forces the
author identity, strips stray trailers, and pushes submodules before the
parent. `framework/app_shell` is our own repository and is committed and
pushed from inside itself, first. `thirdparty/libjxl` is pinned and is never
pushed at all.

## The neural raw denoiser

Photos are denoised on the raw Bayer mosaic, *before* demosaicing, by
**RawNIND UtNet2** — the model darktable 5.6 ships as Neural Restore → Raw
Denoise. It is a joint denoise **and** demosaic, so it replaces the ordinary
demosaic rather than running after it; white balance, the colour matrix and the
tone curve are unchanged either way.

- Model: <https://github.com/darktable-org/darktable-ai> — **GPL-3.0**
- Paper: Brummer & De Vleeschouwer, *Learning Joint Denoising, Demosaicing, and
  Compression from the Raw Natural Image Noise Dataset*,
  [arXiv:2501.08924](https://arxiv.org/abs/2501.08924)
- Training data: RawNIND (real noisy/clean raw pairs), CC BY 4.0 / CC0
- Runtime: ONNX Runtime, CPU. Measured **22 s** for a 12.5 MP frame on an
  S23 Ultra, which is why it runs in the background and swaps in when ready.

Neither the model nor the runtime is in git. Fetch both before building:

```bash
tools/fetch_onnxruntime.sh   # thirdparty/onnxruntime/ (arm64 .so + headers)
tools/fetch_model.sh         # app/src/main/assets/models/model_bayer.onnx
```

Without them the app still builds and runs; photos simply develop with the
ordinary Malvar demosaic and no denoise.

## Exporting

The bottom bar's **Save PNG** writes what is on screen — developed, denoised,
tone-mapped — as an 8-bit sRGB PNG. The DNG stays the archive; the PNG is the
shareable copy.

Under scoped storage (`targetSdk 34`) it lands in the app's own folder rather
than `Pictures/`, and the status line says which. A MediaStore insert would put
it in the gallery proper; that needs Java in `app_shell` and is not done yet.

## License

Copyright (C) 2026 nava. Licensed under the GNU Affero General Public License
version 3 or later, the same terms `vk_canvas` is under — see `LICENSE`.

The bundled denoise model is GPL-3.0 (see above), which AGPL-3.0 is compatible
with. Its attribution is a licence obligation, not a courtesy — keep it in the
app's details panel and here.
