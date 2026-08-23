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

## License

Copyright (C) 2026 nava. Licensed under the GNU Affero General Public License
version 3 or later, the same terms `vk_canvas` is under — see `LICENSE`.
