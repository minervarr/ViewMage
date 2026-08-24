# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in `ViewMage`.

## What this is

A JPEG XL viewer for Android. A file manager opens it with a `.jxl`; it shows
the picture; pinch, drag and double-tap move it around. There is no gallery, no
folder paging, no second format and no settings screen — the name is the scope,
and additions should be argued against it rather than into it.

The application is C++. The only Java is a five-line Activity whose entire job
is `System.loadLibrary`.

## Party rules — the directory says which

| Directory | Party | Rule |
|---|---|---|
| `framework/` | first (`minervarr`) | Ours. Modify freely; commit and push **from inside the submodule, first**. |
| `thirdparty/` | third | **Never modified, never pushed.** Pinned and detached. |

`thirdparty/libjxl` is pinned at **v0.12.0**, detached, with only
`third_party/{highway,brotli,skcms}` initialized — `testdata` alone is about a
gigabyte and nothing needs it. Anything ViewMage wants that libjxl does not
provide is written in `src/jxl_image.cc`, on our side of the line. Updating
libjxl means moving the pin, never editing the tree. `git_wrapper push` skips
pinned submodules, so the rule is enforced by tooling and not only by care.

## Build

```bash
git submodule update --init --recursive
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Needs NDK 29.0.14206865, CMake 3.22.1+, and `slangc` for the shaders — there is
no Vulkan SDK on the dev machine, so `app/build.gradle` passes
`-DVCE_SLANGC=/opt/shader-slang/bin/slangc` through `VceShaders.cmake`'s
existing seam rather than faking a SDK layout.

Desktop tests (no device, no GPU):

```bash
./tests/run_desktop_tests.sh
```

## Four things the build taught, all non-obvious

1. **libjxl must be `-O2` even in Debug.** Gradle's debug variant sets
   `CMAKE_BUILD_TYPE=Debug`, which compiled highway's SIMD kernels at `-O0`: a
   3060x4080 photo took **19.5 seconds** to decode. `CMakeLists.txt` now saves,
   raises and restores `CMAKE_{C,CXX}_FLAGS_DEBUG` around libjxl's
   `add_subdirectory()`. Same photo: **0.37 s**. Never "fix" this back.
2. **The photograph goes in the FOREGROUND image layer.** `Canvas::image()`
   draws before the vector overlay composites, and `Canvas::clear()` is itself
   a full-screen overlay rect — so a background image is painted over by the
   thing drawing the letterbox around it. Use `imageFg()` +
   `Renderer::draw()`'s `foregroundImages`.
3. **AGP builds every CMake target it can see.** `EXCLUDE_FROM_ALL` does not
   stop it, because it asks ninja for targets by name; without
   `targets "viewmage"` in `defaultConfig.externalNativeBuild.cmake` the build
   compiles libjxl's fuzzer runner. Note that `targets` belongs in
   `defaultConfig`, not in the block that sets `path`.
4. **The consumer compiles `android_platform.cc`.** `AndroidSurfaceProvider` /
   `AndroidAssetReader` live in vk_canvas's `platform/` layer, which neither
   `vk_canvas_core` (may not include platform SDK headers) nor
   `app_shell_android` (only uses them) builds. Same arrangement as
   native_app_glue.

## Layout

```
src/
  android_main.cc      read the intent's bytes, build host + app, run
  viewmage_app.cc/hh   the AppView: lifecycle, gestures, one quad
  view_transform.cc/hh fit/pan/zoom. PURE — no Vulkan, no Android, no libjxl
  jxl_image.cc/hh      bytes → RGBA8. The ONLY file that includes libjxl
```

`view_transform.hh` is pure on purpose: the entire feel of the viewer lives in
those clamp rules, so it is the part that must be provable rather than
eyeballed. `tests/view_transform_test.cc` covers fit, pan clamping at all four
edges, focal-point zoom, the floor and ceiling, double-tap, rotation and
degenerate input, on a desktop, in under a second. Keep it dependency-free.

## Changes made to app_shell (its own repo)

Both are generic — neither mentions JXL, images or ViewMage — and both pass
that library's stated test, "could a drawing program use it?"

- `read_intent_data_bytes()` + `AppShellActivity.readIntentData()`: the bytes
  the activity was launched to open. Native code cannot do this itself; a
  `content://` URI has no file behind it and the read grant belongs to the
  Intent, so only a `ContentResolver` reaches it.
- `AppView::onPointerDown/Move/Up(pointerId, x, y)` + `AndroidHost` iterating
  every pointer index. Pinch is unreachable otherwise: the host previously read
  `AMotionEvent_getX(event, 0)` and nothing else. The single-pointer synthesis
  (tap/drag/wheel) is unchanged and still pointer-0 only.

## Testing honestly

`tests/` runs on a desktop and covers the math and the decoder, including
truncated and corrupt input. **Rendering is verified visually on a device** —
vk_canvas itself has no automated rendering tests, and this project does not
claim one it does not have. Two-finger pinch has not been machine-verified
either: `adb shell input` cannot produce a real multi-pointer gesture. Say so
rather than implying coverage that is not there.

## Committing

`./git_wrapper commit "…"` / `./git_wrapper push` — never plain git. Push
`framework/app_shell` from inside itself first.
