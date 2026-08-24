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

## Colour management is not optional, and its absence does not look like its absence

The first version decoded with `JXL_TYPE_UINT8`, never subscribed to
`JXL_DEC_COLOR_ENCODING` and never called `JxlDecoderSetCms()`. libjxl then
hands back the file's own values with **no colour transform at all**, and they
were drawn as though they were sRGB.

The symptom was HDR photos looking washed out, which reads as a screen
limitation and is not one. Two things make this worth remembering:

- **It was wrong for ordinary sRGB images too**, just less visibly. Measured
  against libjxl's own `djxl` reference decode of the same file, the old path
  returned 60 and 188 where the reference says 52 and 194; the current path
  returns 53 and 195. The gradient was being pulled toward mid-grey.
- **The 8-bit truncation happened at the door**, so the extra range was gone
  before any control could have reached it. No slider could have recovered it.

The decoder now sets a CMS and asks for **linear light, sRGB primaries,
unclipped float**. Values below 0 (out of gamut) and far above 1 (highlights)
are meaningful and must survive to the GPU; the single clamp happens in the
fragment shader after exposure and the tone curve.

`jxl_dec` does **not** link `jxl_cms` (`lib/jxl.cmake:226-229` vs `:219`), so
`CMakeLists.txt` links it explicitly. Without that line `JxlGetDefaultCms()` is
an undefined reference.

### HDR output: asked for, granted, and verified on a panel

ViewMage requests `OutputTarget::ExtendedLinearScrgb` and then asks
`Renderer::hdrActive()` what it actually got — the request can be refused by
the driver, the compositor, or a window that was never put into HDR colour
mode, and none of those say so.

Verified on an S23 Ultra (SM-S918B): the driver enumerates 68 surface formats,
and the one taken is `R16G16B16A16_SFLOAT` +
`VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT` (format 97, colorspace 1000104002).
So FP16-extended-linear does exist on Samsung, and PQ is not the only Android
option — which answers two of the three open questions in vk_canvas's
`USAGE_hdr_output.md`. Reported headroom is **2.22x** SDR white.

**Not `Hdr10PQ`, deliberately.** Both are real HDR targets, but under PQ the
fixed-function blend mixes PQ code values by a coverage weight, and PQ is steep
enough that mixing code values is not mixing the luminances they stand for.
Antialiased edges and the letterbox come out wrong in a way that reads as bad
antialiasing rather than bad colour. scRGB blends in linear light.

**The tone curve rolls toward the panel, not toward SDR white.** vk_canvas's
`rolloffCurve()` takes a ceiling; ViewMage passes the display's headroom, so
highlights use the range the panel actually has. autoEV is kept — pinning 0 EV
because the panel is capable opens a dim file dim and calls it honesty.

The ceiling is clamped to the image's own white point inside the curve, and
that clamp is load-bearing rather than defensive: without it the curve
extrapolates instead of compressing, and a photo whose white point sits below
the panel's headroom gets its highlights multiplied (measured: 2.24x, over a
ceiling it also overshot). On a device that looked like coloured speckle over
every specular highlight. With the clamp, an image that fits the panel passes
through as an exact identity.

**Where the numbers come from.** `headroom` is not guessed: app_shell reads
`Display.getHdrCapabilities().getDesiredMaxLuminance()` and divides by BT.2408
graphics white (203). Desired, not maximum — the maximum is a peak the panel
sustains over a small window only, and mapping a whole image to it is how HDR
gets a reputation for being painful to look at.

### Auto-exposure only applies to HDR

libjxl reports `intensity_target = 255` for SDR content. Feeding that to the
BT.2408 203-nit rule darkened every ordinary photo by a third of a stop. An
image whose range the display can already show is returned at exactly 0 EV and
a white point of 1.0, which makes the tone curve a provable identity.
`tests/jxl_image_test.cc` asserts that exactly, not approximately.

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
- `maybeRequestHdrColorMode()` + `activity::display_hdr_headroom()`: an
  **opt-in** HDR window, requested through an `io.nava.appshell.HDR`
  `<meta-data>` and default off, plus the display's headroom. `setColorMode`
  must happen in `onCreate`, before the surface exists, because the colour mode
  can change which `VkSurfaceFormatKHR` pairs the driver enumerates. Native
  code cannot read `Display.HdrCapabilities` — there is no NDK equivalent, and
  the alternative is hardcoding a number and calling it headroom.
- `AppView::onPointerDown/Move/Up(pointerId, x, y)` + `AndroidHost` iterating
  every pointer index. Pinch is unreachable otherwise: the host previously read
  `AMotionEvent_getX(event, 0)` and nothing else. The single-pointer synthesis
  (tap/drag/wheel) is unchanged and still pointer-0 only.

## Testing honestly

`tests/` runs on a desktop and covers the math and the decoder, including
truncated and corrupt input. The tone curve is tested in vk_canvas
(`output_target_test`), not here: `rolloffCurve()` is the C++ authority and
`image_frag.slang` mirrors it, on the same terms as the PQ constants. **No
shader has been executed under test** — the shaders compile and pass
`spirv-val`, which is not the same as running.

**Rendering is verified visually on a device** —
vk_canvas itself has no automated rendering tests, and this project does not
claim one it does not have. Two-finger pinch has not been machine-verified
either: `adb shell input` cannot produce a real multi-pointer gesture. Say so
rather than implying coverage that is not there.

## Committing

`./git_wrapper commit "…"` / `./git_wrapper push` — never plain git. Push
`framework/app_shell` from inside itself first.
