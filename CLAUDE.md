# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in `ViewMage`.

## What this is

**A RAW post app for Android, and a viewer.** A file manager opens it with a
`.dng` (or a `.jxl`); it develops or decodes the file and shows the picture;
pinch, drag and double-tap move it around.

The application is C++. The only Java is a five-line Activity whose entire job
is `System.loadLibrary`.

### Why the scope changed, and what the new one is

This was "a JPEG XL viewer for Android", and that scope statement did real work:
it said there is no gallery, no folder paging, no second format and no settings
screen, and that additions must be argued against the name rather than into it.

What changed is upstream. `camera_without_blood` became **capture-only**: it
writes DNGs and nothing else (its JXL stills, and `libjxl` with them, were
removed). So a JXL-only viewer became an app for a format nothing produces,
while the files that *are* produced — RAW sensor measurements — had nothing to
open them. The old scope was not wrong; its subject moved.

The replacement scope, and it is meant to bind the same way:

> **Open a camera RAW file and show the scene as it was.** Developing is not an
> optional extra here — a DNG is measurements, not a picture, and turning it
> into one is the app's entire job. What earns a place is what serves that
> single render: develop, denoise, tone map, export. What does not is
> everything that turns a viewer into an editor.

Concretely, still true and still worth defending:

- **No editor.** The target is one good automatic render, not curves, layers or
  local adjustments. Sliders are a failure to make the automatic render right.
- **No gallery.** The app is opened *with* a file.
- **No second pipeline.** JXL and DNG both produce `DecodedImage` (linear light
  plus what the file said about itself) and share everything downstream — auto
  exposure, tone curve, GPU upload, details panel. A third format would be a
  third *producer*, never a parallel path.
- **JXL support stays.** It costs nothing now that it is one producer among
  others, and files already shot still open.

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
./tests/run_desktop_tests.sh                # all of them
./tests/run_desktop_tests.sh /path/to.dng   # also develop a real RAW file
```

`dng_image_test` needs **nothing** — no libjxl, no Vulkan, no device — because
it builds its own DNG in memory. It is the one test with no excuse for being
skipped, and it pins the parts of a TIFF reader that break silently: tag
parsing, inline-vs-offset values, the CFA mapping, black/white normalisation,
truncation handling, and the claim that a neutral scene develops neutral.

Pass a real `.dng` to also develop that and print its distribution. That check
is how the develop was validated against the camera: the same S23 Ultra frame
developed here and by the camera's own `raw_develop.cc` agreed to three
significant figures (median 25.9 vs 25.98 nits, p99 1788.7 vs 1788, peak 2345.9
vs 2344.6). Two independent implementations landing on the same numbers is the
only evidence worth having that the port is faithful.

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

Verified on an S23 Ultra (SM-S918B, API 36): the driver enumerates 68 surface
formats, and the one taken is `R16G16B16A16_SFLOAT` +
`VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT` (format 97, colorspace 1000104002).
So FP16-extended-linear exists on **this** phone and PQ is not the only Android
option. That is one device, not a generation sweep — vk_canvas's
`USAGE_hdr_output.md` keeps its third open question open for that reason.

**`setColorMode` is not what makes the format selection work.** Measured by
flipping the opt-in meta-data off: without the HDR colour mode the driver still
advertises the same 68 pairs, still selects format 97, still reports `hdr=1`.
The call is kept because it is what tells the compositor to treat the surface
as HDR — a different question, and one that experiment does not answer. Do not
read it as evidence the call is unnecessary.

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

**Where the numbers come from, and why they move.** `headroom` is not guessed.
app_shell prefers `Display.getHdrSdrRatio()` (API 34+), which is a **live**
measurement: SDR white is whatever the system is currently driving the panel
at, so the same screen has real headroom in a dark room and almost none
outdoors. Where that is unavailable it falls back to
`getHdrCapabilities().getDesiredMaxLuminance() / 203` — desired, not maximum,
because the maximum is a peak the panel sustains over a small window only, and
mapping a whole image to it is how HDR gets a reputation for being painful to
look at.

On the S23, `isHdrSdrRatioAvailable()` returns **false**, so the static
fallback is what actually runs: 450 nits / 203 = **2.22x**. The live path is
written but has not executed on any device here — say so rather than implying
it is exercised.

Because the value is live, `refreshHeadroom()` re-reads it on
`onSurfaceRecreated()` — Android's way of handing the app back after a trip
through the background, which is exactly when the brightness may have changed.

### Auto-exposure only applies to HDR

libjxl reports `intensity_target = 255` for SDR content. Feeding that to the
BT.2408 203-nit rule darkened every ordinary photo by a third of a stop. An
image whose range the display can already show is returned at exactly 0 EV and
a white point of 1.0, which makes the tone curve a provable identity.
`tests/jxl_image_test.cc` asserts that exactly, not approximately.

## The neural raw denoise (RawNIND UtNet2)

`src/ai_denoise.{hh,cc}` runs the model darktable 5.6 ships as Neural Restore ->
Raw Denoise. It is a **joint denoise AND demosaic**: in `[1,4,512,512]` packed
Bayer, out `[1,3,1024,1024]` **camera-native RGB**, ColorMatrix deliberately not
baked in. So it REPLACES the Malvar block in `dng_image.cc` rather than running
beside it, and WB -> highlight reconstruction -> CCM -> transfer -> tone curve are
shared by both paths. Model GPL-3.0, app AGPL-3.0 — compatible; the attribution
in `README.md` and the details panel is a **licence obligation**.

Ported from `darktable-ai`'s `models/rawdenoise-nind/demo.py` and verified
**bit-exact** against it (max abs diff `0.000e+00` over 7.5M values). Three
details are load-bearing, and each one fails looking like a *colour* bug:

- **Plane order `[R, G1, G2, B]`**, G1 being the first green in raster order.
- **Mirror-padded tiles, overlap TRIMMED not blended** — `T=512, overlap=64,
  step=384`, geometry baked into the ONNX export, not tunable.
- **One global gain match after stitching** (`mean(in)/mean(out)`). The net emits
  an arbitrary learned scale (`match_gain=output` at training). Skip it and every
  photo is the wrong brightness. Per tile instead of globally gives each tile its
  own exposure.

**Never feed it white-balanced data.** It was trained unbalanced; `g[]` stays
strictly downstream. This is the single easiest thing to "tidy" into a silent
quality regression.

Highlight reconstruction is **approximate on the AI path**: the pack clips input
to [0,1], so a blown channel arrives at the ceiling rather than above it. The two
paths are not identical there.

**Measured, S23 Ultra, 4080x3060:** model read 167 ms, session build <100 ms,
denoise **22.3 s**, shadow noise down **12x** (roughness 0.00411 -> 0.00035) with
text still sharp. Desktop x86 is ~211 ms/tile, 24 tiles.

Because 22 s is far too long to hold a blank screen, `viewmage_app.cc` does
**progressive refinement**: the ordinary develop is uploaded immediately, the
denoise runs on a worker, and `pollRefinement()` swaps the texture on the main
thread when it lands. A failed refinement keeps the picture that is already up.

**Gotcha that cost a measurement:** the model load was first done on the main
thread and appeared to take **six minutes**. It does not — the app had been
backgrounded with the screen off and Android had frozen the process. Wake the
device (`input keyevent KEYCODE_WAKEUP`, `svc power stayon true`) before timing
anything, or you will measure the scheduler instead of your code. The load is
still on the worker, where it belongs.

Neither the 31 MB model nor the 17 MB ORT `.so` is in git; `tools/fetch_model.sh`
(pinned tag + SHA-256) and `tools/fetch_onnxruntime.sh` fetch them, and CMake
warns and builds without the denoise when they are absent (`VIEWMAGE_WITH_ORT`).

## The POST bar and PNG export

`src/png_export.{hh,cc}` plus the bottom bar in `viewmage_app.cc`'s `draw()`.
One status line and one button, because the render is automatic — there is
nothing to adjust, and anything more would be a photo editor, which this is
deliberately not.

The export writes **8-bit sRGB**, which is a real loss of range and still the
right choice: 16-bit PQ renders as a washed, too-dark picture in anything that
is not colour-managed (the complaint that started all this), and JXL cannot be
decoded by the phone that shot the photo. The DNG remains the archive; the PNG
is the copy you send. What is written is **what was on screen** — same exposure,
same tone curve, same rolloff — so `rolloff()` here MIRRORS `image_frag.slang`
and vk_canvas's `rolloffCurve()`. Change one, change all three.

Two orderings are copied from the shader and matter for the same reasons:
negatives are clamped **before** the luminance dot product (otherwise
out-of-gamut speculars go magenta), and BT.2020 luma weights are used on wide
pixels (709's weights skew the ratio per hue and reappear as a green cast).

**Save is inert until the denoise lands.** The refinement takes ~26 s, and
during it the pixels on screen are the ordinary demosaic. Exporting then hands
the user a PNG with ~7x the shadow noise of the finished render and nothing in
the file to say it was the wrong one — which is exactly what happened in
testing. The button is dimmed while `refine_thread_` is in flight and
`doExport()` refuses independently, because the check that matters belongs next
to the thing it protects, not in `draw()`.

**`roughness_probe()` is the tool that caught it.** Mean `|2c - l - r|` on green,
sampled: Malvar reads ~0.023 on this scene and the neural output ~0.012 in
linear (0.050 vs 0.005 after tone mapping) — an order of magnitude apart in the
shadows. Both buffers are byte-identical in SIZE, so this is the only cheap way
to tell from a log which one is in memory. `export requested (... roughness ...)`
answers "did Save write the denoised pixels?" directly.

**Pixels are KEPT after upload on the DNG path** (`keep_pixels_`), against
`releasePixels()`'s usual rule. They are the app's largest allocation, and they
are also the only thing Save can write without spending another 26 s
re-denoising — for an app whose purpose is leaving with a file, that trade goes
the other way round than it does for a viewer.

**Scoped storage:** at `targetSdk 34` a plain filesystem write into
`/sdcard/Pictures` fails. `export_dir()` tries it, **probes it for writability**
(creating a directory is not proof of being allowed to write in it), and falls
back to the app's own external files dir. The UI then says "Saved to the app
folder" rather than claiming "Pictures" — naming the wrong place is how a user
loses a file. The proper fix is a MediaStore insert through a ContentResolver,
which is Java and belongs in `app_shell`, not here.

**Two touch bugs lived here, and an injected tap hid both.** Worth reading
before adding any control:

- **The surface is not the screen.** Measured on an S23 Ultra: surface
  **1440x2963**, screen **1440x3088**, so the window sits **125 px down**. A bar
  placed flush at the surface's bottom edge lands at screen y 3003..3076 —
  inside the **gesture-navigation strip**, which swallows the touch before the
  app sees it. The button drew, looked pressable, and was completely dead. The
  POST bar therefore keeps a `navGuard` margin above the bottom.
- **`safeInsets()` reports 0 on all four edges**, so it cannot be used to solve
  that. The proper fix is app_shell reporting the navigation inset; the margin
  is the stand-in until then.
- **Pointer and Canvas coordinates ARE the same space** (both surface-relative).
  Verified: a tap the app received at y=2915 hit a rect Canvas placed at
  2878..2951. Do not "correct" pointer coordinates by the insets.

**Why the automated test passed anyway:** `adb input tap` is in SCREEN space and
injects below the gesture layer, so a tap aimed 125 px BELOW the button arrives
as a value inside its rect and fires the export — while a real finger on the
button did nothing. `draw()` logs the rect once (`Save button at x,y wxh`)
because `adb screencap` returns an all-zero PNG on HDR content, so the control
cannot be seen. **Add 125 to the logged y to get where to inject, and never
treat a passing injected tap as proof a control is reachable.**

## The display tone curve

`apply_display_render()` in `decoded_image.cc`, called from the DNG path only —
a JXL arrives already rendered and curving it twice doubles the contrast. It is a
**toe, not an S-curve**: slope `kContrast` at mid grey, smoothstepped back to
exactly 1.0 by diffuse white, so **highlights are numerically untouched** (p99 and
peak are identical pre/post on real frames). The pivot is where auto-exposure
actually puts mid grey, so it is a contrast change and not an exposure shift.

Two bugs live in the first version of this, both caught by measuring:

- A straight log-log line (slope 1.25 everywhere) amplified a night frame's p99
  from 1526 to **3878 nits**. Highlights then get compressed a second time by the
  shader's rolloff, which is the pass that knows the display headroom, and
  stacking the two flattens speculars.
- A fixed *scene-referred* pivot put a dark scene almost entirely on the
  darkening side, reading as an underexposure.

The shared shader (`framework/.../image_frag.slang`) deliberately refuses S-curves
— it is an instrument. The rendering intent belongs here, in the POST app.

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
truncated and corrupt input. The tone curve's authority is `rolloffCurve()` in
vk_canvas, mirrored by `image_frag.slang` on the same terms as the PQ
constants; `run_desktop_tests.sh` builds and runs vk_canvas's
`output_target_test` too, because vk_canvas only builds its own tests from the
Windows build and a test nobody runs is a test that rots. **No
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
