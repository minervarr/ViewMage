# ViewMage — design

**Date:** 2026-08-23
**Status:** implemented and verified on device. Three sections were corrected
against what the build actually showed — §4 (draw order), §7 (no font is
needed) and §9 (a fourth limitation, since removed). Each correction is marked
below.

A JPEG XL image viewer for Android, built on the `app_shell` + `vk_canvas`
C++ framework, decoding with libjxl.

---

## 1. Purpose and scope

Open a `.jxl` file from a file manager and look at it. Fit to screen on open,
pinch to zoom, drag to pan, double-tap to toggle fit and 1:1.

That is the entire application. There is deliberately no gallery, no folder
paging, no thumbnail cache, no second image format and no settings screen.
Each of those was considered and cut: the viewer is reached from a file
manager, so the file manager already *is* the browser, and duplicating it
would mean a thumbnail cache and background decode threads for a feature the
user did not ask for.

### Success criteria

1. Tapping a `.jxl` in a file manager offers ViewMage, and choosing it shows
   the image.
2. The image is fitted to the screen on open, with correct aspect ratio.
3. Pinch zooms about the focal point; drag pans; neither can throw the image
   off-screen or shrink it below fit.
4. Every failure — no URI, unreadable URI, not a JXL, corrupt data, image
   too large for the GPU — produces a readable on-screen message, never a
   blank screen and never a crash.
5. The pan/zoom math is covered by desktop unit tests.

### Non-goals

Animated JXL sequences, JXL encoding, HDR tone mapping, colour management
beyond what libjxl's default sRGB output does, editing, sharing, printing.

---

## 2. Party rules, and why the directory names encode them

Three tiers, and the directory a dependency lives in states which one it is:

| Directory | Party | Rule |
|---|---|---|
| `framework/` | first — `minervarr` | Ours. May be modified, and is: see §5. Committed and pushed from inside itself, before the parent. |
| `firstparty/` | first — `minervarr` | Same, for non-framework libraries. Unused by ViewMage. |
| `thirdparty/` | third | **Never modified, never pushed.** Pinned at a specific commit, left detached. |

libjxl is not ours, so it is third party, so it is untouchable. Anything
ViewMage needs *from* libjxl that libjxl does not already provide is written
on our side of the boundary, in `src/jxl_image.cc`, never as a patch to the
submodule. `git_wrapper push` skips pinned/detached submodules automatically,
so this rule is enforced by the tooling and not only by discipline.

---

## 3. Repository layout

```
ViewMage/
├── framework/
│   ├── app_shell/            submodule → minervarr/App_shell
│   └── vk_canvas/            submodule → minervarr/Vk_Canvas_Lb_LAW
├── thirdparty/
│   └── libjxl/               submodule → libjxl/libjxl   (pinned)
├── src/
│   ├── android_main.cc       entry point; constructs the app and AndroidHost
│   ├── viewmage_app.{hh,cc}  the AppView — lifecycle, input, frame loop, draw
│   ├── view_transform.{hh,cc} fit/pan/zoom math; pure, no Vulkan, no Android
│   └── jxl_image.{hh,cc}     compressed bytes → RGBA8 + dimensions
├── app/
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/io/nava/viewmage/ViewMageActivity.java
│       └── assets/{shaders,fonts}/
├── tests/
│   ├── view_transform_test.cc
│   └── jxl_image_test.cc
├── CMakeLists.txt
├── build.gradle, settings.gradle, gradlew
└── CLAUDE.md, README.md, LICENSE
```

`reference/jxlviewer` (oupson's Kotlin app) is gitignored. It was read for its
libjxl build wiring and is not a dependency; no ViewMage code derives from it.

---

## 4. Data flow

One path, no branches:

```
File manager → Intent ACTION_VIEW, data = content://…/photo.jxl
    ↓
ViewMageActivity (Java)
    ContentResolver.openInputStream(getIntent().getData()) → byte[]
    ↓ JNI                                    [new capability in app_shell, §5a]
std::vector<uint8_t> compressed
    ↓ src/jxl_image.cc
    JxlDecoder, JXL_TYPE_UINT8, 4 channels → RGBA8 + width/height
    ↓
    if width or height > DeviceCaps max 2D image dimension:
        box-downsample by an integer factor until it fits
    ↓
Renderer::create_texture(rgba, w, h, mips = false)
    ↓ every dirty frame
ViewTransform → one ImageDraw quad → Canvas::imageFg() → ImageLayer
```

### CORRECTION: the foreground layer, not the background one

The design said `Canvas::image()`. That draws into ImageLayer's BACKGROUND
pass, which runs before the vector overlay composites — the right place for
album art sitting behind UI chrome, and the wrong place here. `Canvas::clear()`
is itself a full-screen overlay rect, so it painted over the photograph
entirely: the first build on device rendered a correct decode onto a screen
that showed nothing but background colour.

`imageFg()` and `Renderer::draw()`'s `foregroundImages` put it above the
overlay instead. The rule the mistake taught: in this engine the background
image layer is for what UI sits ON, and a viewer's photograph is not that — it
is the subject, and the only thing that may cover it is nothing.

### Why not `art_texture.hh`

`vk_canvas` already has `createTextureFromImageFile()`, and it is the wrong
tool twice over: it takes a **path** through the `AssetReader` seam, and a
`content://` URI is not a path; and it decodes through `img_decode_kit`,
which handles JPEG and the stb formats and has **no JXL support at all**.
Adding one would mean modifying `img_decode_kit` inside `vk_canvas`, which
would drag libjxl into every future `vk_canvas` consumer for a format only
this app cares about. So ViewMage hands `Renderer::create_texture()` its own
decoded bytes and leaves `vk_canvas` untouched.

### Why `mips = false`

A mip chain is what stops a texture drawn *much smaller* than its decode size
from aliasing. Here the image is decoded at full resolution and drawn at fit
scale or larger, and `ImageDraw`'s UVs let zoom go past 1:1 — so level 0 is
the only level worth sampling. Skipping the chain saves a third of the VRAM
and a per-upload blit pass, and avoids the sampler softening the image at
just-below-1:1 zoom, which is exactly where a viewer is looked at hardest.

---

## 5. Changes to `app_shell`

`app_shell` is first-party, so it is the right place for anything genuinely
generic. Both additions below pass that library's own stated test — *"could a
drawing program use it?"* — and neither mentions JXL, images or ViewMage.

### 5a. Reading the launch intent's DATA

`os/launch_intent.{hh,cc}` today offers only `read_string_extra()`, whose own
header comment concedes that a pure-NativeActivity app "cannot get SAF's
result back, because `onActivityResult` needs Java". A `content://` URI has
the same shape of problem: native code cannot `fopen()` it, and only a
`ContentResolver` can read it.

Added:

- `std::vector<uint8_t> read_intent_data_bytes(android_app*)` — returns the
  bytes the activity was launched to view, or empty when there is no data URI
  or it could not be read.
- `AppShellActivity` gains the Java half: open the intent's data URI through
  `ContentResolver`, read it fully, hand back a `byte[]`.

Empty return means "nothing to show" and is a normal state, not an error —
the app decides what that means, which is the same division `launchArgument()`
already follows.

Reading the whole file into memory is correct for this API's purpose: an
image, a document or a config file arriving by intent is bounded by what a
file manager would hand over. A streaming variant is future work, not v1.

### 5b. Multitouch

`AppView` exposes `onMouseMove` / `onLButtonDown` / `onLButtonUp` and nothing
else, and `AndroidHost::handleInputEvent()` reads only pointer index `0`
(`AMotionEvent_getX(event, 0)`). Pinch is therefore not reachable through the
current seam at any level. `core/gesture.hh` does not help: it is listed as
WIP and **not compiled** into `vk_canvas_core`, and it is single-pointer by
construction anyway.

Added to `AppView`, all defaulted empty so no existing consumer changes:

```cpp
virtual void onPointerDown(int pointerId, int x, int y) {}
virtual void onPointerMove(int pointerId, int x, int y) {}
virtual void onPointerUp  (int pointerId, int x, int y) {}
```

`AndroidHost` iterates every pointer index and handles
`ACTION_POINTER_DOWN` / `ACTION_POINTER_UP` in addition to the existing
actions. The existing single-pointer callbacks keep firing for pointer 0
exactly as they do now, so `AndroidHost`'s current behaviour is a strict
subset of its new behaviour.

Win32 and Wayland forward their single pointer as id `0`. That is a stated
narrowing in the same spirit as `registerHotkey` being system-wide on Windows
and focus-local on Wayland — a mouse genuinely has one pointer, and pretending
otherwise would be the fake.

### Push order

`framework/app_shell` is its own repository. Its changes are committed and
pushed **from inside `framework/app_shell/`, first**; ViewMage's own commit
(which records the new submodule SHA) follows. `git_wrapper push` already
pushes submodules before the parent, but the order is done deliberately, as
that repository's `CLAUDE.md` asks.

---

## 6. Components

### `view_transform.{hh,cc}` — the whole interaction model, purely

Holds image dimensions, viewport dimensions, `scale`, and `offsetX/offsetY`.

| Operation | Meaning |
|---|---|
| `fit()` | contain-scale the image and centre it; the state on open and after resize |
| `pan(dx, dy)` | translate, then clamp |
| `pinch(cx, cy, ratio)` | scale about a focal point, then clamp |
| `toggleFitOneToOne(cx, cy)` | double-tap: fit ⇄ 1:1 about the tap |
| `quad()` | the `ImageDraw` rect for this frame |

Clamping rules: scale never below fit scale and never above 32×; when the
image is smaller than the viewport on an axis it stays centred on that axis;
when larger, its edges cannot come inside the viewport edges.

This file knows nothing about Vulkan, Android, libjxl or `app_shell`, which is
what makes `tests/view_transform_test.cc` able to cover the entire feel of the
app on a desktop with no device and no GPU.

### `jxl_image.{hh,cc}` — bytes in, pixels out

```cpp
struct JxlImage {
    std::vector<uint8_t> rgba;      // w*h*4, straight alpha, top-to-bottom
    uint32_t w = 0, h = 0;
    std::string error;              // empty on success
};
JxlImage decode_jxl(const uint8_t* data, size_t size, uint32_t maxDimension);
```

Wraps libjxl's C API (`JxlDecoder`, `JXL_TYPE_UINT8`, 4 channels). Never
throws; a failure is a populated `error` string. `maxDimension` is the
device's Vulkan limit — when the decoded image exceeds it, the image is
box-downsampled by the smallest integer factor that fits, so the viewer shows
a large image slightly softened rather than showing nothing.

Also exposes `looks_like_jxl(data, size)` (signature check) so a file that is
not JXL produces a clear message rather than a decoder error.

### `viewmage_app.{hh,cc}` — the `AppView`

Owns the `Host`, the `Renderer`, the texture handle, the `ViewTransform`, and
a state enum: `kLoading`, `kReady`, `kError`. Gesture state (active pointers,
last positions, last tap time) lives here, feeding `ViewTransform`.

The loop is a dirty-flag loop — `host_->pump(haveWork)` then draw only when
dirty. An image viewer that is being looked at rather than touched idles at
**zero frames per second**, which is precisely the case `app_shell`'s "the
frame loop is the APP's, not the library's" decision exists to allow.

`onSurfaceLost()` / `onSurfaceRecreated()` matter here: the decoded RGBA is
kept in CPU memory so that returning to the app re-uploads the texture without
decoding again. That is the CPU-survives / GPU-does-not split `app_shell`
documents, and it is invisible until the second visit to the app.

---

## 7. Error handling

Every failure ends in `kError` with a specific message drawn as centred MSDF
text. There is no silent failure and no blank screen.

| Condition | Message |
|---|---|
| launched with no data URI | "No image to show" |
| URI could not be opened or read | "Could not read that file" |
| bytes are not JPEG XL | "Not a JPEG XL image" |
| libjxl reported a decode error | "Could not decode this image" |
| decoded, but Vulkan upload failed | "Not enough graphics memory" |

### CORRECTION: no font ships, and none is needed

The design accepted an `font.otf` in the APK and a `Renderer::initMsdf()` call
so that failures could be worded, calling it weight worth paying. It is not
needed at all: `Canvas`'s constructor takes `const Font* font`, and passing
`nullptr` selects the STROKE-FALLBACK glyphs built into the font engine —
which is exactly what vk_canvas's own Android demo does.

So ViewMage ships no font, calls no `initMsdf()`, and compiles no `msdf_*`
shader. The error line is drawn through the curve path that was already being
built for `clear()`. The tradeoff the design agonised over turned out not to
exist, which is the good kind of correction: the cheaper option was also the
one with fewer parts.

---

## 8. Testing, and what is not tested

Plain `assert()`, Debug-only, no framework — the convention this family of
repositories uses.

- `tests/view_transform_test.cc` — fit for wide/tall/square images in wide and
  tall viewports; pan clamping at all four edges; pinch about a focal point
  keeping that point stationary; the zoom floor and ceiling; double-tap
  round-tripping.
- `tests/jxl_image_test.cc` — a known small `.jxl` fixture decodes to the
  expected dimensions with expected pixels at known coordinates; truncated
  input, empty input and a non-JXL file each produce an error rather than a
  crash; an image above `maxDimension` comes back downsampled and in range.

**Rendering is verified visually on a device.** `vk_canvas` itself has no
automated rendering tests, and this project will not claim one it does not
have.

---

## 9. Known limitations

- **Peak memory on very large images.** The image is decoded at full
  resolution and only then downsampled, so peak usage is the full RGBA buffer.
  libjxl can decode progressively at reduced resolution, which is the correct
  fix; it is real work and deliberately not in v1.
- **Colour management is whatever libjxl's default sRGB output gives.** Wide
  gamut and HDR JXLs will be shown, but not tone-mapped to the display.
- **Animated JXL shows its first frame only.**

- **NOT a limitation, but it looked like one:** the first on-device build took
  **19.5 seconds** to decode a 3060x4080 photo. That was not libjxl and not the
  full-resolution decode above — Gradle's debug variant configures CMake with
  `CMAKE_BUILD_TYPE=Debug`, which compiled highway's SIMD kernels at `-O0`.
  ViewMage's `CMakeLists.txt` now forces `-O2` around libjxl's
  `add_subdirectory()` in every configuration; the same photo decodes in
  **0.37 s**, a 52x difference. Worth stating because a codec built at `-O0` is
  not a slow build of the same program, it is a different one, and the symptom
  looks exactly like a bad decoder.
- **Desktop hosts get no pinch**, because a mouse has one pointer. Stated in
  §5b rather than faked.
