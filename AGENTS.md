# AGENTS.md — ViewMage

A RAW post app (DNG develop + neural denoise) and JXL viewer for Android,
C++ end to end. Start by reading **CLAUDE.md** — it is the project's
hard-won engineering log and stays current; this file only adds what it does
not say about working here. Do not duplicate its content back.

## Layout (current)

| Dir | What |
|---|---|
| `core/` | portable static lib (`viewmage_core`): DNG/JXL decode, develop, `ai_denoise`, enhance, export math. No Vulkan, no OS headers |
| `gui/` | Linux Wayland app + POST bar logic shared with Android (`viewmage_app.cc`) |
| `cli/` | `viewmage_cli` headless develop-to-PNG plus measurement tooling |
| `android/` | Gradle project (gradlew lives HERE, not at repo root) |
| `framework/app_shell`, `framework/vk_canvas` | minervarr submodules — modifiable, own repos |
| `thirdparty/` | pinned/detached — **never modify, never push** (libjxl v0.12.0) |

## Build & test commands

```bash
./tests/run_desktop_tests.sh                    # all suites, no device needed
./tests/run_desktop_tests.sh /path/to.dng       # also develops that RAW

cmake -B build -DCMAKE_BUILD_TYPE=Debug -DVCE_SLANGC=/opt/shader-slang/bin/slangc
cmake --build build --target viewmage_cli -j    # or: viewmage (desktop GUI)

(cd android && ./gradlew assembleDebug)         # APK: android/app/build/outputs/apk/debug/
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

Gotchas, all hit the hard way:

- **Desktop GUI builds need `-DVCE_SLANGC=...`** — there is no Vulkan SDK on
  the dev machine; a stale Windows path in the build cache fails the shader
  step silently until reconfigured.
- **CLI denoise needs ORT on the loader path**:
  `LD_LIBRARY_PATH=thirdparty/onnxruntime/lib/x86_64 ./build/cli/viewmage_cli …`
  The x86_64 `.so` is installed manually (gitignored); Android's arm64 comes
  from `tools/fetch_onnxruntime.sh`. CMake warns and builds without the
  denoiser when absent — check `VIEWMAGE_WITH_ORT` before assuming your change
  was exercised.
- **Model + ORT are gitignored binaries**; `tools/fetch_model.sh` +
  `tools/fetch_onnxruntime.sh` fetch them.
- Stale-build paranoia: after editing `core/`, confirm object timestamps
  actually moved before trusting "build successful".

## Measurement workflow (project convention)

Never fix image quality by eye alone; both diagnostic modes are first-class:

```bash
LD_LIBRARY_PATH=… ./build/cli/viewmage_cli FILE.dng --analyze            # clip census, LED bloom rings, CA check
LD_LIBRARY_PATH=… ./build/cli/viewmage_cli FILE.dng --compare [X Y W H]  # Malvar vs AI single vs x4 ensemble
```

Test DNGs live in `test_dngs/photo/`. Record before/after numbers in the
commit message or CLAUDE.md when a claim is being made.

## Commits

`./git_wrapper commit "…" / push` — never plain git. Push `framework/app_shell`
from inside itself first. Never commit unless explicitly asked; screenshots
and exported PNGs pulled during device debugging do not belong in git.

## Device-testing traps (details in CLAUDE.md)

- Injected taps are SCREEN space; the surface sits ~125 px down — add it to
  logged button y.
- After `am start` on a running activity, check `dumpsys input` for
  `touchableRegion=<empty>`: every tap dies silently; force-stop first.
- Wake the phone (`KEYCODE_WAKEUP`, `svc power stayon true`) before timing
  anything, or you measure the scheduler instead of your code.
