#!/usr/bin/env bash
# Put ONNX Runtime's Android arm64 .so and headers in thirdparty/onnxruntime/.
#
# A PREBUILT import, never compiled here: ORT's own build is a multi-hour
# cross-compile. This is the same binary camera_without_blood ships and has
# device-verified, so the two apps run the identical runtime.
#
# Without it the app still builds and runs — CMake warns, VIEWMAGE_WITH_ORT
# stays off, and photos develop without the neural denoise.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${root}/thirdparty/onnxruntime"
camera="${CAMERA_REPO:-${root}/../camera_without_blood}/libs/thirdparty/onnxruntime"

if [[ -f "${camera}/lib/arm64-v8a/libonnxruntime.so" ]]; then
    echo "Copying from ${camera}"
    mkdir -p "${dest}/lib/arm64-v8a"
    cp -r "${camera}/include" "${dest}/"
    cp "${camera}/lib/arm64-v8a/libonnxruntime.so" "${dest}/lib/arm64-v8a/"
else
    echo "Not found: ${camera}" >&2
    echo "Set CAMERA_REPO=/path/to/camera_without_blood, or unpack the" >&2
    echo "onnxruntime-android AAR: jni/arm64-v8a/libonnxruntime.so -> " >&2
    echo "  ${dest}/lib/arm64-v8a/, headers/ -> ${dest}/include/" >&2
    exit 1
fi
echo "Installed $(du -sh "${dest}" | cut -f1) into thirdparty/onnxruntime"
