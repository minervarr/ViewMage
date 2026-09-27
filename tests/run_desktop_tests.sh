#!/usr/bin/env bash
# Desktop test runner. The Android build does NOT build these: they exist so
# the pure logic can be checked without a phone, which is the only reason it
# was kept pure.
#
# view_transform needs nothing at all. jxl_image needs libjxl — the SYSTEM one
# here, via pkg-config, rather than the cross-compiled submodule in
# thirdparty/. Same upstream version, and it keeps a desktop test from having
# to configure an Android toolchain.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${TMPDIR:-/tmp}/viewmage-tests"
mkdir -p "$OUT"

echo "== view_transform_test =="
g++ -std=c++17 -O1 -Wall -o "$OUT/view_transform_test" \
    ../core/tests/view_transform_test.cc ../core/src/view_transform.cc \
    -I ../core/include
"$OUT/view_transform_test"

echo "== outline_place_test =="
g++ -std=c++17 -O1 -Wall -o "$OUT/outline_place_test" \
    ../core/tests/outline_place_test.cc \
    -I ../core/include
"$OUT/outline_place_test"

# The tone curve lives in vk_canvas, and so does its test — but vk_canvas only
# builds its tests from the WINDOWS build, so on a Linux dev machine nothing
# ever ran it. A test nobody runs is a test that rots, and this one guards the
# claim that an SDR image is bit-identical through the HDR-capable curve.
# It is pure math over vulkan.h types: no GPU, no device, no Vulkan loader.
echo "== output_target_test (vk_canvas) =="
VKC=../framework/vk_canvas
if [ ! -f "$VKC/core/output_target.cc" ]; then
    echo "SKIPPED: vk_canvas submodule not initialised"
elif ! echo '#include <vulkan/vulkan.h>' | g++ -x c++ -fsyntax-only - 2>/dev/null; then
    echo "SKIPPED: no Vulkan headers (install vulkan-headers)"
else
    g++ -std=c++17 -O1 -Wall -o "$OUT/output_target_test" \
        "$VKC/core/tests/output_target_test.cc" "$VKC/core/output_target.cc" \
        -I "$VKC/core"
    "$OUT/output_target_test"
fi

# dng_image needs nothing at all -- no libjxl, no GPU, no device. It builds its
# own DNG fixture in memory, so it runs everywhere and there is no excuse for
# skipping it. Pass a real .dng as $1 to also develop that.
echo "== dng_image_test =="
g++ -std=c++17 -O1 -Wall -o "$OUT/dng_image_test" \
    ../core/tests/dng_image_test.cc ../core/src/dng_image.cc ../core/src/decoded_image.cc ../core/src/ai_denoise.cc \
    -I ../core/src -I ../core/include -I ../framework/vk_canvas/core
"$OUT/dng_image_test" ${1:+"$1"}

echo "== document_scan_test =="
ORT=../thirdparty/onnxruntime
PAGE=../android/app/src/main/assets/models/page_quad.onnx
if [ -f "$ORT/lib/x86_64/libonnxruntime.so" ] && [ -f "$PAGE" ]; then
    g++ -std=c++17 -O1 -Wall -DVIEWMAGE_WITH_ORT -o "$OUT/document_scan_test" \
        ../core/tests/document_scan_test.cc ../core/src/document_scan.cc \
        ../core/src/enhance.cc \
        -I ../core/include -I "$ORT/include" \
        -L "$ORT/lib/x86_64" -lonnxruntime \
        -Wl,-rpath,"$ORT/lib/x86_64"
else
    echo "(page model or ONNX Runtime absent — luminance detector only)"
    g++ -std=c++17 -O1 -Wall -o "$OUT/document_scan_test" \
        ../core/tests/document_scan_test.cc ../core/src/document_scan.cc \
        ../core/src/enhance.cc \
        -I ../core/include
fi
"$OUT/document_scan_test"

echo "== jxl_image_test =="
if ! pkg-config --exists libjxl libjxl_threads libjxl_cms; then
    echo "SKIPPED: no system libjxl (install it, or build thirdparty/libjxl for the host)"
    exit 0
fi
g++ -std=c++17 -O1 -Wall -o "$OUT/jxl_image_test" \
    ../core/tests/jxl_image_test.cc ../core/src/jxl_image.cc ../core/src/decoded_image.cc \
    -I ../core/src -I ../core/include -I ../framework/vk_canvas/core \
    $(pkg-config --cflags --libs libjxl libjxl_threads libjxl_cms)
"$OUT/jxl_image_test" ../core/tests
