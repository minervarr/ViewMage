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
    view_transform_test.cc ../src/view_transform.cc
"$OUT/view_transform_test"

echo "== jxl_image_test =="
if ! pkg-config --exists libjxl libjxl_threads; then
    echo "SKIPPED: no system libjxl (install it, or build thirdparty/libjxl for the host)"
    exit 0
fi
g++ -std=c++17 -O1 -Wall -o "$OUT/jxl_image_test" \
    jxl_image_test.cc ../src/jxl_image.cc \
    $(pkg-config --cflags --libs libjxl libjxl_threads)
"$OUT/jxl_image_test" .
