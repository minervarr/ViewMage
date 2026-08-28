#!/usr/bin/env bash
# Desktop Linux build -> <repo>/build/linux (Release) or build/linux_debug (Debug)
# Prereqs: cmake >= 3.22, ninja, a C++17 compiler, wayland-client dev headers,
# a Vulkan loader + headers, and the Slang shader compiler (slangc).
#
# Usage: scripts/linux/build.sh [--debug|--release|--clean] [cmake args...]
set -euo pipefail
cd "$(dirname "$0")/../.."

BUILD_TYPE=Release
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --debug)   BUILD_TYPE=Debug ;;
        --release) BUILD_TYPE=Release ;;
        --clean)   CLEAN=1 ;;
        *)         ;;
    esac
done

if [[ "$BUILD_TYPE" == "Debug" ]]; then
    BUILD_DIR="build/linux_debug"
else
    BUILD_DIR="build/linux"
fi

if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "Cleaning $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
fi

# Find slangc — same probing logic as Matrix_Player.
SLANGC_ARG=()
if [[ -z "${VULKAN_SDK:-}" ]]; then
    if command -v slangc >/dev/null 2>&1; then
        SLANGC_ARG=(-DVCE_SLANGC="$(command -v slangc)")
    else
        for _slangc in /opt/shader-slang-bin/bin/slangc \
                       /opt/shader-slang/bin/slangc; do
            if [[ -x "$_slangc" ]]; then
                SLANGC_ARG=(-DVCE_SLANGC="$_slangc")
                break
            fi
        done
    fi
fi

echo "Configuring CMake (Ninja, $BUILD_TYPE) -> $BUILD_DIR..."
cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    "${SLANGC_ARG[@]}"
cmake --build "$BUILD_DIR"

echo
echo "Binary in $BUILD_DIR/gui/viewmage"
echo "Run: $BUILD_DIR/gui/viewmage /path/to/photo.dng"
