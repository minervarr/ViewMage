#!/usr/bin/env bash
# Fetch the neural raw-denoise model into app/src/main/assets/models/.
#
# RawNIND UtNet2 (Brummer & De Vleeschouwer), the model darktable 5.6 ships as
# Neural Restore -> Raw Denoise. GPL-3.0, compatible with this app's AGPL-3.0.
#   paper:  https://arxiv.org/abs/2501.08924
#   source: https://github.com/darktable-org/darktable-ai
#
# The release tag and the SHA-256 below are PINNED. A model is a dependency like
# any other: silently picking up a different one would change every photo the app
# renders, with nothing in the diff to show why.
set -euo pipefail

TAG="release-5.6.0"
BUNDLE="rawdenoise-nind.dtmodel"
SHA256="d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af"
URL="https://github.com/darktable-org/darktable-ai/releases/download/${TAG}/${BUNDLE}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${root}/app/src/main/assets/models"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

echo "Fetching ${BUNDLE} (${TAG})..."
curl -fsSL -o "${tmp}/${BUNDLE}" "${URL}"

echo "${SHA256}  ${tmp}/${BUNDLE}" | sha256sum -c - || {
    echo "CHECKSUM MISMATCH — refusing to install this model." >&2
    exit 1
}

mkdir -p "${dest}"
# Only the Bayer variant. model_linear.onnx is for X-Trans/Foveon sensors, which
# this camera is not, and it is another 31 MB in the APK for nothing.
unzip -p "${tmp}/${BUNDLE}" "rawdenoise-nind/model_bayer.onnx" > "${dest}/model_bayer.onnx"

echo "Installed ${dest}/model_bayer.onnx ($(du -h "${dest}/model_bayer.onnx" | cut -f1))"
