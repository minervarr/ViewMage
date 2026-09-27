#!/usr/bin/env bash
# DocAligner LC100 heatmap (DocsaidLab). Four corners of a photographed page.
# Apache-2.0. The file id and the SHA-256 are pinned: a different file would
# move every rectangle the scanner finds, with nothing in the diff to say why.
#   https://github.com/DocsaidLab/DocAligner
set -euo pipefail

FILE_ID="1IlbLPkCv-TdaBLOPh_4J97P1_KHYYJ7a"
SHA256="f4117b786e3a18470f3865c93f3c2bd69d9b998edd60f385574a5c665e79594e"
URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${root}/android/app/src/main/assets/models/page_quad.onnx"
tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

echo "Fetching DocAligner LC100..."
curl -fL -o "${tmp}" "${URL}"

echo "${SHA256}  ${tmp}" | sha256sum -c - || {
    echo "CHECKSUM MISMATCH — refusing to install this model." >&2
    exit 1
}

mkdir -p "$(dirname "${dest}")"
cp "${tmp}" "${dest}"
echo "Installed ${dest} ($(du -h "${dest}" | cut -f1))"
