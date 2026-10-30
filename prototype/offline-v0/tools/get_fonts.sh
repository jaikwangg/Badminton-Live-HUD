#!/usr/bin/env bash
# ดาวน์โหลดฟอนต์ (SIL Open Font License) ที่ render_mp4.py ใช้ ไปไว้ที่ ./fonts
set -euo pipefail
mkdir -p fonts
for f in ofl/ibmplexsansthai/IBMPlexSansThai-SemiBold.ttf ofl/ibmplexsansthai/IBMPlexSansThai-Medium.ttf \
         ofl/barlowcondensed/BarlowCondensed-SemiBold.ttf ofl/barlowcondensed/BarlowCondensed-Bold.ttf ofl/barlowcondensed/BarlowCondensed-Medium.ttf; do
  curl -sSfL -o "fonts/$(basename "$f")" "https://raw.githubusercontent.com/google/fonts/main/$f"
done
echo "fonts ready"
