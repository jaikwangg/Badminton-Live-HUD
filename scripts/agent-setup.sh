#!/usr/bin/env bash
# Setup script สำหรับ cloud agent (Codex cloud environment, Claude Code บนเว็บ ฯลฯ)
# ใน Codex: Settings → Environments → เลือก repo นี้ → Setup script: bash scripts/agent-setup.sh
# หมายเหตุ: cloud agent ไม่มี GPU ให้ใช้ onnxruntime (CPU) สำหรับ test เท่านั้น
set -euo pipefail

python3 --version

if [ -f pyproject.toml ]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -e ".[dev]"
else
  echo "ยังไม่มี pyproject.toml (รอ T01) — ข้ามการติดตั้ง"
fi
