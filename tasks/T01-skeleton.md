# T01 — โครง repo, frame bus, supervisor

**Goal:** สร้างโครงพื้นฐานที่การ์ดอื่นทุกใบต้องใช้
**Owns:** `pyproject.toml`, `config/`, `src/hud/bus/`, `src/hud/config.py`, `src/hud/supervisor.py`, `src/hud/health.py`, `tests/`, `.gitignore`, `docker/`
**Depends:** —

## ทำ
- สร้าง `pyproject.toml` (Python 3.11), `ruff`, `pytest`, `.gitignore` (ไม่เอาไฟล์โมเดล วิดีโอ และ `recordings/` เข้า repo)
- `config/default.yaml` + `config.py` อ่าน config เป็น dataclass มีค่า default ครบทุกตัว
- `bus/messages.py`: สร้าง dataclass ตาม CONTRACTS ครบทุก message พร้อม `to_json` และ `from_json`
- `bus/shm_ring.py`: ring buffer 4 ช่องบน shared memory ใช้ seqlock และมีฟังก์ชัน `write(frame, ts)` กับ `read_latest()`
- `supervisor.py`: เปิด worker ตามที่ระบุใน config, รับ heartbeat, restart worker ที่เงียบเกิน 3 วินาที
- `docker/Dockerfile` (CUDA 12 runtime, Ubuntu 22.04)

## Acceptance
- [ ] ทดสอบ shm_ring: มี writer 1 ตัวกับ reader 2 ตัว นาน 10 วินาที ต้องไม่มีภาพที่อ่านขณะยังเขียนไม่เสร็จ
- [ ] แปลง message ทุกชนิดไป-กลับ (JSON → dataclass → JSON) แล้วได้ค่าเท่าเดิม
- [ ] supervisor restart worker จำลองที่ถูก kill ได้ภายใน 3 วินาที
- [ ] `ruff check` และ `pytest` ผ่าน

**Out of scope:** โมเดล, หน้าเว็บ
