# ใช้ Codex กับ repo นี้

Codex อ่าน `AGENTS.md` เองอัตโนมัติ ไม่ต้องตั้งค่าเพิ่ม

## Codex CLI (บนเครื่อง)
```bash
git clone https://github.com/jaikwangg/Badminton-Live-HUD && cd Badminton-Live-HUD
codex "ทำ tasks/T01-skeleton.md ตามกติกาใน AGENTS.md แล้วเปิด PR"
```
ค่าใน `.codex/config.toml`:
- `approval_policy = "on-request"`: Codex จะถามก่อนทำอะไรที่เสี่ยง
- `sandbox_mode = "workspace-write"`: แก้ไฟล์ได้เฉพาะใน repo นี้

ถ้าใช้ GPU benchmark (`tools/benchmark.py`) ต้องรันบนเครื่องที่มี GPU

## Codex cloud
1. เชื่อม GitHub กับ Codex แล้วสร้าง environment ของ repo นี้
2. ใส่ setup script เป็น `bash scripts/agent-setup.sh`
3. สั่งทีละการ์ด เช่น `ทำ tasks/T05-overlay.md`
4. Codex cloud ไม่มี GPU ให้ใช้ ตามกติกาใน AGENTS.md ต้องเขียน "needs GPU benchmark" ไว้ใน PR

## รีวิวข้ามค่าย
- PR ที่ Claude หรือ Gemini เขียน: ให้ Codex รีวิว (comment `@codex review` ใน PR ถ้าเปิดใช้ Codex GitHub review ไว้ หรือใช้ `/review` ใน CLI)
- PR ที่ Codex เขียน: ให้ Claude หรือ Gemini รีวิวเทียบกับการ์ดงาน + `docs/CONTRACTS.md`

## แบ่งงานแนะนำ (หลัง T01 เสร็จ)

| Agent | การ์ด |
|---|---|
| Codex | T02 sources, T03 players |
| Claude | T04 fusion + server, T05 overlay |

แต่ละการ์ดแก้คนละโฟลเดอร์ จึงทำพร้อมกันได้โดยไม่ชนกัน
