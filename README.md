# Badminton Live HUD

ระบบ HUD แบดมินตันแบบเรียลไทม์จากกล้องตัวเดียว ตามผู้เล่น ตามลูก แสดง court map และสถิติ rally แล้วส่งเป็น overlay ใน OBS (ตอนนี้รองรับประเภทเดี่ยว)

> สถานะ: spec พร้อม, มี prototype แบบประมวลผลหลังถ่ายใน `prototype/offline-v0/` ระบบเรียลไทม์เริ่มจาก `tasks/T01-skeleton.md`

| ไฟล์ | เนื้อหา |
|---|---|
| [docs/GOALS.md](docs/GOALS.md) | เป้าหมายที่วัดได้ (G1–G7) และ non-goals |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | โครงสร้างระบบ หลักความเสถียร และ latency budget |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | รูปแบบข้อมูลระหว่างโมดูล |
| [docs/ROADMAP.md](docs/ROADMAP.md) | milestone S0–S4 และลำดับการ์ดงาน |
| [docs/REFERENCES.md](docs/REFERENCES.md) | งานอ้างอิง สิ่งที่ควรและไม่ควรทำตาม และเรื่อง license |
| [tasks/](tasks/) | การ์ดงาน T01–T11 |
| [prototype/offline-v0/](prototype/offline-v0/) | prototype จากคลิปจริง (ประเภทคู่): โค้ด, ข้อมูลที่วิเคราะห์แล้ว, หน้า HUD, ตัวเรนเดอร์ MP4 16:9 |
| [AGENTS.md](AGENTS.md) | กติกาสำหรับ AI agent ทุกค่าย |
| [docs/CODEX.md](docs/CODEX.md) | วิธีใช้ Codex กับ repo นี้ |
