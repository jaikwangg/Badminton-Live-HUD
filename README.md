# Badminton Live HUD

ระบบ HUD แบดมินตันแบบเรียลไทม์จากกล้องตัวเดียว ตามผู้เล่น ตามลูก แสดง court map และสถิติ rally แล้วส่งเป็น overlay ใน OBS (ตอนนี้รองรับประเภทเดี่ยว)

> สถานะ: spec พร้อม, มี prototype แบบประมวลผลหลังถ่าย 2 ชุด (`prototype/offline-v0/`, `prototype/doubles-v2/`) ระบบเรียลไทม์เริ่มจาก `tasks/T01-skeleton.md` ดูภาพรวมที่ [docs/PROGRESS.md](docs/PROGRESS.md)

| ไฟล์ | เนื้อหา |
|---|---|
| [docs/PROGRESS.md](docs/PROGRESS.md) | ความคืบหน้าทีละ stage, ผล, และสิ่งที่ต้องทำต่อ |
| [docs/LESSONS.md](docs/LESSONS.md) | สิ่งที่ได้เรียนรู้และข้อบกพร่องจาก prototype |
| [docs/GOALS.md](docs/GOALS.md) | เป้าหมายที่วัดได้ (G1–G7) และ non-goals |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | โครงสร้างระบบ หลักความเสถียร และ latency budget |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | รูปแบบข้อมูลระหว่างโมดูล |
| [docs/ROADMAP.md](docs/ROADMAP.md) | milestone S0–S4 และลำดับการ์ดงาน |
| [docs/REFERENCES.md](docs/REFERENCES.md) | งานอ้างอิง สิ่งที่ควรและไม่ควรทำตาม และเรื่อง license |
| [tasks/](tasks/) | การ์ดงาน T01–T11 |
| [prototype/offline-v0/](prototype/offline-v0/) | stage 1: คลิปแนวนอน ประเภทคู่ ตามผู้เล่น + HUD เว็บ/MP4 16:9 และ 9:16 |
| [prototype/doubles-v2/](prototype/doubles-v2/) | stage 2: คลิปมือถือแนวตั้ง ประเภทคู่ มีตามลูก ช็อต และความเร็วออกไม้ พร้อม HUD 1080×1920 |
| [design/mockup-v0/](design/mockup-v0/) | mockup HUD และหน้า operator (ข้อมูลจำลอง) |
| branch [`media`](https://github.com/jaikwangg/Badminton-Live-HUD/tree/media) | คลิปต้นฉบับ วิดีโอ HUD และภาพดีบักของแต่ละ stage (แยกจากโค้ด) |
| [AGENTS.md](AGENTS.md) | กติกาสำหรับ AI agent ทุกค่าย |
| [docs/CODEX.md](docs/CODEX.md) | วิธีใช้ Codex กับ repo นี้ |
