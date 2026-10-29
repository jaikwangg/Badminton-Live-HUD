# T03 — Players worker (YOLO + ByteTrack)

**Goal:** ตามผู้เล่น 2 คนแล้วส่งตำแหน่งเท้าออกไปทุกเฟรม
**Owns:** `src/hud/models/detector*.py`, `src/hud/workers/players.py`, `tests/test_players.py`
**Depends:** T01

## ทำ
- `models/detector.py`: interface `Detector.infer(img) -> list[Det]` แยก backend เป็น `onnxruntime` และ `tensorrt` (เลือกจาก config)
- ใช้ **pose model รอบเดียว** (เริ่มจาก YOLO11s-pose ขนาดภาพเข้า 640) ได้ทั้ง bbox และข้อเท้า ห้ามรัน detect กับ pose แยก 2 รอบ
  - **license:** YOLO เป็น AGPL ให้เรียกผ่าน interface ใน `models/` เพื่อสลับไปใช้ RTMPose (Apache-2.0) ได้ภายหลัง (ดู `docs/REFERENCES.md`)
- `foot_px`: ใช้ค่าเฉลี่ยข้อเท้า (keypoint 15, 16) ถ้าความมั่นใจต่ำ ให้ใช้จุดกึ่งกลางขอบล่างของ bbox แทน
- กรองคนนอกคอร์ทด้วย polygon คอร์ทที่ขยายขอบ 20px และตัดกล่องที่อยู่สูงกว่าขอบคอร์ทด้านไกล
- ใช้ ByteTrack ตาม track ID แต่**กำหนดฝั่ง P1/P2 จากฝั่งของเส้นเน็ตทุกเฟรม** (ใช้ cross product) ไม่ยึดตาม track ID
- ต้องเขียนโค้ดเอง ห้ามก๊อปจาก repo อ้างอิงที่ไม่มี license
- อ่านเฉพาะเฟรมล่าสุดจากบัส แล้วส่ง message `players` และ heartbeat

## Acceptance
- [ ] ใช้คลิปทดสอบ 1 นาที: P1/P2 ต้อง**ไม่สลับเลย** (ประเภทเดี่ยว) นับด้วยตาแล้วบันทึกไว้ใน PR
- [ ] unit test การแบ่งฝั่งจากเส้นเน็ต ครอบคลุมกรณียืนบนเส้นพอดี
- [ ] เวลาประมวลผลต่อเฟรม (`proc_ms`) ≤8ms บน RTX 3060 (ใช้ ONNX ก่อนได้ ถ้าไม่มี GPU ให้เขียนว่า "needs GPU benchmark")
- [ ] สลับโมเดลได้ด้วยการแก้ `models.players.path` ใน config อย่างเดียว
