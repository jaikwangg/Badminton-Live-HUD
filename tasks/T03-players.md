# T03 — Players worker (YOLO + ByteTrack)

**Goal:** ตามผู้เล่น 2 คนแล้วส่งตำแหน่งเท้าออกไปทุกเฟรม
**Owns:** `src/hud/models/detector*.py`, `src/hud/workers/players.py`, `tests/test_players.py`
**Depends:** T01

## ทำ
- `models/detector.py`: interface `Detector.infer(img) -> list[Det]` แยก backend เป็น `onnxruntime` และ `tensorrt` (เลือกจาก config)
- ใช้ YOLO11s (COCO class person) ขนาดภาพเข้า 640
- กรองเฉพาะคนที่ยืนอยู่ในคอร์ท (ใช้ homography จาก config ถ้ามี ถ้าไม่มีให้กรองด้วย ROI)
- ใช้ ByteTrack ตาม track ID แล้วคำนวณ `foot_px` จากจุดกึ่งกลางขอบล่างของ bbox
- อ่านเฉพาะเฟรมล่าสุดจากบัส แล้วส่ง message `players` และ heartbeat

## Acceptance
- [ ] ใช้คลิปทดสอบ 1 นาที: ID ของผู้เล่น 2 คนต้องไม่สลับกันเกิน 2 ครั้ง (นับด้วยตา แล้วบันทึกไว้ใน PR)
- [ ] เวลาประมวลผลต่อเฟรม (`proc_ms`) ≤8ms บน RTX 3060 (ใช้ ONNX ก่อนได้ ถ้าไม่มี GPU ให้เขียนว่า "needs GPU benchmark")
- [ ] สลับโมเดลได้ด้วยการแก้ `models.players.path` ใน config อย่างเดียว
