# T11 — TensorRT, benchmark, soak test

**Goal:** พิสูจน์ว่าระบบผ่าน G1 และ G2
**Owns:** `tools/export_trt.py`, `tools/benchmark.py`, `tools/soak_test.py`
**Depends:** S2

## ทำ
- `export_trt.py`: แปลง ONNX เป็น TensorRT FP16 ทั้ง YOLO และ TrackNet
- `benchmark.py --replay <clip>`: วัด fps ต่อ worker, latency end-to-end p50/p99 (นับจาก capture_ts จนส่ง WS ออก), จำนวนเฟรมที่ drop และ GPU% แล้วส่งออกเป็น JSON + ตาราง
- `soak_test.py --hours 3`: เล่นคลิปวน, kill worker แบบสุ่มทุก 15 นาที, บันทึก RAM/VRAM ทุก 1 นาที, สรุปผลผ่านหรือไม่ผ่านตาม G2/G3

## Acceptance
- [ ] รายงาน benchmark บน RTX 3060 ผ่าน G1
- [ ] soak test 3 ชม. ผ่าน G2 และ G3 แนบกราฟหน่วยความจำไว้ใน PR
