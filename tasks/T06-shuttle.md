# T06 — Shuttle worker (TrackNetV3)

**Goal:** หาตำแหน่งลูกขนไก่ในภาพทุกเฟรม
**Owns:** `src/hud/models/shuttle*.py`, `src/hud/workers/shuttle.py`
**Depends:** S1

## ทำ
- แปลงโมเดล TrackNetV3 ที่เทรนเสร็จแล้วเป็น ONNX (เขียนขั้นตอนไว้ใน `tools/export_trt.py`) ขนาดภาพเข้า 512×288 หน้าต่าง 3 เฟรม
- interface `ShuttleTracker.infer(frames) -> (visible, px, conf)` แยก backend เป็น ONNX และ TensorRT
- เก็บ 3 เฟรมล่าสุดไว้ในหน่วยความจำของ worker เอง ถ้าเฟรมข้ามให้ reset หน้าต่าง
- **เช็กลำดับสี (RGB/BGR) และวิธี normalize ให้ตรงกับ repo ทางการของ TrackNetV3** แล้วบันทึกไว้ใน PR
- หลังประมวลผล: รวม heatmap 3 ใบโดยให้น้ำหนักตามความมั่นใจ, threshold และ morph open, กรอง contour ตามขนาด (ค่าทั้งหมดอยู่ใน config) ดูแนวคิดใน `docs/REFERENCES.md`
- ส่ง message `shuttle` และ heartbeat

## Acceptance
- [ ] `proc_ms` ≤10ms บน RTX 3060 (ถ้าไม่มี GPU ให้เขียนว่า "needs GPU benchmark")
- [ ] kill worker นี้แล้ว HUD ยังแสดงข้อมูลผู้เล่นได้ปกติ
- [ ] ใช้คลิปทดสอบ 1 นาที แล้วทำวิดีโอที่วาดจุดลูกทับไว้ แนบใน PR (ยังไม่ต้องวัดความแม่น)
