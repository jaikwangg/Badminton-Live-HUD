# T06 — Shuttle worker (TrackNetV3)

**Goal:** หาตำแหน่งลูกขนไก่ในภาพทุกเฟรม
**Owns:** `src/hud/models/shuttle*.py`, `src/hud/workers/shuttle.py`, `tests/test_shuttle*.py`, `tools/export_trt.py`
**Depends:** S1

## โมเดล: ใช้ checkpoint ทางการของ TrackNetV3
- ไฟล์ `TrackNet_best.pt` จาก https://github.com/qaz812345/TrackNetV3 (MIT) อยู่ใน `TrackNetV3_ckpts.zip` ที่ `D:\Badminton-Live-HUD-data` ห้าม commit เข้า repo
- checkpoint นี้ตรวจแล้วว่าเก็บข้อมูลแบบนี้: dict ที่มี key `model`, `param_dict` โดยมี `param_dict.seq_len = 8` และ `param_dict.bg_mode = "concat"`
- **ภาพเข้า 27 channel** = ภาพพื้นหลัง (median) 1 ภาพ + 8 เฟรมติดกัน (3 channel ต่อภาพ) ขนาด 512×288 **ภาพออก 8 heatmap** (1 ใบต่อเฟรม)
- ลำดับ channel, ลำดับสี, วิธี normalize และวิธีคำนวณภาพพื้นหลัง ต้องตรงกับโค้ดใน repo ทางการ (`dataset.py`, `test.py`) ให้อ่านจากโค้ดจริงแล้วบันทึกไว้ใน PR ห้ามเดา

## ทำ
- `tools/export_trt.py`: โหลดคลาส `TrackNet` จาก repo ทางการที่ clone ไว้ (ใช้ `--repo` ห้าม copy โค้ดเข้ามา) โดยอ่าน `seq_len` และ `bg_mode` จาก `param_dict` แล้วโหลด `ckpt["model"]` ด้วย `strict=True` ถ้าโหลดไม่ได้ **ต้องหยุดพร้อม error** ห้าม export โมเดลที่ weight ยังเป็นค่าสุ่ม จากนั้น export เป็น ONNX และถ้าสั่งไว้ให้ build `.engine` ด้วย `trtexec`
- interface `ShuttleTracker.infer(frames, background) -> (visible, px, conf)` แยก backend เป็น ONNX และ TensorRT ขนาด input อ่านจากโมเดล ไม่ hard-code
- worker เก็บ 8 เฟรมล่าสุดไว้ในหน่วยความจำของตัวเอง ถ้าเฟรมข้ามไม่เกิน `max_gap` ให้ใช้เฟรมล่าสุดเติมช่องที่ขาด ถ้าข้ามเกินกว่านั้นให้ reset หน้าต่าง ค่าทั้งหมดอยู่ใน config
- เก็บภาพทุกเฟรมที่ย่อแล้วไว้ใน deque ไม่ต้องย่อใหม่ซ้ำทุกครั้ง และใช้ `cv2.resize` แบบเดียวกับ repo ทางการ ไม่ใช้การหยิบ pixel แบบ nearest
- ภาพพื้นหลัง: คำนวณ median จากเฟรมสุ่ม N เฟรมตอนเริ่ม และอัปเดตทุก M วินาทีใน thread แยก ห้ามคำนวณใน loop เฟรม (N, M อยู่ใน config) ระหว่างที่ยังไม่มีภาพพื้นหลัง ให้ส่ง `visible=false, conf=null`
- ใช้ heatmap ของ **เฟรมล่าสุด** จึงไม่เพิ่ม latency จากการรอหน้าต่าง จากนั้น threshold, morph open, กรอง contour ตามขนาด แล้วเอาจุดกึ่งกลางของ contour ที่มั่นใจที่สุด (ค่าทั้งหมดอยู่ใน config) แล้วแปลงพิกัดกลับเป็นพิกเซลของภาพต้นฉบับด้วย `(x+0.5)*W/w`
- เฟรมที่ไม่ได้ infer (หน้าต่างยังไม่ครบหรือยังไม่มีภาพพื้นหลัง) ให้ส่ง `conf=null` เพื่อให้แยกได้จากกรณีที่ "infer แล้วแต่ไม่เห็นลูก" (`conf` เป็นตัวเลข)
- ถ้า infer error ห้ามให้ worker ล่ม ให้ log ไว้แล้วส่ง `visible=false, conf=null` และส่ง heartbeat ต่อตามปกติ ระหว่างโหลดโมเดลและ warm-up ต้องส่ง heartbeat ด้วย
- ส่ง message `shuttle` และ heartbeat (ส่งออก stdout ด้วย เพราะ supervisor อ่านจาก stdout)

## คลิปทดสอบ (ชั่วคราว)
ใช้ `IMG_3354.mov` / `IMG_3355.mov` (4K HEVC 60fps) ไปก่อน ถึงจะเป็นคลิปประเภทคู่ แต่ใช้ทดสอบการตามลูกได้ ห้าม commit วิดีโอ เมื่อมีคลิปประเภทเดี่ยวแล้วให้เปลี่ยนไปใช้คลิปนั้น

## Acceptance
- [ ] export จาก `TrackNet_best.pt` ได้ ONNX ที่โหลด weight ครบทุกตัว (`strict=True`) และแนบ log ไว้ใน PR
- [ ] test: ใส่ checkpoint ผิดรูปแบบแล้วต้อง export ไม่ผ่าน และ infer error แล้ว worker ต้องไม่ล่ม
- [ ] `proc_ms` ≤10ms บน RTX 3060 (ถ้าไม่มี GPU ให้เขียนว่า "needs GPU benchmark")
- [ ] kill worker นี้แล้ว HUD ยังแสดงข้อมูลผู้เล่นได้ปกติ
- [ ] ใช้คลิปทดสอบ 1 นาที แล้วทำวิดีโอที่วาดจุดลูกทับไว้ แนบใน PR (ยังไม่ต้องวัดความแม่น)
