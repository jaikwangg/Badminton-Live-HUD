# T02 — แหล่งภาพ (ไฟล์ / กล้อง / SRT)

**Goal:** ให้ทุกแหล่งภาพใช้ interface เดียวกัน และกล้องหลุดแล้วต่อกลับเองได้
**Owns:** `src/hud/sources/`, `tests/test_sources.py`
**Depends:** T01

## ทำ
- `base.py`: `class Source: open(), read() -> Frame | None, close()`
- `file_replay.py`: เล่นไฟล์ตามความเร็วจริงของคลิป (ไม่เร่งเกิน fps ของไฟล์) และมีตัวเลือก `loop: true`
- `camera.py`: กล้อง USB/V4L2 ผ่าน OpenCV
- `srt.py`: รับภาพ SRT ผ่าน GStreamer pipeline (ถ้าไม่มี GStreamer ให้ fallback ไปใช้ ffmpeg)
- watchdog: ถ้าไม่ได้เฟรมเกิน 1 วินาทีให้ reconnect โดยหน่วงเวลาเพิ่มขึ้นทีละขั้น สูงสุด 5 วินาที
- process `capture`: อ่านจาก source แล้วเขียนเข้า frame bus พร้อมส่ง heartbeat

## Acceptance
- [ ] เล่นไฟล์ 60fps แล้วได้เฟรมเข้าบัส 60±1 fps
- [ ] จำลองว่า source ไม่ส่งเฟรม 2 วินาที แล้วระบบต้องต่อกลับเองได้ภายใน 5 วินาที
- [ ] เปลี่ยนแหล่งภาพได้ด้วยการแก้ `source.type` ใน config อย่างเดียว
