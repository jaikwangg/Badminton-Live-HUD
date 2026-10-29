# T07 — Audio hit worker

**Goal:** จับจังหวะตีลูกจากเสียง
**Owns:** `src/hud/workers/audio.py`
**Depends:** S1

## ทำ
- รับเสียงจากไมค์หรือ track เสียงของ source (ถ้าเป็นไฟล์ให้ดึง track เสียงจากไฟล์)
- ใช้ onset detection แบบ band-pass ช่วง 2–8 kHz ให้ threshold ปรับตามเสียงรบกวนรอบข้างเองได้ และไม่จับซ้ำภายใน 150ms
- แปลงเวลาเสียงให้เป็นนาฬิกาเดียวกับภาพ (`ts` monotonic) โดยมีค่า offset ใน config
- ส่ง message `hit` และ heartbeat

## Acceptance
- [ ] ใช้คลิปทดสอบ 1 นาที แล้วนับด้วยตาเทียบกับที่ระบบจับได้ ใส่ตัวเลขใน PR
- [ ] ถ้าไม่มีไมค์ worker ต้องขึ้นสถานะ `down` โดยไม่ทำให้ระบบล่ม
