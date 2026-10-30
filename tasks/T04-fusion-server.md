# T04 — Fusion v0 + WS server + operator API

**Goal:** รวมผลจาก worker ทั้งหมดเป็น GameState แล้วส่งออกผ่าน WebSocket
**Owns:** `src/hud/fusion/homography.py`, `src/hud/fusion/state.py`, `src/hud/server/`, `tools/calibrate.py`
**Depends:** T01

## ทำ
- `homography.py`: รับมุมคอร์ท 4 มุม (pixel) แล้วคำนวณ H เพื่อแปลง `foot_px` เป็น `court_xy_m`, บันทึกเป็น `config/court_calib.json`
- `state.py`: subscribe ข้อมูลจาก worker แล้วสร้าง GameState ตาม CONTRACTS ทุกฟิลด์ต้องมีค่า (ถ้าไม่รู้ให้ใส่ `null`), คำนวณ `speed_mps` และ `dist_m` แบบ smoothing, ถ้า worker ไหนไม่ส่ง heartbeat ให้ตั้งสถานะเป็น `down`
  - ระบุตัวตน: หา `side` จากเส้นเน็ตทุกเฟรม แล้วจับคู่คนในฝั่งเดียวกันกับ `id` ด้วย `app` + ตำแหน่งเฟรมก่อน (Hungarian) **ห้าม map จาก `track_id` ตรงๆ** ประเภทเดี่ยวมีฝั่งละคนจึงใช้ `side` อย่างเดียวได้
- `ws_server.py`: ส่ง GameState 30Hz ที่ `:8765`
- `operator_api.py`: ทำ endpoint ตาม CONTRACTS (ขั้นนี้ทำแค่ calibrate, players, score, swap_ids)
- `tools/calibrate.py`: หน้าคลิก 4 มุมบนภาพนิ่ง

## Acceptance
- [ ] ทดสอบ homography กับจุดที่รู้ตำแหน่งจริง ต้องคลาดเคลื่อน <1 cm
- [ ] มี test ล็อกขนาดคอร์ทไว้ที่ **13.40 × 6.10 ม.** (repo อ้างอิงใช้ 15 × 7.1 ผิด ดู REFERENCES)
- [ ] คำนวณ `speed_mps` และ `dist_m` แบบสะสมทีละเฟรม (O(1) ต่อเฟรม) ห้ามคำนวณจากประวัติทั้งหมดใหม่ทุกเฟรม
- [ ] `shuttle.landing_xy_m_est` เป็น `null` ระหว่างที่ลูกลอย
- [ ] ถ้า kill players worker แล้ว WS ยังส่งข้อมูลต่อ และ `workers.players` เปลี่ยนเป็น `down`
- [ ] GameState ทุก message ผ่านการตรวจ schema
- [ ] คลิปประเภทคู่ 1 นาที: `id` ของทั้ง 4 คนไม่สลับ นับด้วยตาแล้วบันทึกใน PR
