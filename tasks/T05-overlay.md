# T05 — Overlay สำหรับ OBS

**Goal:** หน้า HUD โปร่งใสขนาด 1920×1080 ที่ใช้เป็น Browser Source ใน OBS ได้เลย
**Owns:** `web/overlay/`, `tools/mock_ws.py`
**Depends:** T01 (ใช้แค่ CONTRACTS ก็เริ่มได้)

## ทำ
- HTML + JS ธรรมดา พื้นหลังโปร่งใส รับ GameState จาก `ws://localhost:8765`
- แสดงตามดีไซน์ prototype (Badminton Live HUD): ป้ายชื่อ + วงแหวนใต้เท้า, แผง rally time / shots / shot speed (ติดป้าย Est.), การ์ดผู้เล่น (MOVE m/s, ระยะ m) 2 หรือ 4 ใบตาม `match.mode` จัดกลุ่มตาม `team`, court map, สถานะ LIVE
- court map แสดงลูกเฉพาะจุดตกโดยประมาณ (`landing_xy_m_est`) พร้อมป้าย est. ห้ามวาดลูกที่กำลังลอย
- ถ้าค่าเป็น `null` ให้แสดง "—" ถ้าสถานะลูกเป็น `predicted` ให้แสดงจาง ถ้า WS หลุดให้ขึ้นป้าย "สัญญาณขาด" แล้ว reconnect เอง
- ใช้ `requestAnimationFrame` ทำ interpolation ระหว่าง message ให้ภาพลื่นที่ 60fps
- `tools/mock_ws.py`: เขียนตัวส่ง GameState ปลอมไว้ทดสอบหน้า overlay โดยไม่ต้องมีโมเดล เลือกได้ทั้ง `singles` และ `doubles`

## Acceptance
- [ ] เปิดใน OBS Browser Source แล้วพื้นหลังโปร่งใสจริง
- [ ] เปิดคู่กับ mock 1 ชม. แล้วหน่วยความจำของหน้าเว็บไม่โตขึ้นเกิน 10%
- [ ] ปิด mock แล้วเปิดใหม่ หน้าเว็บต้องกลับมาแสดงผลได้เองโดยไม่ต้อง refresh
