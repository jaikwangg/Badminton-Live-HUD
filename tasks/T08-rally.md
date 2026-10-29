# T08 — Rally state machine + Kalman + สถิติ

**Goal:** แปลงข้อมูลลูก + เสียง + ผู้เล่น เป็น rally, shot และคะแนนที่ระบบแนะนำ
**Owns:** `src/hud/fusion/kalman.py`, `src/hud/fusion/rally_fsm.py`, `src/hud/fusion/stats.py`
**Depends:** S1, T06/T07 (ใช้ message ปลอมทดสอบก่อนได้)

## ทำ
- Kalman สำหรับลูก: ถ้าลูกหายไม่เกิน 10 เฟรมให้เปลี่ยนสถานะเป็น `predicted` ถ้าหายนานกว่านั้นให้เป็น `lost`
- นับ shot: ใช้ทั้งเสียง `hit` และจังหวะที่ลูกเปลี่ยนทิศ (ต้องเกิดใกล้กันไม่เกิน 100ms ถึงจะนับ)
- state machine ของ rally: IDLE → SERVE → RALLY → END, คำนวณ rally time และจำนวน shot
- ตอนจบ rally ให้ประมาณจุดลูกตกจากตำแหน่งสุดท้าย แล้วส่ง event `point_suggested` (**ห้ามเพิ่มคะแนนเอง** ต้องรอให้คนกดยืนยัน)
- ประมาณความเร็วลูก (`speed_kmh_est`) จากระยะบนคอร์ทในช่วง 3–5 เฟรมแรกหลังตีลูก

## Acceptance
- [ ] unit test ของ state machine ครอบคลุมทุก transition และกรณีลูกหายกลาง rally
- [ ] ถ้าไม่มีเสียง ระบบยังนับ shot จากภาพได้ (สถานะเป็น degraded)
