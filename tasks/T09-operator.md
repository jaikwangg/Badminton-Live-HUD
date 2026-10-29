# T09 — หน้าควบคุม (Operator)

**Goal:** ให้คนที่ไม่ใช่ dev คุมระบบได้ทั้งแมตช์
**Owns:** `web/operator/`, endpoint ที่ยังขาดใน `src/hud/server/operator_api.py`
**Depends:** S1

## ทำ
- คาลิเบรตคอร์ท: แสดงภาพนิ่งจากกล้องแล้วลากจุด 4 มุม ต้องทำเสร็จใน ≤60 วินาที
- ตั้งค่าผู้เล่น: ชื่อ, สี, ฝั่ง
- ปุ่มคะแนน +/−, ปุ่มยืนยันหรือปฏิเสธ `point_suggested`, ปุ่มสลับ ID, ปุ่ม reset rally
- แผง health: fps, latency, สถานะ worker แต่ละตัว
- ทำตามดีไซน์ prototype (หน้า Operator)

## Acceptance
- [ ] ให้คนที่ไม่ใช่ dev ลองคาลิเบรตแล้วจับเวลา บันทึกผลใน PR
- [ ] ทุกปุ่มมีผลกับ GameState ภายใน 100ms
