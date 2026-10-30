# Mockup v0 (stage 0)

ดีไซน์แรกของ HUD ก่อนมี prototype ที่ใช้ข้อมูลจริง ทำใน Claude Design canvas

| ไฟล์ | ขนาด | เนื้อหา |
|---|---|---|
| `Main.dc.html` | 1920×1080 | HUD ซ้อนบนภาพสดสำหรับ OBS: กรอบผู้เล่น, LOCK ON ลูก, สกอร์, สถิติแรลลี่, court map |
| `Operator.dc.html` | 1440×900 | หน้าควบคุมสำหรับคนคุม: สถานะ fps/latency, คาลิเบรตคอร์ท, แก้คะแนน |
| `canvas.json` | — | การจัดวางสองหน้าบน canvas |

- ข้อมูลในหน้าเป็น**ค่าจำลอง** สุ่มช็อตและตำแหน่งผู้เล่นขึ้นมา (ชื่อผู้เล่นเป็นชื่อสมมติ)
- ไฟล์ `.dc.html` ต้องเปิดใน design canvas เพราะอาศัย runtime `support.js` ของ canvas เปิดตรงใน browser จะไม่ทำงาน
- ใช้เป็นแนวทาง layout ของ T05 (overlay) และ T09 (operator) ส่วน layout ที่ใช้กับข้อมูลจริงแล้วดู `prototype/doubles-v2/render_v3.py`
