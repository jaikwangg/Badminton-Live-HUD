# Roadmap

ช่วงนี้เน้นทำ **ระบบให้นิ่งก่อน** ใช้โมเดลที่เทรนเสร็จแล้ว ส่วน test set และ fine-tune ทำทีหลัง (ดูเป้าหมายใน `GOALS.md`)

## Milestones

| ขั้น | ผลลัพธ์ | การ์ดงาน | Goal ที่ต้องผ่าน |
|---|---|---|---|
| **S0** | โครง repo, frame bus, supervisor | T01 | `pytest` ผ่าน |
| **S1** | เล่นไฟล์ → ตามผู้เล่น → court map → overlay ใน OBS | T02 T03 T04 T05 | G1 (เฉพาะส่วนผู้เล่น), G5 |
| **S2** | เพิ่มลูก + เสียง + rally + หน้าควบคุม + recorder | T06 T07 T08 T09 T10 | G1 ทั้งระบบ, G3, G4, G6, G7 |
| **S3** | TensorRT + benchmark + soak test 3 ชม. | T11 | G1, G2 |
| **S4** | ต่อกล้องจริงผ่าน SRT + ทดสอบหน้างาน | (เพิ่มการ์ดตามปัญหาที่เจอหน้างาน) | ครบทุก goal |
| ภายหลัง | ทำ test set จาก `recordings/` → fine-tune → สลับ `.engine` | — | เป้าความแม่นใน GOALS |

## ลำดับและงานที่ทำขนานกันได้
```
T01 ──┬─▶ T02 sources ─────┐
      ├─▶ T03 players ─────┤
      ├─▶ T04 fusion+server┼─▶ S1 ──┬─▶ T06 shuttle ─┐
      └─▶ T05 overlay ─────┘        ├─▶ T07 audio ───┤
                                    ├─▶ T08 rally ───┼─▶ S2 ─▶ T11 ─▶ S3 ─▶ S4
                                    ├─▶ T09 operator ┤
                                    └─▶ T10 recorder ┘
```
T01 ต้องเสร็จก่อนการ์ดอื่น หลังจากนั้นแจกให้หลาย agent ทำพร้อมกันได้ เพราะแต่ละการ์ดแก้คนละโฟลเดอร์

## วิธีแจกงานให้ agent
1. สั่งว่า `ทำ tasks/T03-players.md ตามกติกาใน AGENTS.md`
2. agent สร้าง branch แล้วเปิด PR
3. ใช้ agent **คนละค่าย** รีวิว PR (เช่น Codex เขียน ให้ Claude รีวิว) ด้วยคำสั่ง `รีวิว PR นี้เทียบกับ AGENTS.md, CONTRACTS.md และ acceptance ในการ์ด`
4. รัน benchmark บนเครื่องที่มี GPU เอง เพราะ agent บน cloud ไม่มี GPU
5. merge แล้วติ๊กการ์ดในตารางนี้
