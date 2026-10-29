# T10 — Recorder

**Goal:** ทุก session มีวิดีโอดิบและ GameState เก็บไว้ (เป็น dataset ในอนาคต)
**Owns:** `src/hud/recorder/`
**Depends:** S1

## ทำ
- อ่านเฟรมจากบัสแล้ว encode เป็น mp4 ด้วย NVENC (ถ้าไม่มีให้ fallback ไปใช้ x264 veryfast)
- เขียน GameState ทุกเฟรมลง `events.jsonl` และเก็บ `config.snapshot.yaml`
- ถ้าดิสก์เหลือน้อยกว่า 5 GB ให้หยุดอัดแล้วแจ้งใน health โดยไม่ทำให้ระบบหลักช้าลง
- แบ่งไฟล์ทุก 30 นาที

## Acceptance
- [ ] อัด 10 นาทีแล้วจำนวนเฟรมในวิดีโอตรงกับจำนวนบรรทัดใน `events.jsonl` (ต่างกันไม่เกิน 1%)
- [ ] จำลองดิสก์เต็มแล้ว pipeline หลักยังได้ 60fps
