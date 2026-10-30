# Media archive (branch `media`)

branch นี้เก็บไฟล์วิดีโอ เสียง และภาพดีบักของ prototype แต่ละ stage แยกจากโค้ด เพื่อให้ branch โค้ด (`main`, `prototype/*`, `task/*`) ยังเบาและไม่ผิดกติกา "ห้าม commit `*.mp4`" ใน `AGENTS.md`

> คลิปมีหน้าผู้เล่นและคนในสนาม และ repo นี้เป็น public

ดูความคืบหน้าและบทเรียนของแต่ละ stage ได้ที่ `docs/PROGRESS.md` และ `docs/LESSONS.md` บน branch `prototype/offline-v0` (PR #8)

## ดึงไฟล์

```bash
# เอาเฉพาะโค้ด (ไม่โหลดวิดีโอ)
git clone --single-branch -b main https://github.com/jaikwangg/Badminton-Live-HUD.git

# เอาวิดีโอด้วย
git clone --single-branch -b media https://github.com/jaikwangg/Badminton-Live-HUD.git hud-media

# ประกอบคลิปต้นฉบับ stage 1 กลับเป็นไฟล์เดียว (แบ่งไว้เพราะ GitHub รับไฟล์ละไม่เกิน 100 MB)
cd hud-media/stage1-offline-v0
cat source_IMG_5341.mov.part* > source_IMG_5341.mov
sha256sum -c source_IMG_5341.mov.sha256
```

ตรวจไฟล์อื่นได้ด้วย `sha256sum -c SHA256SUMS.txt`

## Stage 1: offline-v0 (`stage1-offline-v0/`)

คลิป 1:18 นาที กล้องมือถือตั้งนิ่งแนวนอน ประเภทคู่ (ปั้น, พฤกษ / ปลื้ม, ภูสิท) · โค้ดอยู่ที่ `prototype/offline-v0/`

| ไฟล์ | ขนาด | คืออะไร | ใช้กับ |
|---|---|---|---|
| `source_IMG_5341.mov.part00–02` | 205 MB รวม | คลิปต้นฉบับ HEVC 1920×1078 มีเสียง 2 แทร็ก (ใช้แทร็ก `0:a:1`) | ต้นทางของทุกขั้น วางเป็น `data/source.mov` |
| `clip720.mp4` | 13 MB | คลิปย่อ 1280×720 สำหรับหน้าเว็บ | วางที่ `site/clip720.mp4` แล้วเปิด `site/index.html` |
| `audio.wav` | 7.5 MB | เสียงแทร็ก 2 แบบ mono 48 kHz | `pipeline/hits.py` |
| `hud_16x9_1080p.mp4` | 27 MB | **ผลลัพธ์** HUD แบบจอถ่ายทอดสด 1920×1080 | — |
| `hud_9x16_vertical.mp4` | 22 MB | **ผลลัพธ์** HUD แนวตั้ง 1080×1920 | — |
| `raw_detections.zip` | 1.9 MB | ผลตรวจจับดิบจาก YOLO หลายขนาด/ความละเอียด และ log | เทียบโมเดล |
| `data/dets2.json` | 8.3 MB | ผลตรวจจับทุกเฟรม: bbox, keypoints 17 จุด, ตำแหน่งบนคอร์ท, track id, ค่าสี (ผลของ `pipeline/color2.py`) | `pipeline/identify.py` วางเป็น `data/dets2.json` |
| `debug_frames.zip` | 28 MB | ภาพที่ใช้ตรวจด้วยตาระหว่างทำ (contact sheet, เฟรมพื้นหลัง) | ดีบัก |

## Stage 2: doubles-v2 (`stage2-doubles-v2/`)

คลิป 1:57 นาที มือถือแนวตั้ง 540×960 วางหลังคอร์ท กล้องขยับเล็กน้อย ประเภทคู่ (บอส, เทมป์ / เป้, ตี๋) · โค้ดอยู่ที่ `prototype/doubles-v2/`

| ไฟล์ | ขนาด | คืออะไร | ใช้กับ |
|---|---|---|---|
| `source.mov` | 28 MB | คลิปต้นฉบับ | ต้นทางของทุกขั้น |
| `a540.mp4` | 41 MB | 540×960 30 fps ไม่มีเสียง (ตัวที่ใช้ตรวจจับจริง) | `detect.py`, `camera.py`, `shuttle_cands.py` |
| `master1080.mp4` | 78 MB | 1080×1920 30 fps ไม่มีเสียง | พื้นหลังของ `render_v3.py` และ `report.py` |
| `audio.wav` | 10 MB | เสียง mono 44.1 kHz | `audio_onsets.py`, ใส่เสียงในวิดีโอผลลัพธ์ |
| `hud_v3_statui.mp4` | 40 MB | **ผลลัพธ์ล่าสุด** HUD แบบ stat UI (ไม่มีพลังกาย) | — |
| `hud_v2_panels.mp4` | 47 MB | ผลลัพธ์รุ่นก่อน: แผงซ้อนในวิดีโอ มีพลังกายบนหัว | เทียบ |
| `raw_logs.zip` | 0.7 MB | ผลตรวจจับดิบก่อนแยกตัวตน (`dets.jsonl`), homography ก่อน refine และ log ทุกขั้น | ดีบัก |
| `data/dets2.jsonl` | 8.3 MB | ผลตรวจจับทุกเฟรมพร้อมสีหน้าอกที่คำนวณใหม่ (ผลของ `chest2.py`) | `players.py`, `shuttle_track.py` วางเป็น `data/dets2.jsonl` |
| `data/cands.json` | 8.1 MB | จุดที่อาจเป็นลูกขนไก่ในแต่ละเฟรม (ผลของ `shuttle_cands.py`) | `shuttle_track.py` วางเป็น `data/cands.json` |
| `debug_frames.zip` | 68 MB | ภาพตรวจทีละเฟรม: แถบภาพลูก, ภาพตอนตี, preview HUD | ดีบัก |

## ไม่ได้เก็บ

- ไฟล์โมเดล `*.pt` และ OpenVINO: โหลดใหม่ได้จาก Ultralytics (YOLO11s-pose, YOLO11m, YOLO11m-pose) และติด license AGPL-3.0
- ภาพตัวอย่าง HUD จาก TikTok ที่ใช้เป็นแรงบันดาลใจ เพราะเป็นงานของคนอื่น
