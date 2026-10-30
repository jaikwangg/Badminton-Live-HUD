# Prototype offline-v0: HUD จากคลิปจริง (ประเภทคู่)

ลองระบบกับคลิปจริงความยาว 1:18 นาที ถ่ายด้วยมือถือตั้งนิ่ง ประมวลผล**หลังถ่าย**บน CPU แล้วสร้างหน้าเว็บที่เล่นวิดีโอพร้อม HUD ซ้อนจากข้อมูลที่วัดได้จริง

> โค้ดชุดนี้เป็น prototype สำหรับพิสูจน์แนวคิด **ไม่ได้ทำตาม** `docs/CONTRACTS.md` และโครง `src/hud/` ใช้เป็นตัวอ้างอิงได้ แต่ไม่ใช่โค้ดจริงของระบบ

| ผู้เล่น | ฝั่ง | จุดสังเกต | ตามได้ | ระยะวิ่ง |
|---|---|---|---|---|
| ปั้น | ทีมเรา | เสื้อกรมท่า รองเท้าขาว | 84% | 53.7 ม. |
| พฤกษ | ทีมเรา | เสื้อน้ำเงิน #11 (PRUKHU) รองเท้าดำ | 100% | 57.8 ม. |
| ปลื้ม | คู่แข่ง | เสื้อดำ กางเกงขาว | 87% | 53.6 ม. |
| ภูสิท | คู่แข่ง | เสื้อขาว | 61% | 48.1 ม. |

ตรวจพบ 6 rally และ 32 shot (shot ยังพลาดอยู่มาก ดูข้อจำกัดด้านล่าง)

## Pipeline

```
source.mov ─▶ calib.py ─▶ detect.py ─▶ color2.py ─▶ detect_far.py ─▶ identify.py ─▶ hits.py ─▶ build.py ─▶ site/index.html
             homography   pose+track    สีเสื้อ/รองเท้า  สแกนฝั่งไกลซ้ำ    ใครเป็นใคร      shot/rally    หน้า HUD
```

| ขั้น | ไฟล์ | ทำอะไร | ใช้เวลา (CPU 2 core) |
|---|---|---|---|
| 1 | `pipeline/calib.py` | homography จากจุดตัดเส้นคอร์ท 10 จุด (คลาดเคลื่อนไม่เกิน 6 ซม.) → `data/calib.json` | ทันที |
| 2 | `pipeline/detect.py yolo11s-pose.pt 960` | YOLO11s-pose + ByteTrack ทุกเฟรมที่ 30 fps ใช้ข้อเท้าเป็นตำแหน่งเท้า | ~16 นาที |
| 3 | `pipeline/color2.py` | วัดสัดส่วนลายพิมพ์ขาวบนเสื้อ, ความขาวของรองเท้า, สีกางเกง → `data/dets2.json` | ~2.5 นาที |
| 4 | `pipeline/detect_far.py` | ครอปฝั่งไกลแล้วขยายภาพตรวจซ้ำ (conf 0.15) เพื่อเก็บคนที่ถูกบังหรือตัวเล็ก → `data/far_dets.json` | ~16 นาที |
| 5 | `pipeline/identify.py data/dets2.json` | แบ่งฝั่งจากเส้นเน็ตทุกเฟรม แล้วแยกคนจากสี + ความต่อเนื่องของตำแหน่ง (Hungarian), smoothing, ความเร็ว, ระยะวิ่ง → `data/players.json` | ไม่กี่วินาที |
| 6 | `pipeline/hits.py 5 7 1.6` | shot = เสียงกระทบลูก (2.5–9 kHz) ตรงกับการเหวี่ยงแขน, rally = ช่วงที่ฝั่งเราเคลื่อนที่ต่อเนื่องและมี shot ≥ 2 → `data/events.json` | ไม่กี่วินาที |
| 7 | `pipeline/build.py` | ฝังข้อมูลลง `web/template.html` → `site/index.html` (หน้าเว็บเล่นวิดีโอ + HUD) | ทันที |
| 8 | `pipeline/render_vertical.py out.mp4` | เรนเดอร์วิดีโอแนวตั้ง 1080×1920 (9:16) สำหรับมือถือ/TikTok: ชื่อโปรเจกต์, ช่องสถิติ, วิดีโอครอปเฉพาะคอร์ท, การ์ดผู้เล่น 4 คน, court map, timeline ของ rally (ตั้งชื่อหัวได้ด้วย `HUD_TITLE`) | ~5 นาที |
| 8b | `pipeline/render_mp4.py out.mp4` | เรนเดอร์วิดีโอ 1920×1080 แบบจอถ่ายทอดสด: วิดีโอ + แผงสถิติ + court map + การ์ดผู้เล่น (ต้องมีฟอนต์จาก `tools/get_fonts.sh`) | ~6 นาที |
| ตรวจ | `pipeline/verify.py 3,20,52` | วาดชื่อลงเฟรมจริงเพื่อตรวจด้วยตา → `out/verify.jpg` | — |

`tools/` มีสคริปต์ที่ใช้ระหว่างสำรวจ: หาเส้นคอร์ทด้วย Hough (`lines.py`), ทดลองจับเสียง (`audio_hits.py`), และพล็อตเสียงเทียบกับการเหวี่ยงแขน (`plot_sig.py`)

## รัน

```bash
pip install -r requirements.txt          # โมเดล YOLO จะถูกโหลดอัตโนมัติตอนรันครั้งแรก
cd prototype/offline-v0
# วางคลิปต้นฉบับไว้ที่ data/source.mov (หรือกำหนด HUD_SRC=/path/to/clip.mov)
ffmpeg -i data/source.mov -map 0:a:1 -ac 1 -ar 48000 data/audio.wav
ffmpeg -i data/source.mov -map 0:v:0 -map 0:a:1 -vf "scale=1280:720,fps=30" -c:v libx264 -preset veryfast -b:v 1300k -c:a aac -b:a 64k -ac 1 -movflags +faststart data/clip720.mp4
python3 pipeline/calib.py
python3 pipeline/detect.py yolo11s-pose.pt 960 && python3 pipeline/color2.py && python3 pipeline/detect_far.py
python3 pipeline/identify.py data/dets2.json && python3 pipeline/hits.py 5 7 1.6 && python3 pipeline/build.py
bash tools/get_fonts.sh && python3 pipeline/render_vertical.py out/hud_vertical.mp4   # MP4 9:16
python3 pipeline/render_mp4.py out/hud_1080p.mp4                                  # MP4 16:9
HUD_PREVIEW=20.4 python3 pipeline/render_mp4.py out/preview.png                 # เฟรมเดียวสำหรับตรวจ layout
# เปิด site/index.html (ต้องมี site/clip720.mp4 อยู่ข้างกัน build.py จะก๊อปจาก data/ ให้ถ้ามี)
```

ถ้าจะรันแค่ขั้น 5–7 ใช้ข้อมูลใน `data/` ได้เลย ขั้น 5 กับ 7 รันได้ทันที ส่วนขั้น 6 ต้องมี `data/audio.wav`

## ไฟล์ใน `data/`

| ไฟล์ | เนื้อหา |
|---|---|
| `calib.json` | homography 3×3 (ภาพ 1920×1078 → เมตรบนพื้น, X = ความกว้าง 0–6.10, Y = ระยะจากเส้นหลังฝั่งเรา 0–13.40) |
| `dets2.json` | ผลตรวจจับทุกเฟรม: bbox, keypoints 17 จุด, ตำแหน่งบนคอร์ท, track id, ค่าสี |
| `far_dets.json` | ผลตรวจจับเพิ่มจากการสแกนฝั่งไกล |
| `players.json` | ตำแหน่ง/ความเร็ว/ระยะของ 4 คนทุกเฟรม (30 fps) |
| `events.json` | shot, rally, สัญญาณการเหวี่ยงแขน |

**ภาพตรวจ** อยู่ใน `check/`: `verify*.jpg` (ชื่อบนเฟรมจริง), `hud16x9_*.jpg` และ `hud9x16_*.jpg` (preview HUD)

**คลิปและวิดีโอ** อยู่ใน branch [`media`](https://github.com/jaikwangg/Badminton-Live-HUD/tree/media/stage1-offline-v0):
- คลิปต้นฉบับ 205 MB แบ่งเป็น 3 ส่วนเพราะ GitHub รับไฟล์ละไม่เกิน 100 MB รวมกลับด้วย `cat source_IMG_5341.mov.part* > data/source.mov`
- `clip720.mp4` (วางที่ `site/` แล้วเปิด `site/index.html` ได้เลย), `audio.wav`, วิดีโอ HUD 16:9 และ 9:16

ไม่มีไฟล์โมเดล `*.pt` ใน repo ตามกติกาใน `AGENTS.md`

ความคืบหน้าและบทเรียนรวมของทุก prototype: `docs/PROGRESS.md`, `docs/LESSONS.md`

## ข้อจำกัดที่เจอ

- **ไม่มีการตามลูก:** ไฟล์โมเดล TrackNetV3 อยู่บน Google Drive ซึ่งเครื่องที่ใช้ประมวลผลเข้าไม่ได้ ส่วนไฟล์ใน repo จีนก็เสีย
  - ความเร็วลูกจึงเป็นค่าประมาณ (est.) = ระยะจากคนตีถึงคนรับ ÷ เวลาระหว่างสองช็อต
- **Shot recall ต่ำ:** ใช้เสียงร่วมกับการเหวี่ยงแขน แต่พลาดช็อตเบาและช็อตฝั่งไกล rally ละได้ 3–8 shot ทั้งที่จริงน่าจะมากกว่านั้น ต้องใช้ TrackNet ช่วย
- **กล้องตั้งต่ำ:**
  - ผู้เล่นฝั่งเราบังฝั่งไกลบ่อย ภูสิทจึงตามได้แค่ 61%
  - ช่วงท้ายคอร์ทฝั่งเรา ~0.9 ม. อยู่นอกเฟรม
  - ฝั่งไกล 1 px ≈ 8 ซม. ต้อง smoothing หนักกว่า
- **แยกคนด้วยสีใช้ได้ดีกับคลิปนี้** เพราะรองเท้าขาว vs ดำ และเสื้อมีลาย vs เรียบ ต่างกันชัด ถ้าใส่ชุดคล้ายกันต้องใช้ re-ID model
- **License:** Ultralytics YOLO เป็น AGPL-3.0 (ดู `docs/REFERENCES.md`)

## สิ่งที่ได้เรียนรู้สำหรับระบบจริง

1. คาลิเบรตจากจุดตัดเส้นหลายจุด (≥ 8) แม่นกว่า 4 มุม และใช้กับคอร์ทที่มองไม่เห็นเส้นหลังได้
2. กำหนดฝั่งจากเส้นเน็ตทุกเฟรม แล้วแยกคนในฝั่งด้วยลักษณะภายนอก ได้ผลดีมากสำหรับประเภทคู่
3. ฝั่งไกลต้องใช้การสแกนแบบครอปและขยายภาพ หรือติดกล้องให้สูงขึ้น
4. ตั้งกล้องสูง 2.5–3 ม. จะแก้ปัญหาการบังได้มากที่สุด
