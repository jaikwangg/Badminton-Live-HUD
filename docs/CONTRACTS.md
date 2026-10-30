# Contracts (v1)

> ข้อตกลงร่วมระหว่างโมดูล ถ้าจะแก้ต้องเปิด PR ที่แก้แค่ไฟล์นี้ก่อน แล้วรอคนอนุมัติ

## หน่วยและพิกัด
- เวลา: `ts` = `time.monotonic_ns()` ตอนจับภาพ (int, ns)
- พิกัดภาพ: pixel ของภาพต้นฉบับ `[x, y]` จุดเริ่มอยู่มุมซ้ายบน
- พิกัดคอร์ท: เมตร `[x, y]` จุดเริ่มอยู่มุมคอร์ทฝั่ง "ทีมเรา" ด้านซ้าย, `x` วัดตามความยาว 0–13.40, `y` วัดตามความกว้าง 0–6.10
  - ประเภทเดี่ยวใช้เส้นข้างใน y = 0.46 และ 5.64
  - ประเภทคู่ใช้เต็มความกว้าง 0–6.10 และเส้นเสิร์ฟยาวอยู่ที่ x = 0.76 และ 12.64
- ความเร็ว: ผู้เล่นใช้ m/s ส่วนลูกใช้ km/h และต้องลงท้ายชื่อฟิลด์ด้วย `_est`

## Frame bus (shared memory)
- ชื่อ: `hud_frames`, มี 4 ช่อง, แต่ละช่องเก็บ header + ภาพ BGR uint8 ขนาด H×W×3
- header: `frame_id:u64, ts:i64, width:u16, height:u16, seq:u64` (ใช้ `seq` แบบ seqlock กันอ่านภาพที่ยังเขียนไม่เสร็จ)

## Worker → Fusion (ZeroMQ PUB, `ipc:///tmp/hud_<name>`, JSON)
```json
{"v":1,"type":"players","frame_id":123,"ts":0,
 "players":[{"track_id":3,"bbox":[x1,y1,x2,y2],"foot_px":[x,y],"conf":0.91,"app":[0.12,0.53,0.31]}]}

{"v":1,"type":"shuttle","frame_id":123,"ts":0,"visible":true,"px":[x,y],"conf":0.83}

{"v":1,"type":"hit","ts":0,"strength":0.72}

{"v":1,"type":"heartbeat","worker":"players","ts":0,"fps":59.8,"proc_ms":6.1}
```

- `players[].app` = เวกเตอร์ลักษณะภายนอกสำหรับแยกตัวตน (เช่น สีเสื้อช่วงอกที่ตัดสีพื้นคอร์ทออก หรือ re-ID embedding) ความยาวคงที่ตาม config ถ้าไม่มีให้ส่ง `null`

## Fusion → Clients (WS `:8765`, ส่ง 30Hz, JSON)
```json
{
  "v": 1, "ts": 0,
  "match":  {"mode":"singles|doubles","score":[11,9],"games":[1,0],"server":0},
  "rally":  {"active":true,"n":21,"t_ms":4810,"shots":5},
  "players":[
    {"id":0,"name":"น้ำ","color":"#3fd0f0","team":0,"side":0,"tracked":true,
     "img_xy":[812,640],"bbox":[760,420,860,650],
     "court_xy_m":[2.1,3.0],"speed_mps":2.4,"dist_m":23.6}
  ],
  "shuttle":{"status":"tracked|predicted|lost","img_xy":[900,300],
             "landing_xy_m_est":null,"last_shot":{"by":0,"speed_kmh_est":150,"type":null}},
  "events": [{"ts":0,"kind":"shot|rally_start|rally_end|point_suggested","data":{}}],
  "health": {"fps":59.9,"latency_ms_p50":128,"latency_ms_p99":190,"dropped":0,
             "workers":{"players":"ok","shuttle":"ok|degraded|down","audio":"ok"}}
}
```
- `events` ส่งเฉพาะ event ใหม่ตั้งแต่รอบที่แล้ว
- ฟิลด์ที่ไม่รู้ค่าให้ส่ง `null` ห้ามละฟิลด์ทิ้ง
- **ลูกไม่มีพิกัดบนคอร์ทระหว่างลอย** เพราะกล้องตัวเดียวไม่รู้ความสูงของลูก
  - `landing_xy_m_est` มีค่าเฉพาะตอนจบ shot หรือ rally (จุดตกโดยประมาณ) ช่วงอื่นให้เป็น `null`
  - court map ห้ามวาดลูกจาก `img_xy`
- `players[].img_xy` = ตำแหน่งเท้า โดยใช้ค่าเฉลี่ยข้อเท้าจาก pose ถ้าไม่มีให้ใช้จุดกึ่งกลางขอบล่างของ bbox
- `match.mode`
  - `singles`: ผู้เล่น 2 คน (`id` 0–1)
  - `doubles`: ผู้เล่น 4 คน (`id` 0–3)
  - ส่ง `players` ครบทุกคนทุกรอบ คนที่ตามไม่ได้ให้ `tracked:false` และค่าตำแหน่งเป็น `null`
- `players[].id` ต้องคงที่ต่อคนตลอดแมตช์ ห้ามเปลี่ยนตาม track ID
- `players[].team` = ทีม (0 = ทีมเรา, 1 = คู่แข่ง) คงที่ตลอดแมตช์ `score` และ `games` เรียงตาม team
- `players[].side` = ฝั่งคอร์ทที่ยืนอยู่ตอนนั้น ทุกประเภทให้คำนวณจากฝั่งของเส้นเน็ต**ทุกเฟรม** ห้ามยึดตาม track ID
- `shuttle.last_shot.by` = `id` ของคนตี

_Changelog:_
- _2026-10-29 เปลี่ยน `shuttle.court_xy_m` เป็น `landing_xy_m_est` ตอนนี้ยังไม่มีโค้ดที่ใช้ฟิลด์นี้ จึงไม่ต้องเพิ่มเลขเวอร์ชัน (`v`)_
- _2026-10-30 รองรับประเภทคู่: `match.mode` = `doubles`, เพิ่ม `players[].team`, `app` ใน message `players` ของ worker, `mode` ใน `/api/players` และ `team` ใน `/api/score` (ยังรับ `side` แบบเดิม) เข้ากับของเดิมทั้งหมด จึงไม่เพิ่มเลขเวอร์ชัน_

## Operator API (HTTP `:8080`)

| Method | Path | Body |
|---|---|---|
| POST | `/api/calibrate` | `{"corners_px":[[x,y]×4]}` เรียงตามเข็มนาฬิกา เริ่มจากมุมซ้ายไกล |
| POST | `/api/players` | `{"mode":"singles\|doubles","players":[{"id","name","color","team","side"}]}` ไม่ส่ง `mode` ถือเป็น `singles` · `side` = ฝั่งตอนเริ่ม |
| POST | `/api/score` | `{"team":0,"delta":1}` (ยังรับ `side` แบบเดิมได้ ถือเป็น team) |
| POST | `/api/point/confirm` | `{"event_ts":0,"accept":true}` |
| POST | `/api/swap_ids` | `{"a":0,"b":1}` |
| POST | `/api/rally/reset` | `{}` |
| GET | `/health` | → ส่ง `health` ตามรูปแบบข้างบน |

## Recording
`recordings/<YYYYMMDD-HHMMSS>/` เก็บ `video.mp4`, `events.jsonl` (GameState ทุกเฟรม) และ `config.snapshot.yaml`
