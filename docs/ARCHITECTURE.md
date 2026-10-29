# Architecture

## ภาพรวม
```
[Source] กล้อง / SRT / ไฟล์ replay        ← interface เดียวกัน: read() -> Frame
    │  frame + capture_ts
    ▼
[Frame bus] shared memory ring buffer (4 ช่อง เก็บเฉพาะเฟรมล่าสุด)
    ├─▶ [players worker]  YOLO + ByteTrack      ─┐
    ├─▶ [shuttle worker]  TrackNetV3 (3 เฟรม)   ─┼─▶ ZeroMQ PUB/SUB (ipc)
    └─▶ [audio worker]    จับเสียงตีลูก          ─┘          │
                                                            ▼
                                   [fusion] sync ด้วย capture_ts → homography → Kalman
                                            → rally state machine → stats → GameState
                                                            │
                     ┌──────────────────────────────────────┼──────────────┐
                     ▼                                      ▼              ▼
          [server] WS :8765 (GameState 30Hz)         [recorder]       [health]
                   HTTP :8080 (web + operator API)   mp4 + jsonl      fps/latency/GPU
                     │
           web/overlay (OBS Browser Source)  ·  web/operator
```
ทุกกล่องรันเป็น process แยกกัน มี `supervisor.py` คอยเปิดและ restart

## โมดูลและเจ้าของ

| โฟลเดอร์ | หน้าที่ |
|---|---|
| `src/hud/sources/` | camera.py, srt.py, file_replay.py (เล่นไฟล์ตามความเร็วจริงตาม fps) |
| `src/hud/bus/` | shm_ring.py (frame bus), messages.py (dataclass ตาม CONTRACTS) |
| `src/hud/models/` | interface `Detector` / `ShuttleTracker` แยก backend เป็น ONNX Runtime และ TensorRT |
| `src/hud/workers/` | players.py, shuttle.py, audio.py |
| `src/hud/fusion/` | homography.py, kalman.py, rally_fsm.py, stats.py, state.py |
| `src/hud/server/` | ws_server.py, operator_api.py (FastAPI) |
| `src/hud/recorder/` | writer.py |
| `src/hud/health.py`, `supervisor.py` | ดู heartbeat, restart worker, เก็บ metrics |
| `web/overlay/`, `web/operator/` | HTML + JS รับข้อมูลผ่าน WebSocket |
| `tools/` | calibrate.py, export_trt.py, benchmark.py, soak_test.py |

## หลักความเสถียร
1. แยกแต่ละส่วนเป็นคนละ process และให้ supervisor restart ทีละตัว
2. ทิ้งเฟรมเก่าเสมอ worker อ่านเฉพาะเฟรมล่าสุด ไม่ต่อคิว
3. ถ้าส่วนไหนเสียให้ลดฟีเจอร์ fusion ต้องทำงานต่อได้แม้ไม่มีข้อมูลลูกหรือเสียง
4. Source มี watchdog ถ้าไม่ได้เฟรมเกิน 1 วินาทีให้ reconnect โดยหน่วงเวลาเพิ่มขึ้นทีละขั้น (backoff) สูงสุด 5 วินาที
5. Recorder เป็นงานรอง ถ้าดิสก์ช้าหรือเต็มให้หยุดอัด แต่ห้ามทำให้ pipeline หลักช้าลง

## Stack
Python 3.11 · Ultralytics YOLO11s · TrackNetV3 (ONNX → TensorRT FP16) · OpenCV + GStreamer (SRT) · pyzmq · FastAPI + websockets · Ubuntu 22.04 + Docker (nvidia runtime) · OBS Browser Source

## Latency budget (เป้า p50 ≤150ms)

| ขั้น | ms |
|---|---|
| กล้อง → PC (SRT) | 40–60 |
| decode + เขียนเข้า frame bus | 5 |
| players / shuttle (ทำงานขนานกัน) | 10–15 |
| หน้าต่าง 3 เฟรมของ TrackNet | +33 |
| fusion + ส่ง WS | 5 |
| render overlay | 16 |

ตั้ง video delay ใน OBS ให้เท่ากับ latency p50 ภาพกับกราฟิกจะตรงกัน
