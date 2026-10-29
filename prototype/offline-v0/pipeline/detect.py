"""Pass 1: run pose+tracking on every 2nd frame (60->30 fps), save raw detections.
Usage: python3 detect.py <model> <imgsz> [max_frames]
"""
import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))

import cv2, numpy as np, json, sys, time
from ultralytics import YOLO

STEP = 2
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'yolo11s-pose.pt'
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 960
MAXF = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

H = np.array(json.load(open(os.path.join(DATA, 'calib.json')))['H'])

def to_court(x, y):
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])

def region_stats(hsv, x1, y1, x2, y2):
    h, w = hsv.shape[:2]
    x1, x2 = int(max(0, min(x1, x2))), int(min(w, max(x1, x2)))
    y1, y2 = int(max(0, min(y1, y2))), int(min(h, max(y1, y2)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    r = hsv[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    return [float(v) for v in np.median(r, axis=0)]

model = YOLO(MODEL)
cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
out = []
fi = -1
t0 = time.time()
n = 0
while True:
    ok = cap.grab()
    if not ok:
        break
    fi += 1
    if fi % STEP:
        continue
    ok, frame = cap.retrieve()
    if not ok:
        break
    res = model.track(frame, imgsz=IMGSZ, conf=0.25, persist=True, tracker='bytetrack.yaml', verbose=False, classes=[0])[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dets = []
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(boxes)
        kps = res.keypoints.data.cpu().numpy()  # (n,17,3)
        for b, c, tid, k in zip(boxes, confs, ids, kps):
            x1, y1, x2, y2 = [float(v) for v in b]
            la, ra = k[15], k[16]
            if la[2] > 0.4 and ra[2] > 0.4:
                fx, fy, fsrc = (la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2, 'ankle'
            elif la[2] > 0.4 or ra[2] > 0.4:
                a = la if la[2] > ra[2] else ra
                fx, fy, fsrc = a[0], a[1], 'ankle1'
            else:
                fx, fy, fsrc = (x1 + x2) / 2, y2, 'bbox'
            cut = y2 >= frame.shape[0] - 4
            X, Y = to_court(fx, fy)
            # torso: shoulders 5,6 hips 11,12 ; shorts: hips -> knees 13,14
            torso = shorts = None
            bw, bh = x2 - x1, y2 - y1
            torso_box = region_stats(hsv, x1 + bw * 0.3, y1 + bh * 0.2, x2 - bw * 0.3, y1 + bh * 0.42)
            if min(k[5][2], k[6][2], k[11][2], k[12][2]) > 0.2:
                sx1, sx2 = min(k[5][0], k[6][0], k[11][0], k[12][0]), max(k[5][0], k[6][0], k[11][0], k[12][0])
                sy1, sy2 = min(k[5][1], k[6][1]), max(k[11][1], k[12][1])
                padx = (sx2 - sx1) * 0.2; pady = (sy2 - sy1) * 0.15
                torso = region_stats(hsv, sx1 + padx, sy1 + pady, sx2 - padx, sy2 - pady)
                if min(k[13][2], k[14][2]) > 0.2:
                    ky = min(k[13][1], k[14][1])
                    shorts = region_stats(hsv, sx1 + padx, sy2, sx2 - padx, sy2 + (ky - sy2) * 0.6)
            dets.append(dict(tid=int(tid), box=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)], conf=round(float(c), 3),
                             foot=[round(float(fx), 1), round(float(fy), 1)], fsrc=fsrc, cut=bool(cut),
                             court=[round(X, 3), round(Y, 3)], torso=torso, torso_box=torso_box, shorts=shorts,
                             kp=[[round(float(a), 1), round(float(b_), 1), round(float(cc), 2)] for a, b_, cc in k]))
    out.append(dict(f=fi, t=round(fi / fps, 4), dets=dets))
    n += 1
    if n % 100 == 0:
        el = time.time() - t0
        print(f'{n} frames {el:.0f}s ({el / n * 1000:.0f} ms/frame)', flush=True)
    if n >= MAXF:
        break
json.dump(dict(fps=fps, step=STEP, model=MODEL, imgsz=IMGSZ, frames=out), open(os.path.join(DATA, f'dets_{MODEL.split(".")[0]}_{IMGSZ}.json'), 'w'))
print('DONE', n, 'frames', f'{time.time() - t0:.0f}s', flush=True)
