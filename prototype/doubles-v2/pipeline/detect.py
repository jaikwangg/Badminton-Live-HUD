"""Step 3: YOLO11m + ByteTrack on every frame (30 fps CFR, 540x960 analysis copy).
Inference runs on rows Y0..960 (players never appear above the far wall) at imgsz 832,
which is the same pixel scale as running the full frame at imgsz 1280.
Output: dets.jsonl (one line per frame), coords in 540x960 source pixels.
"""
import cv2, json, sys, time, numpy as np
from ultralytics import YOLO

SRC = 'a540.mp4'; Y0 = 330; IMGSZ = 832
model = YOLO('yolo11m.pt')
cap = cv2.VideoCapture(SRC)
out = open('dets.jsonl', 'w')

def chest(hsv, b):
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + w * 0.3), int(x2 - w * 0.3); cy1, cy2 = int(y1 + h * 0.18), int(y1 + h * 0.40)
    cx1, cy1 = max(cx1, 0), max(cy1, 0); cx2, cy2 = min(cx2, hsv.shape[1]), min(cy2, hsv.shape[0])
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    r = hsv[cy1:cy2, cx1:cx2].reshape(-1, 3).astype(np.int16)
    H, S, V = r[:, 0], r[:, 1], r[:, 2]
    return dict(dark=round(float((V < 75).mean()), 3),
                light=round(float(((V > 165) & (S < 70)).mean()), 3),
                skin=round(float(((H <= 22) & (S > 35) & (S < 175) & (V > 85)).mean()), 3),
                green=round(float(((H >= 40) & (H <= 95) & (S > 60) & (V > 35)).mean()), 3),
                sat=round(float(((S > 110) & (V > 80)).mean()), 3),
                h=float(np.median(H)), s=float(np.median(S)), v=float(np.median(V)))

i = 0; t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok:
        break
    crop = f[Y0:]
    r = model.track(crop, imgsz=IMGSZ, conf=0.15, iou=0.6, persist=True, tracker='bytetrack.yaml', verbose=False, classes=[0])[0]
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    dets = []
    if r.boxes is not None and len(r.boxes):
        xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(xy)
        for b, c, tid in zip(xy, cf, ids):
            b = [float(b[0]), float(b[1]) + Y0, float(b[2]), float(b[3]) + Y0]
            dets.append(dict(id=int(tid), b=[round(v, 1) for v in b], c=round(float(c), 3), ch=chest(hsv, b)))
    out.write(json.dumps(dict(i=i, d=dets)) + '\n')
    i += 1
    if i % 100 == 0:
        out.flush(); el = time.time() - t0
        print(f'{i} frames {el:.0f}s {el / i:.2f}s/f', flush=True)
out.close()
print('DONE', i, f'{time.time() - t0:.0f}s', flush=True)
