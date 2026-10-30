"""Second pass on the far half-court only (crop, upscaled), low conf, to recover occluded/small far players."""
import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))

import cv2, json, numpy as np, time
from ultralytics import YOLO
import importlib.util
H = np.array(json.load(open(os.path.join(DATA, 'calib.json')))['H'])
X0, Y0, X1, Y1 = 420, 250, 1560, 720
model = YOLO('yolo11s-pose.pt')
cap = cv2.VideoCapture(SRC)
spec = importlib.util.spec_from_file_location('c2', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'color2_feats.py')); c2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c2)
out = {}; fi = -1; n = 0; t0 = time.time()
while True:
    if not cap.grab(): break
    fi += 1
    if fi % 2: continue
    ok, img = cap.retrieve()
    crop = img[Y0:Y1, X0:X1]
    r = model.predict(crop, imgsz=1152, conf=0.15, iou=0.6, verbose=False, classes=[0])[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dets = []
    if r.boxes is not None and len(r.boxes):
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.data.cpu().numpy()):
            b = b + np.array([X0, Y0, X0, Y0]); k = k.copy(); k[:, 0] += X0; k[:, 1] += Y0
            la, ra = k[15], k[16]
            if la[2] > 0.4 and ra[2] > 0.4: fx, fy = (la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2
            elif max(la[2], ra[2]) > 0.4: a = la if la[2] > ra[2] else ra; fx, fy = a[0], a[1]
            else: fx, fy = (b[0] + b[2]) / 2, b[3]
            p = H @ np.array([fx, fy, 1.0]); X, Y = p[0] / p[2], p[1] / p[2]
            if not (-1.2 <= X <= 7.3 and 6.7 <= Y <= 15): continue
            kp = [[round(float(a), 1), round(float(bb), 1), round(float(cc), 2)] for a, bb, cc in k]
            d = dict(tid=-1, box=[round(float(v), 1) for v in b], conf=round(float(c), 3), foot=[round(float(fx), 1), round(float(fy), 1)],
                     fsrc='far', cut=False, court=[round(float(X), 3), round(float(Y), 3)], kp=kp)
            d['f2'] = c2.feats(hsv, d)
            dets.append(d)
    out[fi] = dets
    n += 1
    if n % 200 == 0: print(n, f'{(time.time()-t0)/n*1000:.0f} ms/frame', flush=True)
json.dump(out, open(os.path.join(DATA, 'far_dets.json'), 'w'))
print('DONE', n, time.time() - t0, flush=True)
