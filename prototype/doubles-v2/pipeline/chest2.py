"""Recompute chest colour per detection (non-blue saturation, so the blue floor seen through a cut-off box does not
look like a printed shirt). -> dets2.jsonl"""
import cv2, json, numpy as np
rows = [json.loads(l) for l in open('dets.jsonl')]
cap = cv2.VideoCapture('a540.mp4')
for i, r in enumerate(rows):
    ok, f = cap.read()
    if not r['d']: continue
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    for d in r['d']:
        x1, y1, x2, y2 = d['b']; w, h = x2 - x1, y2 - y1
        if y2 < 420: d['ch2'] = None; continue
        cx1, cx2 = int(max(0, x1 + w * 0.28)), int(min(540, x2 - w * 0.28)); cy1, cy2 = int(max(0, y1 + h * 0.18)), int(min(960, y1 + h * 0.42))
        if cx2 - cx1 < 2 or cy2 - cy1 < 2: d['ch2'] = None; continue
        a = hsv[cy1:cy2, cx1:cx2].reshape(-1, 3).astype(np.int16); H, S, V = a[:, 0], a[:, 1], a[:, 2]
        floor = (H >= 100) & (H <= 125) & (S > 120) & (V > 90)                      # court blue
        keep = ~floor
        n = max(1, keep.sum())
        d['ch2'] = dict(dark=round(float(((V < 75) & keep).sum() / n), 3),
                        color=round(float(((S > 90) & (V > 80) & keep).sum() / n), 3),
                        light=round(float(((V > 165) & (S < 70) & keep).sum() / n), 3),
                        green=round(float(((H >= 40) & (H <= 95) & (S > 60) & (V > 35) & keep).sum() / n), 3),
                        floor=round(float(floor.mean()), 3), edge=bool(x1 <= 2 or x2 >= 538))
open('dets2.jsonl', 'w').write('\n'.join(json.dumps(r) for r in rows) + '\n')
print('done')
