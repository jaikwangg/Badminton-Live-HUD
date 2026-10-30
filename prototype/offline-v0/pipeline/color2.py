"""Extra appearance features per on-court detection: white-print fraction on torso, shoe whiteness, shorts brightness."""
import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))

import json, cv2, numpy as np, time
D = json.load(open(os.path.join(DATA, 'dets_yolo11s-pose_960.json')))
H = None
cap = cv2.VideoCapture(SRC)
byf = {fr['f']: fr for fr in D['frames']}
fi = -1; t0 = time.time()

def crop(hsv, x1, y1, x2, y2):
    h, w = hsv.shape[:2]
    x1, x2 = int(max(0, min(x1, x2))), int(min(w, max(x1, x2)))
    y1, y2 = int(max(0, min(y1, y2))), int(min(h, max(y1, y2)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return hsv[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)

while True:
    ok = cap.grab()
    if not ok:
        break
    fi += 1
    fr = byf.get(fi)
    if fr is None:
        continue
    todo = [d for d in fr['dets'] if -1.2 <= d['court'][0] <= 7.3 and -1.2 <= d['court'][1] <= 15]
    if not todo:
        continue
    ok, img = cap.retrieve()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for d in todo:
        x1, y1, x2, y2 = d['box']; bw, bh = x2 - x1, y2 - y1
        k = d['kp']
        if min(k[5][2], k[6][2], k[11][2], k[12][2]) > 0.2:
            tx1 = min(k[5][0], k[6][0]); tx2 = max(k[5][0], k[6][0])
            ty1 = min(k[5][1], k[6][1]); ty2 = max(k[11][1], k[12][1])
        else:
            tx1, tx2, ty1, ty2 = x1 + bw * .25, x2 - bw * .25, y1 + bh * .18, y1 + bh * .48
        T = crop(hsv, tx1, ty1, tx2, ty2)
        f = {}
        if T is not None:
            white = (T[:, 2] > 165) & (T[:, 1] < 70)
            f['white'] = float(white.mean())
            col = T[~white]
            if len(col) > 10:
                f['h'] = float(np.median(col[:, 0])); f['s'] = float(np.median(col[:, 1])); f['v'] = float(np.median(col[:, 2]))
        # shoes: below each ankle
        sh = []
        r = max(4, bh * 0.035)
        for a in (15, 16):
            if k[a][2] > 0.3:
                S = crop(hsv, k[a][0] - r * 1.6, k[a][1] - r * 0.3, k[a][0] + r * 1.6, k[a][1] + r * 1.4)
                if S is not None:
                    sh.append(float(((S[:, 2] > 175) & (S[:, 1] < 60)).mean()))
        if sh:
            f['shoe'] = float(np.mean(sh))
        if min(k[11][2], k[12][2], k[13][2], k[14][2]) > 0.2:
            sy1 = max(k[11][1], k[12][1]); sy2 = min(k[13][1], k[14][1])
            Sx = crop(hsv, min(k[11][0], k[12][0]), sy1, max(k[11][0], k[12][0]), sy1 + (sy2 - sy1) * 0.6)
            if Sx is not None:
                f['shortsV'] = float(np.median(Sx[:, 2])); f['shortsWhite'] = float(((Sx[:, 2] > 165) & (Sx[:, 1] < 70)).mean())
        d['f2'] = f
print('done', time.time() - t0)
json.dump(D, open(os.path.join(DATA, 'dets2.json'), 'w'))
