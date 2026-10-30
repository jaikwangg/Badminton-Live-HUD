import numpy as np
def crop(hsv, x1, y1, x2, y2):
    h, w = hsv.shape[:2]
    x1, x2 = int(max(0, min(x1, x2))), int(min(w, max(x1, x2)))
    y1, y2 = int(max(0, min(y1, y2))), int(min(h, max(y1, y2)))
    if x2 - x1 < 3 or y2 - y1 < 3: return None
    return hsv[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
def feats(hsv, d):
    x1, y1, x2, y2 = d['box']; bw, bh = x2 - x1, y2 - y1; k = d['kp']
    if min(k[5][2], k[6][2], k[11][2], k[12][2]) > 0.2:
        tx1 = min(k[5][0], k[6][0]); tx2 = max(k[5][0], k[6][0]); ty1 = min(k[5][1], k[6][1]); ty2 = max(k[11][1], k[12][1])
    else:
        tx1, tx2, ty1, ty2 = x1 + bw * .25, x2 - bw * .25, y1 + bh * .18, y1 + bh * .48
    T = crop(hsv, tx1, ty1, tx2, ty2); f = {}
    if T is not None:
        white = (T[:, 2] > 165) & (T[:, 1] < 70); f['white'] = float(white.mean()); col = T[~white]
        if len(col) > 10: f['h'] = float(np.median(col[:, 0])); f['s'] = float(np.median(col[:, 1])); f['v'] = float(np.median(col[:, 2]))
    if min(k[11][2], k[12][2], k[13][2], k[14][2]) > 0.2:
        sy1 = max(k[11][1], k[12][1]); sy2 = min(k[13][1], k[14][1])
        Sx = crop(hsv, min(k[11][0], k[12][0]), sy1, max(k[11][0], k[12][0]), sy1 + (sy2 - sy1) * 0.6)
        if Sx is not None: f['shortsV'] = float(np.median(Sx[:, 2])); f['shortsWhite'] = float(((Sx[:, 2] > 165) & (Sx[:, 1] < 70)).mean())
    return f
