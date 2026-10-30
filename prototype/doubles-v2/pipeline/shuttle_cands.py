"""Step 7a: shuttle candidates = small blobs brighter than the same spot in frames t-10, t-3, t+3, t+10.
Neighbours are warped into frame t twice: with the background (ceiling/wall) homography for pixels above the far
baseline and with the floor (court) homography for pixels on the court. -> cands.json"""
import cv2, numpy as np, json, time
from collections import deque
from courtlib import project

CAM = np.load('cam.npy'); COURT = np.load('court.npy'); N = len(CAM)
OFFS = (-10, -3, 3, 10); THR = 20
cap = cv2.VideoCapture('a540.mp4')
buf = {}; out = []; t0 = time.time()
def gray(i):
    return buf[i]
nxt = 0
def load_until(j):
    global nxt
    while nxt <= min(j, N - 1):
        ok, f = cap.read()
        if not ok: break
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        buf[nxt] = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16), hsv[..., 1])
        nxt += 1
rows = np.arange(960)[:, None]
for t in range(N):
    load_until(t + 10)
    for k in list(buf):
        if k < t - 10: del buf[k]
    G, S = buf[t]
    ftop = project(COURT[t], [[3.05, 13.4]])[0, 1]
    above = rows < ftop
    D = None
    for o in OFFS:
        s = t + o
        if s < 0 or s >= N: continue
        Gs = buf[s][0].astype(np.float32)
        Hb = CAM[t] @ np.linalg.inv(CAM[s]); Hf = COURT[t] @ np.linalg.inv(COURT[s])
        Wb = cv2.warpPerspective(Gs, Hb, (540, 960), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Wf = cv2.warpPerspective(Gs, Hf, (540, 960), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = G - np.where(above, Wb, Wf)
        D = d if D is None else np.minimum(D, d)
    m = ((D > THR) & (S < 90)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    cs = []
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]; w = st[k, cv2.CC_STAT_WIDTH]; h = st[k, cv2.CC_STAT_HEIGHT]
        if a < 2 or a > 90 or max(w, h) > 26: continue
        ys, xs = np.where(lab[st[k, 1]:st[k, 1] + h, st[k, 0]:st[k, 0] + w] == k)
        vals = D[st[k, 1] + ys, st[k, 0] + xs]
        cs.append([round(float(cen[k][0]), 1), round(float(cen[k][1]), 1), int(a), int(vals.max()), int(G[st[k, 1] + ys, st[k, 0] + xs].mean()), int(max(w, h))])
    out.append(cs)
    if (t + 1) % 300 == 0:
        print(t + 1, f'{(time.time() - t0) / (t + 1):.3f}s/f', 'avg cands', round(np.mean([len(c) for c in out[-300:]]), 1), flush=True)
json.dump(out, open('cands.json', 'w'))
print('DONE', flush=True)
