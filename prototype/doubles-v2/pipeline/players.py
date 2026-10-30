"""Step 6: who is who. Per frame, detections standing inside our court are split by the net (side), then matched to
the two names of that side using chest colour (dark / light / skin / green / saturated) + previous position.
Colour scores are averaged along each ByteTrack id so one bad frame does not swap names.
-> players.json : per player per frame foot (540px), court XY (m), box; plus speed, distance, stamina."""
import json, numpy as np, cv2, sys
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from courtlib import project

CFG = json.load(open('config.json'))
NAMES = [p['name'] for p in CFG['players']]                  # 0 บอส near, 1 เทมป์ near, 2 เป๋ far, 3 ตี๋ far
SIDE = [p['side'] for p in CFG['players']]                    # 0 = near camera, 1 = far
COURT = np.load('court.npy') if len(sys.argv) < 2 else None
if COURT is None:
    COURT = np.array([C @ np.load('Hc0.npy') for C in np.load('cam.npy')])
INV = np.array([np.linalg.inv(H) for H in COURT])
FPS = 30.0; NET = 6.70
rows = [json.loads(l) for l in open('dets2.jsonl')]
N = len(rows)

def appearance(ch, side):
    """>0 favours the first name of the side (near: บอส black, far: เป๋ dark green). Chest colour without the blue floor."""
    if ch is None or ch['floor'] > 0.6: return 0.0
    wgt = 0.5 if ch['edge'] else 1.0
    if side == 0:      # บอส: black sleeveless  vs  เทมป์: multicolour print
        return wgt * float(np.clip(4.0 * (ch['dark'] - 0.45) - 3.0 * ch['color'] - 1.5 * ch['light'], -2.5, 2.5))
    else:              # เป๋: dark green  vs  ตี๋: black
        return wgt * float(np.clip(6.0 * (ch['green'] - 0.12) - 1.5 * (ch['dark'] - 0.5), -2.5, 2.5))

dets = []
for i, r in enumerate(rows):
    keep = []
    for d in r['d']:
        x1, y1, x2, y2 = d['b']; fx, fy = (x1 + x2) / 2, y2
        X, Y = project(INV[i], [[fx, fy]])[0]
        h = y2 - y1
        if not (-0.9 <= X <= 7.0 and -1.6 <= Y <= 14.3) or d['c'] < 0.25 or h < 30: continue
        # body height in metres from box height, rough sanity check (rejects people sitting behind the far line)
        top = project(INV[i], [[fx, y1]])[0]
        side = 0 if Y < NET else 1
        keep.append(dict(id=d['id'], b=d['b'], c=d['c'], foot=[fx, fy], X=float(X), Y=float(Y), side=side,
                         a=appearance(d.get('ch2'), side)))
    # drop duplicate boxes on the same person (keep the more confident one)
    keep.sort(key=lambda d: -d['c']); kept = []
    for d in keep:
        bx = d['b']; dup = False
        for k in kept:
            kb = k['b']
            ix = max(0, min(bx[2], kb[2]) - max(bx[0], kb[0])); iy = max(0, min(bx[3], kb[3]) - max(bx[1], kb[1]))
            a1 = (bx[2] - bx[0]) * (bx[3] - bx[1]); a2 = (kb[2] - kb[0]) * (kb[3] - kb[1])
            if ix * iy / min(a1, a2) > 0.6: dup = True; break
        if not dup: kept.append(d)
    dets.append(kept)

# smooth appearance along ByteTrack ids (+-20 frames)
byid = defaultdict(list)
for i, ds in enumerate(dets):
    for d in ds:
        if d['id'] >= 0: byid[(d['id'], d['side'])].append((i, d))
for k, lst in byid.items():
    idx = np.array([i for i, _ in lst]); a = np.array([d['a'] for _, d in lst])
    for i, d in lst:
        m = np.abs(idx - i) <= 45; d['as'] = float(np.median(a[m]))
for ds in dets:
    for d in ds: d.setdefault('as', d['a'])

foot = np.full((N, 4, 2), np.nan); court = np.full((N, 4, 2), np.nan); box = np.full((N, 4, 4), np.nan); conf = np.zeros((N, 4))
last = [None] * 4
for i, ds in enumerate(dets):
    for side in (0, 1):
        ids = [j for j in range(4) if SIDE[j] == side]
        sc = sorted([d for d in ds if d['side'] == side], key=lambda d: -d['c'])[:4]
        if not sc: continue
        C = np.zeros((len(sc), 2))
        for a, d in enumerate(sc):
            for b, j in enumerate(ids):
                z = d['as'] if b == 0 else -d['as']
                cost = 2.6 / (1 + np.exp(1.8 * z))                     # appearance: 0 good .. 2.6 bad
                if last[j] is not None and i - last[j][0] <= 20:
                    cost += min(np.hypot(d['X'] - last[j][1][0], d['Y'] - last[j][1][1]), 3.0) * 0.5
                else:
                    cost += 0.4
                C[a, b] = cost
        r_, c_ = linear_sum_assignment(C)
        for a, b in zip(r_, c_):
            if C[a, b] > 2.6: continue
            j = ids[b]; d = sc[a]
            foot[i, j] = d['foot']; court[i, j] = [d['X'], d['Y']]; box[i, j] = d['b']; conf[i, j] = d['c']
            last[j] = (i, (d['X'], d['Y']))

# feet hidden (net post, a near player in front): the box is short -> put the foot one typical body height below the head
for j in range(4):
    hs = box[:, j, 3] - box[:, j, 1]
    ok = ~np.isnan(hs)
    if ok.sum() < 30: continue
    idx = np.where(ok)[0]
    for t in idx:
        m = idx[np.abs(idx - t) <= 90]
        med = np.median(hs[m])
        if hs[t] < 0.82 * med and (SIDE[j] == 1 or box[t, j, 3] >= 955):
            fx = foot[t, j, 0]; fy = box[t, j, 1] + med
            foot[t, j, 1] = fy
            court[t, j] = project(INV[t], [[fx, fy]])[0]
# remove single-frame spikes before smoothing
for j in range(4):
    for k in range(2):
        col = court[:, j, k].copy(); ok = ~np.isnan(col)
        if ok.sum() > 9: col[ok] = median_filter(col[ok], size=7); court[:, j, k] = col

def interp(arr, maxgap):
    arr = arr.copy(); ok = ~np.isnan(arr[:, 0]); idx = np.where(ok)[0]
    for k in range(len(idx) - 1):
        s, e = idx[k], idx[k + 1]
        if 1 < e - s <= maxgap:
            for t in range(s + 1, e):
                w = (t - s) / (e - s); arr[t] = arr[s] * (1 - w) + arr[e] * w
    return arr

out = []
for j in range(4):
    c = interp(court[:, j], 15); f = interp(foot[:, j], 15); b = interp(box[:, j], 15)
    ok = ~np.isnan(c[:, 0]); cs = c.copy()
    win = 9 if SIDE[j] == 0 else 31
    s0 = None
    for t in range(N + 1):
        if t < N and ok[t]:
            if s0 is None: s0 = t
        elif s0 is not None:
            if t - s0 >= win:
                for k in range(2): cs[s0:t, k] = savgol_filter(c[s0:t, k], win, 2)
            s0 = None
    # image foot point consistent with the smoothed court position
    fs = np.full_like(f, np.nan)
    for t in np.where(ok)[0]: fs[t] = project(COURT[t], [cs[t]])[0]
    sp = np.zeros(N); dist = np.zeros(N); acc = 0.0
    for t in range(1, N):
        if ok[t] and ok[t - 1]:
            st = np.hypot(*(cs[t] - cs[t - 1])); v = st * FPS
            if v > 8.5: st = 0; v = 0
            sp[t] = v
            if v > (0.4 if SIDE[j] == 0 else 0.7): acc += st
        dist[t] = acc
    sp = np.convolve(median_filter(sp, size=9 if SIDE[j] == 0 else 15), np.ones(9) / 9, mode='same')
    # stamina index (not a measurement): drains with running load, recovers when walking/standing
    stam = np.zeros(N); s = 100.0
    for t in range(N):
        v = sp[t]
        s -= ((0.6 * v if v > 0.8 else 0.0) + 1.0 * max(0.0, v - 2.5) ** 1.5) / FPS
        if v < 0.5: s += 0.12 / FPS
        s = min(100.0, max(40.0, s)); stam[t] = s
    out.append(dict(name=NAMES[j], side=SIDE[j], ok=ok.tolist(),
                    court=np.where(np.isnan(cs), None, np.round(cs, 3)).tolist(),
                    foot=np.where(np.isnan(fs), None, np.round(fs, 1)).tolist(),
                    box=np.where(np.isnan(b), None, np.round(b, 1)).tolist(),
                    speed=np.round(sp, 2).tolist(), dist=np.round(dist, 2).tolist(), stamina=np.round(stam, 1).tolist(),
                    coverage=round(float(ok.mean()), 3)))
    print(f'{NAMES[j]}: coverage {ok.mean() * 100:.0f}%  distance {acc:.1f} m  max speed {sp.max():.1f} m/s  stamina end {stam[-1]:.0f}')
json.dump(dict(fps=FPS, n=N, players=out), open('players.json', 'w'))
