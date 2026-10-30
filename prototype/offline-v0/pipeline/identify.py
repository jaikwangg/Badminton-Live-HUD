"""Pass 2a: assign the 4 named players per frame, smooth positions, compute speed/distance.
Usage: python3 identify.py dets.json  -> players.json
"""
import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))

import json, sys, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

D = json.load(open(sys.argv[1]))
frames = D['frames']
import os
if os.path.exists(os.path.join(DATA, 'far_dets.json')):
    FD = json.load(open(os.path.join(DATA, 'far_dets.json'))); added = 0
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u > 0 else 0
    for fr in frames:
        for d in FD.get(str(fr['f']), []):
            if d['conf'] < 0.2:
                continue
            if any(iou(d['box'], e['box']) > 0.45 for e in fr['dets']):
                continue
            d['conf'] = max(d['conf'], 0.3)  # passes the shared validity gate
            fr['dets'].append(d); added += 1
    print('far-pass detections added:', added)
FPS = D['fps'] / D['step']
NET_Y = 6.70

# identity order: 0 ปั้น (near, navy), 1 พฤกษ (near, royal-blue jersey #11), 2 ปลื้ม (far, black shirt), 3 ภูสิท (far)
NAMES = ['ปั้น', 'พฤกษ', 'ปลื้ม', 'ภูสิท']
SIDE = [0, 0, 1, 1]

def qscore(d):
    """>0 means the darker/plain identity of the side (near: ปั้น navy+white shoes, far: ปลื้ม black shirt)."""
    f = d.get('f2') or {}
    if d['court'][1] < NET_Y:
        q = 0.0
        if 'shoe' in f: q += 3.0 * (f['shoe'] - 0.4)
        if 'white' in f: q += -12.0 * (f['white'] - 0.05)
        if 'v' in f: q += -(f['v'] - 66) / 20.0
        return q
    if 'white' in f:
        return 4.0 * (0.4 - f['white'])
    return 0.0

def valid(d):
    X, Y = d['court']
    bh = d['box'][3] - d['box'][1]
    return d['conf'] >= 0.3 and -0.9 <= X <= 7.0 and -1.0 <= Y <= 14.8 and bh > 60

# smooth q along each ByteTrack id over +-15 frames
from collections import defaultdict
bytid = defaultdict(list)
for i, fr in enumerate(frames):
    for d in fr['dets']:
        if valid(d):
            d['q'] = qscore(d)
            bytid[d['tid']].append((i, d))
for tid, lst in bytid.items():
    idx = np.array([i for i, _ in lst]); qs = np.array([d['q'] for _, d in lst])
    for i, d in lst:
        m = np.abs(idx - i) <= 15
        d['qs'] = float(qs[m].mean()) * 0.6 + d['q'] * 0.4 if tid >= 0 else d['q']

def idcost(d, j):
    q = d.get('qs', 0.0)
    plain = (j in (0, 2))
    z = q if plain else -q
    return 2.0 / (1.0 + np.exp(2.0 * z))   # 0 (confident match) .. 2 (confident mismatch)

# --- per-frame assignment ---
N = len(frames)
pos = np.full((N, 4, 2), np.nan)      # court XY
foot = np.full((N, 4, 2), np.nan)     # image foot px
box = np.full((N, 4, 4), np.nan)
kps = [[None] * 4 for _ in range(N)]
cut = np.zeros((N, 4), bool)
last = [None] * 4
for i, fr in enumerate(frames):
    cands = [d for d in fr['dets'] if valid(d)]
    # de-duplicate heavily overlapping boxes
    cands.sort(key=lambda d: -d['conf'])
    keep = []
    for d in cands:
        bx = d['box']
        dup = False
        for k in keep:
            kb = k['box']
            ix = max(0, min(bx[2], kb[2]) - max(bx[0], kb[0])); iy = max(0, min(bx[3], kb[3]) - max(bx[1], kb[1]))
            inter = ix * iy
            a = (bx[2] - bx[0]) * (bx[3] - bx[1]); b = (kb[2] - kb[0]) * (kb[3] - kb[1])
            if inter / min(a, b) > 0.7:
                dup = True; break
        if not dup:
            keep.append(d)
    for side in (0, 1):
        ids = [j for j in range(4) if SIDE[j] == side]
        sc = [d for d in keep if (d['court'][1] < NET_Y) == (side == 0)][:3]
        if not sc:
            continue
        C = np.zeros((len(sc), 2))
        for a, d in enumerate(sc):
            for b, j in enumerate(ids):
                cost = idcost(d, j) * 1.5
                if last[j] is not None and i - last[j][0] <= 15:
                    dist = np.hypot(d['court'][0] - last[j][1][0], d['court'][1] - last[j][1][1])
                    cost += min(dist, 3.0) * 0.6
                C[a, b] = cost
        r, c = linear_sum_assignment(C)
        for a, b in zip(r, c):
            if C[a, b] > 3.0:
                continue
            j = ids[b]; d = sc[a]
            pos[i, j] = d['court']; foot[i, j] = d['foot']; box[i, j] = d['box']; kps[i][j] = d['kp']; cut[i, j] = d['cut']
            last[j] = (i, d['court'])

# --- label smoothing: majority over a sliding window to suppress single-frame swaps (near side) ---
# swap check: if two identities on the same side swap positions for < 6 frames, revert
for side in (0, 1):
    a, b = [j for j in range(4) if SIDE[j] == side]
    i = 1
    while i < N - 1:
        if not (np.isnan(pos[i - 1, a, 0]) or np.isnan(pos[i - 1, b, 0]) or np.isnan(pos[i, a, 0]) or np.isnan(pos[i, b, 0])):
            stay = np.hypot(*(pos[i, a] - pos[i - 1, a])) + np.hypot(*(pos[i, b] - pos[i - 1, b]))
            swap = np.hypot(*(pos[i, a] - pos[i - 1, b])) + np.hypot(*(pos[i, b] - pos[i - 1, a]))
            if swap + 0.3 < stay * 0.5:
                for arr in (pos, foot, box):
                    arr[i, [a, b]] = arr[i, [b, a]]
                kps[i][a], kps[i][b] = kps[i][b], kps[i][a]
                cut[i, [a, b]] = cut[i, [b, a]]
        i += 1

# --- interpolate short gaps, smooth, speed & distance ---
def interp(arr, maxgap=12):
    arr = arr.copy()
    valid_ = ~np.isnan(arr[:, 0])
    idx = np.where(valid_)[0]
    filled = np.zeros(len(arr), bool)
    for k in range(len(idx) - 1):
        s, e = idx[k], idx[k + 1]
        if 1 < e - s <= maxgap:
            for t in range(s + 1, e):
                w = (t - s) / (e - s)
                arr[t] = arr[s] * (1 - w) + arr[e] * w
                filled[t] = True
    return arr, filled

out_players = []
for j in range(4):
    p, filled = interp(pos[:, j])
    fp, _ = interp(foot[:, j])
    bx, _ = interp(box[:, j])
    ok = ~np.isnan(p[:, 0])
    ps = p.copy()
    # smooth contiguous segments
    seg_start = None
    for t in range(N + 1):
        if t < N and ok[t]:
            if seg_start is None:
                seg_start = t
        else:
            if seg_start is not None:
                L = t - seg_start
                win = 9 if SIDE[j] == 0 else 21
                if L >= win:
                    for c in range(2):
                        ps[seg_start:t, c] = savgol_filter(p[seg_start:t, c], win, 2)
                elif L >= 5:
                    for c in range(2):
                        ps[seg_start:t, c] = savgol_filter(p[seg_start:t, c], 5 if L < 9 else 9, 1)
                seg_start = None
    spd = np.zeros(N); dist = np.zeros(N)
    acc = 0.0
    for t in range(1, N):
        if ok[t] and ok[t - 1]:
            step = np.hypot(*(ps[t] - ps[t - 1]))
            v = step * FPS
            if v > 9.0:  # physically implausible for a badminton player -> tracking jump
                step = 0.0; v = 0.0
            spd[t] = v
            if v > (0.35 if SIDE[j] == 0 else 0.6):
                acc += step
        dist[t] = acc
    # smooth speed for display
    from scipy.ndimage import median_filter as _mf
    spd_s = np.convolve(_mf(spd, size=9 if SIDE[j] == 0 else 15), np.ones(7) / 7, mode='same')
    out_players.append(dict(name=NAMES[j], side=SIDE[j], ok=ok.tolist(), filled=filled.tolist(),
                            court=np.where(np.isnan(ps), None, np.round(ps, 3)).tolist(),
                            foot=np.where(np.isnan(fp), None, np.round(fp, 1)).tolist(),
                            box=np.where(np.isnan(bx), None, np.round(bx, 1)).tolist(),
                            cut=cut[:, j].tolist(),
                            speed=np.round(spd_s, 2).tolist(), dist=np.round(dist, 2).tolist(),
                            coverage=float(ok.mean())))
    print(f'{NAMES[j]}: coverage {ok.mean() * 100:.0f}%  dist {acc:.1f} m  max speed {spd_s.max():.1f} m/s')

json.dump(dict(fps=FPS, n=N, t=[fr['t'] for fr in frames], players=out_players, kps=kps), open(os.path.join(DATA, 'players.json'), 'w'))
