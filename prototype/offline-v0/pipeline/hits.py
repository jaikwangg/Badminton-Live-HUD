"""Pass 2b: shots from audio transients + arm-swing of the 4 tracked players, then rallies.
Usage: python3 hits.py  (reads players.json, audio.wav) -> events.json
"""
import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))

import json, sys, numpy as np, scipy.signal as ss
RISE, SNR, SWING = (float(a) for a in (sys.argv[1:4] if len(sys.argv) > 3 else (7, 9, 2.2)))
from scipy.io import wavfile
from scipy.ndimage import median_filter

P = json.load(open(os.path.join(DATA, 'players.json')))
FPS = P['fps']; N = P['n']; T = np.array(P['t'])
players = P['players']; KP = P['kps']

# ---------- audio transient candidates ----------
sr, x = wavfile.read(os.path.join(DATA, 'audio.wav')); x = x.astype(np.float32) / 32768
sos = ss.butter(4, [2500, 9000], btype='band', fs=sr, output='sos')
y = ss.sosfiltfilt(sos, x)
hop = int(sr * 0.004); n = len(y) // hop
e = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-7
db = 20 * np.log10(e)
base = median_filter(db, size=301)
rise = db - np.array([db[max(0, i - 5):i].min() if i > 0 else db[0] for i in range(n)])
ta = np.arange(n) * 0.004
aud = []
for i in range(1, n - 1):
    if rise[i] >= RISE and db[i] - base[i] >= SNR and db[i] >= db[i - 1] and db[i] >= db[i + 1]:
        aud.append([ta[i], db[i] - base[i]])
merged = []
for a in aud:
    if merged and a[0] - merged[-1][0] < 0.12:
        if a[1] > merged[-1][1]:
            merged[-1] = a
    else:
        merged.append(a)
aud = np.array(merged)
print('audio candidates', len(aud))

# ---------- swing signal per player: wrist speed in body-heights per second ----------
def wrist_speed(j):
    sp = np.full(N, np.nan); up = np.zeros(N)
    prev = None
    for i in range(N):
        k = KP[i][j]
        bx = players[j]['box'][i]
        if k is None or bx is None or bx[0] is None:
            prev = None; continue
        bh = max(40.0, bx[3] - bx[1])
        best = None
        for w, s in ((9, 5), (10, 6)):
            if k[w][2] > 0.3:
                best = max(best or 0, 1.0)
        pts = [np.array(k[w][:2]) if k[w][2] > 0.3 else None for w in (9, 10)]
        if prev is not None:
            v = 0.0
            for a, b in zip(pts, prev[0]):
                if a is not None and b is not None:
                    v = max(v, np.linalg.norm(a - b) / bh * FPS)
            sp[i] = v
        # wrist above shoulder line -> overhead stroke indicator
        if k[5][2] > 0.3 and k[6][2] > 0.3:
            sh = min(k[5][1], k[6][1])
            for w in (9, 10):
                if k[w][2] > 0.3 and k[w][1] < sh:
                    up[i] = 1
        prev = (pts,)
    return sp, up

SW = []; UP = []
for j in range(4):
    s, u = wrist_speed(j)
    SW.append(np.nan_to_num(s)); UP.append(u)
SW = np.array(SW); UP = np.array(UP)

def idx_of(t):
    return int(np.clip(np.searchsorted(T, t), 0, N - 1))

shots = []
for t, snr in aud:
    i0, i1 = idx_of(t - 0.15), idx_of(t + 0.10)
    if i1 <= i0:
        continue
    win = SW[:, i0:i1 + 1]
    peak = win.max(axis=1)
    j = int(np.argmax(peak))
    # require a clear arm swing on our court
    if peak[j] < SWING:
        continue
    side = players[j]['side']
    shots.append(dict(t=round(float(t), 3), p=j, side=side, snr=round(float(snr), 1), swing=round(float(peak[j]), 2),
                      overhead=bool(UP[j, i0:i1 + 1].max() > 0)))
print('audio+swing shots', len(shots))

# ---------- clean-up: alternate sides, drop doubles ----------
clean = []
for s in shots:
    if clean:
        prev = clean[-1]
        dt = s['t'] - prev['t']
        if dt < 0.35:
            if s['snr'] + s['swing'] > prev['snr'] + prev['swing']:
                clean[-1] = s
            continue
        if s['side'] == prev['side'] and dt < 0.9:
            if s['snr'] + s['swing'] > prev['snr'] + prev['swing']:
                clean[-1] = s
            continue
    clean.append(s)
shots = clean

# ---------- rallies: windows of high movement on our side that contain >=2 shots ----------
S = np.array([players[j]['speed'] for j in (0, 1)])
act = np.convolve(S.sum(0), np.ones(30) / 30, mode='same')
on = act > 1.35
wins = []; i = 0
while i < N:
    if on[i]:
        j = i
        while j < N and on[j]: j += 1
        wins.append([T[i], T[j - 1]]); i = j
    else:
        i += 1
merged = []
for w in wins:
    if merged and w[0] - merged[-1][1] < 1.2: merged[-1][1] = w[1]
    else: merged.append(w)
wins = [w for w in merged if w[1] - w[0] >= 2.5]
rallies = []
for w in wins:
    r = [s for s in shots if w[0] - 0.6 <= s['t'] <= w[1] + 0.3]
    if len(r) >= 2:
        rallies.append((w, r))
kept = set(id(s) for _, r in rallies for s in r)
shots = [s for s in shots if id(s) in kept]

# ---------- estimated average shuttle ground speed between consecutive hitters ----------
def court_at(j, t):
    i = idx_of(t)
    for d in range(0, 8):
        for k in (i - d, i + d):
            if 0 <= k < N and players[j]['court'][k] is not None and players[j]['court'][k][0] is not None:
                return np.array(players[j]['court'][k])
    return None

R = []
for ri, (w, r) in enumerate(rallies):
    for a, b in zip(r, r[1:]):
        if a['side'] != b['side']:
            pa, pb = court_at(a['p'], a['t']), court_at(b['p'], b['t'])
            dt = b['t'] - a['t']
            if pa is not None and pb is not None and 0.25 < dt < 2.5:
                d = float(np.hypot(*(pb - pa)))
                a['dist_m'] = round(d, 2)
                a['avg_kmh_est'] = round(d / dt * 3.6, 0)
    for s in r:
        s['rally'] = ri + 1
    R.append(dict(n=ri + 1, start=round(min(w[0], r[0]['t'] - 0.3), 2), end=round(max(w[1], r[-1]['t'] + 0.5), 2), shots=len(r),
                  first=r[0]['p'], last=r[-1]['p']))

for r in R:
    print(f"rally {r['n']}: {r['start']:.1f}-{r['end']:.1f}s shots {r['shots']}")
cnt = np.bincount([s['p'] for s in shots], minlength=4)
print('shots per player', cnt.tolist())
json.dump(dict(shots=shots, rallies=R, audio_candidates=len(aud),
               swing=np.round(SW, 2).tolist()), open(os.path.join(DATA, 'events.json'), 'w'))
