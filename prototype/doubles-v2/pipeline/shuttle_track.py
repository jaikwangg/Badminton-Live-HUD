"""Step 7b: link shuttle candidates into tracklets, then per rally pick ONE path whose direction changes happen only at
the racket of one of the 4 players (shuttles from neighbouring courts never qualify). -> shuttle.json"""
import json, numpy as np
from courtlib import project

C = json.load(open('cands.json')); P = json.load(open('players.json'))
COURT = np.load('court.npy'); N = min(len(C), P['n']); FPS = 30.0
PL = P['players']
rows = [json.loads(l) for l in open('dets2.jsonl')]

# ---------- 1. candidate filtering: drop blobs on legs / shoes of anybody, and below the near-court floor horizon ----------
def in_lower_body(x, y, i):
    # any detected person (players, neighbours, spectators): blobs on bodies are limbs / shirt prints, not the shuttle
    for d in rows[i]['d']:
        x1, y1, x2, y2 = d['b']; w, h = x2 - x1, y2 - y1
        if x1 - 3 <= x <= x2 + 3 and y1 - 3 <= y <= y2 + 0.08 * h + 3: return True
        if x1 - 0.35 * w <= x <= x2 + 0.35 * w and y2 - 0.15 * h <= y <= y2 + 0.1 * h: return True     # shoes of a stretched leg
    return False
cand = []
for i in range(N):
    cs = []
    for x, y, a, pk, v, sz in C[i]:
        if pk < 24: continue
        if in_lower_body(x, y, i): continue
        cs.append((x, y, a, pk))
    cand.append(cs)

# ---------- 2. tracklets: constant-velocity nearest neighbour, gaps up to 2 frames ----------
tracks = []; active = []
for i in range(N):
    used = set()
    pairs = []
    for ti, tr in enumerate(active):
        (li, lx, ly) = tr['pts'][-1]; vx, vy = tr['v']; g = i - li
        px, py = lx + vx * g, ly + vy * g
        spd = np.hypot(vx, vy)
        R = 55 + 10 * g if len(tr['pts']) == 1 else 8 + 0.6 * spd * g + 4 * g
        for ci, (x, y, a, pk) in enumerate(cand[i]):
            dd = np.hypot(x - px, y - py)
            if dd < R: pairs.append((dd / R, ti, ci))
    pairs.sort(); took = set()
    for sc, ti, ci in pairs:
        if ti in took or ci in used: continue
        tr = active[ti]; (li, lx, ly) = tr['pts'][-1]; x, y = cand[i][ci][:2]; g = i - li
        nv = ((x - lx) / g, (y - ly) / g)
        tr['v'] = nv if len(tr['pts']) == 1 else (0.5 * tr['v'][0] + 0.5 * nv[0], 0.5 * tr['v'][1] + 0.5 * nv[1])
        tr['pts'].append((i, x, y)); took.add(ti); used.add(ci)
    for ci, c in enumerate(cand[i]):
        if ci not in used:
            active.append(dict(pts=[(i, c[0], c[1])], v=(0.0, 0.0)))
    still = []
    for tr in active:
        if i - tr['pts'][-1][0] > 3: tracks.append(tr)
        else: still.append(tr)
    active = still
tracks += active

def quality(tr):
    p = np.array(tr['pts'], float)
    if len(p) < 5: return None
    t = p[:, 0]
    disp = np.hypot(p[-1, 1] - p[0, 1], p[-1, 2] - p[0, 2])
    if disp < 15: return None                                   # static flicker / limbs
    # smoothness: second differences on consecutive frames (shuttle flight is smooth, limbs/noise are jittery)
    acc = []
    for k in range(1, len(p) - 1):
        if p[k, 0] - p[k - 1, 0] == 1 and p[k + 1, 0] - p[k, 0] == 1:
            acc.append(np.hypot(p[k + 1, 1] - 2 * p[k, 1] + p[k - 1, 1], p[k + 1, 2] - 2 * p[k, 2] + p[k - 1, 2]))
    res = float(np.median(acc)) if acc else 0.0
    if acc and res > 6.0: return None
    return dict(t0=int(t[0]), t1=int(t[-1]), pts=p.tolist(), n=len(p), disp=float(disp), res=res)
good = [q for q in (quality(tr) for tr in tracks) if q]
good.sort(key=lambda q: q['t0'])
print('tracklets', len(tracks), 'smooth', len(good))

# ---------- 3. racket zones of the 4 players ----------
def zone_hit(x, y, i):
    """player whose racket reach contains (x, y): reach scaled by box height (see shots.py zone_score)."""
    i = min(max(int(i), 0), N - 1); best = None
    for j, p in enumerate(PL):
        b = p['box'][i]
        if not b or b[0] is None: continue
        w, h = b[2] - b[0], b[3] - b[1]; cx = (b[0] + b[2]) / 2
        dx = max(0.0, abs(x - cx) - 0.5 * w) / (0.45 * h + 4)
        top, low = b[1] - 0.3 * h, b[1] + 0.6 * h
        dy = (top - y) / (0.25 * h + 4) if y < top else max(0.0, y - low) / (0.25 * h + 4)
        sc = float(np.hypot(dx, dy))
        if sc <= 1.0 and (best is None or sc < best[1]): best = (j, sc)
    return best

def fit_seg(p):
    k = len(p)
    cx = np.polyfit(p[:, 0], p[:, 1], 1 if k < 5 else 2); cy = np.polyfit(p[:, 0], p[:, 2], 1 if k < 5 else 2)
    return cx, cy

def vel(c, t): return np.polyval(np.polyder(c), t)

def link(a, b):
    """('cont', cut) if b continues a's flight, ('hit', cut, j, t, x, y) if they meet at a player's racket and the
    direction changes, else None.  cut = first frame of b; a's points from there on are dropped (overlap)."""
    tb = b['t0']; gap = tb - a['t1']
    if gap > 18: return None
    pa = np.array([q for q in a['pts'] if q[0] < tb])
    if len(pa) < 4 or len(a['pts']) - len(pa) > 14: return None
    gap = tb - pa[-1, 0]
    if gap > 18: return None
    ea = pa[-6:]; eb = np.array(b['pts'][:6])
    cxa, cya = fit_seg(ea); cxb, cyb = fit_seg(eb)
    ta = pa[-1, 0]; xa, ya = pa[-1, 1:]; xb, yb = eb[0, 1:]
    va = np.array([vel(cxa, ta), vel(cya, ta)]); vb = np.array([vel(cxb, tb), vel(cyb, tb)])
    cosang = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
    px, py = np.polyval(cxa, tb), np.polyval(cya, tb)
    if gap <= 10 and cosang > 0.75 and np.hypot(px - xb, py - yb) < 10 + 2.5 * max(gap, 1):
        return ('cont', tb)
    if gap <= 12 and cosang < 0.6 and np.hypot(xa - xb, ya - yb) < min(30 + 50 * max(gap, 1), 180):
        mx, my, mt = (xa + xb) / 2, (ya + yb) / 2, int((ta + tb) / 2)
        z = zone_hit(mx, my, mt) or zone_hit(xa, ya, int(ta)) or zone_hit(xb, yb, int(tb))
        if z:
            return ('hit', tb, z[0], (ta + tb) / 2, mx, my)
    return None

# ---------- 4. best path (DP over tracklets, value = frames covered) ----------
n = len(good); best = [q['n'] for q in good]; prev = [-1] * n; how = [None] * n
for b in range(n):
    for a in range(b):
        if good[a]['t0'] >= good[b]['t0'] - 3 or good[b]['t0'] - good[a]['t1'] > 18: continue
        L = link(good[a], good[b])
        if L is None: continue
        lost = sum(1 for q in good[a]['pts'] if q[0] >= good[b]['t0'])
        v = best[a] - 2 * lost + good[b]['n'] - (2 if L[0] == 'cont' else 0)
        if v > best[b]: best[b] = v; prev[b] = a; how[b] = L
# extract chains greedily, best first; a chain that overlaps an accepted one in time is dropped
used = set(); chains = []; spans = []
for e in np.argsort(best)[::-1]:
    if e in used: continue
    ch = []; k = e; ok = True
    while k != -1:
        if k in used: ok = False; break
        ch.append(k); k = prev[k]
    if not ok: continue
    ch = ch[::-1]
    t0, t1 = good[ch[0]]['t0'], good[ch[-1]]['t1']
    if sum(good[c]['n'] for c in ch) < 20: continue
    allp = np.array([q for c in ch for q in good[c]['pts']])
    if max(np.ptp(allp[:, 1]), np.ptp(allp[:, 2])) < 60: continue      # shuttle held / bounced in hand, not a flight
    if any(min(t1, b_) - max(t0, a_) > 0.3 * (t1 - t0) for a_, b_ in spans): continue
    for c in ch: used.add(c)
    chains.append(ch); spans.append((t0, t1))
chains.sort(key=lambda ch: good[ch[0]]['t0'])
out = []
for ch in chains:
    pts = []; hits = []
    for idx, c in enumerate(ch):
        nxt_cut = how[ch[idx + 1]][1] if idx + 1 < len(ch) else 10 ** 9
        pts += [q for q in good[c]['pts'] if q[0] < nxt_cut]
        if idx > 0 and how[c][0] == 'hit':
            _, _, j, t, x, y = how[c]; hits.append(dict(t=t / FPS, frame=t, player=j, x=x, y=y, src='track'))
    out.append(dict(t0=good[ch[0]]['t0'] / FPS, t1=good[ch[-1]]['t1'] / FPS, pts=pts, hits=hits, n_tracklets=len(ch)))
    print(f"path {good[ch[0]]['t0'] / FPS:6.2f}-{good[ch[-1]]['t1'] / FPS:6.2f}s  tracklets {len(ch):2d}  points {len(pts):4d}  hits " +
          ' '.join(f"{h['t']:.2f}:{PL[h['player']]['name']}" for h in hits))
json.dump(dict(paths=out, tracklets=good), open('shuttle.json', 'w'))
