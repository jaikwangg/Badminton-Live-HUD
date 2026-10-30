"""Step 8: shots, rallies and shot speed.
- shot evidence: the chosen shuttle path turns back / is knocked sideways at a racket; the path starts at a racket
  (serve); an audio hit while the shuttle is not visible.
- sides must alternate. Each shot gets the sides it is geometrically compatible with (a near player's racket zone is
  large and can cover far players in the image, so a shot can be compatible with both); a small dynamic programme
  picks the alternating sequence that needs the fewest corrections (insert a hidden shot, drop a doubtful one).
- hitter = the team-mate of that side nearer the shuttle.
- speed off the racket v0 = (e^(kD) - 1) / (kT), k = g / 6.8^2, D = hitter -> next hitter on the floor (m), T = flight time.
- config.json: add_shots / remove_shots / rally_end.  -> events.json"""
import json, numpy as np
from courtlib import project

CFG = json.load(open('config.json')); S = json.load(open('shuttle.json')); P = json.load(open('players.json'))
ON = json.load(open('onsets.json'))['onsets']; COURT = np.load('court.npy')
PL = P['players']; N = P['n']; FPS = 30.0; NAMES = [p['name'] for p in PL]; SIDE = [p['side'] for p in PL]
K = 9.81 / 6.8 ** 2
AOFF = CFG.get('audio_offset', 0.0)

spos = {}
for path in S['paths']:
    for f, x, y in path['pts']: spos[int(f)] = (x, y)
def shuttle_at(f, win=3):
    for d in range(win + 1):
        for g in (f - d, f + d):
            if g in spos: return spos[g]
    return None
def visible(t): return shuttle_at(int(round(t * FPS)), 3) is not None
def box(j, f, lim=30):
    """player box at frame f; if the player is hidden (e.g. behind a near player) use the nearest box within lim frames"""
    f = int(min(max(f, 0), N - 1))
    for d in range(lim + 1):
        for g in (f - d, f + d):
            if 0 <= g < N:
                b = PL[j]['box'][g]
                if b and b[0] is not None: return b
    return None
def ppos(j, f):
    f = int(min(max(f, 0), N - 1))
    for d in range(0, 25):
        for g in (f - d, f + d):
            if 0 <= g < N and PL[j]['court'][g] and PL[j]['court'][g][0] is not None: return np.array(PL[j]['court'][g])
    return None
def zone_score(j, f, xy):
    """0 = shuttle within reach of player j's racket; 1 = edge of reach. Reach is scaled by the player's box height
    (arm + racket ~0.55 box-heights above the head, ~0.5 box-heights sideways)."""
    b = box(j, f)
    if b is None or xy is None: return None
    w, h = b[2] - b[0], b[3] - b[1]
    cx = (b[0] + b[2]) / 2
    dx = max(0.0, abs(xy[0] - cx) - 0.5 * w) / (0.45 * h + 4)
    top, low = b[1] - 0.3 * h, b[1] + 0.6 * h
    dy = (top - xy[1]) / (0.25 * h + 4) if xy[1] < top else (max(0.0, xy[1] - low) / (0.25 * h + 4))
    return float(np.hypot(dx, dy))
def sides_for(f, xy, lim=1.0):
    if xy is None: return {0, 1}, None
    sc = {j: zone_score(j, f, xy) for j in range(4)}
    ok = {SIDE[j] for j, s in sc.items() if s is not None and s <= lim}
    return (ok if ok else {0, 1}), sc
def last_seen(f, lim=40):
    for g in range(f - 1, f - lim, -1):
        if g in spos: return spos[g]
    for g in range(f + 1, f + lim):
        if g in spos: return spos[g]
    return None
def hitter(side, f, xy):
    best = None
    for j in range(4):
        if SIDE[j] != side: continue
        b = box(j, f)
        if b is None: continue
        if xy is not None:
            h = b[3] - b[1]; d = np.hypot(xy[0] - (b[0] + b[2]) / 2, xy[1] - (b[1] + 0.2 * h)) / h
        else:                                   # shuttle hidden: side-to-side distance to where it was last seen
            ref = shuttle_at(f, 10) or last_seen(f)
            d = 0.0 if ref is None else abs(ref[0] - (b[0] + b[2]) / 2)
        if best is None or d < best[1]: best = (j, d)
    return best[0] if best else [j for j in range(4) if SIDE[j] == side][0]

# ---------- shot evidence from the shuttle paths ----------
raw = []
for path in S['paths']:
    for h in path['hits']:
        raw.append(dict(t=h['t'], f=int(round(h['frame'])), xy=(h['x'], h['y']), src='track'))
    pts = np.array(path['pts'])
    for k in range(4, len(pts) - 4):                       # turn-back inside one tracklet
        if pts[k + 4, 0] - pts[k - 4, 0] > 12: continue
        va = (pts[k] - pts[k - 4])[1:] / max(pts[k, 0] - pts[k - 4, 0], 1); vb = (pts[k + 4] - pts[k])[1:] / max(pts[k + 4, 0] - pts[k, 0], 1)
        if min(np.linalg.norm(va), np.linalg.norm(vb)) < 1.5: continue
        # a hit is a sharp cusp: velocity jumps between the 2 frames before and the 2 frames after
        v1 = (pts[k] - pts[k - 2])[1:] / max(pts[k, 0] - pts[k - 2, 0], 1); v2 = (pts[k + 2] - pts[k])[1:] / max(pts[k + 2, 0] - pts[k, 0], 1)
        if np.linalg.norm(v2 - v1) < 8: continue
        if va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)) < -0.2:
            f = int(pts[k, 0]); xy = tuple(pts[k, 1:]); s, _ = sides_for(f, xy)
            if any(zone_score(j, f, xy) is not None and zone_score(j, f, xy) <= 1.0 for j in range(4)):
                raw.append(dict(t=f / FPS, f=f, xy=xy, src='track'))
    # kink at a racket confirmed by a hit sound: a far-side hit often keeps the shuttle moving down the image
    # (it comes toward the camera), so there is no turn-back, only a sudden change of speed / direction
    for o in ON:
        f = int(round((o['t'] + AOFF) * FPS))
        idx = np.where(np.abs(pts[:, 0] - f) <= 4)[0]
        if len(idx) == 0: continue
        k = idx[np.argmin(np.abs(pts[idx, 0] - f))]
        if k < 3 or k + 3 >= len(pts): continue
        if pts[k + 3, 0] - pts[k - 3, 0] > 9: continue
        v1 = (pts[k] - pts[k - 3])[1:] / max(pts[k, 0] - pts[k - 3, 0], 1); v2 = (pts[k + 3] - pts[k])[1:] / max(pts[k + 3, 0] - pts[k, 0], 1)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if min(n1, n2) < 0.5: continue
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
        if (n2 / n1 > 1.8 or n1 / n2 > 2.5 or ang > 35):
            xy = tuple(pts[k, 1:])
            if any(zone_score(j, int(pts[k, 0]), xy) is not None and zone_score(j, int(pts[k, 0]), xy) <= 1.2 for j in range(4)):
                raw.append(dict(t=pts[k, 0] / FPS, f=int(pts[k, 0]), xy=xy, src='track-kink'))
    f0, x0, y0 = path['pts'][0]                             # path starts at a racket (only used for a rally's first path)
    if any(zone_score(j, int(f0), (x0, y0)) is not None and zone_score(j, int(f0), (x0, y0)) <= 1.0 for j in range(4)):
        raw.append(dict(t=f0 / FPS, f=int(f0), xy=(x0, y0), src='track-start', path_t0=path['t0']))
# audio hits heard while the shuttle is not visible: weaker evidence (cheaper to drop), side unknown
for o in ON:
    t = o['t'] + AOFF; f = int(round(t * FPS))
    if not visible(t):
        raw.append(dict(t=t, f=f, xy=None, src='audio', snr=o['snr']))
    else:                                   # heard while the shuttle is in view: side from where the shuttle is
        xy = shuttle_at(f, 3)
        if any(zone_score(j, f, xy) is not None and zone_score(j, f, xy) <= 1.3 for j in range(4)):
            raw.append(dict(t=t, f=f, xy=xy, src='audio-seen', snr=o['snr']))
raw.sort(key=lambda s: s['t'])
dd = []
for s in raw:
    if dd and s['t'] - dd[-1]['t'] < 0.25 and (s['xy'] is None or dd[-1]['xy'] is None or
                                               np.hypot(s['xy'][0] - dd[-1]['xy'][0], s['xy'][1] - dd[-1]['xy'][1]) < 40):
        rank = {'track': 3, 'track-kink': 2, 'track-start': 2, 'audio-seen': 1, 'audio': 1}
        if rank.get(s['src'], 0) > rank.get(dd[-1]['src'], 0): dd[-1] = s
        continue
    dd.append(s)
raw = dd
ons = np.array([o['t'] + AOFF for o in ON])
def near_onset(t): return float(np.min(np.abs(ons - t))) if len(ons) else 9.0
dd = []
for s in raw:
    if dd and s['xy'] is not None and dd[-1]['xy'] is not None and s['t'] - dd[-1]['t'] < 0.5 and \
            np.hypot(s['xy'][0] - dd[-1]['xy'][0], s['xy'][1] - dd[-1]['xy'][1]) < 30:
        if near_onset(s['t']) < near_onset(dd[-1]['t']): dd[-1] = s
        continue
    dd.append(s)
raw = dd
for s in raw:
    s['sides'], s['zs'] = sides_for(s['f'], s['xy'], 1.3 if s['src'] == 'audio-seen' else 1.0)

# ---------- rally windows ----------
spans = sorted([[p['t0'], p['t1']] for p in S['paths']])
win = []
for a, b in spans:
    if win and a - win[-1][1] < 1.8: win[-1][1] = max(win[-1][1], b)
    else: win.append([a, b])
win = [w for w in win if w[1] - w[0] > 0.6]
win = [w for w in win if not any(a_ <= (w[0] + w[1]) / 2 <= b_ for a_, b_ in CFG.get('ignore_ranges', []))]
for ad in CFG.get('add_shots', []):          # a configured shot outside every window starts / extends one
    if not any(w[0] - 2.5 <= ad['t'] <= w[1] + 2.5 for w in win): win.append([ad['t'], ad['t'] + 0.5])
win.sort()

def resolve(shots, onsets):
    """DP: keep/drop each shot, insert hidden shots between neighbours, sides alternate. Returns final list."""
    n = len(shots)
    if n == 0: return []
    INF = 1e9
    # state: (index of last kept shot, its side); value = cost
    best = {}
    dropc = [0.5 if s['src'] in ('audio', 'audio-seen') else (50.0 if s['src'] == 'config' else 1.0) for s in shots]
    for i, s in enumerate(shots):
        for side in (0, 1):
            c = 0.0 if side in s['sides'] else 2.0
            c += sum(dropc[:i])                                # dropping all earlier shots
            best[(i, side)] = (c, None, None)
    for i in range(n):
        for side in (0, 1):
            if (i, side) not in best: continue
            ci = best[(i, side)][0]
            for k in range(i + 1, n):
                s = shots[k]; drop = float(sum(dropc[i + 1:k]))
                mg = 0.2 if (s['src'] == 'config' and shots[i]['src'] == 'config') else 0.3
                if s['t'] - shots[i]['t'] < mg: continue         # the shuttle cannot cross the court that fast
                for side2 in (0, 1):
                    c = ci + drop + (0.0 if side2 in s['sides'] else (50.0 if s['src'] == 'config' else 2.0))
                    ins = None
                    if side2 == side:                          # need one hidden shot of the other side in between
                        gap = s['t'] - shots[i]['t']
                        cand = [t for t in onsets if shots[i]['t'] + 0.3 < t < s['t'] - 0.3 and not visible(t)]
                        if cand:
                            c += 0.5; ins = (min(cand, key=lambda t: abs(t - (shots[i]['t'] + s['t']) / 2)), 'audio')
                        elif gap >= 0.66:
                            c += 1.3; ins = ((shots[i]['t'] + s['t']) / 2, 'inferred')
                        else:
                            continue
                    if c < best.get((k, side2), (INF,))[0]:
                        best[(k, side2)] = (c, (i, side), ins)
    end = min(((v[0] + sum(dropc[key[0] + 1:]), key) for key, v in best.items()))[1]
    seq = []; key = end
    while key is not None:
        c, pk, ins = best[key]; s = dict(shots[key[0]]); s['side'] = key[1]; seq.append(s)
        if ins: seq.append(dict(t=ins[0], f=int(round(ins[0] * FPS)), xy=None, src=ins[1], side=1 - key[1], sides={1 - key[1]}))
        key = pk
    return seq[::-1]

rallies = []; USED_ADDS = set()
for ri, (a, b) in enumerate(win):
    n = ri + 1
    RE = {float(k): v for k, v in CFG.get('rally_end', {}).items()}      # key = any time inside the rally
    fixed_end = next((v for k, v in RE.items() if a - 1.0 <= k <= b + 2.5), None)
    end = fixed_end if fixed_end is not None else b + 0.4
    adds_after = [ad['t'] for ad in CFG.get('add_shots', []) if b < ad['t'] <= b + 2.5]
    if adds_after and fixed_end is None: end = max(end, max(adds_after) + 0.6)
    shots = [dict(s) for s in raw if a - 0.4 <= s['t'] <= end and (s['src'] != 'track-start' or abs(s['path_t0'] - a) < 0.05)]
    onsets = [o['t'] + AOFF for o in ON if a - 0.6 <= o['t'] + AOFF <= end]
    for rm in CFG.get('remove_shots', []):
        shots = [s for s in shots if abs(s['t'] - rm) > 0.15]
    adds = [ad for ad in CFG.get('add_shots', []) if a - 2.5 <= ad['t'] <= end + 2.5 and ad['t'] not in USED_ADDS]
    USED_ADDS.update(ad['t'] for ad in adds)
    shots = [s for s in shots if not any(abs(s['t'] - ad['t']) < 0.25 for ad in adds)]
    for ad in adds:
        if True:
            j = NAMES.index(ad['player']); f = int(round(ad['t'] * FPS))
            shots.append(dict(t=ad['t'], f=f, xy=shuttle_at(f, 6), src='config', sides={SIDE[j]}, player=j))
    shots.sort(key=lambda s: s['t'])
    seq = resolve(shots, onsets)
    # a heard-only shot at the end of a rally needs the shuttle to show up again afterwards (else: landing / other court)
    while seq and seq[-1]['src'] == 'audio' and not any(seq[-1]['f'] < f <= seq[-1]['f'] + 45 for f in spos):
        seq.pop()
    # serve before the first seen shot, heard but not seen
    if seq:
        pre = [t for t in onsets if seq[0]['t'] - 2.4 < t < seq[0]['t'] - 0.4 and not visible(t)]
        if pre:
            t = max(pre); seq.insert(0, dict(t=t, f=int(round(t * FPS)), xy=None, src='audio', side=1 - seq[0]['side']))
    for s in seq:
        if 'player' not in s: s['player'] = hitter(s['side'], s['f'], s['xy'])
        s['name'] = NAMES[s['player']]
    for k, s in enumerate(seq):
        s['speed_kmh'] = s['D'] = s['T'] = None
        pa = ppos(s['player'], s['f']); pb = None; T = None
        if k + 1 < len(seq):
            nx = seq[k + 1]; pb = ppos(nx['player'], nx['f']); T = nx['t'] - s['t']
        else:
            last = [f for f in spos if s['f'] < f <= int(end * FPS) + 3]
            if last:
                f = max(last); x, y = spos[f]
                XY = project(np.linalg.inv(COURT[f]), [[x, y]])[0]
                if -1.5 < XY[0] < 7.6 and -1 < XY[1] < 15 and (XY[1] > 6.7) == (s['side'] == 0) and (f - s['f']) / FPS >= 0.3:
                    pb = XY; T = (f - s['f']) / FPS
        if pa is not None and pb is not None and T and T > 0.15:
            D = float(np.hypot(*(pb - pa)))
            if k + 1 == len(seq) and (D < 3.0 or abs(pb[1] - 6.7) < 1.0): continue   # last shot into / just over the net
            v = (np.exp(K * D) - 1) / (K * T) * 3.6
            s['D'] = round(D, 2); s['T'] = round(T, 3); s['speed_kmh'] = round(float(v))
    if seq and fixed_end is None:
        lt = seq[-1]['t']; after = [f / FPS for f in spos if lt < f / FPS <= lt + 2.4]
        end = round(min(end, (max(after) if after else lt + 1.0) + 0.3), 2)
        seq = [q for q in seq if q['t'] <= end]
    st = min(a, seq[0]['t'] - 0.3) if seq else a
    rallies.append(dict(n=n, start=round(st, 2), end=round(end, 2),
                        shots=[dict(t=round(s['t'], 3), frame=s['f'], player=s['player'], name=s['name'], side=s['side'], src=s['src'],
                                    speed_kmh=s['speed_kmh'], D=s['D'], T=s['T'], xy=[round(v, 1) for v in s['xy']] if s['xy'] else None)
                               for s in seq]))

warn = []
for r in rallies:
    sh = r['shots']
    for k, s in enumerate(sh):
        if s['speed_kmh'] is not None and (s['speed_kmh'] > 250 or s['speed_kmh'] < 25):
            warn.append(dict(t=s['t'], rally=r['n'], kind='speed', msg=f"ความเร็ว {s['speed_kmh']} km/h ({s['name']})"))
        if k + 1 < len(sh):
            g = sh[k + 1]['t'] - s['t']
            if g < 0.33: warn.append(dict(t=s['t'], rally=r['n'], kind='gap', msg=f'ช็อตถัดไปห่าง {g:.2f} วิ (สั้นเกิน)'))
            if g > 2.4: warn.append(dict(t=s['t'], rally=r['n'], kind='gap', msg=f'ช็อตถัดไปห่าง {g:.2f} วิ (นานเกิน)'))
        if s['src'] == 'inferred':
            warn.append(dict(t=s['t'], rally=r['n'], kind='inferred', msg=f"เดาจากการสลับฝั่ง ไม่เห็นลูก ไม่มีเสียง ({s['name']})"))
json.dump(dict(rallies=rallies, warnings=warn), open('events.json', 'w'), ensure_ascii=False, indent=1)
for r in rallies:
    print(f"rally {r['n']}: {r['start']:.2f}-{r['end']:.2f}s  shots {len(r['shots'])}  " +
          ' '.join(f"{s['t']:.2f}{s['name']}[{s['src'][:2]}]{'' if s['speed_kmh'] is None else s['speed_kmh']}" for s in r['shots']))
print('warnings', len(warn))
for w in warn: print('  ', w['t'], w['msg'])
