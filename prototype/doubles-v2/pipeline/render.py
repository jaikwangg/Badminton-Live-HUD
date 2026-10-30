"""Step 9: draw the HUD on the 1080x1920 master with Pillow (Thai fonts) and put the original audio back.
Usage: python3 render.py out.mp4          HUD_PREVIEW=12.3 python3 render.py preview.png"""
import os, sys, json, subprocess, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from courtlib import project

OUT = sys.argv[1] if len(sys.argv) > 1 else 'hud.mp4'
PREV = os.environ.get('HUD_PREVIEW')
FONTS = os.environ.get('HUD_FONTS', 'fonts')
CFG = json.load(open('config.json')); P = json.load(open('players.json')); E = json.load(open('events.json'))
S = json.load(open('shuttle.json')); COURT = np.load('court.npy')
PL = P['players']; N = P['n']; FPS = 30.0
NAMES = [p['name'] for p in PL]; COL = [tuple(p['color']) for p in CFG['players']]
RALLIES = E['rallies']; SHOTS = [dict(s, rally=r['n']) for r in RALLIES for s in r['shots']]
spos = {}
for path in S['paths']:
    for f, x, y in path['pts']: spos[int(f)] = (2 * x, 2 * y)
# drop points where the 'shuttle' is not moving (a blob resting on the floor / a shoe), they are not in flight
moving = {}
for f, (x, y) in spos.items():
    nb = [spos[g] for g in (f - 3, f + 3) if g in spos]
    moving[f] = any(np.hypot(x - a, y - b) > 6 for a, b in nb)
spos = {f: p for f, p in spos.items() if moving[f]}

def F(n, s): return ImageFont.truetype(os.path.join(FONTS, n), s, layout_engine=ImageFont.Layout.RAQM)
TH = lambda s: F('IBMPlexSansThai-SemiBold.ttf', s); THM = lambda s: F('IBMPlexSansThai-Medium.ttf', s)
NB = lambda s: F('BarlowCondensed-Bold.ttf', s); NS = lambda s: F('BarlowCondensed-SemiBold.ttf', s); NM = lambda s: F('BarlowCondensed-Medium.ttf', s)
W, H = 1080, 1920
NAVY = (14, 26, 44); EDGE = (64, 120, 170); FG = (238, 243, 248); MUTED = (158, 178, 200); CYAN = (88, 196, 236); YEL = (250, 214, 64)

# ---------- static panel layer (RGBA) ----------
panel = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(panel)
TOP_Y, TOP_H = 64, 136
TOPC = [(28, 360), (400, 300), (712, 340)]
for x, w in TOPC:
    d.rectangle((x, TOP_Y, x + w, TOP_Y + TOP_H), fill=NAVY + (205,))
    d.rectangle((x, TOP_Y, x + 5, TOP_Y + TOP_H), fill=CYAN + (255,))
    d.rectangle((x, TOP_Y + TOP_H - 2, x + w, TOP_Y + TOP_H), fill=EDGE + (200,))
for (x, w), lab in zip(TOPC, ('RALLY TIME', 'RALLY COUNT', 'SHOT SPEED')):
    d.text((x + 20, TOP_Y + 10), lab, font=NS(30), fill=MUTED + (255,))
d.text((TOPC[1][0] + TOPC[1][1] - 70, TOP_Y + TOP_H - 48), 'SHOT', font=NS(30), fill=MUTED + (255,))
d.text((TOPC[2][0] + TOPC[2][1] - 82, TOP_Y + TOP_H - 48), 'km/h', font=NS(30), fill=MUTED + (255,))
BOT_Y, BOT_H = 1560, 330
LP = (20, 520); RP = (556, 504)
ROW_H = (BOT_H - 16) // 4
for j in range(4):
    y = BOT_Y + 8 + j * ROW_H
    d.rectangle((LP[0], y, LP[0] + LP[1], y + ROW_H - 8), fill=NAVY + (205,))
    d.rectangle((LP[0], y, LP[0] + 5, y + ROW_H - 8), fill=COL[j] + (255,))
    d.text((LP[0] + 18, y + 2), NAMES[j], font=TH(36), fill=COL[j] + (255,))
    d.text((LP[0] + 250, y + 12), 'พลังกาย', font=THM(24), fill=MUTED + (255,))
    d.text((LP[0] + 18, y + 54), 'MOVE', font=NS(20), fill=MUTED + (255,))
d.rectangle((RP[0], BOT_Y + 8, RP[0] + RP[1], BOT_Y + BOT_H - 8), fill=NAVY + (205,))
d.rectangle((RP[0], BOT_Y + 8, RP[0] + 5, BOT_Y + BOT_H - 8), fill=CYAN + (255,))
d.text((RP[0] + 20, BOT_Y + 14), 'COURT MAP', font=NS(30), fill=FG + (255,))
d.text((RP[0] + 340, BOT_Y + 20), 'ตำแหน่งผู้เล่น', font=THM(20), fill=MUTED + (255,))
MS = 34.0; MX0 = RP[0] + (RP[1] - 13.4 * MS) / 2; MY0 = BOT_Y + 64
def mp(X, Y): return (MX0 + Y * MS, MY0 + (6.1 - X) * MS)     # horizontal map: our baseline on the left
segs = ([(0, y, 6.1, y) for y in (0, 0.76, 4.72, 8.68, 12.64, 13.4)] + [(x, 0, x, 13.4) for x in (0, 0.46, 5.64, 6.1)] +
        [(3.05, 0, 3.05, 4.72), (3.05, 8.68, 3.05, 13.4)])
for x1, y1, x2, y2 in segs:
    d.line([mp(x1, y1), mp(x2, y2)], fill=(200, 214, 230, 200), width=2)
nx = MX0 + 6.7 * MS
for yy in range(int(MY0 - 8), int(MY0 + 6.1 * MS + 8), 12): d.line([(nx, yy), (nx, yy + 6)], fill=(255, 255, 255, 230), width=2)
d.text((MX0, MY0 + 6.1 * MS + 8), 'ทีมเรา', font=THM(20), fill=MUTED + (255,))
d.text((MX0 + 13.4 * MS - 52, MY0 + 6.1 * MS + 8), 'คู่แข่ง', font=THM(20), fill=MUTED + (255,))
PANEL = panel

def rally_at(t):
    for r in RALLIES:
        if r['start'] <= t <= r['end']: return r, 'live'
    past = [r for r in RALLIES if r['end'] < t]
    return (past[-1], 'done') if past else (None, None)

def ring(Hm, X, Y, r):
    a = np.linspace(0, 2 * np.pi, 40)
    return (project(Hm, np.c_[X + r * np.cos(a), Y + r * np.sin(a)]) * 2).astype(np.int32)

def shuttle_xy(f):
    if f in spos: return spos[f]
    for g in range(1, 3):
        if f - g in spos and f + g in spos:
            a, b = spos[f - g], spos[f + g]; return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return None

cap = cv2.VideoCapture('master1080.mp4')
ff = None if PREV else subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                                          '-i', 'src.mov', '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                                          '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
i0 = 0
if PREV: i0 = int(round(float(PREV) * FPS)); cap.set(1, i0)
t0 = time.time(); i = i0
while i < N:
    ok, fr = cap.read()
    if not ok: break
    t = i / FPS; Hm = COURT[i]
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    # --- rings with glow (on floor plane) ---
    glow = np.zeros_like(rgb)
    on = [j for j in range(4) if PL[j]['court'][i] and PL[j]['court'][i][0] is not None]
    for j in on:
        X, Y = PL[j]['court'][i]
        pts = ring(Hm, X, Y, 0.45)
        far = PL[j]['side'] == 1
        cv2.polylines(glow, [pts], True, COL[j], 6 if far else 14, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 7)
    rgb = cv2.add(rgb, (glow * 0.9).astype(np.uint8))
    for j in on:
        X, Y = PL[j]['court'][i]; far = PL[j]['side'] == 1
        cv2.polylines(rgb, [ring(Hm, X, Y, 0.45)], True, COL[j], 2 if far else 4, cv2.LINE_AA)
    im = Image.fromarray(rgb).convert('RGBA')
    im.alpha_composite(PANEL)
    d = ImageDraw.Draw(im)
    # --- name tags (pushed up when two tags overlap) ---
    placed = []
    for j in sorted(on, key=lambda j: -(PL[j]['box'][i][1] if PL[j]['box'][i] and PL[j]['box'][i][0] is not None else 0)):
        b = PL[j]['box'][i]
        if not b or b[0] is None: continue
        far = PL[j]['side'] == 1
        cx = (b[0] + b[2]); top = 2 * b[1]
        fn = TH(26 if far else 34); fv = NB(26 if far else 34)
        st = PL[j]['stamina'][i]; txt = NAMES[j]; val = f'{st:.0f}'
        w1 = d.textlength(txt, font=fn); w2 = d.textlength(val, font=fv)
        pw = w1 + w2 + 40; ph = (44 if far else 56)
        x = min(max(cx - pw / 2, 8), W - pw - 8); y = max(top - ph - 22, 220)
        for _ in range(4):
            hit = [q for q in placed if x < q[2] + 6 and x + pw > q[0] - 6 and y < q[3] + 6 and y + ph > q[1] - 6]
            if not hit: break
            y = min(q[1] for q in hit) - ph - 8
        placed.append((x, y, x + pw, y + ph))
        d.rectangle((x, y, x + pw, y + ph), fill=NAVY + (215,), outline=EDGE + (255,), width=2)
        d.text((x + 12, y + (2 if far else 2)), txt, font=fn, fill=FG)
        d.text((x + pw - 12 - w2, y + (6 if far else 6)), val, font=fv, fill=FG)
        d.rectangle((x + 10, y + ph - 9, x + 10 + (pw - 20) * st / 100, y + ph - 5), fill=COL[j] + (255,))
        d.polygon([(cx - 9, y + ph + 2), (cx + 9, y + ph + 2), (cx, y + ph + 14)], fill=CYAN + (255,))
    # --- rally state / top bar ---
    r, state = rally_at(t)
    rshots = [s for s in SHOTS if r and s['rally'] == r['n'] and s['t'] <= t]
    if r:
        rt = (min(t, r['end']) - r['start'])
        rtxt = f'{int(rt // 60):02d}:{rt % 60:05.2f}'
        cnt = str(len(rshots))
        last = rshots[-1] if rshots else None
        spd = f"{last['speed_kmh']:.0f}" if last and last['speed_kmh'] is not None else '—'
    else:
        rtxt, cnt, spd, last = '00:00.00', '0', '—', None
    dim = FG if state == 'live' else MUTED
    d.text((TOPC[0][0] + 20, TOP_Y + 42), rtxt, font=NB(80), fill=dim)
    d.text((TOPC[1][0] + 20, TOP_Y + 42), cnt, font=NB(80), fill=dim)
    d.text((TOPC[2][0] + 20, TOP_Y + 42), spd, font=NB(80), fill=YEL if state == 'live' else MUTED)
    # --- LOCK ON ---
    sxy = shuttle_xy(i) if state == 'live' else None
    if sxy and TOP_Y + TOP_H + 24 < sxy[1] < BOT_Y - 24:
        x, y = sxy; s = 30; L = 12
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cx_, cy_ = x + dx * s, y + dy * s
            d.line([(cx_, cy_), (cx_ - dx * L, cy_)], fill=YEL, width=4); d.line([(cx_, cy_), (cx_, cy_ - dy * L)], fill=YEL, width=4)
        d.ellipse((x - 13, y - 13, x + 13, y + 13), outline=(255, 255, 255, 220), width=2)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=YEL)
        LW, LH = 260, 70
        cands = [(x + 52, y - 40 - LH), (x - 52 - LW, y - 40 - LH), (x + 52, y + 40), (x - 52 - LW, y + 40)]
        def bad(c):
            cx0, cy0 = c
            if cx0 < 6 or cx0 + LW > W - 6 or cy0 < TOP_Y + TOP_H + 6 or cy0 + LH > BOT_Y - 6: return 10
            return sum(1 for q in placed if cx0 < q[2] + 6 and cx0 + LW > q[0] - 6 and cy0 < q[3] + 6 and cy0 + LH > q[1] - 6)
        lx, ly = min(cands, key=bad)
        lx = min(max(lx, 6), W - LW - 6); ly = min(max(ly, TOP_Y + TOP_H + 6), BOT_Y - LH - 6)
        ax_ = x + (30 if lx > x else -30); ay_ = y + (-30 if ly < y else 30)
        d.line([(ax_, ay_), (lx if lx > x else lx + LW, ly + (LH if ly < y else 0))], fill=YEL, width=3)
        d.rectangle((lx, ly, lx + 260, ly + 70), fill=NAVY + (220,))
        d.rectangle((lx, ly, lx + 5, ly + 70), fill=YEL)
        d.text((lx + 14, ly + 2), 'LOCK ON', font=NS(24), fill=YEL)
        d.text((lx + 14, ly + 24), f'{spd} km/h', font=NB(40), fill=FG)
        if last:
            nm = last['name']; d.text((lx + 252 - d.textlength(nm, font=TH(26)), ly + 2), nm, font=TH(26), fill=COL[last['player']])
    # --- bottom-left: stamina + distance ---
    for j in range(4):
        y = BOT_Y + 8 + j * ROW_H
        st = PL[j]['stamina'][i]; dist = PL[j]['dist'][i]
        v = f'{st:.0f}'; d.text((LP[0] + LP[1] - 18 - d.textlength(v, font=NB(44)), y - 2), v, font=NB(44), fill=FG)
        d.rectangle((LP[0] + 250, y + 44, LP[0] + LP[1] - 18, y + 48), fill=(50, 70, 96))
        d.rectangle((LP[0] + 250, y + 44, LP[0] + 250 + (LP[1] - 268) * st / 100, y + 48), fill=COL[j])
        n_on = int(min(1.0, dist / 120.0) * 22)
        for k in range(22):
            d.rectangle((LP[0] + 80 + k * 13, y + 58, LP[0] + 89 + k * 13, y + 66), fill=COL[j] if k < n_on else (50, 70, 96))
        dv = f'{dist:.1f} m'; d.text((LP[0] + LP[1] - 18 - d.textlength(dv, font=NM(26)), y + 50), dv, font=NM(26), fill=FG)
    # --- court map ---
    for j in range(4):
        trail = []
        for k in range(max(0, i - 40), i + 1):
            c = PL[j]['court'][k]
            if c and c[0] is not None: trail.append(mp(min(max(c[0], -0.3), 6.4), min(max(c[1], -0.4), 13.8)))
        if len(trail) > 1: d.line(trail, fill=COL[j] + (170,), width=3)
    if last and state == 'live':
        nxt = next((s for s in SHOTS if s['rally'] == last['rally'] and s['t'] > last['t']), None)
        a = PL[last['player']]['court'][last['frame']]
        if a and a[0] is not None:
            pa = mp(a[0], a[1]); d.ellipse((pa[0] - 13, pa[1] - 13, pa[0] + 13, pa[1] + 13), outline=YEL, width=2)
    for j in on:
        c = PL[j]['court'][i]; p = mp(min(max(c[0], -0.3), 6.4), min(max(c[1], -0.4), 13.8))
        d.ellipse((p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9), fill=COL[j], outline=NAVY, width=2)
    frame = np.asarray(im.convert('RGB'))
    if PREV:
        Image.fromarray(frame).save(OUT); print('preview', OUT); sys.exit(0)
    ff.stdin.write(frame.tobytes()); i += 1
    if i % 300 == 0: print(i, f'{(time.time() - t0) / (i - i0):.3f}s/f', flush=True)
ff.stdin.close(); ff.wait(); print('DONE', i, f'{time.time() - t0:.0f}s', flush=True)
