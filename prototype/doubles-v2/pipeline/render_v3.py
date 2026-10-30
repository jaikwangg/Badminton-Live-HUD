"""HUD v3: stat UI of the previous version (title, RALLY / RALLY TIME / SHOTS / LAST SHOT, player cards, vertical court map,
rally timeline) around the vertical clip. In-video: floor rings, name tags (name + m/s), LOCK ON on the shuttle.
Usage: python3 render_v3.py out.mp4      HUD_PREVIEW=12.3 python3 render_v3.py preview.png"""
import os, sys, json, subprocess, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from courtlib import project

OUT = sys.argv[1] if len(sys.argv) > 1 else 'hud_v3.mp4'
PREV = os.environ.get('HUD_PREVIEW')
TITLE = os.environ.get('HUD_TITLE', 'BADMINTON LIVE HUD')
FONTS = os.environ.get('HUD_FONTS', 'fonts')
CFG = json.load(open('config.json')); P = json.load(open('players.json')); E = json.load(open('events.json'))
S = json.load(open('shuttle.json')); COURT = np.load('court.npy')
PL = P['players']; N = P['n']; FPS = 30.0; DUR = N / FPS
NAMES = [p['name'] for p in PL]; SIDE = [p['side'] for p in PL]; COL = [tuple(p['color']) for p in CFG['players']]
SIDE_TH = ['ทีมเรา', 'ทีมเรา', 'คู่แข่ง', 'คู่แข่ง']
RALLIES = E['rallies']; SHOTS = sorted([dict(s, rally=r['n']) for r in RALLIES for s in r['shots']], key=lambda s: s['t'])

spos = {}
for path in S['paths']:
    for f, x, y in path['pts']: spos[int(f)] = (x, y)
moving = {f: any(np.hypot(x - spos[g][0], y - spos[g][1]) > 3 for g in (f - 3, f + 3) if g in spos) for f, (x, y) in spos.items()}
spos = {f: p for f, p in spos.items() if moving[f]}

BG = (10, 15, 22); PANEL = (17, 26, 38); LINE = (44, 60, 84); COURTC = (18, 53, 91)
FG = (238, 243, 248); MUTED = (163, 182, 203); ACCENT = (255, 215, 90)
def F(n, s): return ImageFont.truetype(os.path.join(FONTS, n), s, layout_engine=ImageFont.Layout.RAQM)
TH_SB = lambda s: F('IBMPlexSansThai-SemiBold.ttf', s); TH_M = lambda s: F('IBMPlexSansThai-Medium.ttf', s)
NB = lambda s: F('BarlowCondensed-Bold.ttf', s); NSB = lambda s: F('BarlowCondensed-SemiBold.ttf', s); NM = lambda s: F('BarlowCondensed-Medium.ttf', s)
def rrect(d, box, r, fill=None, outline=None, width=1): d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

# ---------- layout (1080 x 1920) ----------
W, Hh, M = 1080, 1920, 24
HEAD_Y = 44; STAT_Y, STAT_H = 132, 132
VX, VY, VW = M, STAT_Y + STAT_H + 20, W - 2 * M
SRC_Y0, SRC_Y1 = 720, 1680                         # rows of the 1080x1920 master shown (ceiling and bottom edge cut)
SC = VW / 1080.0; VH = round((SRC_Y1 - SRC_Y0) * SC)
def to_view(x540, y540): return (x540 * 2 * SC, (y540 * 2 - SRC_Y0) * SC)
LOW_Y = VY + VH + 22; TL_H = 132; TL_Y = Hh - 36 - TL_H; LOW_H = TL_Y - 18 - LOW_Y
CARD_W = 640; CARD_GAP = 14; CARD_H = (LOW_H - 3 * CARD_GAP) // 4
MAP_X = M + CARD_W + 16; MAP_W = W - M - MAP_X
MS = min((MAP_W - 48) / 6.1, (LOW_H - 112) / 13.4)
MAP_OX = MAP_X + (MAP_W - 6.1 * MS) / 2; MAP_OY = LOW_Y + 56
def m2(X, Y): return (MAP_OX + X * MS, MAP_OY + (13.4 - Y) * MS)
def mpos(q): return m2(min(6.5, max(-0.4, q[0])), min(13.8, max(-0.4, q[1])))
STATS = [('RALLY', 190), ('RALLY TIME', 250), ('SHOTS', 190)]
CELLS = []; sx_ = M
for lab, w in STATS: CELLS.append((lab, sx_, w)); sx_ += w + 12
CELLS.append(('LAST SHOT', sx_, W - M - sx_))

# ---------- static background ----------
bg = Image.new('RGB', (W, Hh), BG); d = ImageDraw.Draw(bg)
d.text((M, HEAD_Y), TITLE, font=NB(64), fill=FG); d.rectangle((M, HEAD_Y + 74, M + 64, HEAD_Y + 78), fill=ACCENT)
tag = 'ประเภทคู่'; f_tag = TH_M(24); w = d.textlength(tag, font=f_tag) + 32
rrect(d, (W - M - w, HEAD_Y + 18, W - M, HEAD_Y + 60), 8, outline=LINE, width=2); d.text((W - M - w + 16, HEAD_Y + 20), tag, font=f_tag, fill=MUTED)
f_lab = NSB(24)
for lab, x, w in CELLS:
    rrect(d, (x, STAT_Y, x + w, STAT_Y + STAT_H), 12, fill=PANEL, outline=LINE); d.text((x + 18, STAT_Y + 12), lab, font=f_lab, fill=MUTED)
lab, x, w = CELLS[3]; ex = x + 18 + d.textlength('LAST SHOT', font=f_lab) + 10
rrect(d, (ex, STAT_Y + 16, ex + 50, STAT_Y + 40), 4, outline=ACCENT, width=2); d.text((ex + 8, STAT_Y + 14), 'EST.', font=NSB(19), fill=ACCENT)
f_name = TH_SB(30); f_side = TH_M(19); f_clab = NSB(17)
for j in range(4):
    y = LOW_Y + j * (CARD_H + CARD_GAP)
    rrect(d, (M, y, M + CARD_W, y + CARD_H), 12, fill=PANEL, outline=LINE)
    d.rectangle((M + 1, y + 10, M + 6, y + CARD_H - 10), fill=COL[j])
    d.text((M + 24, y + 0), NAMES[j], font=f_name, fill=FG)
    d.text((M + 32 + d.textlength(NAMES[j], font=f_name), y + 12), SIDE_TH[j], font=f_side, fill=MUTED)
    colw = (CARD_W - 48) / 4
    for k, l in enumerate(('SPEED', 'DISTANCE', 'SHOTS', 'TOP')):
        d.text((M + 24 + k * colw, y + 54), l, font=f_clab, fill=MUTED)
rrect(d, (MAP_X, LOW_Y, MAP_X + MAP_W, LOW_Y + LOW_H), 12, fill=PANEL, outline=LINE)
d.text((MAP_X + 18, LOW_Y + 12), 'COURT MAP', font=f_lab, fill=MUTED)
f_small = TH_M(20)
d.text((MAP_X + 18, LOW_Y + LOW_H - 36), '↑ คู่แข่ง', font=f_small, fill=MUTED)
d.text((MAP_X + MAP_W - 18 - d.textlength('ทีมเรา ↓', font=f_small), LOW_Y + LOW_H - 36), 'ทีมเรา ↓', font=f_small, fill=MUTED)
a, b = m2(0, 13.4), m2(6.1, 0); d.rectangle((a[0], a[1], b[0], b[1]), fill=COURTC)
rrect(d, (M, TL_Y, W - M, TL_Y + TL_H), 12, fill=PANEL, outline=LINE); d.text((M + 18, TL_Y + 12), 'TIMELINE', font=f_lab, fill=MUTED)
TLX0, TLX1, TLY = M + 24, W - M - 24, TL_Y + 78
d.rounded_rectangle((TLX0, TLY - 5, TLX1, TLY + 5), radius=5, fill=(36, 50, 70))
def tlx(t): return TLX0 + (TLX1 - TLX0) * t / DUR
for rr in RALLIES:
    d.rounded_rectangle((tlx(rr['start']), TLY - 12, tlx(rr['end']), TLY + 12), radius=4, fill=(74, 66, 32), outline=(150, 128, 52), width=2)
    d.text((tlx(rr['start']) + 2, TLY + 16), str(rr['n']), font=NSB(20), fill=MUTED)
for s in SHOTS:
    x = tlx(s['t']); d.rectangle((x - 1, TLY - 22, x + 1, TLY - 14), fill=COL[s['player']])
rrect(d, (VX - 1, VY - 1, VX + VW, VY + VH), 12, outline=LINE, width=2)
BGN = np.array(bg)
SEGS = [(0, Y, 6.1, Y) for Y in (0, 0.76, 4.72, 8.68, 12.64, 13.4)] + [(X, 0, X, 13.4) for X in (0, 0.46, 5.64, 6.1)] + [(3.05, 0, 3.05, 4.72), (3.05, 8.68, 3.05, 13.4)]
for (x1, y1, x2, y2) in SEGS:
    p, q = m2(x1, y1), m2(x2, y2); cv2.line(BGN, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (205, 214, 224), 1, cv2.LINE_AA)
n1, n2 = m2(-0.4, 6.7), m2(6.5, 6.7)
for k in range(0, int(n2[0] - n1[0]), 10):
    cv2.line(BGN, (int(n1[0] + k), int(n1[1])), (int(min(n2[0], n1[0] + k + 6)), int(n1[1])), (255, 255, 255), 2, cv2.LINE_AA)
VM = np.zeros((VH, VW), np.uint8); cv2.rectangle(VM, (12, 0), (VW - 13, VH - 1), 255, -1); cv2.rectangle(VM, (0, 12), (VW - 1, VH - 13), 255, -1)
for c in ((12, 12), (VW - 13, 12), (12, VH - 13), (VW - 13, VH - 13)): cv2.circle(VM, c, 12, 255, -1, cv2.LINE_AA)
VM3 = (VM[..., None] / 255.0).astype(np.float32)

TOP = np.array([np.maximum.accumulate(np.array(p['speed'])) for p in PL])
RING = np.linspace(0, 2 * np.pi, 40)
def ring_pts(Hm, X, Y, r):
    q = project(Hm, np.c_[X + r * np.cos(RING), Y + r * np.sin(RING)])
    return np.array([to_view(x, y) for x, y in q], np.int32)
def rally_at(t): return next((r for r in RALLIES if r['start'] <= t <= r['end'] + 0.6), None)
def ended(t): return next((r for r in RALLIES if r['end'] < t <= r['end'] + 2.2), None)
def shuttle_xy(f):
    if f in spos: return spos[f]
    for g in range(1, 3):
        if f - g in spos and f + g in spos:
            a, b = spos[f - g], spos[f + g]; return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return None
f_big = NSB(78); f_unit = NM(30); f_last = TH_SB(34); f_cnum = NSB(38); f_cunit = NM(19)
f_tn = TH_SB(26); f_tnum = NM(26); f_map = TH_M(19); f_b1 = NSB(38); f_b2 = TH_SB(34)

cap = cv2.VideoCapture('master1080.mp4')
ff = None if PREV else subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{Hh}', '-r', '30', '-i', '-',
    '-i', 'src.mov', '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
i = 0
if PREV: i = int(round(float(PREV) * FPS)); cap.set(1, i)
t0 = time.time(); i0 = i
while i < N:
    ok, src = cap.read()
    if not ok: break
    t = i / FPS; Hm = COURT[i]; fr = BGN.copy()
    vid = cv2.cvtColor(cv2.resize(src[SRC_Y0:SRC_Y1], (VW, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    reg = fr[VY:VY + VH, VX:VX + VW].astype(np.float32)
    fr[VY:VY + VH, VX:VX + VW] = (vid * VM3 + reg * (1 - VM3)).astype(np.uint8)
    view = fr[VY:VY + VH, VX:VX + VW]
    on = [j for j in range(4) if PL[j]['court'][i] and PL[j]['court'][i][0] is not None]
    recent = {s['player']: (t - s['t']) / 0.5 for s in SHOTS if 0 <= t - s['t'] < 0.5}
    glow = np.zeros_like(view)
    for j in on:
        X, Y = PL[j]['court'][i]
        cv2.polylines(glow, [ring_pts(Hm, X, Y, 0.45)], True, COL[j], 6 if SIDE[j] else 14, cv2.LINE_AA)
    view[:] = cv2.add(view, (cv2.GaussianBlur(glow, (0, 0), 7) * 0.9).astype(np.uint8))
    for j in on:
        X, Y = PL[j]['court'][i]
        cv2.polylines(view, [ring_pts(Hm, X, Y, 0.45)], True, COL[j], 2 if SIDE[j] else 4, cv2.LINE_AA)
        if j in recent: cv2.polylines(view, [ring_pts(Hm, X, Y, 0.45 + 0.9 * recent[j])], True, ACCENT, 2, cv2.LINE_AA)
    for j in range(4):                                     # map trails + dots
        seg = []
        for k in range(max(0, i - 45), i + 1):
            q = PL[j]['court'][k]
            if q and q[0] is not None: p = mpos(q); seg.append((int(p[0]), int(p[1])))
            else:
                if len(seg) > 1: cv2.polylines(fr, [np.array(seg, np.int32)], False, tuple(int(c * .6) for c in COL[j]), 2, cv2.LINE_AA)
                seg = []
        if len(seg) > 1: cv2.polylines(fr, [np.array(seg, np.int32)], False, tuple(int(c * .6) for c in COL[j]), 2, cv2.LINE_AA)
    for j in on:
        p = mpos(PL[j]['court'][i]); c = (int(p[0]), int(p[1]))
        cv2.circle(fr, c, 12, COURTC, -1, cv2.LINE_AA); cv2.circle(fr, c, 10, COL[j], -1, cv2.LINE_AA)
        if j in recent: cv2.circle(fr, c, int(14 + 16 * recent[j]), ACCENT, 2, cv2.LINE_AA)
    for j in range(4):                                     # card MOVE bars
        y = LOW_Y + j * (CARD_H + CARD_GAP) + 44
        cv2.rectangle(fr, (M + 24, y), (M + CARD_W - 24, y + 6), (36, 50, 70), -1)
        sp = PL[j]['speed'][i] if j in on else 0
        cv2.rectangle(fr, (M + 24, y), (M + 24 + int((CARD_W - 48) * min(1, sp / 5)), y + 6), COL[j], -1)
    x = tlx(t); cv2.rectangle(fr, (int(x) - 2, TLY - 26), (int(x) + 2, TLY + 26), FG, -1)

    im = Image.fromarray(fr); d = ImageDraw.Draw(im)
    rr = rally_at(t)
    if rr:
        sh = [s for s in SHOTS if s['rally'] == rr['n'] and s['t'] <= t]
        vals = [str(rr['n']), f"{max(0, min(t, rr['end']) - rr['start']):.1f}", str(len(sh))]
        last = next((s for s in reversed(sh) if s.get('speed_kmh')), None)
        lt = f"{last['name']} · {int(last['speed_kmh'])} km/h" if last else (sh[-1]['name'] if sh else '—')
        lastshot = sh[-1] if sh else None
    else:
        vals = ['—', '—', '—']; lt = 'ระหว่าง rally'; lastshot = None
    # name tags (name + m/s), pushed up when they overlap
    placed = []
    for j in sorted(on, key=lambda j: PL[j]['court'][i][1]):
        b = PL[j]['box'][i]
        if not b or b[0] is None: continue
        nm = NAMES[j]; sp = f"{PL[j]['speed'][i]:.1f} m/s"
        w1 = d.textlength(nm, font=f_tn); w2 = d.textlength(sp, font=f_tnum); pw, ph = w1 + w2 + 30, 42
        cx, top = to_view((b[0] + b[2]) / 2, b[1])
        x = VX + cx - pw / 2; y = VY + top - ph - 12
        x = max(VX + 6, min(VX + VW - pw - 6, x)); y = max(VY + 6, y)
        for _ in range(4):
            hit = [q for q in placed if x < q[2] + 4 and x + pw > q[0] - 4 and y < q[3] + 4 and y + ph > q[1] - 4]
            if not hit: break
            y = min(q[1] for q in hit) - ph - 6
        y = max(VY + 6, y); placed.append((x, y, x + pw, y + ph))
        rrect(d, (x, y, x + pw, y + ph), 7, fill=(8, 14, 22)); d.rectangle((x + 1, y, x + pw - 1, y + 3), fill=COL[j])
        d.text((x + 10, y + 1), nm, font=f_tn, fill=FG); d.text((x + 20 + w1, y + 6), sp, font=f_tnum, fill=MUTED)
        if j in recent: d.text((x, y - 32), 'SHOT', font=NB(28), fill=ACCENT)
    # LOCK ON
    sxy = shuttle_xy(i) if rr and t <= rr['end'] else None
    if sxy:
        vx, vy = to_view(*sxy); x, y = VX + vx, VY + vy
        if VY + 30 < y < VY + VH - 30:
            s_, L = 28, 11
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                cx_, cy_ = x + dx * s_, y + dy * s_
                d.line([(cx_, cy_), (cx_ - dx * L, cy_)], fill=ACCENT, width=4); d.line([(cx_, cy_), (cx_, cy_ - dy * L)], fill=ACCENT, width=4)
            d.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(255, 255, 255), width=2); d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=ACCENT)
            LW, LH = 250, 66
            cands = [(x + 48, y - 40 - LH), (x - 48 - LW, y - 40 - LH), (x + 48, y + 40), (x - 48 - LW, y + 40),
                     (x + 60, y - LH / 2), (x - 60 - LW, y - LH / 2), (x + 48, y - 120 - LH), (x - 48 - LW, y - 120 - LH), (x + 48, y + 120), (x - 48 - LW, y + 120)]
            def bad(c):
                if c[0] < VX + 6 or c[0] + LW > VX + VW - 6 or c[1] < VY + 6 or c[1] + LH > VY + VH - 6: return 10
                return sum(1 for q in placed if c[0] < q[2] + 6 and c[0] + LW > q[0] - 6 and c[1] < q[3] + 6 and c[1] + LH > q[1] - 6)
            lx, ly = min(cands, key=bad); lx = min(max(lx, VX + 6), VX + VW - LW - 6); ly = min(max(ly, VY + 6), VY + VH - LH - 6)
            d.line([(x + (28 if lx > x else -28), y + (-28 if ly < y else 28)), (lx if lx > x else lx + LW, ly + (LH if ly < y else 0))], fill=ACCENT, width=3)
            d.rectangle((lx, ly, lx + LW, ly + LH), fill=(8, 14, 22)); d.rectangle((lx, ly, lx + 5, ly + LH), fill=ACCENT)
            d.text((lx + 14, ly + 2), 'LOCK ON', font=NSB(22), fill=ACCENT)
            spd = f"{int(lastshot['speed_kmh'])} km/h" if lastshot and lastshot.get('speed_kmh') else '— km/h'
            d.text((lx + 14, ly + 22), spd, font=NB(38), fill=FG)
            if lastshot:
                nm = lastshot['name']; d.text((lx + LW - 10 - d.textlength(nm, font=TH_SB(24)), ly + 2), nm, font=TH_SB(24), fill=COL[lastshot['player']])
    for j in on:
        p = mpos(PL[j]['court'][i]); d.text((p[0] + 15, p[1] - 14), NAMES[j], font=f_map, fill=FG, stroke_width=2, stroke_fill=COURTC)
    mm, ss = divmod(t, 60); tt = f'{int(mm)}:{ss:04.1f}'
    d.text((MAP_X + MAP_W - 18 - d.textlength(tt, font=f_lab), LOW_Y + 12), tt, font=f_lab, fill=MUTED)
    tot = f'{tt} / {int(DUR // 60)}:{DUR % 60:04.1f}'; d.text((W - M - 18 - d.textlength(tot, font=f_lab), TL_Y + 12), tot, font=f_lab, fill=MUTED)
    for k, v in enumerate(vals):
        lab, x, w = CELLS[k]; d.text((x + 18, STAT_Y + 38), v, font=f_big, fill=FG)
        if k == 1 and v != '—': d.text((x + 26 + d.textlength(v, font=f_big), STAT_Y + 78), 's', font=f_unit, fill=MUTED)
    lab, x, w = CELLS[3]; fl = f_last if d.textlength(lt, font=f_last) < w - 30 else TH_SB(28)
    d.text((x + 18, STAT_Y + 62), lt, font=fl, fill=ACCENT if (rr and recent) else FG)
    for j in range(4):
        y = LOW_Y + j * (CARD_H + CARD_GAP); colw = (CARD_W - 48) / 4
        ns = sum(1 for s in SHOTS if s['player'] == j and s['t'] <= t)
        items = [(f"{PL[j]['speed'][i]:.1f}" if j in on else '—', 'm/s'), (f"{PL[j]['dist'][i]:.1f}", 'm'), (str(ns), ''), (f'{TOP[j][i]:.1f}', 'm/s')]
        for k, (v, u) in enumerate(items):
            xx = M + 24 + k * colw; d.text((xx, y + 70), v, font=f_cnum, fill=FG)
            if u: d.text((xx + d.textlength(v, font=f_cnum) + 4, y + 86), u, font=f_cunit, fill=MUTED)
    re_ = ended(t)
    if re_:
        a1 = f"RALLY {re_['n']}"; a2 = f" จบ · {re_['end'] - re_['start']:.1f} วิ · {len(re_['shots'])} shots"
        w = d.textlength(a1, font=f_b1) + d.textlength(a2, font=f_b2) + 48; x = VX + VW / 2 - w / 2; y = VY + VH - 84
        rrect(d, (x, y, x + w, y + 62), 12, fill=(8, 14, 22), outline=ACCENT, width=2)
        d.text((x + 24, y + 9), a1, font=f_b1, fill=ACCENT); d.text((x + 24 + d.textlength(a1, font=f_b1), y + 5), a2, font=f_b2, fill=FG)
    if PREV:
        im.save(OUT); print('preview', OUT); sys.exit(0)
    ff.stdin.write(np.asarray(im).tobytes()); i += 1
    if i % 300 == 0: print(i, f'{(time.time() - t0) / (i - i0):.3f}s/f', flush=True)
ff.stdin.close(); ff.wait(); print('DONE', i, f'{time.time() - t0:.0f}s', OUT, flush=True)
