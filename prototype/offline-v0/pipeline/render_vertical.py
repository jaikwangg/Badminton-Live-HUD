"""Render a vertical 9:16 (1080x1920) MP4: cropped video + HUD panels from players.json / events.json.
Usage: python3 render_vertical.py out.mp4      |  HUD_PREVIEW=20.4 python3 render_vertical.py preview.png
"""
import os, sys, json, subprocess, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
FONTS = os.environ.get("HUD_FONTS", "fonts")
OUT = sys.argv[1] if len(sys.argv) > 1 else "hud_vertical.mp4"
PREV = os.environ.get("HUD_PREVIEW")
TITLE = os.environ.get("HUD_TITLE", "BADMINTON LIVE HUD")

P = json.load(open(os.path.join(DATA, "players.json")))
E = json.load(open(os.path.join(DATA, "events.json")))
H = np.array(json.load(open(os.path.join(DATA, "calib.json")))["H"]); Hinv = np.linalg.inv(H)
T = np.array(P["t"]); N = len(T); FPS = P["fps"]; DUR = float(T[-1])
PL = P["players"]; SHOTS = E["shots"]; RALLIES = E["rallies"]
NAMES = [p["name"] for p in PL]
SIDE = ["ทีมเรา", "ทีมเรา", "คู่แข่ง", "คู่แข่ง"]

BG = (10, 15, 22); PANEL = (17, 26, 38); LINE = (44, 60, 84); COURT = (18, 53, 91)
FG = (238, 243, 248); MUTED = (163, 182, 203); ACCENT = (255, 215, 90)
COL = [(63, 208, 240), (75, 227, 160), (195, 155, 255), (255, 138, 107)]

def F(name, size): return ImageFont.truetype(os.path.join(FONTS, name), size, layout_engine=ImageFont.Layout.RAQM)
TH_SB = lambda s: F("IBMPlexSansThai-SemiBold.ttf", s)
TH_M = lambda s: F("IBMPlexSansThai-Medium.ttf", s)
NB = lambda s: F("BarlowCondensed-Bold.ttf", s)
NSB = lambda s: F("BarlowCondensed-SemiBold.ttf", s)
NM = lambda s: F("BarlowCondensed-Medium.ttf", s)

# ---------- layout (1080 x 1920) ----------
W, Hh, M = 1080, 1920, 24
CW_ = W - 2 * M
CX0, CX1 = 200, 1720                      # horizontal crop of the 1920x1078 source
VW = CW_; SX = VW / (CX1 - CX0); VH = round(1078 * SX); SY = VH / 1078
HEAD_Y = 44
STAT_Y, STAT_H = 132, 132
VX, VY = M, STAT_Y + STAT_H + 20
LOW_Y = VY + VH + 22
TL_H = 132
LOW_H = Hh - 36 - TL_H - 18 - LOW_Y
CARD_W = 640; CARD_GAP = 14; CARD_H = (LOW_H - 3 * CARD_GAP) // 4
MAP_X = M + CARD_W + 16; MAP_W = W - M - MAP_X
TL_Y = LOW_Y + LOW_H + 18
MS = min((MAP_W - 48) / 6.1, (LOW_H - 112) / 13.4)
MAP_OX = MAP_X + (MAP_W - 6.1 * MS) / 2; MAP_OY = LOW_Y + 56
def m2(X, Y): return (MAP_OX + X * MS, MAP_OY + (13.4 - Y) * MS)
STATS = [("RALLY", 190), ("RALLY TIME", 250), ("SHOTS", 190)]
sx_ = M; CELLS = []
for lab, w in STATS:
    CELLS.append((lab, sx_, w)); sx_ += w + 12
CELLS.append(("LAST SHOT", sx_, W - M - sx_))

def rrect(d, box, r, fill=None, outline=None, width=1): d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

# ---------- static background ----------
bg = Image.new("RGB", (W, Hh), BG); d = ImageDraw.Draw(bg)
f_title = NB(64)
d.text((M, HEAD_Y), TITLE, font=f_title, fill=FG)
tw = d.textlength(TITLE, font=f_title)
d.rectangle((M, HEAD_Y + 74, M + 64, HEAD_Y + 78), fill=ACCENT)
tag = "ประเภทคู่"; f_tag = TH_M(24); w = d.textlength(tag, font=f_tag) + 32
rrect(d, (W - M - w, HEAD_Y + 18, W - M, HEAD_Y + 60), 8, outline=LINE, width=2); d.text((W - M - w + 16, HEAD_Y + 20), tag, font=f_tag, fill=MUTED)
f_lab = NSB(24)
for lab, x, w in CELLS:
    rrect(d, (x, STAT_Y, x + w, STAT_Y + STAT_H), 12, fill=PANEL, outline=LINE)
    d.text((x + 18, STAT_Y + 12), lab, font=f_lab, fill=MUTED)
lab, x, w = CELLS[3]; ex = x + 18 + d.textlength("LAST SHOT", font=f_lab) + 10
rrect(d, (ex, STAT_Y + 16, ex + 50, STAT_Y + 40), 4, outline=ACCENT, width=2); d.text((ex + 8, STAT_Y + 14), "EST.", font=NSB(19), fill=ACCENT)
f_name = TH_SB(36); f_side = TH_M(20); f_clab = NSB(20)
for j in range(4):
    y = LOW_Y + j * (CARD_H + CARD_GAP)
    rrect(d, (M, y, M + CARD_W, y + CARD_H), 12, fill=PANEL, outline=LINE)
    d.rectangle((M + 1, y + 10, M + 6, y + CARD_H - 10), fill=COL[j])
    d.text((M + 24, y + 8), NAMES[j], font=f_name, fill=FG)
    d.text((M + 32 + d.textlength(NAMES[j], font=f_name), y + 24), SIDE[j], font=f_side, fill=MUTED)
    colw = (CARD_W - 48) / 4
    for k, l in enumerate(("SPEED", "DISTANCE", "SHOTS", "TOP")):
        d.text((M + 24 + k * colw, y + CARD_H - 92), l, font=f_clab, fill=MUTED)
rrect(d, (MAP_X, LOW_Y, MAP_X + MAP_W, LOW_Y + LOW_H), 12, fill=PANEL, outline=LINE)
d.text((MAP_X + 18, LOW_Y + 12), "COURT MAP", font=f_lab, fill=MUTED)
f_small = TH_M(20)
d.text((MAP_X + 18, LOW_Y + LOW_H - 38), "↑ คู่แข่ง", font=f_small, fill=MUTED)
d.text((MAP_X + MAP_W - 18 - d.textlength("ทีมเรา ↓", font=f_small), LOW_Y + LOW_H - 38), "ทีมเรา ↓", font=f_small, fill=MUTED)
a, b = m2(0, 13.4), m2(6.1, 0); d.rectangle((a[0], a[1], b[0], b[1]), fill=COURT)
rrect(d, (M, TL_Y, W - M, TL_Y + TL_H), 12, fill=PANEL, outline=LINE)
d.text((M + 18, TL_Y + 12), "TIMELINE", font=f_lab, fill=MUTED)
TLX0, TLX1, TLY = M + 24, W - M - 24, TL_Y + 78
d.rounded_rectangle((TLX0, TLY - 5, TLX1, TLY + 5), radius=5, fill=(36, 50, 70))
def tlx(t): return TLX0 + (TLX1 - TLX0) * t / DUR
for rr in RALLIES:
    d.rounded_rectangle((tlx(rr["start"]), TLY - 12, tlx(rr["end"]), TLY + 12), radius=4, fill=(74, 66, 32), outline=(150, 128, 52), width=2)
    d.text((tlx(rr["start"]) + 4, TLY + 16), str(rr["n"]), font=NSB(20), fill=MUTED)
for s in SHOTS:
    x = tlx(s["t"]); d.rectangle((x - 1, TLY - 22, x + 1, TLY - 14), fill=COL[s["p"]])
rrect(d, (VX - 1, VY - 1, VX + VW, VY + VH), 12, outline=LINE, width=2)
BGN = np.array(bg)
COURT_LINES = [(0, Y, 6.1, Y) for Y in (0, 0.76, 4.72, 8.68, 12.64, 13.4)] + [(X, 0, X, 13.4) for X in (0, 0.46, 5.64, 6.1)] + [(3.05, 0, 3.05, 4.72), (3.05, 8.68, 3.05, 13.4)]
for (x1, y1, x2, y2) in COURT_LINES:
    p, q = m2(x1, y1), m2(x2, y2); cv2.line(BGN, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (205, 214, 224), 1, cv2.LINE_AA)
n1, n2 = m2(-0.4, 6.7), m2(6.5, 6.7)
for k in range(0, int(n2[0] - n1[0]), 10):
    cv2.line(BGN, (int(n1[0] + k), int(n1[1])), (int(min(n2[0], n1[0] + k + 6)), int(n1[1])), (255, 255, 255), 2, cv2.LINE_AA)
VM = np.zeros((VH, VW), np.uint8); cv2.rectangle(VM, (12, 0), (VW - 13, VH - 1), 255, -1); cv2.rectangle(VM, (0, 12), (VW - 1, VH - 13), 255, -1)
for c in ((12, 12), (VW - 13, 12), (12, VH - 13), (VW - 13, VH - 13)): cv2.circle(VM, c, 12, 255, -1, cv2.LINE_AA)
VM3 = (VM[..., None] / 255.0).astype(np.float32)

TOP = np.array([np.maximum.accumulate(np.array(p["speed"])) for p in PL])
def proj(X, Y):
    p = Hinv @ np.array([X, Y, 1.0]); return ((p[0] / p[2]) - CX0) * SX, (p[1] / p[2]) * SY
RING = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 33)]
def ring_pts(X, Y, r): return np.array([proj(X + r * c, Y + r * s) for c, s in RING], np.int32)
def rally_at(t): return next((r for r in RALLIES if r["start"] <= t <= r["end"] + 0.6), None)
def ended(t): return next((r for r in RALLIES if r["end"] < t <= r["end"] + 2.2), None)
def mpos(q): return m2(min(6.5, max(-0.4, q[0])), min(13.8, max(-0.4, q[1])))
f_big = NSB(78); f_unit = NM(30); f_last = TH_SB(34); f_cnum = NSB(48); f_cunit = NM(22)
f_tn = TH_SB(24); f_tnum = NM(24); f_map = TH_M(19); f_b1 = NSB(38); f_b2 = TH_SB(34)

ff = None if PREV else subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", f"{FPS:.4f}", "-i", "-",
    "-i", SRC, "-map", "0:v", "-map", "1:2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
cap = cv2.VideoCapture(SRC); fi = -1; i = 0; t0 = time.time()
if PREV:
    i = int(np.argmin(np.abs(T - float(PREV)))); cap.set(cv2.CAP_PROP_POS_FRAMES, 2 * i); fi = 2 * i - 1
while i < N:
    if not cap.grab(): break
    fi += 1
    if fi % 2: continue
    ok, src = cap.retrieve()
    if not ok: break
    t = T[i]; fr = BGN.copy()
    vid = cv2.cvtColor(cv2.resize(src[:, CX0:CX1], (VW, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    reg = fr[VY:VY + VH, VX:VX + VW].astype(np.float32)
    fr[VY:VY + VH, VX:VX + VW] = (vid * VM3 + reg * (1 - VM3)).astype(np.uint8)
    view = fr[VY:VY + VH, VX:VX + VW]
    on = [j for j in range(4) if PL[j]["court"][i] and PL[j]["court"][i][0] is not None]
    order = sorted(on, key=lambda j: -PL[j]["court"][i][1])
    recent = {s["p"]: (t - s["t"]) / 0.5 for s in SHOTS if 0 <= t - s["t"] < 0.5}
    for j in order:
        X, Y = PL[j]["court"][i]; big = Y < 6.7
        cv2.polylines(view, [ring_pts(X, Y, 0.44)], True, tuple(int(c * .35) for c in COL[j]), 8 if big else 4, cv2.LINE_AA)
        cv2.polylines(view, [ring_pts(X, Y, 0.44)], True, COL[j], 3 if big else 2, cv2.LINE_AA)
        if j in recent:
            cv2.polylines(view, [ring_pts(X, Y, 0.44 + 0.9 * recent[j])], True, ACCENT, 2, cv2.LINE_AA)
    for j in range(4):
        seg = []
        for k in range(max(0, i - 45), i + 1):
            q = PL[j]["court"][k]
            if q and q[0] is not None: p = mpos(q); seg.append((int(p[0]), int(p[1])))
            else:
                if len(seg) > 1: cv2.polylines(fr, [np.array(seg, np.int32)], False, tuple(int(c * .6) for c in COL[j]), 2, cv2.LINE_AA)
                seg = []
        if len(seg) > 1: cv2.polylines(fr, [np.array(seg, np.int32)], False, tuple(int(c * .6) for c in COL[j]), 2, cv2.LINE_AA)
    for j in on:
        p = mpos(PL[j]["court"][i]); c = (int(p[0]), int(p[1]))
        cv2.circle(fr, c, 12, COURT, -1, cv2.LINE_AA); cv2.circle(fr, c, 10, COL[j], -1, cv2.LINE_AA)
        if j in recent: cv2.circle(fr, c, int(14 + 16 * recent[j]), ACCENT, 2, cv2.LINE_AA)
    for j in range(4):
        y = LOW_Y + j * (CARD_H + CARD_GAP) + 64
        cv2.rectangle(fr, (M + 24, y), (M + CARD_W - 24, y + 7), (36, 50, 70), -1)
        sp = PL[j]["speed"][i] if j in on else 0
        cv2.rectangle(fr, (M + 24, y), (M + 24 + int((CARD_W - 48) * min(1, sp / 5)), y + 7), COL[j], -1)
    x = tlx(t); cv2.rectangle(fr, (int(x) - 2, TLY - 26), (int(x) + 2, TLY + 26), FG, -1)

    im = Image.fromarray(fr); d = ImageDraw.Draw(im)
    placed = []
    for j in sorted(on, key=lambda j: PL[j]["court"][i][1]):
        b = PL[j]["box"][i]
        if not b or b[0] is None: continue
        nm = NAMES[j]; sp = f"{PL[j]['speed'][i]:.1f} m/s"
        w1 = d.textlength(nm, font=f_tn); w2 = d.textlength(sp, font=f_tnum); pw, ph = w1 + w2 + 30, 38
        x = VX + ((b[0] + b[2]) / 2 - CX0) * SX - pw / 2; y = VY + b[1] * SY - ph - 10
        x = max(VX + 6, min(VX + VW - pw - 6, x)); y = max(VY + 6, y)
        for _ in range(4):
            hit = [q for q in placed if x < q[2] + 4 and x + pw > q[0] - 4 and y < q[3] + 4 and y + ph > q[1] - 4]
            if not hit: break
            y = min(q[1] for q in hit) - ph - 6
        y = max(VY + 6, y); placed.append((x, y, x + pw, y + ph))
        rrect(d, (x, y, x + pw, y + ph), 7, fill=(8, 14, 22)); d.rectangle((x + 1, y, x + pw - 1, y + 3), fill=COL[j])
        d.text((x + 10, y + 1), nm, font=f_tn, fill=FG); d.text((x + 20 + w1, y + 5), sp, font=f_tnum, fill=MUTED)
        if j in recent: d.text((x, y - 32), "SHOT", font=NB(28), fill=ACCENT)
    for j in on:
        p = mpos(PL[j]["court"][i]); d.text((p[0] + 15, p[1] - 14), NAMES[j], font=f_map, fill=FG, stroke_width=2, stroke_fill=COURT)
    mm, ss = divmod(t, 60); tt = f"{int(mm)}:{ss:04.1f}"
    d.text((MAP_X + MAP_W - 18 - d.textlength(tt, font=f_lab), LOW_Y + 12), tt, font=f_lab, fill=MUTED)
    d.text((W - M - 18 - d.textlength(tt + f" / {int(DUR // 60)}:{DUR % 60:04.1f}", font=f_lab), TL_Y + 12), tt + f" / {int(DUR // 60)}:{DUR % 60:04.1f}", font=f_lab, fill=MUTED)
    rr = rally_at(t)
    if rr:
        sh = [s for s in SHOTS if s["rally"] == rr["n"] and s["t"] <= t]
        vals = [str(rr["n"]), f"{max(0, min(t, rr['end']) - rr['start']):.1f}", str(len(sh))]
        last = next((s for s in reversed(sh) if s.get("avg_kmh_est")), None)
        lt = f"{NAMES[last['p']]} · {int(last['avg_kmh_est'])} km/h" if last else (NAMES[sh[-1]["p"]] if sh else "—")
    else:
        vals = ["—", "—", "—"]; lt = "ระหว่าง rally"
    for k, v in enumerate(vals):
        lab, x, w = CELLS[k]; d.text((x + 18, STAT_Y + 38), v, font=f_big, fill=FG)
        if k == 1 and v != "—": d.text((x + 26 + d.textlength(v, font=f_big), STAT_Y + 78), "s", font=f_unit, fill=MUTED)
    lab, x, w = CELLS[3]; fl = f_last if d.textlength(lt, font=f_last) < w - 30 else TH_SB(28)
    d.text((x + 18, STAT_Y + 62), lt, font=fl, fill=ACCENT if (rr and recent) else FG)
    for j in range(4):
        y = LOW_Y + j * (CARD_H + CARD_GAP); colw = (CARD_W - 48) / 4
        ns = sum(1 for s in SHOTS if s["p"] == j and s["t"] <= t)
        items = [(f"{PL[j]['speed'][i]:.1f}" if j in on else "—", "m/s"), (f"{PL[j]['dist'][i]:.1f}", "m"), (str(ns), ""), (f"{TOP[j][i]:.1f}", "m/s")]
        for k, (v, u) in enumerate(items):
            xx = M + 24 + k * colw; d.text((xx, y + CARD_H - 66), v, font=f_cnum, fill=FG)
            if u: d.text((xx + d.textlength(v, font=f_cnum) + 4, y + CARD_H - 46), u, font=f_cunit, fill=MUTED)
    re_ = ended(t)
    if re_:
        a1 = f"RALLY {re_['n']}"; a2 = f" จบ · {re_['end'] - re_['start']:.1f} วิ · {re_['shots']} shots"
        w = d.textlength(a1, font=f_b1) + d.textlength(a2, font=f_b2) + 48; x = VX + VW / 2 - w / 2; y = VY + VH - 84
        rrect(d, (x, y, x + w, y + 62), 12, fill=(8, 14, 22), outline=ACCENT, width=2)
        d.text((x + 24, y + 9), a1, font=f_b1, fill=ACCENT); d.text((x + 24 + d.textlength(a1, font=f_b1), y + 5), a2, font=f_b2, fill=FG)
    if PREV:
        im.save(OUT); print("preview", OUT); sys.exit(0)
    ff.stdin.write(np.asarray(im).tobytes()); i += 1
    if i % 300 == 0: print(i, f"{(time.time() - t0) / i * 1000:.0f} ms/frame", flush=True)
ff.stdin.close(); ff.wait(); print("DONE", i, f"{time.time() - t0:.0f}s", OUT, flush=True)
