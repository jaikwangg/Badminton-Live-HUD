"""Render a 16:9 (1920x1080) broadcast-style MP4: video + HUD panels from players.json / events.json.
Usage: python3 render_mp4.py out.mp4 [max_frames]
"""
import os, sys, json, subprocess, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
FONTS = os.environ.get("HUD_FONTS", "fonts")
OUT = sys.argv[1] if len(sys.argv) > 1 else "hud_1080p.mp4"
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

P = json.load(open(os.path.join(DATA, "players.json")))
E = json.load(open(os.path.join(DATA, "events.json")))
H = np.array(json.load(open(os.path.join(DATA, "calib.json")))["H"]); Hinv = np.linalg.inv(H)
T = np.array(P["t"]); N = len(T); FPS = P["fps"]
PL = P["players"]; SHOTS = E["shots"]; RALLIES = E["rallies"]
NAMES = [p["name"] for p in PL]
DESC = ["ทีมเรา · เสื้อกรมท่า", "ทีมเรา · เสื้อน้ำเงิน #11", "คู่แข่ง · เสื้อดำ", "คู่แข่ง · เสื้อขาว"]

# ---------- palette (RGB) ----------
BG = (10, 15, 22); PANEL = (17, 26, 38); PANEL2 = (13, 21, 32); LINE = (44, 60, 84)
FG = (238, 243, 248); MUTED = (163, 182, 203); ACCENT = (255, 215, 90); LIVE = (255, 93, 108)
COL = [(63, 208, 240), (75, 227, 160), (195, 155, 255), (255, 138, 107)]
def bgr(c): return (c[2], c[1], c[0])

def F(name, size): return ImageFont.truetype(os.path.join(FONTS, name), size, layout_engine=ImageFont.Layout.RAQM)
TH_SB = lambda s: F("IBMPlexSansThai-SemiBold.ttf", s)
TH_M = lambda s: F("IBMPlexSansThai-Medium.ttf", s)
NUM_B = lambda s: F("BarlowCondensed-Bold.ttf", s)
NUM_SB = lambda s: F("BarlowCondensed-SemiBold.ttf", s)
NUM_M = lambda s: F("BarlowCondensed-Medium.ttf", s)
f_title = TH_SB(40); f_vs = NUM_B(30); f_tag = TH_M(20); f_lab = NUM_SB(20); f_big = NUM_SB(64); f_unit = NUM_M(26)
f_last = TH_SB(30); f_name = TH_SB(30); f_desc = TH_M(18); f_cnum = NUM_SB(38); f_cunit = NUM_M(20)
f_tagname = TH_SB(22); f_tagnum = NUM_M(22); f_map = TH_M(17); f_banner = TH_SB(34); f_bnum = NUM_SB(34); f_small = TH_M(16)

# ---------- layout ----------
W, Hh = 1920, 1080
VX, VY, VW = 24, 104, 1360
VH = round(VW * 1078 / 1920)            # 764
SX, SY = VW / 1920, VH / 1078
RX, RW = 1408, 488                       # right column
STAT_Y, STAT_H = VY, 112
MAP_Y = STAT_Y + 2 * STAT_H + 16 + 16
MAP_H = VY + VH - MAP_Y
CARD_Y = VY + VH + 18; CARD_H = Hh - CARD_Y - 20
CARD_W = (W - 48 - 3 * 16) // 4
# court map geometry (vertical, opponents on top)
MS = 33.0
MAP_CW, MAP_CH = 6.1 * MS, 13.4 * MS
MAP_OX = RX + (RW - MAP_CW) / 2; MAP_OY = MAP_Y + 52
def m2(X, Y): return (MAP_OX + X * MS, MAP_OY + (13.4 - Y) * MS)

def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

COURT_LINES = []
for Y in (0, 0.76, 4.72, 8.68, 12.64, 13.4): COURT_LINES.append((0, Y, 6.1, Y))
for X in (0, 0.46, 5.64, 6.1): COURT_LINES.append((X, 0, X, 13.4))
COURT_LINES += [(3.05, 0, 3.05, 4.72), (3.05, 8.68, 3.05, 13.4)]

# ---------- static background ----------
bg = Image.new("RGB", (W, Hh), BG); d = ImageDraw.Draw(bg)
x = 24
for j in (0, 1):
    d.ellipse((x, 44, x + 16, 60), fill=COL[j]); x += 22
d.text((x + 6, 26), "ปั้น · พฤกษ", font=f_title, fill=FG); x += 18 + d.textlength("ปั้น · พฤกษ", font=f_title)
d.text((x + 10, 36), "VS", font=f_vs, fill=MUTED); x += 10 + d.textlength("VS", font=f_vs) + 24
for j in (2, 3):
    d.ellipse((x, 44, x + 16, 60), fill=COL[j]); x += 22
d.text((x + 6, 26), "ปลื้ม · ภูสิท", font=f_title, fill=FG)
tx = W - 24
for txt, col in (("PROTOTYPE · ประมวลผลจากคลิปจริง", ACCENT), ("ประเภทคู่", MUTED)):
    w = d.textlength(txt, font=f_tag) + 28
    rrect(d, (tx - w, 34, tx, 70), 6, outline=col if col == ACCENT else LINE, width=2)
    d.text((tx - w + 14, 37), txt, font=f_tag, fill=col); tx -= w + 12
# stat cells
CELLS = [("RALLY", RX, STAT_Y), ("RALLY TIME", RX + RW // 2 + 8, STAT_Y), ("SHOTS", RX, STAT_Y + STAT_H + 16), ("LAST SHOT", RX + RW // 2 + 8, STAT_Y + STAT_H + 16)]
CW = RW // 2 - 8
for lab, cx, cy in CELLS:
    rrect(d, (cx, cy, cx + CW, cy + STAT_H), 10, fill=PANEL, outline=LINE, width=1)
    d.text((cx + 18, cy + 12), lab, font=f_lab, fill=MUTED)
lx = CELLS[3][1] + 18 + d.textlength("LAST SHOT", font=f_lab) + 10
rrect(d, (lx, CELLS[3][2] + 14, lx + 46, CELLS[3][2] + 36), 4, outline=ACCENT, width=2)
d.text((lx + 7, CELLS[3][2] + 12), "EST.", font=NUM_SB(17), fill=ACCENT)
# map panel
rrect(d, (RX, MAP_Y, RX + RW, MAP_Y + MAP_H), 10, fill=PANEL, outline=LINE, width=1)
d.text((RX + 18, MAP_Y + 12), "COURT MAP", font=f_lab, fill=MUTED)
d.text((RX + 18, MAP_Y + MAP_H - 34), "↑ คู่แข่ง", font=f_small, fill=MUTED)
d.text((RX + RW - 18 - d.textlength("ทีมเรา ↓", font=f_small), MAP_Y + MAP_H - 34), "ทีมเรา ↓", font=f_small, fill=MUTED)
a = m2(0, 13.4); b = m2(6.1, 0)
d.rectangle((a[0], a[1], b[0], b[1]), fill=(18, 53, 91))
# player cards
for j in range(4):
    cx = 24 + j * (CARD_W + 16)
    rrect(d, (cx, CARD_Y, cx + CARD_W, CARD_Y + CARD_H), 10, fill=PANEL, outline=LINE, width=1)
    d.rectangle((cx + 1, CARD_Y + 1, cx + CARD_W - 1, CARD_Y + 5), fill=COL[j])
    d.text((cx + 18, CARD_Y + 12), NAMES[j], font=f_name, fill=FG)
    d.text((cx + 26 + d.textlength(NAMES[j], font=f_name), CARD_Y + 22), DESC[j], font=f_desc, fill=MUTED)
    for k, lab in enumerate(("SPEED", "DISTANCE", "SHOTS", "TOP")):
        d.text((cx + 18 + k * (CARD_W - 36) / 4, CARD_Y + 80), lab, font=NUM_SB(17), fill=MUTED)
# video frame border
rrect(d, (VX - 1, VY - 1, VX + VW, VY + VH), 10, outline=LINE, width=2)
BG_NP = np.array(bg)
# static court lines on map (drawn with AA later each frame onto np array is costly; bake here)
for (x1, y1, x2, y2) in COURT_LINES:
    p, q = m2(x1, y1), m2(x2, y2)
    cv2.line(BG_NP, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (205, 214, 224), 1, cv2.LINE_AA)
n1, n2 = m2(-0.35, 6.7), m2(6.45, 6.7)
for k in range(0, int(n2[0] - n1[0]), 10):
    cv2.line(BG_NP, (int(n1[0] + k), int(n1[1])), (int(min(n2[0], n1[0] + k + 6)), int(n1[1])), (255, 255, 255), 2, cv2.LINE_AA)
VMASK = np.zeros((VH, VW), np.uint8); cv2.rectangle(VMASK, (0, 0), (VW - 1, VH - 1), 255, -1)
# rounded corners mask for video
r = 10
VMASK[:r, :r] = 0; VMASK[:r, -r:] = 0; VMASK[-r:, :r] = 0; VMASK[-r:, -r:] = 0
for (cx_, cy_) in ((r, r), (VW - r - 1, r), (r, VH - r - 1), (VW - r - 1, VH - r - 1)):
    cv2.circle(VMASK, (cx_, cy_), r, 255, -1)
VM3 = (VMASK[..., None] / 255.0).astype(np.float32)

# ---------- precomputed series ----------
TOP = np.zeros((4, N))
for j in range(4):
    sp = np.array(PL[j]["speed"]); TOP[j] = np.maximum.accumulate(sp)
shot_t = np.array([s["t"] for s in SHOTS])

def proj(X, Y):
    p = Hinv @ np.array([X, Y, 1.0]); return p[0] / p[2], p[1] / p[2]
RING = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 33)]

def rally_at(t):
    for rr in RALLIES:
        if rr["start"] <= t <= rr["end"] + 0.6: return rr
    return None
def ended_recently(t):
    for rr in RALLIES:
        if rr["end"] < t <= rr["end"] + 2.2: return rr
    return None

# ---------- encode ----------
PREV = os.environ.get("HUD_PREVIEW")
ff = None if PREV else subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", f"{FPS:.4f}", "-i", "-",
                       "-i", SRC, "-map", "0:v", "-map", "1:2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
cap = cv2.VideoCapture(SRC)
fi = -1; i = 0; t0 = time.time()
if PREV:
    i = int(np.argmin(np.abs(T - float(PREV)))); cap.set(cv2.CAP_PROP_POS_FRAMES, 2 * i); fi = 2 * i - 1
while i < N and i < MAXF:
    if not cap.grab(): break
    fi += 1
    if fi % 2: continue
    ok, src = cap.retrieve()
    if not ok: break
    t = T[i]
    frame = BG_NP.copy()
    vid = cv2.cvtColor(cv2.resize(src, (VW, VH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    region = frame[VY:VY + VH, VX:VX + VW].astype(np.float32)
    frame[VY:VY + VH, VX:VX + VW] = (vid * VM3 + region * (1 - VM3)).astype(np.uint8)
    view = frame[VY:VY + VH, VX:VX + VW]
    # rings (AA, in court plane)
    order = sorted([j for j in range(4) if PL[j]["court"][i] and PL[j]["court"][i][0] is not None], key=lambda j: -PL[j]["court"][i][1])
    recent = {}
    for s in SHOTS:
        if 0 <= t - s["t"] < 0.5: recent[s["p"]] = (t - s["t"]) / 0.5
    for j in order:
        X, Y = PL[j]["court"][i]
        big = Y < 6.7
        for rad, wid, colr in ((0.44, 7 if big else 4, tuple(int(c * 0.35) for c in COL[j])), (0.44, 3 if big else 2, COL[j])):
            pts = np.array([[proj(X + rad * c, Y + rad * s_)[0] * SX, proj(X + rad * c, Y + rad * s_)[1] * SY] for c, s_ in RING], np.int32)
            cv2.polylines(view, [pts], True, colr, wid, cv2.LINE_AA)
        if j in recent:
            k = recent[j]; rad = 0.44 + 0.9 * k
            pts = np.array([[proj(X + rad * c, Y + rad * s_)[0] * SX, proj(X + rad * c, Y + rad * s_)[1] * SY] for c, s_ in RING], np.int32)
            c2 = tuple(int(ACCENT[q] * (1 - k) + view[0, 0, q] * 0) for q in range(3))
            cv2.polylines(view, [pts], True, c2, 2, cv2.LINE_AA)
    # map trails + dots
    for j in range(4):
        pts = []
        for k in range(max(0, i - 45), i + 1):
            q = PL[j]["court"][k]
            if q and q[0] is not None:
                p = m2(min(6.45, max(-0.35, q[0])), min(13.75, max(-0.35, q[1]))); pts.append((int(p[0]), int(p[1])))
            elif pts:
                if len(pts) > 1: cv2.polylines(frame, [np.array(pts, np.int32)], False, tuple(int(c * 0.6) for c in COL[j]), 2, cv2.LINE_AA)
                pts = []
        if len(pts) > 1: cv2.polylines(frame, [np.array(pts, np.int32)], False, tuple(int(c * 0.6) for c in COL[j]), 2, cv2.LINE_AA)
    for j in range(4):
        q = PL[j]["court"][i]
        if q and q[0] is not None:
            p = m2(min(6.45, max(-0.35, q[0])), min(13.75, max(-0.35, q[1])))
            cv2.circle(frame, (int(p[0]), int(p[1])), 10, BG, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(p[0]), int(p[1])), 8, COL[j], -1, cv2.LINE_AA)
            if j in recent: cv2.circle(frame, (int(p[0]), int(p[1])), int(12 + 14 * recent[j]), ACCENT, 2, cv2.LINE_AA)
    # card speed bars
    for j in range(4):
        cx = 24 + j * (CARD_W + 16); by = CARD_Y + 62
        cv2.rectangle(frame, (cx + 18, by), (cx + CARD_W - 18, by + 6), (36, 50, 70), -1)
        sp = PL[j]["speed"][i] if (PL[j]["court"][i] and PL[j]["court"][i][0] is not None) else 0
        cv2.rectangle(frame, (cx + 18, by), (cx + 18 + int((CARD_W - 36) * min(1, sp / 5)), by + 6), COL[j], -1)

    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    # name tags on video (nudge up when overlapping a tag already placed)
    placed = []
    for j in sorted(order, key=lambda j: PL[j]["court"][i][1]):
        b = PL[j]["box"][i]
        if not b or b[0] is None: continue
        nm = NAMES[j]; sp = f"{PL[j]['speed'][i]:.1f} m/s"
        w1 = d.textlength(nm, font=f_tagname); w2 = d.textlength(sp, font=f_tagnum)
        pw, ph = w1 + w2 + 30, 34
        x = VX + (b[0] + b[2]) / 2 * SX - pw / 2; y = VY + b[1] * SY - ph - 10
        x = max(VX + 6, min(VX + VW - pw - 6, x)); y = max(VY + 6, y)
        for _ in range(4):
            hit = [q for q in placed if x < q[2] + 4 and x + pw > q[0] - 4 and y < q[3] + 4 and y + ph > q[1] - 4]
            if not hit: break
            y = min(q[1] for q in hit) - ph - 6
        y = max(VY + 6, y); placed.append((x, y, x + pw, y + ph))
        rrect(d, (x, y, x + pw, y + ph), 6, fill=(8, 14, 22))
        d.rectangle((x + 1, y, x + pw - 1, y + 3), fill=COL[j])
        d.text((x + 10, y + 1), nm, font=f_tagname, fill=FG)
        d.text((x + 20 + w1, y + 4), sp, font=f_tagnum, fill=MUTED)
        if j in recent:
            d.text((x, y - 30), "SHOT", font=NUM_B(26), fill=ACCENT)
    # map names
    for j in range(4):
        q = PL[j]["court"][i]
        if q and q[0] is not None:
            p = m2(min(6.45, max(-0.35, q[0])), min(13.75, max(-0.35, q[1])))
            d.text((p[0] + 13, p[1] - 12), NAMES[j], font=f_map, fill=FG, stroke_width=2, stroke_fill=(18, 53, 91))
    mm, ss = divmod(t, 60)
    tt = f"{int(mm)}:{ss:04.1f}"; d.text((RX + RW - 18 - d.textlength(tt, font=f_lab), MAP_Y + 12), tt, font=f_lab, fill=MUTED)
    # stats
    rr = rally_at(t)
    if rr:
        sh = [s for s in SHOTS if s["rally"] == rr["n"] and s["t"] <= t]
        vals = [str(rr["n"]), f"{max(0, min(t, rr['end']) - rr['start']):.1f}", str(len(sh))]
        last = next((s for s in reversed(sh) if s.get("avg_kmh_est")), None)
        lt = f"{NAMES[last['p']]} · {int(last['avg_kmh_est'])} km/h" if last else (NAMES[sh[-1]["p"]] if sh else "—")
    else:
        vals = ["—", "—", "—"]; lt = "ระหว่าง rally"
    for k, v in enumerate(vals):
        cx, cy = CELLS[k][1], CELLS[k][2]
        d.text((cx + 18, cy + 34), v, font=f_big, fill=FG)
        if k == 1 and v != "—":
            d.text((cx + 26 + d.textlength(v, font=f_big), cy + 66), "s", font=f_unit, fill=MUTED)
    cx, cy = CELLS[3][1], CELLS[3][2]
    d.text((cx + 18, cy + 52), lt, font=f_last if d.textlength(lt, font=f_last) < CW - 30 else TH_SB(24), fill=ACCENT if (rr and recent) else FG)
    if rr:
        cv2_x = 0
    # cards numbers
    for j in range(4):
        cx = 24 + j * (CARD_W + 16); colw = (CARD_W - 36) / 4
        on = PL[j]["court"][i] and PL[j]["court"][i][0] is not None
        nshot = int(np.sum([(s["p"] == j and s["t"] <= t) for s in SHOTS]))
        items = [(f"{PL[j]['speed'][i]:.1f}" if on else "—", "m/s"), (f"{PL[j]['dist'][i]:.1f}", "m"), (str(nshot), ""), (f"{TOP[j][i]:.1f}", "m/s")]
        for k, (v, u) in enumerate(items):
            x = cx + 18 + k * colw
            d.text((x, CARD_Y + 100), v, font=f_cnum, fill=FG)
            if u: d.text((x + d.textlength(v, font=f_cnum) + 4, CARD_Y + 114), u, font=f_cunit, fill=MUTED)
    # rally-end banner
    re = ended_recently(t)
    if re:
        txt1 = f"RALLY {re['n']}"; txt2 = f" จบ · {re['end'] - re['start']:.1f} วินาที · {re['shots']} shots"
        w = d.textlength(txt1, font=f_bnum) + d.textlength(txt2, font=f_banner) + 48
        x = VX + VW / 2 - w / 2; y = VY + VH - 86
        rrect(d, (x, y, x + w, y + 58), 10, fill=(8, 14, 22), outline=ACCENT, width=2)
        d.text((x + 24, y + 8), txt1, font=f_bnum, fill=ACCENT)
        d.text((x + 24 + d.textlength(txt1, font=f_bnum), y + 5), txt2, font=f_banner, fill=FG)
    if PREV:
        im.save(OUT); print('preview', OUT); sys.exit(0)
    ff.stdin.write(np.asarray(im).tobytes())
    i += 1
    if i % 300 == 0: print(i, f"{(time.time() - t0) / i * 1000:.0f} ms/frame", flush=True)
ff.stdin.close(); ff.wait()
print("DONE", i, f"{time.time() - t0:.0f}s", OUT, flush=True)
