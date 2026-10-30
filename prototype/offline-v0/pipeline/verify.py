import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
import json, sys, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
P = json.load(open(os.path.join(DATA, 'players.json')))
T = np.array(P['t'])
times = [float(a) for a in sys.argv[1].split(',')]
out = sys.argv[2] if len(sys.argv) > 2 else 'out/verify.jpg'
COL = [(240, 208, 63), (160, 227, 75), (255, 155, 195), (107, 138, 255)]  # BGR-ish for cv2
font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf', 40)
tiles = []
cap = cv2.VideoCapture(SRC)
for t in times:
    i = int(np.argmin(np.abs(T - t)))
    cap.set(cv2.CAP_PROP_POS_MSEC, T[i] * 1000)
    ok, fr = cap.read()
    if not ok:
        continue
    for j, p in enumerate(P['players']):
        b = p['box'][i]; f = p['foot'][i]
        if b and b[0] is not None:
            cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COL[j], 4)
        if f and f[0] is not None:
            cv2.circle(fr, (int(f[0]), int(f[1])), 10, COL[j], -1)
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(im)
    for j, p in enumerate(P['players']):
        b = p['box'][i]
        if b and b[0] is not None:
            dr.text((b[0], max(0, b[1] - 44)), p['name'], fill=COL[j][::-1], font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    dr.text((10, 10), f't={T[i]:.2f}', fill=(255, 255, 0), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    tiles.append(cv2.resize(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), (640, 360)))
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
cv2.imwrite(out, np.vstack(rows))
print('wrote', out)
