"""Step 10: check images + report.
 check/court_8frames.jpg      projected court lines on 8 frames
 check/shots_rally<N>.jpg     every shot: frame at the hit and 0.2 s later (shuttle path, hitter box)
 check/gaps.jpg               long stretches inside rallies with no shot
 check/report.md              rallies, shots, warnings, how to fix via config.json"""
import json, cv2, numpy as np, os
from PIL import Image, ImageDraw, ImageFont
E = json.load(open('events.json')); P = json.load(open('players.json')); S = json.load(open('shuttle.json')); CFG = json.load(open('config.json'))
PL = P['players']; N = P['n']; FPS = 30.0
COL = [tuple(p['color']) for p in CFG['players']]
FONT = ImageFont.truetype('fonts/IBMPlexSansThai-SemiBold.ttf', 20, layout_engine=ImageFont.Layout.RAQM)
FONTS = ImageFont.truetype('fonts/IBMPlexSansThai-Medium.ttf', 17, layout_engine=ImageFont.Layout.RAQM)
spos = {}
for path in S['paths']:
    for f, x, y in path['pts']: spos[int(f)] = (x, y)
cap = cv2.VideoCapture('a540.mp4')
def frame(f):
    cap.set(1, int(min(max(f, 0), N - 1))); ok, im = cap.read(); return im
Y0, Y1 = 200, 820

def annotate(f, shot=None, label=None):
    im = frame(f)
    for g in range(f - 12, f + 13):
        if g in spos:
            x, y = spos[g]; c = (0, 230, 255) if g <= f else (255, 200, 0)
            cv2.circle(im, (int(x), int(y)), 3 if g != f else 7, c, -1 if g != f else 2, cv2.LINE_AA)
    if shot is not None and shot.get('xy'):
        x, y = shot['xy']; cv2.circle(im, (int(x), int(y)), 14, (0, 0, 255), 2, cv2.LINE_AA)
    rgb = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(rgb)
    for j in range(4):
        b = PL[j]['box'][min(f, N - 1)]
        if not b or b[0] is None: continue
        wdt = 3 if shot is not None and shot['player'] == j else 1
        d.rectangle(b, outline=COL[j], width=wdt)
        d.text((b[0], b[1] - 22), PL[j]['name'], font=FONTS, fill=COL[j], stroke_width=2, stroke_fill=(0, 0, 0))
    if label: d.text((6, Y0 + 4), label, font=FONT, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(rgb)[Y0:Y1]

os.makedirs('check', exist_ok=True)
lines = ['# รายงานตรวจ', '', f"วิดีโอ {N / FPS:.1f} วิ · {len(E['rallies'])} แรลลี่ · {sum(len(r['shots']) for r in E['rallies'])} ช็อต", '']
src_th = {'track': 'เห็นลูกกลับทิศที่ไม้', 'track-kink': 'ลูกเปลี่ยนความเร็ว/ทิศที่ไม้ + เสียงตี', 'audio-seen': 'เสียงตี ตอนลูกอยู่ที่ไม้', 'track-start': 'เส้นทางลูกเริ่มที่ไม้', 'audio': 'เสียงตี (ไม่เห็นลูก)', 'inferred': 'เดาจากการสลับฝั่ง', 'config': 'กำหนดใน config หลังตรวจภาพ'}
for r in E['rallies']:
    tiles = []
    lines += [f"## แรลลี่ {r['n']}  ({r['start']:.2f}–{r['end']:.2f} วิ, {len(r['shots'])} ช็อต)", '', '| # | เวลา (วิ) | คนตี | ที่มา | km/h | D (ม.) | T (วิ) |', '|---|---|---|---|---|---|---|']
    for k, s in enumerate(r['shots']):
        f = s['frame']
        a = annotate(f, s, f"#{k + 1} {s['t']:.2f}s {s['name']} {'' if s['speed_kmh'] is None else str(s['speed_kmh']) + ' km/h'}")
        b = annotate(f + 6, s, f"+0.2s  [{s['src']}]")
        tiles.append(np.hstack([a, b]))
        lines.append(f"| {k + 1} | {s['t']:.2f} | {s['name']} | {src_th.get(s['src'], s['src'])} | {'—' if s['speed_kmh'] is None else s['speed_kmh']} | {s['D'] or '—'} | {s['T'] or '—'} |")
    lines.append('')
    if tiles:
        th = [cv2.resize(t, (720, 413)) for t in tiles]
        while len(th) % 2: th.append(np.zeros_like(th[0]))
        grid = np.vstack([np.hstack(th[k:k + 2]) for k in range(0, len(th), 2)])
        cv2.imwrite(f"check/shots_rally{r['n']}.jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
# long gaps without shots inside rallies
gaps = []
for r in E['rallies']:
    ts = [r['start']] + [s['t'] for s in r['shots']] + [r['end']]
    for a, b in zip(ts, ts[1:]):
        if b - a > 2.4: gaps.append((r['n'], a, b))
if gaps:
    rows = []
    for n, a, b in gaps:
        fs = np.linspace(a * FPS + 6, b * FPS - 6, 6).astype(int)
        rows.append(np.hstack([cv2.resize(annotate(int(f), None, f'R{n} {f / FPS:.1f}s'), (270, 270)) for f in fs]))
    cv2.imwrite('check/gaps.jpg', cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
lines += ['## คำเตือน', '']
lines += [f"- {w['t']:.2f} วิ (แรลลี่ {w['rally']}): {w['msg']}" for w in E['warnings']] or ['- ไม่มี']
lines += ['', '## ช่วงในแรลลี่ที่ไม่มีช็อตเกิน 2.4 วิ', ''] + ([f'- แรลลี่ {n}: {a:.2f}–{b:.2f} วิ' for n, a, b in gaps] or ['- ไม่มี'])
lines += ['', '## แก้ไขผลเอง', '', 'แก้ `config.json` แล้วรัน `python3 shots.py && python3 report.py && python3 render.py out.mp4`', '',
          '- เพิ่มช็อต: `"add_shots": [{"t": 12.34, "player": "บอส"}]`', '- ลบช็อต: `"remove_shots": [12.34]` (ลบช็อตที่ห่างไม่เกิน 0.15 วิ)',
          '- เวลาจบแรลลี่: `"rally_end": {"3": 41.2}`', '- เสียงช้า/เร็วกว่าภาพ: `"audio_offset": 0.05`']
open('check/report.md', 'w').write('\n'.join(lines))
print('report written:', len(E['rallies']), 'rallies,', len(gaps), 'gaps,', len(E['warnings']), 'warnings')
