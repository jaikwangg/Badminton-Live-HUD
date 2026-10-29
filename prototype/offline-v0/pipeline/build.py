import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
import json, numpy as np, os
P = json.load(open(os.path.join(DATA, 'players.json')))
E = json.load(open(os.path.join(DATA, 'events.json')))
H = np.array(json.load(open(os.path.join(DATA, 'calib.json')))['H'])
Hinv = np.linalg.inv(H)
Hinv = Hinv / Hinv[2, 2]
DESC = ['เสื้อกรมท่า', 'เสื้อน้ำเงิน #11', 'เสื้อดำ กางเกงขาว', 'เสื้อขาว']

def compact(arr, nd):
    return [None if (a is None or a[0] is None) else [round(x, nd) for x in a] for a in arr]

players = []
for j, p in enumerate(P['players']):
    players.append(dict(name=p['name'], desc=DESC[j], side=p['side'], coverage=round(p['coverage'], 3),
                        court=compact(p['court'], 2), box=compact(p['box'], 0),
                        speed=[round(s, 1) for s in p['speed']], dist=[round(d, 1) for d in p['dist']]))
data = dict(t=[round(t, 3) for t in P['t']], img=[1920, 1078], Hinv=Hinv.round(8).tolist(),
            players=players, shots=E['shots'], rallies=E['rallies'])
js = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web', 'template.html')).read().replace('/*DATA*/null', js)
os.makedirs('site', exist_ok=True)
open('site/index.html', 'w').write(html)
if not os.path.exists('site/clip720.mp4') and os.path.exists(os.path.join(DATA, 'clip720.mp4')):
    import shutil; shutil.copy(os.path.join(DATA, 'clip720.mp4'), 'site/clip720.mp4')
print('index.html', len(html) // 1024, 'KB')
