import os
DATA = os.environ.get("HUD_DATA", "data")
SRC = os.environ.get("HUD_SRC", os.path.join(DATA, "source.mov"))
import json, numpy as np, scipy.signal as ss, sys
from scipy.io import wavfile
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E = json.load(open(os.path.join(DATA, 'events.json'))); P = json.load(open(os.path.join(DATA, 'players.json')))
T = np.array(P['t']); SW = np.array(E['swing'])
sr, x = wavfile.read(os.path.join(DATA, 'audio.wav')); x = x.astype(np.float32) / 32768
sos = ss.butter(4, [2500, 9000], btype='band', fs=sr, output='sos'); y = ss.sosfiltfilt(sos, x)
hop = int(sr * 0.004); n = len(y) // hop
e = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-7; db = 20 * np.log10(e); base = median_filter(db, size=301)
ta = np.arange(n) * 0.004
a, b = float(sys.argv[1]), float(sys.argv[2])
fig, ax = plt.subplots(2, 1, figsize=(24, 7), sharex=True)
m = (ta >= a) & (ta <= b); ax[0].plot(ta[m], (db - base)[m], lw=0.6); ax[0].set_ylim(-5, 35); ax[0].grid(alpha=.3)
cols = ['c', 'g', 'm', 'r']
for j in range(4):
    mm = (T >= a) & (T <= b); ax[1].plot(T[mm], SW[j][mm], cols[j], lw=1, label=P['players'][j]['name'] if False else str(j))
for s in E['shots']:
    if a <= s['t'] <= b:
        for k in range(2): ax[k].axvline(s['t'], color=cols[s['p']], alpha=.6)
ax[1].legend(); ax[1].set_xticks(np.arange(a, b, 0.5)); ax[1].grid(alpha=.3); ax[1].set_ylim(0, 12)
plt.tight_layout(); plt.savefig(sys.argv[3], dpi=55)
