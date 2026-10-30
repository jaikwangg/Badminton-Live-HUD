"""Step 1b: racket-hit candidates from audio transients in the 2-8 kHz band. -> onsets.json"""
import json, numpy as np, scipy.signal as ss
from scipy.io import wavfile
from scipy.ndimage import median_filter, minimum_filter1d

sr, x = wavfile.read('audio.wav'); x = x.astype(np.float32) / 32768
sos = ss.butter(4, [2000, 8000], btype='band', fs=sr, output='sos'); y = ss.sosfiltfilt(sos, x)
hop = int(sr * 0.004); n = len(y) // hop
e = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-7
db = 20 * np.log10(e)
base = median_filter(db, size=375)                       # 1.5 s rolling background
prev = np.concatenate([[db[0]] * 6, minimum_filter1d(db, 6, origin=2)[:-6]])  # min over previous ~24 ms
rise = db - prev
t = np.arange(n) * 0.004
cand = []
for i in range(2, n - 2):
    if db[i] == db[i - 2:i + 3].max() and rise[i] >= 6 and db[i] - base[i] >= 8:
        cand.append([float(t[i]), float(db[i] - base[i]), float(rise[i])])
merged = []
for c in cand:
    if merged and c[0] - merged[-1][0] < 0.10:
        if c[1] > merged[-1][1]: merged[-1] = c
    else:
        merged.append(c)
json.dump(dict(onsets=[dict(t=round(a, 3), snr=round(b, 1), rise=round(r, 1)) for a, b, r in merged],
               env=dict(dt=0.004, db=np.round(db - base, 1).tolist())), open('onsets.json', 'w'))
print(len(merged), 'onsets')
