"""Step 5: court homography for every frame.
prediction = camera motion (cam.npy) applied to the frame-0 calibration; then pulled onto the real white lines of the
near half-court; every 2nd frame the court is also re-read from the image from scratch. The candidate whose projected
lines sit on more white pixels wins. Finally the 4 court corners are smoothed over time. -> court.npy (N,3,3) court(m)->540px"""
import cv2, numpy as np, json, time
from courtlib import white_mask, refine, refine_similarity, overlap, calibrate_first, project
from scipy.ndimage import uniform_filter1d

Hc0 = np.load('Hc0.npy'); CAM = np.load('cam.npy')
cap = cv2.VideoCapture('a540.mp4'); N = len(CAM)
out = []; src = []; t0 = time.time(); stats = dict(pred=0, refined=0, redetect=0)
REF = np.float32([[3.05, 0.3], [3.05, 4.72], [5.64, 4.72], [0.46, 4.72], [3.05, 8.68], [6.1, 13.4], [0, 13.4], [5.64, 0.76]])  # all inside the frame
for i in range(N):
    ok, f = cap.read()
    if not ok: break
    m = white_mask(f)
    pred = CAM[i] @ Hc0; pred /= pred[2, 2]
    cands = [('pred', pred, overlap(pred, m, 0, 6.7))]
    try:
        r = refine_similarity(pred, m); cands.append(('refined', r, overlap(r, m, 0, 6.7)))
    except Exception:
        pass
    if i % 2 == 0:
        try:
            Hd, _ = calibrate_first(f, log=lambda *a: None)
            Hd = refine(Hd, m, 0.0, 13.4, iters=1); cands.append(('redetect', Hd, overlap(Hd, m, 0, 6.7)))
        except Exception:
            pass
    # reject candidates that jump far from the camera-motion prediction (occlusion / wrong line)
    pp = project(pred, REF)
    cands = [c for c in cands if np.abs(project(c[1], REF) - pp).max() < 8]
    name, H, ov = max(cands, key=lambda c: c[2] + (0.002 if c[0] != 'pred' else 0))
    stats[name] += 1; out.append(H / H[2, 2]); src.append((name, round(ov, 3)))
    if (i + 1) % 300 == 0:
        print(i + 1, f'{(time.time() - t0) / (i + 1):.3f}s/f', stats, flush=True)
Hs = np.array(out)
# temporal smoothing of the projected reference points, then re-fit
P = np.array([project(H, REF) for H in Hs])                      # (N,5,2)
Ps = uniform_filter1d(P, size=7, axis=0, mode='nearest')
Hsm = np.array([cv2.findHomography(REF, p.astype(np.float32), 0)[0] for p in Ps])
np.save('court.npy', Hsm)
ovs = np.array([s[1] for s in src])
json.dump(dict(stats=stats, overlap_p10=float(np.percentile(ovs, 10)), overlap_median=float(np.median(ovs)),
               jitter_px=float(np.abs(np.diff(P, axis=0)).max())), open('court_stats.json', 'w'), indent=1)
print('DONE', stats, 'overlap median %.3f p10 %.3f' % (np.median(ovs), np.percentile(ovs, 10)), flush=True)
