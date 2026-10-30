"""Step 2: camera motion. Homography from frame 0 to every frame, using corners on the ceiling and upper wall
(rows < 400 in 540x960; no people there). -> cam.npy  (N,3,3) maps frame-0 pixels to frame-i pixels."""
import cv2, numpy as np
cap = cv2.VideoCapture('a540.mp4')
ok, f0 = cap.read(); g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(g0); mask[0:400] = 255
p0 = cv2.goodFeaturesToTrack(g0, maxCorners=600, qualityLevel=0.01, minDistance=8, mask=mask)
Hs = [np.eye(3)]; inl = []; prevH = np.eye(3)
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    init = cv2.perspectiveTransform(p0, prevH)
    p1, st, err = cv2.calcOpticalFlowPyrLK(g0, g, p0, init.copy(), winSize=(21, 21), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    good = st.ravel() == 1
    H, m = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 1.5)
    if H is None or m.sum() < 40:
        H = prevH
    Hs.append(H); inl.append(int(m.sum()) if m is not None else 0); prevH = H
Hs = np.array(Hs); np.save('cam.npy', Hs)
c = np.float32([[[270, 700]], [[270, 200]], [[50, 900]], [[500, 900]]])
d = np.array([np.abs(cv2.perspectiveTransform(c, H) - c).max() for H in Hs])
print('frames', len(Hs), 'inliers min/median', min(inl), int(np.median(inl)), 'max corner displacement px (540 scale): p50 %.2f p99 %.2f max %.2f' % (np.median(d), np.percentile(d, 99), d.max()))
