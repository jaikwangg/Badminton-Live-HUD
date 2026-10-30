"""Report: projected court lines over 8 frames spread through the clip -> check/court_8frames.jpg"""
import cv2, numpy as np
from courtlib import SEGS, project
CO = np.load('court.npy'); N = len(CO); cap = cv2.VideoCapture('a540.mp4'); tiles = []
for i in np.linspace(0, N - 1, 8).astype(int):
    cap.set(1, int(i)); ok, f = cap.read()
    for x1, y1, x2, y2 in SEGS:
        P = project(CO[i], [[x1 + (x2 - x1) * t, y1 + (y2 - y1) * t] for t in np.linspace(0, 1, 80)])
        cv2.polylines(f, [P.astype(np.int32)], False, (0, 0, 255), 1, cv2.LINE_AA)
    p = project(CO[i], [[-0.3, 6.7], [6.4, 6.7]]).astype(int); cv2.line(f, tuple(p[0]), tuple(p[1]), (0, 255, 255), 1)
    cv2.putText(f, f'{i / 30:.1f}s', (8, 28), 0, 0.9, (0, 255, 255), 2)
    tiles.append(f[420:960])
cv2.imwrite('check/court_8frames.jpg', np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
