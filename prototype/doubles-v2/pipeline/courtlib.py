"""Court geometry helpers: white-line mask, Hough line clustering, first-frame calibration, DT refinement."""
import cv2, numpy as np, itertools
from scipy.optimize import least_squares

W_COURT, L_COURT = 6.10, 13.40
XS = [0.0, 0.46, 3.05, 5.64, 6.10]                       # longitudinal lines (sidelines + centre)
YS = [0.0, 0.76, 4.72, 6.70, 8.68, 12.64, 13.40]          # baselines, service lines (6.70 = net, not painted)
SEGS = ([(0, y, 6.10, y) for y in (0, 0.76, 4.72, 8.68, 12.64, 13.40)] +
        [(x, 0, x, 13.40) for x in (0, 0.46, 5.64, 6.10)] + [(3.05, 0, 3.05, 4.72), (3.05, 8.68, 3.05, 13.40)])

def white_mask(bgr, floor_top=480):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 170) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    m[:floor_top] = 0
    return m

def hough_lines(mask):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 720, 60, minLineLength=40, maxLineGap=20)
    if segs is None: return []
    segs = segs.reshape(-1, 4).astype(float)
    items = []
    for x1, y1, x2, y2 in segs:
        th = np.arctan2(y2 - y1, x2 - x1) % np.pi
        n = np.array([-np.sin(th), np.cos(th)]); rho = n @ [x1, y1]
        items.append((th, rho, np.hypot(x2 - x1, y2 - y1), (x1, y1, x2, y2)))
    items.sort(key=lambda a: -a[2])
    cl = []
    for th, rho, ln, s in items:
        for c in cl:
            dth = abs(c['th'] - th); dth = min(dth, np.pi - dth)
            # compare rho after aligning orientation
            r2 = rho if abs(c['th'] - th) < np.pi / 2 else -rho
            if dth < np.radians(3.0) and abs(c['rho'] - r2) < 16:
                c['segs'].append(s); c['len'] += ln; break
        else:
            cl.append(dict(th=th, rho=rho, segs=[s], len=ln))
    out = []
    for c in cl:
        pts = np.array([[s[0], s[1]] for s in c['segs']] + [[s[2], s[3]] for s in c['segs']], np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        L = np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0]); L /= np.hypot(L[0], L[1])
        out.append(dict(L=L, len=c['len'], ang=np.degrees(np.arctan2(vy, vx)) % 180, ymin=pts[:, 1].min(), ymax=pts[:, 1].max(),
                        xmin=pts[:, 0].min(), xmax=pts[:, 0].max()))
    return out

def inter(L1, L2):
    p = np.cross(L1, L2); return p[:2] / p[2] if abs(p[2]) > 1e-9 else None

def cross_ratio(a, b, c, d):
    return ((c - a) * (d - b)) / ((c - b) * (d - a))

def project(H, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]

def calibrate_first(bgr, log=print):
    """Court -> image homography from one frame, following: VP of longitudinal lines, two lowest horizontals
    = baseline (Y=0) and doubles long service line (Y=0.76), cross-ratio matching of longitudinal lines."""
    m = white_mask(bgr); lines = hough_lines(m)
    hor = [l for l in lines if (l['ang'] < 12 or l['ang'] > 168) and l['len'] > 120]
    lon = [l for l in lines if 20 < l['ang'] < 160 and l['len'] > 60 and l['ymax'] > 520]
    # vanishing point of longitudinal lines (robust: RANSAC over pairs)
    best = None
    for a, b in itertools.combinations(range(len(lon)), 2):
        vp = inter(lon[a]['L'], lon[b]['L'])
        if vp is None or vp[1] > 480: continue
        inl = [k for k, l in enumerate(lon) if abs(l['L'] @ [vp[0], vp[1], 1]) < 4]
        score = sum(lon[k]['len'] for k in inl)
        if best is None or score > best[0]: best = (score, vp, inl)
    _, vp, inl = best
    A = np.array([lon[k]['L'] for k in inl]); u, s, vt = np.linalg.svd(A); v = vt[-1]; vp = v[:2] / v[2]
    lon = [lon[k] for k in inl]
    log(f'longitudinal lines through VP {np.round(vp, 1)}: {len(lon)}')
    # two lowest long horizontals
    def y_at_center(l): return -(l['L'][0] * 270 + l['L'][2]) / l['L'][1]
    hor.sort(key=lambda l: -y_at_center(l))
    base, lsv = hor[0], hor[1]
    log(f'baseline y@270={y_at_center(base):.1f}, long service y@270={y_at_center(lsv):.1f}')
    hvp = np.cross(base['L'], lsv['L'])                        # horizontal vanishing point (homogeneous)
    # param along the baseline: x coordinate of intersections; 1D projective with ideal point = hvp
    def xb(L):
        p = inter(L, base['L']); return p[0]
    xs = sorted([(xb(l['L']), l) for l in lon], key=lambda a: a[0])
    hx = hvp[0] / hvp[2] if abs(hvp[2]) > 1e-12 else 1e12
    # choose assignment of observed lines to model X values (ordered, monotone) minimising 1D projective residual
    obs = np.array([a[0] for a in xs]); bestA = None
    n = len(obs)
    for k in range(min(n, 5), 3, -1):
        for oi in itertools.combinations(range(n), k):
            for idx in itertools.combinations(range(5), k):
                Xm = np.array([XS[q] for q in idx]); ob = obs[list(oi)]
                M = [[X, 1, -x * X] for X, x in zip(Xm, ob)] + [[1, 0, -hx]]
                r = list(ob) + [0]
                sol, *_ = np.linalg.lstsq(np.array(M, float), np.array(r, float), rcond=None)
                a_, b_, c_ = sol
                pred = (a_ * Xm + b_) / (c_ * Xm + 1)
                res = np.abs(pred - ob).max() / (abs(ob.max() - ob.min()) + 1e-9)   # relative residual
                cost = res - 0.02 * k
                if bestA is None or cost < bestA[0]: bestA = (cost, idx, oi, res)
    _, idx, oi, res = bestA
    log('baseline intersections x =', np.round(obs, 1).tolist())
    log('used lines', list(oi), '-> X =', [XS[k] for k in idx], 'relative residual %.4f' % res)
    src, dst = [], []
    for (x, l), k in zip([xs[q] for q in oi], idx):
        for Y, HL in ((0.0, base), (0.76, lsv)):
            p = inter(l['L'], HL['L']); src.append([XS[k], Y]); dst.append(p)
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
    return H, dict(vp=vp.tolist(), n_lon=len(lon), assign=[XS[k] for k in idx], residual=float(res), base=base['L'].tolist(), lsv=lsv['L'].tolist())

def model_points(step=0.05, ymax=13.4, ymin=0.0):
    pts = []
    for x1, y1, x2, y2 in SEGS:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
        for t in np.linspace(0, 1, n):
            X, Y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            if ymin <= Y <= ymax: pts.append((X, Y))
    return np.array(pts)

def refine(H, mask, region_ymin=0.0, region_ymax=13.4, img_rows=(480, 960), iters=2):
    """Pull the projected model lines onto the white pixels: minimise distance-transform values (robust)."""
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    P = model_points(0.04, region_ymax, region_ymin)
    h, w = mask.shape
    def resid(p):
        Hm = np.append(p, 1).reshape(3, 3)
        q = project(Hm, P)
        ok = (q[:, 0] >= 1) & (q[:, 0] < w - 2) & (q[:, 1] >= img_rows[0]) & (q[:, 1] < min(h - 2, img_rows[1]))
        x, y = q[:, 0].clip(0, w - 2), q[:, 1].clip(0, h - 2)
        x0, y0 = x.astype(int), y.astype(int); fx, fy = x - x0, y - y0
        v = (dt[y0, x0] * (1 - fx) * (1 - fy) + dt[y0, x0 + 1] * fx * (1 - fy) + dt[y0 + 1, x0] * (1 - fx) * fy + dt[y0 + 1, x0 + 1] * fx * fy)
        return np.where(ok, np.minimum(v, 12.0), 0.0)
    p = (H / H[2, 2]).ravel()[:8]
    for _ in range(iters):
        r = least_squares(resid, p, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=200)
        p = r.x
    Hn = np.append(p, 1).reshape(3, 3)
    return Hn

def overlap(H, mask, ymin=0.0, ymax=13.4, img_rows=(480, 960)):
    """Fraction of visible model points that land on (dilated) white pixels."""
    P = model_points(0.04, ymax, ymin); q = project(H, P); h, w = mask.shape
    ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= img_rows[0]) & (q[:, 1] < min(h, img_rows[1]))
    if ok.sum() < 20: return 0.0
    d = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    qi = q[ok].astype(int)
    return float((d[qi[:, 1], qi[:, 0]] > 0).mean())


def refine_similarity(H, mask, region_ymin=0.0, region_ymax=6.7, img_rows=(480, 960)):
    """Small image-space correction A (shift, rotation, scale about the image centre) so that A@H sits on the white
    lines of the given court region. 4 parameters: well conditioned even when the sidelines are off-frame."""
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    P = model_points(0.04, region_ymax, region_ymin); q0 = project(H, P); h, w = mask.shape
    ok0 = (q0[:, 0] >= 1) & (q0[:, 0] < w - 2) & (q0[:, 1] >= img_rows[0]) & (q0[:, 1] < min(h - 2, img_rows[1]))
    q0 = q0[ok0]; c = np.array([w / 2, (img_rows[0] + img_rows[1]) / 2])
    if len(q0) < 30: return H
    def A(p):
        tx, ty, th, s = p; cs, sn = (1 + s) * np.cos(th), (1 + s) * np.sin(th)
        M = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1.0]])
        T1 = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]]); T2 = np.array([[1, 0, c[0] + tx], [0, 1, c[1] + ty], [0, 0, 1]])
        return T2 @ M @ T1
    def resid(p):
        q = project(A(p), q0); x, y = q[:, 0].clip(0, w - 2), q[:, 1].clip(0, h - 2)
        x0, y0 = x.astype(int), y.astype(int); fx, fy = x - x0, y - y0
        v = (dt[y0, x0] * (1 - fx) * (1 - fy) + dt[y0, x0 + 1] * fx * (1 - fy) + dt[y0 + 1, x0] * (1 - fx) * fy + dt[y0 + 1, x0 + 1] * fx * fy)
        return np.minimum(v, 10.0)
    r = least_squares(resid, np.zeros(4), loss='soft_l1', f_scale=2.0, x_scale=[1, 1, 0.01, 0.01], max_nfev=100)
    Hn = A(r.x) @ H
    return Hn / Hn[2, 2]
