"""Click-to-crown segmentation (v2).

Idea: learn what "this tree" and "everything around it" look like from the
pixels near the click and the ring around the window, score every pixel as
tree / not-tree, then find the best star-shaped (roughly round) outline around
the click with dynamic programming in polar coordinates. The model is then
re-learned from the outline it found (inside = tree, just outside = background,
which includes the cast shadow) and the outline is refined.

numpy only.
"""
import numpy as np


# ---------------------------------------------------------------- helpers
def box_blur(a, r):
    if r <= 0:
        return a.astype(np.float32)
    k = 2 * r + 1
    pad_w = ((r + 1, r), (r + 1, r)) + ((0, 0),) * (a.ndim - 2)
    c = np.pad(a.astype(np.float64), pad_w, mode='edge').cumsum(0).cumsum(1)
    out = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (out / (k * k)).astype(np.float32)


def features(rgb):
    """Per-pixel features, standardised over the window: brightness,
    two chromaticities, local texture (std of brightness)."""
    f = box_blur(rgb[..., :3].astype(np.float32), 1)
    tot = f.sum(axis=2) + 1.0
    lum = f.mean(axis=2)
    cr = f[..., 0] / tot
    cg = f[..., 1] / tot
    tex = np.sqrt(np.maximum(box_blur(lum ** 2, 2) - box_blur(lum, 2) ** 2, 0))
    tex = box_blur(tex, 1)
    feats = np.stack([lum, cr, cg, tex], axis=2)
    h, w, d = feats.shape
    flat = feats.reshape(-1, d)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-3)
    return flat.reshape(h, w, d)


def _fit_mix(X, k):
    """Tiny diagonal Gaussian mixture (hard-EM / k-means init)."""
    n = len(X)
    if n > 4000:
        X = X[np.random.default_rng(0).choice(n, 4000, replace=False)]
        n = 4000
    k = max(1, min(k, n // 8))
    order = np.argsort(X[:, 0])
    centers = X[order[((np.arange(k) + 0.5) * n / k).astype(int)]].copy()
    for _ in range(10):
        d = ((X[:, None, :] - centers[None]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                centers[j] = X[lab == j].mean(0)
    lab = ((X[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
    means, vars_, ws = [], [], []
    for j in range(k):
        m = lab == j
        if m.sum() < 3:
            continue
        means.append(X[m].mean(0))
        vars_.append(X[m].var(0) + 0.15)
        ws.append(m.sum() / n)
    return np.array(means), np.array(vars_), np.log(np.array(ws))


def _loglik(F, mix):
    means, vars_, logw = mix
    out = []
    for m, v, lw in zip(means, vars_, logw):
        out.append(lw - 0.5 * (((F - m) ** 2) / v + np.log(v)).sum(1))
    out = np.stack(out, 1)
    mx = out.max(1, keepdims=True)
    return (mx + np.log(np.exp(out - mx).sum(1, keepdims=True)))[:, 0]


def _polar_dp(S, lam, dmax):
    """S[a, r]: score of putting the boundary at radius r on ray a.
    Returns the radius per ray maximising sum(S) - lam*sum|dr| (closed curve)."""
    A, Rn = S.shape
    S3 = np.concatenate([S, S, S])
    score = S3[0].copy()
    back = np.zeros((3 * A, Rn), np.int8)
    for i in range(1, 3 * A):
        best = np.full(Rn, -np.inf)
        arg = np.zeros(Rn, np.int8)
        for d in range(-dmax, dmax + 1):
            cand = np.full(Rn, -np.inf)
            lo, hi = max(0, -d), min(Rn, Rn - d)
            cand[lo:hi] = score[lo + d:hi + d] - lam * abs(d)
            better = cand > best
            best = np.where(better, cand, best)
            arg = np.where(better, d, arg)
        score = best + S3[i]
        back[i] = arg
    r = int(np.argmax(score))
    path = np.zeros(3 * A, int)
    for i in range(3 * A - 1, -1, -1):
        path[i] = r
        r = r + int(back[i][r])
        r = min(max(r, 0), Rn - 1)
    return path[A:2 * A].astype(np.float32)


def _shift_and(m, cross=True):
    p = np.pad(m, 1, constant_values=True)
    return m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def _shift_or(m):
    p = np.pad(m, 1, constant_values=False)
    return m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def _flood(mask, seed):
    h, w = mask.shape
    if not mask[seed]:
        return None
    seen = np.zeros_like(mask)
    seen[seed] = True
    stack = [seed]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return seen


def _fill_holes(mask):
    pad = np.pad(~mask, 1, constant_values=True)
    return ~_flood(pad, (0, 0))[1:-1, 1:-1]


def _bilinear(img, xs, ys):
    """Sample img at continuous coords (pixel centres at +0.5)."""
    h, w = img.shape
    x = np.clip(xs - 0.5, 0, w - 1.001)
    y = np.clip(ys - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def _snap_to_edge(logit, rr, ang_idx, rt, th, cy, cx, R):
    """Pixel-level refinement: inside the polar outline's core plus every
    tree-like pixel in a band around it, keep what is connected to the click,
    then read the boundary along each ray with sub-pixel precision."""
    h, w = rr.shape
    rtp = rt[ang_idx]
    core = rr < 0.75 * rtp
    cand = core | ((logit > 0) & (rr < 1.2 * rtp + 2))
    cand = _shift_and(_shift_or(_shift_and(cand)))        # drop 1-px specks/necks
    cand = _fill_holes(cand)
    seed = (int(cy), int(cx))
    comp = _flood(cand, seed) if cand[seed] else None
    if comp is None:
        return rt
    soft = box_blur(comp.astype(np.float32), 1)
    steps = np.arange(0.0, R + 8, 0.25)
    xs = cx + np.outer(np.cos(th), steps)
    ys = cy + np.outer(np.sin(th), steps)
    vals = _bilinear(soft, xs, ys)
    out = rt.copy()
    for a in range(len(th)):
        below = np.nonzero(vals[a] < 0.5)[0]
        if len(below) == 0:
            out[a] = steps[-1]
        else:
            j = below[0]
            if j == 0:
                continue
            v0, v1 = vals[a, j - 1], vals[a, j]
            t = (v0 - 0.5) / max(v0 - v1, 1e-6)
            out[a] = steps[j - 1] + t * 0.25
    out = np.clip(out, 0.75 * rt, 1.2 * rt + 2)               # stay close to the polar fit
    mw = max(2, len(out) // 60)
    out = np.median(np.stack([np.roll(out, k) for k in range(-mw, mw + 1)]), axis=0)  # drop spikes
    return out


def _hull_radius(rt, th, cy, cx):
    """Convex hull of the outline, returned again as one radius per ray."""
    n = len(rt)
    pts = np.stack([rt * np.cos(th), rt * np.sin(th)], axis=1)
    P = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    hull = np.array(lo[:-1] + up[:-1])
    # densify the hull edges and bin them by angle (centre is inside the hull)
    dense = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        for t in np.linspace(0, 1, 40, endpoint=False):
            dense.append(a + (b - a) * t)
    dense = np.array(dense)
    ang = np.arctan2(dense[:, 1], dense[:, 0]) % (2 * np.pi)
    idx = (ang / (2 * np.pi / n)).astype(int) % n
    out = np.zeros(n, np.float32)
    for i, r in zip(idx, np.hypot(dense[:, 0], dense[:, 1])):
        out[i] = max(out[i], r)
    empty = out == 0
    if empty.any() and (~empty).any():
        xs = np.arange(n)
        out[empty] = np.interp(xs[empty], xs[~empty], out[~empty], period=n)
    return out


# ------------------------------------------------------------------- main
def segment(rgb, bias=0.0, refine=2, n_ang=96, lam=2.0, dmax=None, shadow_aware=True, snap=True,
            smooth_level=5, fg_frac=0.15, fg_k=3, logit_blur=2,
            hull=False, grow=1.0, cap_ratio=1.3):
    """rgb: HxWx3 uint8 window, click at the centre.
    Returns an (n_ang, 2) array of outline points in continuous pixel
    coordinates (x right, y down, pixel edges at integers), or None."""
    n_ang = int(max(24, n_ang))
    if dmax is None:  # more rays -> neighbouring rays are closer together -> smaller steps
        dmax = 4 if n_ang <= 120 else (3 if n_ang <= 240 else 2)
    h, w = rgb.shape[:2]
    R = min(h, w) // 2 - 1
    if R < 12:
        return None
    cy = cx = float(R + 1) if h == w else h / 2.0
    cy, cx = h / 2.0, w / 2.0
    F = features(rgb).reshape(-1, 4)
    yy, xx = np.mgrid[:h, :w]
    dy, dx = yy + 0.5 - cy, xx + 0.5 - cx
    rr = np.hypot(dy, dx)
    ang_idx = ((np.arctan2(dy, dx) % (2 * np.pi)) / (2 * np.pi / n_ang)).astype(int) % n_ang

    th = np.arange(n_ang) * 2 * np.pi / n_ang
    rad = np.arange(R)
    py = np.clip((cy + np.outer(np.sin(th), rad)).astype(int), 0, h - 1)
    px = np.clip((cx + np.outer(np.cos(th), rad)).astype(int), 0, w - 1)

    fg = rr <= max(3.0, fg_frac * R)
    ring = rr >= 0.88 * R
    bg = ring.copy()
    rt = None

    # cast shadows: clearly darker than anything at the click and flat (no leaf texture)
    Fl = F[:, 0].reshape(h, w)
    Ft = F[:, 3].reshape(h, w)
    thr_dark = np.percentile(Fl[fg], 10)
    thr_tex = np.median(Ft[fg])
    shadow = (Fl < thr_dark) & (Ft < thr_tex) & (rr > 0.15 * R) if shadow_aware else np.zeros_like(fg)

    for it in range(int(refine) + 1):
        fg_mix = _fit_mix(F[fg.ravel()], fg_k)
        bg_mix = _fit_mix(F[(bg | shadow).ravel()], 3)
        logit = _loglik(F, fg_mix) - _loglik(F, bg_mix) + bias
        logit = box_blur(np.clip(logit, -4, 4).reshape(h, w), logit_blur)
        S = np.cumsum(logit[py, px], axis=1)
        rt = _polar_dp(S, lam, dmax)
        if it == int(refine):
            break
        rtp = rt[ang_idx]
        new_fg = rr < 0.75 * rtp
        if new_fg.sum() >= 20:
            fg = new_fg
        new_bg = ((rr > 1.3 * rtp + 2) & (rr < np.minimum(2.2 * rtp + 8, R))) | ring
        if new_bg.sum() >= 50:
            bg = new_bg

    cap = None
    if cap_ratio:                                         # crowns are roughly round: no arm much longer than the typical radius
        cap = cap_ratio * float(np.median(rt))
        rt = np.minimum(rt, cap)
    if snap:
        rt = _snap_to_edge(logit, rr, ang_idx, rt, th, cy, cx, R)
        rounds = 1
    else:
        rounds = max(2, n_ang // 48)
    for _ in range(rounds):                               # light smoothing around the circle
        rt = (np.roll(rt, 1) + rt + np.roll(rt, -1)) / 3.0
    if cap is not None:
        rt = np.minimum(rt, cap)
    if hull:                                              # freehand look: fill the notches, keep it convex
        rt = _hull_radius(rt, th, cy, cx)
    if grow != 1.0:
        rt = rt * grow
    if smooth_level > 0:                                  # low-pass the outline (rounds sharp corners)
        keep = max(5, int(round(80 * 0.8 ** smooth_level)))
        spec = np.fft.rfft(rt)
        spec[keep + 1:] = 0
        rt = np.maximum(np.fft.irfft(spec, n=len(rt)), 1.0).astype(np.float32)
    if np.median(rt) < 2.5:
        return None
    return np.stack([cx + rt * np.cos(th), cy + rt * np.sin(th)], axis=1)
