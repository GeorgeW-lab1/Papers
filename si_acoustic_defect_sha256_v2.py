"""
si_acoustic_defect_sha256_v2.py -- faster, cheaper-to-build version of the acoustic defect-pad SHA-256 computer.
Changes vs v1:  (1) threshold-logic pads: majority / full-adder / XOR3 pads replace 9-NAND adders (about 4x fewer firings);
(2) carry-save adders cut logic depth;  (3) 2-period firing at lower Q (was 4);  (4) multi-mode pads: f0, 2f0, 3f0 resonator modes
carry 3 gates per firing (cross-mode nonlinearity kappa is an ASSUMPTION; a sweep is printed);  (5) micro-pad array
(100 um pitch, ~785k pads) with many hashes pipelined, all digests verified vs hashlib.
Run: python si_acoustic_defect_sha256_v2.py   (python ... --anim re-renders the GIF)
"""
import hashlib, pickle, sys, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ wafer / pads
WAFER_R, SITE, DX = 50.0, 8.0, 0.5            # mm
PROBES = [(0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2)]
NG = int(2 * WAFER_R / DX)                     # 200 x 200 cells

def make_wafer(seed=123):
    R = WAFER_R; rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(120)); th = 2 * np.pi * rng.random(120)
    tsd = np.column_stack((r * np.cos(th), r * np.sin(th)))
    s = rng.uniform(-R * .8, R * .8, 45)
    x = s * np.cos(np.pi / 4) + rng.normal(0, 2, 45); y = s * np.sin(np.pi / 4) + rng.normal(0, 8, 45)
    bpd = np.column_stack((x, y))[x**2 + y**2 <= R**2]
    sites = []; n = int(R // SITE) + 1
    for i in range(-n, n):
        for j in range(-n, n):
            x0, y0 = i * SITE, j * SITE
            if all(cx**2 + cy**2 <= (R - 2)**2 for cx, cy in
                   [(x0, y0), (x0 + SITE, y0), (x0, y0 + SITE), (x0 + SITE, y0 + SITE)]):
                sites.append((x0 + SITE / 2, y0 + SITE / 2, x0, y0))
    sites.sort(key=lambda q: (-q[3], q[2]))
    sites = np.array(sites)
    alld = np.vstack((tsd, bpd))
    ndef = np.array([int(((alld[:, 0] >= s_[2]) & (alld[:, 0] < s_[2] + SITE) &
                          (alld[:, 1] >= s_[3]) & (alld[:, 1] < s_[3] + SITE)).sum()) for s_ in sites])
    return sites, tsd, bpd, ndef

# ------------------------------------------------------------------ STAGE A: acoustic FDTD
def acoustic_blast(sites, tsd, bpd, nsteps=1400, nsnap=40, snap_stride=14, P=16):
    xs = (np.arange(NG) + .5) * DX - WAFER_R
    X, Y = np.meshgrid(xs, xs, indexing="xy"); Rr = np.hypot(X, Y)
    c2 = np.full((NG, NG), 0.25)                              # (c dt/dx)^2, CFL-stable
    ix = np.floor((X + WAFER_R) / SITE).astype(int); iy = np.floor((Y + WAFER_R) / SITE).astype(int)
    edge = ((X + WAFER_R) % SITE < DX) | ((Y + WAFER_R) % SITE < DX)
    c2[edge] *= 0.55                                          # patterned trenches between pads
    for p in np.vstack((tsd, bpd)):                           # defect scatterers (2x2 cells, soft)
        i = int((p[0] + WAFER_R) / DX); j = int((p[1] + WAFER_R) / DX)
        c2[max(j, 0):j + 2, max(i, 0):i + 2] *= 0.2
    g = np.where(Rr <= WAFER_R, 0.0015, 0.25)                 # material damping / absorbing surround
    ring = (Rr > WAFER_R - 4.5) & (Rr <= WAFER_R - 3.0)
    u = np.zeros((NG, NG)); up = np.zeros_like(u)
    pidx = lambda x, y: (int((y + WAFER_R) / DX), int((x + WAFER_R) / DX))
    probe_cells = [pidx(s_[0] + dx, s_[1] + dy) for s_ in sites for dx, dy in PROBES]
    pj = np.array([c[0] for c in probe_cells]); pi_ = np.array([c[1] for c in probe_cells])
    cj = np.array([pidx(s_[0], s_[1])[0] for s_ in sites]); ci = np.array([pidx(s_[0], s_[1])[1] for s_ in sites])
    rec = np.zeros((nsteps, len(pj)), np.float32); recpad = np.zeros((nsteps, len(sites)), np.float32); snaps = []
    ncyc = 3
    for n in range(nsteps):
        lap = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
        src = 0.0
        if n < ncyc * P:
            src = np.sin(2 * np.pi * n / P) * np.sin(np.pi * n / (ncyc * P))**2
        un = (2 * u - (1 - g / 2) * up + c2 * lap + 0.8 * src * ring) / (1 + g / 2)
        up, u = u, un
        rec[n] = u[pj, pi_]; recpad[n] = u[cj, ci]
        if n % snap_stride == 0 and len(snaps) < nsnap: snaps.append(u.astype(np.float32).copy())
    E = (rec.astype(np.float64)**2).sum(0)
    return E, np.array(snaps), recpad

def bits_to_message(bits):
    bits = bits[: len(bits) // 8 * 8]
    return bytes(int("".join(map(str, bits[k:k + 8])), 2) for k in range(0, len(bits), 8))


# ------------------------------------------------------------------ STAGE B (v2): threshold-logic pads, multi-mode resonators
OM_ALL = np.array([1.0, 2.0, 3.0])               # mode frequencies in units of the fundamental f0
Q, DT, NSTEP, NLOCK, FDRV = 2.0, 2 * np.pi / 24, 48, 24, 0.25     # 48 steps = 2 fundamental periods per firing
B_NOM, KAPPA, M_MODES = 0.05, 0.02, 3

def pad_params(ndef):
    ndef = np.asarray(ndef, float); return 1 - 0.03 * ndef, B_NOM * (1 + 0.15 * ndef)

def fire(s, w0, beta, M, kappa=None, trace=False):
    """One firing of G pads, each with M acoustic modes.  s: (G,M) weighted input drive per mode.
    RK4 of  x_m'' + (w_m/Q) x_m' + w_m^2 x_m + beta x_m^3 + kappa x_m sum_{n!=m} x_n^2 = F_m sin(Om_m t);
    amplitude of each mode read by lock-in over the last fundamental period."""
    kappa = KAPPA if kappa is None else kappa
    omd = OM_ALL[:M][None, :]; om = omd * w0[:, None]; om2 = om * om; gam = om / Q; b = beta[:, None]
    F = FDRV * s * (omd**2)
    x = np.zeros_like(F); v = np.zeros_like(F); I = np.zeros_like(F); Qd = np.zeros_like(F); tr = []
    def acc(t, x, v):
        cross = (x * x).sum(1, keepdims=True) - x * x
        return F * np.sin(omd * t) - gam * v - om2 * x - b * x**3 - kappa * x * cross
    h = DT
    for st in range(NSTEP):
        t = st * h
        k1x = v;               k1v = acc(t, x, v)
        k2x = v + .5*h*k1v;    k2v = acc(t + .5*h, x + .5*h*k1x, v + .5*h*k1v)
        k3x = v + .5*h*k2v;    k3v = acc(t + .5*h, x + .5*h*k2x, v + .5*h*k2v)
        k4x = v + h*k3v;       k4v = acc(t + h, x + h*k3x, v + h*k3v)
        x = x + h/6*(k1x + 2*k2x + 2*k3x + k4x); v = v + h/6*(k1v + 2*k2v + 2*k3v + k4v)
        if st >= NSTEP - NLOCK:
            tt = (st + 1) * h; I += x * np.cos(omd * tt); Qd += x * np.sin(omd * tt)
        if trace: tr.append(x[:, 0].copy())
    amp = (2.0 / NLOCK) * np.sqrt(I * I + Qd * Qd)
    return (amp, np.array(tr)) if trace else amp

def calibrate(M, kappa=None):
    """Detector thresholds Th[m, theta]: midway between drive sums theta-1 and theta on a nominal pad
    (other modes held at mid drive 2)."""
    Th = np.zeros((M, 4))
    for m in range(M):
        s = np.full((6, M), 2.0); s[:, m] = np.arange(6)
        amp = fire(s, np.ones(6), np.full(6, B_NOM), M, kappa)[:, m]
        for th in (1, 2, 3): Th[m, th] = 0.5 * (amp[th - 1] + amp[th])
    return Th

def pad_test(M, Th, kappa=None, K=6, nmax=9, seed=1):
    """Self-test by defect count n: every drive sum 0..5 on every mode, other modes random, versus ideal [s >= theta]."""
    rng = np.random.default_rng(seed); rows = []; meta = []
    for n in range(nmax + 1):
        for m in range(M):
            for sv in range(6):
                for _ in range(K if M > 1 else 1):
                    r = rng.integers(0, 6, M).astype(float); r[m] = sv; rows.append(r); meta.append((n, m, sv))
    meta = np.array(meta); S = np.array(rows); w0, be = pad_params(meta[:, 0])
    amp = fire(S, w0, be, M, kappa)[np.arange(len(S)), meta[:, 1]]
    good = np.ones(len(S), bool)
    for th in (1, 2, 3): good &= ((amp >= Th[meta[:, 1], th]) == (meta[:, 2] >= th))
    return np.array([good[meta[:, 0] == n].all() for n in range(nmax + 1)])

# ------------------------------------------------------------------ threshold-logic netlist for SHA-256
MASK = 0xFFFFFFFF
K = [int(x, 16) for x in """428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5 d807aa98 12835b01 243185be 550c7dc3
72be5d74 80deb1fe 9bdc06a7 c19bf174 e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da
983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967 27b70a85 2e1b2138 4d2c6dfc 53380d13
650a7354 766a0abb 81c2c92e 92722c85 a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070
19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3 748f82ee 78a5636f 84c87814 8cc70208
90befffa a4506ceb bef9a3f7 c67178f2""".split()]
H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

class Net:
    """Wires 0/1 are constants.  A gate = one pad firing: out = [sum_i w_i b_i >= theta] (weights 0..2, <=4 inputs).
    Complementary outputs come free from the differential detector (neg())."""
    def __init__(s): s.lvl = [0, 0]; s.gi = []; s.gw = []; s.gth = []; s.gout = []; s.alias = []; s.negmap = {0: 1, 1: 0}
    def inp(s): s.lvl.append(0); return len(s.lvl) - 1
    def neg(s, w):
        if w in s.negmap: return s.negmap[w]
        nw = len(s.lvl); s.lvl.append(s.lvl[w]); s.alias.append((nw, w)); s.negmap[nw] = w; s.negmap[w] = nw; return nw
    def th(s, ins, theta):
        ins = list(ins) + [(0, 0)] * (4 - len(ins)); w = len(s.lvl)
        s.lvl.append(1 + max(s.lvl[i] for i, _ in ins)); s.gi.append([i for i, _ in ins]); s.gw.append([k for _, k in ins])
        s.gth.append(theta); s.gout.append(w); return w
    def and2(s, a, b): return s.th([(a, 1), (b, 1)], 2)
    def or2(s, a, b): return s.th([(a, 1), (b, 1)], 1)
    def maj(s, a, b, c): return s.th([(a, 1), (b, 1), (c, 1)], 2)
    def fa(s, a, b, c):
        co = s.maj(a, b, c); sm = s.th([(a, 1), (b, 1), (c, 1), (s.neg(co), 2)], 3); return sm, co
    def xor3w(s, A, B, C): return [s.fa(A[i], B[i], C[i])[0] for i in range(32)]
    def chw(s, E, F_, G): return [s.or2(s.and2(E[i], F_[i]), s.and2(s.neg(E[i]), G[i])) for i in range(32)]
    def majw(s, A, B, C): return [s.maj(A[i], B[i], C[i]) for i in range(32)]
    def csa(s, A, B, C):
        out = [s.fa(A[i], B[i], C[i]) for i in range(32)]; return [o[0] for o in out], [0] + [o[1] for o in out][:31]
    def add(s, A, B):
        c = 0; o = []
        for i in range(32):
            sm, c = s.fa(A[i], B[i], c); o.append(sm)
        return o

def rotr(w, n): return [w[(i + n) % 32] for i in range(32)]
def shr(w, n): return [w[i + n] if i + n < 32 else 0 for i in range(32)]
def cword(v): return [(v >> i) & 1 for i in range(32)]

def build_sha_net(nbits=440):
    net = Net(); inw = [net.inp() for _ in range(nbits)]
    blk = inw + [1] + [0] * (512 - 64 - nbits - 1) + [(nbits >> (63 - k)) & 1 for k in range(64)]
    W = [[blk[32 * t + 31 - i] for i in range(32)] for t in range(16)]
    a, b, c, d, e, f, g, h = [cword(v) for v in H0]; rend, stw = [], []
    for t in range(64):
        if t >= 16:
            w15, w2 = W[t - 15], W[t - 2]
            s0 = net.xor3w(rotr(w15, 7), rotr(w15, 18), shr(w15, 3)); s1 = net.xor3w(rotr(w2, 17), rotr(w2, 19), shr(w2, 10))
            p, q = net.csa(s0, W[t - 16], W[t - 7]); p, q = net.csa(p, q, s1); W.append(net.add(p, q))
        S1 = net.xor3w(rotr(e, 6), rotr(e, 11), rotr(e, 25)); chv = net.chw(e, f, g)
        p, q = net.csa(h, S1, chv); p, q = net.csa(p, q, cword(K[t])); p, q = net.csa(p, q, W[t]); t1 = net.add(p, q)
        S0 = net.xor3w(rotr(a, 2), rotr(a, 13), rotr(a, 22)); t2 = net.add(S0, net.majw(a, b, c))
        en = net.add(d, t1); an = net.add(t1, t2)
        h, g, f, e, d, c, b, a = g, f, e, en, c, b, a, an
        rend.append(len(net.gout) - 1); stw.append([a, b, c, d, e, f, g, h])
    dig = [net.add(cword(H0[i]), w) for i, w in enumerate([a, b, c, d, e, f, g, h])]
    return net, inw, rend, stw, dig

def word_val(val, w): return sum(int(val[wi]) << i for i, wi in enumerate(w))

def run_net(net, inw, bits_list, pad_ok, pad_w0, pad_be, Th, M, log=True):
    B = len(bits_list); nw = len(net.lvl); val = np.zeros((B, nw), np.int8); val[:, 1] = 1
    for b, bits in enumerate(bits_list): val[b, inw] = bits
    gi = np.array(net.gi); gw = np.array(net.gw, float); gth = np.array(net.gth); go = np.array(net.gout)
    lv = np.array([net.lvl[w] for w in go]); order = np.argsort(lv, kind="stable"); lvs = lv[order]; maxL = int(lvs.max())
    bounds = np.searchsorted(lvs, np.arange(1, maxL + 2))
    al = {}
    for w, src in net.alias: al.setdefault(net.lvl[w], []).append((w, src))
    al = {L: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for L, v in al.items()}
    def do_alias(L):
        if L in al: val[:, al[L][0]] = 1 - val[:, al[L][1]]
    do_alias(0); pad_of = np.zeros(len(go), int); off = 0; t0 = time.time()
    for L in range(1, maxL + 1):
        idx = order[bounds[L - 1]:bounds[L]]
        if len(idx):
            n = len(idx); tot = B * n
            sdr = (val[:, gi[idx]] * gw[idx][None]).sum(-1).reshape(-1)               # (B*n,) weighted drive
            ng = -(-tot // M); pads = pad_ok[(off + np.arange(ng)) % len(pad_ok)]; off += ng
            sp = np.zeros(ng * M); sp[:tot] = sdr
            amp = fire(sp.reshape(ng, M), pad_w0[pads], pad_be[pads], M).reshape(-1)[:tot]
            thr = Th[np.arange(tot) % M, np.tile(gth[idx], B)]
            val[:, go[idx]] = (amp >= thr).reshape(B, n).astype(np.int8)
            pad_of[idx] = pads[np.arange(n) // M]
        do_alias(L)
        if log and L % 500 == 0: print(f"  level {L}/{maxL} ({time.time()-t0:.0f}s)", flush=True)
    return val, pad_of, lv

def slots(lv, P, M, B=1):
    w = np.bincount(lv)[1:]; return float(np.ceil(B * w / (P * M)).sum())

def pad_traces(ndef, Th):
    out = np.zeros((len(ndef), 4, NSTEP)); w0, be = pad_params(ndef)
    for s in range(4):
        _, tr = fire(np.full((len(ndef), 1), float(s)), w0, be, 1, trace=True); out[:, s, :] = tr.T
    return out

# ------------------------------------------------------------------ animation
def animate(sites, tsd, bpd, ndef, E, bits, snaps, recpad, msg, ref, ok, traces, Th, net_res, M):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Circle, Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    val, pad_of, lv, round_end, state_w, dig, ngates = net_res
    NS = len(sites); med = np.median(E)
    Epad = E.reshape(NS, 5).mean(1); Epn = (Epad - Epad.min()) / (Epad.max() - Epad.min())
    amp = np.percentile(np.abs(snaps), 99.7)
    wavecm = matplotlib.colormaps["seismic"]; heatcm = matplotlib.colormaps["inferno"]; viri = matplotlib.colormaps["viridis"]
    xs = (np.arange(NG) + .5) * DX - WAFER_R; Xg, Yg = np.meshgrid(xs, xs); inside = np.hypot(Xg, Yg) <= WAFER_R
    block = msg + b"\x80" + b"\x00" * (55 - len(msg)) + (8 * len(msg)).to_bytes(8, "big")
    block_bits = np.array([(by >> (7 - k)) & 1 for by in block for k in range(8)])
    digest = [word_val(val[0], w) for w in dig]; dhex = "".join(f"{w:08x}" for w in digest)
    NB, NR, NSF, NC, ND = len(snaps), 8, 8, 48, 12
    F1, F2, F3, F4 = NB, NB + NR, NB + NR + NSF, NB + NR + NSF + NC; NF = F4 + ND
    maxL = int(lv.max()); Lf = np.ceil(maxL * np.arange(NC + 1) / NC).astype(int)
    ids = np.arange(len(lv)); permin = np.full(maxL + 2, len(lv)); np.minimum.at(permin, lv, ids)
    minid = np.minimum.accumulate(permin[::-1])[::-1]; ranks = np.array(round_end)
    heat_f = []; ng_f = []
    for c in range(NC):
        sel = (lv > Lf[c]) & (lv <= Lf[c + 1]); heat_f.append(np.bincount(pad_of[sel], minlength=NS)); ng_f.append(int(sel.sum()))
    cum_f = np.cumsum(ng_f); rdone_f = [int(np.searchsorted(ranks, minid[min(Lf[c + 1] + 1, maxL + 1)], side="left")) for c in range(NC)]
    t_ax = np.arange(NSTEP) * DT / (2 * np.pi)
    fig = plt.figure(figsize=(12.5, 7.8))
    axw = fig.add_axes([0.02, 0.28, 0.42, 0.63]); axr = fig.add_axes([0.47, 0.69, 0.51, 0.22])
    axm = fig.add_axes([0.52, 0.47, 0.46, 0.14]); axt = fig.add_axes([0.47, 0.265, 0.51, 0.17]); axt.axis("off")
    title = fig.text(0.5, 0.955, "", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.008, "v2: acoustic stage = real 2-D wave FDTD; each pad firing = RK4 of a %d-mode defect-set resonator with lock-in readout; "
             "constants assumed; signals regenerated/routed by a software scheduler." % M, fontsize=7, color="#555")
    axw.add_patch(Circle((0, 0), WAFER_R, fill=False, lw=3, color="#2c3e50", zorder=6)); patches = []
    for k, (sx, sy, x0, y0) in enumerate(sites):
        p = Rectangle((x0, y0), SITE, SITE, fc="#f2f3f4", ec="#566573", lw=1.2, zorder=1); axw.add_patch(p); patches.append(p)
        axw.text(sx, sy, str(ndef[k]), ha="center", va="center", fontsize=6, color="#2c3e50", zorder=7)
    axw.scatter(*tsd.T, marker="x", c="#c0392b", s=9, lw=.8, zorder=5); axw.scatter(*bpd.T, marker="s", c="#e67e22", s=7, zorder=5)
    pr = np.array([[s_[0] + dx, s_[1] + dy] for s_ in sites for dx, dy in PROBES])
    probe_sc = axw.scatter(pr[:, 0], pr[:, 1], s=3, c="#2c3e50", zorder=8, visible=False)
    im = axw.imshow(np.zeros((NG, NG, 4)), extent=[-WAFER_R, WAFER_R, -WAFER_R, WAFER_R], origin="lower", zorder=3, interpolation="bilinear")
    axw.set_xlim(-WAFER_R * 1.05, WAFER_R * 1.05); axw.set_ylim(-WAFER_R * 1.05, WAFER_R * 1.05); axw.set_aspect("equal"); axw.set_xticks([]); axw.set_yticks([])
    axw.set_title("Patterned Si wafer: pads (digit = defects in pad), TSD x, BPD \u25a0", fontsize=10)
    regcmap = ListedColormap(["#ecf0f1", "#ffffff", "#00bcd4", "#e8daef", "#8e44ad"])
    reg = axr.imshow(np.zeros((16, 32)), cmap=regcmap, vmin=0, vmax=4, aspect="auto")
    axr.set_xticks(np.arange(-.5, 32, 1), minor=True); axr.set_yticks(np.arange(-.5, 16, 1), minor=True)
    axr.grid(which="minor", color="#aab7b8", lw=.4); axr.tick_params(which="both", length=0)
    axr.set_xticks([0, 31]); axr.set_xticklabels(["b31", "b0"], fontsize=7); axr.set_yticks([0, 15]); axr.set_yticklabels(["W0", "W15"], fontsize=7)
    axr.set_title("SHA-256 512-bit input block (16 words x 32 bits)", fontsize=9)
    handles = [
        Patch(fc="#f2f3f4", ec="#566573", label="Pad (idle); digit = defects inside"),
        Line2D([], [], marker="x", ls="", c="#c0392b", label="TSD defect"), Line2D([], [], marker="s", ls="", c="#e67e22", label="BPD defect"),
        Patch(fc="#d62728", alpha=.7, label="Acoustic wave: +u"), Patch(fc="#1f3fbf", alpha=.7, label="Acoustic wave: -u"),
        Line2D([], [], marker=".", ls="", c="#2c3e50", label="Probe points (5 per pad)"),
        Patch(fc=viri(.8), label="Pad coloured by received acoustic energy"),
        Patch(fc="#2ecc71", ec="#1e8449", label="Pad passes threshold-logic self-test"),
        Patch(fc="#bdc3c7", ec="#7f8c8d", hatch="///", label="Pad masked (defects break logic)"),
        Patch(fc=heatcm(.85), label="Pad firing (brighter = more gates)"),
        Line2D([], [], c="#2980b9", label="Pad mode-1 x(t): drive sum 0"), Line2D([], [], c="#27ae60", label="drive sum 1"),
        Line2D([], [], c="#e67e22", label="drive sum 2"), Line2D([], [], c="#c0392b", label="drive sum 3"),
        Line2D([], [], c="k", ls="--", label="detector thresholds (theta=1,2,3)"),
        Patch(fc="#ecf0f1", ec="#aab7b8", label="Input bit: empty"), Patch(fc="#ffffff", ec="#aab7b8", label="Input bit = 0"),
        Patch(fc="#00bcd4", ec="#aab7b8", label="Input bit = 1"), Patch(fc="#8e44ad", ec="#aab7b8", label="Padding bit = 1 (light = 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.8, title="Legend", title_fontsize=7.5, bbox_to_anchor=(0.5, 0.022), frameon=True)
    order_def = np.argsort(-ndef); tcol = ["#2980b9", "#27ae60", "#e67e22", "#c0392b"]
    wave_pads = [int(np.argmin(ndef)), int(np.argsort(ndef)[NS // 2]), int(np.argmax(ndef))]

    def th_lines(ax):
        for th in (1, 2, 3): ax.axhline(Th[0, th], c="k", ls="--", lw=.8)

    def update(f):
        axm.clear(); text = []; nbits = 0; im.set_visible(False); probe_sc.set_visible(False)
        for k, p in enumerate(patches): p.set_facecolor("#f2f3f4"); p.set_hatch(None)
        if f < F1:
            u = np.where(inside, snaps[f], 0) / amp; rgba = wavecm(0.5 + 0.5 * np.clip(u, -1, 1)); rgba[..., 3] = np.clip(np.abs(u) * 1.6, 0, .9)
            im.set_data(rgba); im.set_visible(True); title.set_text("1. Blast the patterned wafer with sound (ring transducer, real wave FDTD)")
            tmax = len(recpad); tcur = f * 14
            for wp, c in zip(wave_pads, ["#2980b9", "#27ae60", "#c0392b"]): axm.plot(recpad[:, wp], c=c, lw=.8, label=f"pad {wp}: {ndef[wp]} defects")
            axm.axvline(tcur, c="k", lw=1); axm.set_xlim(0, tmax); axm.legend(fontsize=6, loc="upper right")
            axm.set_title("displacement u(t) at pad centres (x = time step); defects scatter and reshape the echo", fontsize=8); axm.tick_params(labelsize=7)
            text = [f"time step {tcur}/{tmax}   tone burst 3 cycles, wavelength 8 cells = 4 mm", "scalar wave eq.  u_tt = c^2 lap(u) - damping;  trenches slow c, defects scatter strongly"]
        elif f < F2:
            n = f - F1 + 1; nbits = min(440, 55 * n); npads = min(NS, int(np.ceil(NS * n / NR)))
            for k in range(NS): patches[k].set_facecolor(viri(Epn[k]) if k < npads else "#f2f3f4")
            probe_sc.set_visible(True); title.set_text("2. Pads listen: received acoustic energy at 5 probes per pad -> bits")
            order = np.argsort(E); axm.scatter(range(len(E)), E[order], s=5, c=np.where(bits[order] == 1, "#00bcd4", "#7f8c8d"))
            axm.axhline(med, c="k", ls="--", lw=1); axm.set_title("probe energies (sorted): above median -> 1", fontsize=8); axm.tick_params(labelsize=7)
            text = [f"bits so far: {nbits}/440   ones: {bits[:max(nbits,1)].mean()*100:.1f}%", "E = sum_t u(t)^2 at each probe;  bit = [E >= median]"]
        elif f < F3:
            nbits = 440; n = f - F2 + 1; nshow = min(NS, int(np.ceil(NS * n / NSF)))
            for k in range(NS):
                if k < nshow: patches[k].set_facecolor("#2ecc71" if ok[k] else "#bdc3c7"); patches[k].set_hatch(None if ok[k] else "///")
            pp = order_def[(n - 1) % NS] if n < NSF else order_def[0]
            for c in range(4): axm.plot(t_ax, traces[pp, c], c=tcol[c], lw=1.1)
            th_lines(axm); axm.tick_params(labelsize=7); axm.set_xlabel("time (fundamental periods)", fontsize=7)
            axm.set_title(f"self-test pad {pp} ({ndef[pp]} defects): mode-1 x(t) for drive sums 0-3; lock-in amp vs thresholds", fontsize=8)
            title.set_text("3. Each defect pad self-tests as a multi-mode threshold-logic gate")
            text = [f"pads usable: {ok.sum()}/{NS}   masked: {(~ok).sum()}", f"each pad carries {M} acoustic modes (f0, 2f0, 3f0) = {M} gates per firing",
                    "out = [lock-in amplitude >= theta threshold]; full adder = 2 pads (carry, sum)"]
        elif f < F4:
            nbits = 440; c = f - F3
            for k in range(NS):
                if ok[k]: patches[k].set_facecolor(heatcm(min(1, heat_f[c][k] / max(1, heat_f[c].max()) * .95)))
                else: patches[k].set_facecolor("#bdc3c7"); patches[k].set_hatch("///")
            pp = [p for p in np.argsort(-heat_f[c]) if ok[p]][0]
            for cc in range(4): axm.plot(t_ax, traces[pp, cc], c=tcol[cc], lw=1.1)
            th_lines(axm); axm.tick_params(labelsize=7)
            axm.set_title(f"busiest pad now: pad {pp} ({ndef[pp]} defects), fired {heat_f[c][pp]}x in this slice", fontsize=8)
            r = rdone_f[c]; title.set_text("4. Defect pads fire gates: SHA-256 compression, 64 rounds")
            text = [f"gate firings: {int(cum_f[c]):,}/{ngates:,}  ({M} gates per pad firing)", f"rounds completed: {r:2d}/64   " + "#" * (r // 2) + "." * (32 - r // 2)]
            if r >= 1:
                st = [word_val(val[0], w) for w in state_w[r - 1]]
                text += [f"a={st[0]:08x} b={st[1]:08x} c={st[2]:08x} d={st[3]:08x}", f"e={st[4]:08x} f={st[5]:08x} g={st[6]:08x} h={st[7]:08x}  (read from pads)"]
        else:
            nbits = 440; nw = min(8, f - F4 + 1)
            for k in range(NS):
                if ok[k]: patches[k].set_facecolor("#2ecc71")
                else: patches[k].set_facecolor("#bdc3c7"); patches[k].set_hatch("///")
            axm.axis("off"); title.set_text("5. Digest read from pad outputs - checked against hashlib")
            text = [" ".join(f"{digest[i]:08x}" if i < nw else "........" for i in range(4)), " ".join(f"{digest[i]:08x}" if i < nw else "........" for i in range(4, 8))]
            if nw == 8: text.append("pads == hashlib:  " + ("PASS" if dhex == ref else "FAIL"))
        data = np.zeros(512, dtype=int); data[:nbits] = 1 + bits[:nbits]
        if f >= F3: data[440:] = 3 + block_bits[440:]
        reg.set_data(data.reshape(16, 32)); axt.clear(); axt.axis("off")
        axt.text(0.0, 1.0, "\n".join(text), va="top", ha="left", family="monospace", fontsize=8.3); return []
    anim = FuncAnimation(fig, update, frames=NF, interval=90, blit=False)
    anim.save("si_acoustic_sha256_v2.gif", writer=PillowWriter(fps=11), dpi=62)
    update(8); fig.savefig("v2_blast.png", dpi=80); update(F3 + 24); fig.savefig("v2_compute.png", dpi=80); update(NF - 1); fig.savefig("v2_digest.png", dpi=80)
    print(f"saved si_acoustic_sha256_v2.gif ({NF} frames)", flush=True)

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    M = M_MODES; sites, tsd, bpd, ndef = make_wafer()
    if "--anim" in sys.argv:
        D = pickle.load(open("cache_v2.pkl", "rb")); animate(sites, tsd, bpd, ndef, *D); sys.exit()
    print(f"pads {len(sites)}, defects/pad max {ndef.max()} mean {ndef.mean():.2f}", flush=True)
    # ---- acoustic inputs: 16 messages (one per blast tone period); message 0 keeps the animation snapshots
    E0, snaps, recpad = acoustic_blast(sites, tsd, bpd, P=16)
    bits0 = (E0 >= np.median(E0)).astype(int); msgs = [bits_to_message(bits0)]; bitlist = [bits0]
    for P in [10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        Ep, _, _ = acoustic_blast(sites, tsd, bpd, nsteps=1200, nsnap=0, P=P)
        b = (Ep >= np.median(Ep)).astype(int); bitlist.append(b); msgs.append(bits_to_message(b))
    refs = [hashlib.sha256(m).hexdigest() for m in msgs]
    print(f"16 acoustic messages ready; message 0: {msgs[0].hex()}\nhashlib: {refs[0]}", flush=True)
    # ---- pad physics: thresholds, defect self-test, crosstalk tolerance
    Th = calibrate(M); ok_n = pad_test(M, Th); ok1 = pad_test(1, calibrate(1))
    print(f"self-test by defect count 0..9 (M={M} modes): {ok_n.astype(int).tolist()}   single-mode: {ok1.astype(int).tolist()}", flush=True)
    print("crosstalk (cross-mode nonlinearity kappa) sweep, nominal-pad pass:")
    for kp in (0.02, 0.1, 0.3, 1.0, 3.0):
        T_ = calibrate(M, kp); print(f"  kappa={kp:<5} nominal pad logic correct in all modes: {bool(pad_test(M, T_, kp)[0])}", flush=True)
    ok = ok_n[ndef]; w0w, bew = pad_params(ndef)
    net, inw, rend, stw, dig = build_sha_net(440); ng = len(net.gout); depth = int(max(net.lvl))
    print(f"netlist: {ng:,} pad firings (was 250,624 NANDs), depth {depth} (was 4,949)", flush=True)
    # ---- run 1: the 88-pad wafer, multi-mode
    val, pad_of, lv = run_net(net, inw, [bits0], np.where(ok)[0], w0w, bew, Th, M)
    d0 = "".join(f"{word_val(val[0], w):08x}" for w in dig)
    print(f"[wafer, 88 pads, {M} modes] pad digest {d0}\n  hashlib       {refs[0]}  -> {'PASS' if d0 == refs[0] else 'FAIL'}", flush=True)
    # ---- run 2: micro-pad array (100 um pitch), 16 hashes pipelined together
    Pm = int(np.pi * WAFER_R**2 / 0.1**2); rng = np.random.default_rng(5); ndm = np.zeros(Pm, int)
    for k in rng.integers(0, Pm, len(tsd) + len(bpd)): ndm[k] += 1
    okm = ok_n[ndm]; w0m, bem = pad_params(ndm)
    print(f"micro-pad array: {Pm:,} pads, {(~okm).sum()} masked, defect counts present {sorted(set(ndm.tolist()))}", flush=True)
    valm, _, lvm = run_net(net, inw, bitlist, np.where(okm)[0], w0m, bem, Th, M)
    dm = ["".join(f"{word_val(valm[b], w):08x}" for w in dig) for b in range(16)]
    npass = sum(dm[b] == refs[b] for b in range(16)); print(f"[micro-pad array, 16 pipelined hashes] {npass}/16 digests match hashlib", flush=True)
    # ---- speed table (f0 = fundamental; gate time = 2 periods)
    tg = NSTEP * DT / (2 * np.pi)
    print(f"\nSPEED  (one firing = {tg:.0f} fundamental periods;  {ng:,} firings/hash, depth {depth})")
    print("  config                                   f0      latency/hash      throughput")
    cfgs = [("v1 baseline (NAND, 82 pads, 1 mode)", 82, 1, 250624, None)]
    for name, P, MM in (("v2 same wafer 82 pads, 1 mode", 82, 1), (f"v2 same wafer 82 pads, {M} modes", 82, M),
                        (f"v2 micro-pads {Pm:,}, 1 mode", Pm, 1), (f"v2 micro-pads {Pm:,}, {M} modes", Pm, M)):
        for f0 in (1e9, 10e9):
            lat = slots(lv, P, MM) * tg / f0; thr = P * MM * f0 / (ng * tg)
            print(f"  {name:<40} {f0/1e9:.0f} GHz  {lat*1e6:10.2f} us  {thr:14,.0f} H/s")
    print("  v1 baseline (NAND, 82 pads, 1 mode)      1 GHz       23.50 us        80,303 H/s  (earlier result)")
    tr = pad_traces(ndef, Th)
    D = (E0, bits0, snaps, recpad, msgs[0], refs[0], ok, tr, Th, (val, pad_of, lv, rend, stw, dig, ng), M)
    pickle.dump(D, open("cache_v2.pkl", "wb")); animate(sites, tsd, bpd, ndef, *D)
