"""
si_acoustic_defect_sha256.py
Patterned silicon wafer, blasted with SOUND (not IR).  Defect pads compute SHA-256 as acoustic NAND gates.

STAGE A  acoustic blast (real 2-D wave-equation FDTD)
    - patterned wafer: 8 mm pads separated by slow trenches; TSD/BPD defects are local scatterers
    - an annular transducer rings a tone burst inward; probes on every pad record the displacement u(t)
    - probe energy E = sum u^2;  bit = 1 if E >= median  ->  440 bits = 55 bytes = one SHA-256 block
STAGE B  defect pads compute (real nonlinear-resonator dynamics, one pad = one NAND gate)
    - each pad is a Duffing resonator whose stiffness shift and cubic nonlinearity come from ITS OWN defect count
        x'' + (w0/Q) x' + w0^2 x + beta x^3 = F (b1 + b2) sin(w t)       (RK4 integrated for every single gate firing)
    - b1,b2 in {0,1} are the two input acoustic bursts; output bit = [peak amplitude < THR]  = NAND
    - THR is set on a nominal defect-free pad.  Every pad self-tests all 4 input combos; pads whose defects break
      NAND are masked out.  Output bursts are regenerated at full amplitude (transducer/amplifier per pad).
    - The complete SHA-256 compression (about 250k NANDs, level-scheduled) is executed by firing pads,
      each gate evaluated by integrating the pad ODE.  Digest is read from pad outputs and checked vs hashlib.
Honest limits: scalar acoustic model, assumed constants, per-gate signal regeneration, a software scheduler routes signals.

Run:  python si_acoustic_defect_sha256.py          (~10 min)      python ... --anim   (re-render GIF from cache)
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
def acoustic_blast(sites, tsd, bpd, nsteps=1400, nsnap=40, snap_stride=14):
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
    P = 16; ncyc = 3
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

# ------------------------------------------------------------------ STAGE B: defect-pad NAND (Duffing resonator)
Q, WD, DT, NSTEP, FDRV = 4.0, 1.0, 0.4, 64, 0.25
W0_NOM, B_NOM = 1.0, 0.30
def pad_params(ndef):
    return W0_NOM * (1 - 0.03 * ndef), B_NOM * (1 + 0.15 * ndef)

def nand_sim(b1, b2, w0, beta, trace=False):
    """RK4 integration of the pad resonator for every gate in the batch; returns peak amplitude (and |x| trace)."""
    F = FDRV * (b1 + b2).astype(float)
    x = np.zeros_like(F); v = np.zeros_like(F); peak = np.zeros_like(F); tr = []
    w02 = w0 * w0; gam = w0 / Q
    def acc(t, x, v): return F * np.sin(WD * t) - gam * v - w02 * x - beta * x**3
    for s in range(NSTEP):
        t = s * DT; h = DT
        k1x = v;                 k1v = acc(t, x, v)
        k2x = v + .5*h*k1v;      k2v = acc(t + .5*h, x + .5*h*k1x, v + .5*h*k1v)
        k3x = v + .5*h*k2v;      k3v = acc(t + .5*h, x + .5*h*k2x, v + .5*h*k2v)
        k4x = v + h*k3v;         k4v = acc(t + h, x + h*k3x, v + h*k3v)
        x = x + h/6*(k1x + 2*k2x + 2*k3x + k4x); v = v + h/6*(k1v + 2*k2v + 2*k3v + k4v)
        if s >= NSTEP - 16: peak = np.maximum(peak, np.abs(x))
        if trace: tr.append(np.abs(x))
    return (peak, np.array(tr)) if trace else peak

_p1 = nand_sim(np.array([1]), np.array([0]), np.array([W0_NOM]), np.array([B_NOM]))[0]
_p2 = nand_sim(np.array([1]), np.array([1]), np.array([W0_NOM]), np.array([B_NOM]))[0]
THR = 0.5 * (_p1 + _p2)

def pad_selftest(ndef):
    w0, be = pad_params(ndef); ok = np.ones(len(ndef), bool)
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]; traces = np.zeros((len(ndef), 4, NSTEP)); outs = np.zeros((len(ndef), 4), int)
    for c, (a, b) in enumerate(combos):
        pk, tr = nand_sim(np.full(len(ndef), a), np.full(len(ndef), b), w0, be, trace=True)
        out = (pk < THR).astype(int); outs[:, c] = out; traces[:, c, :] = tr.T
        ok &= (out == (1 - (a & b)))
    return ok, traces, outs

# ------------------------------------------------------------------ NAND netlist for SHA-256
MASK = 0xFFFFFFFF
K = [int(x, 16) for x in """428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5 d807aa98 12835b01 243185be 550c7dc3
72be5d74 80deb1fe 9bdc06a7 c19bf174 e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da
983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967 27b70a85 2e1b2138 4d2c6dfc 53380d13
650a7354 766a0abb 81c2c92e 92722c85 a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070
19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3 748f82ee 78a5636f 84c87814 8cc70208
90befffa a4506ceb bef9a3f7 c67178f2""".split()]
H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

class Net:
    def __init__(s): s.lvl = [0, 0]; s.ga = []; s.gb = []; s.gout = []     # wire 0 = const 0, wire 1 = const 1
    def inp(s): s.lvl.append(0); return len(s.lvl) - 1
    def nand(s, a, b):
        w = len(s.lvl); s.lvl.append(1 + max(s.lvl[a], s.lvl[b]))
        s.ga.append(a); s.gb.append(b); s.gout.append(w); return w
    def NOT(s, a): return s.nand(a, a)
    def xor(s, a, b):
        n1 = s.nand(a, b); return s.nand(s.nand(a, n1), s.nand(b, n1))
    def fa(s, a, b, c):
        n1 = s.nand(a, b); x1 = s.nand(s.nand(a, n1), s.nand(b, n1))
        m1 = s.nand(x1, c); sm = s.nand(s.nand(x1, m1), s.nand(c, m1)); return sm, s.nand(n1, m1)
    def add(s, A, B):
        c = 0; out = []
        for i in range(32):
            sm, c = s.fa(A[i], B[i], c); out.append(sm)
        return out
    def xor3(s, A, B, C): return [s.xor(s.xor(A[i], B[i]), C[i]) for i in range(32)]
    def ch(s, E, F_, G): return [s.nand(s.nand(E[i], F_[i]), s.nand(s.NOT(E[i]), G[i])) for i in range(32)]
    def maj(s, A, B, C):
        out = []
        for i in range(32):
            nab, nac, nbc = s.nand(A[i], B[i]), s.nand(A[i], C[i]), s.nand(B[i], C[i])
            u = s.nand(nab, nac); out.append(s.nand(s.nand(u, u), nbc))
        return out

def rotr(w, n): return [w[(i + n) % 32] for i in range(32)]
def shr(w, n): return [w[i + n] if i + n < 32 else 0 for i in range(32)]
def cword(v): return [(v >> i) & 1 for i in range(32)]            # constants map to wire 0 / wire 1

def build_sha_net(msg_bits_count=440):
    net = Net(); in_wires = [net.inp() for _ in range(msg_bits_count)]
    blk = in_wires + [1] + [0] * (512 - 64 - msg_bits_count - 1)
    blk += [(msg_bits_count >> (63 - k)) & 1 for k in range(64)]
    W = [[blk[32 * t + 31 - i] for i in range(32)] for t in range(16)]      # LSB first
    a, b, c, d, e, f, g, h = [cword(v) for v in H0]
    round_end, state_w = [], []
    for t in range(64):
        if t >= 16:
            w15, w2 = W[t - 15], W[t - 2]
            s0 = net.xor3(rotr(w15, 7), rotr(w15, 18), shr(w15, 3)); s1 = net.xor3(rotr(w2, 17), rotr(w2, 19), shr(w2, 10))
            W.append(net.add(net.add(W[t - 16], s0), net.add(W[t - 7], s1)))
        S1 = net.xor3(rotr(e, 6), rotr(e, 11), rotr(e, 25)); chv = net.ch(e, f, g)
        kw = net.add(cword(K[t]), W[t]); t1 = net.add(net.add(net.add(h, S1), chv), kw)
        S0 = net.xor3(rotr(a, 2), rotr(a, 13), rotr(a, 22)); t2 = net.add(S0, net.maj(a, b, c))
        e_new = net.add(d, t1); a_new = net.add(t1, t2)
        h, g, f, e, d, c, b, a = g, f, e, e_new, c, b, a, a_new
        round_end.append(len(net.gout) - 1); state_w.append([a, b, c, d, e, f, g, h])
    dig = [net.add(cword(H0[i]), w) for i, w in enumerate([a, b, c, d, e, f, g, h])]
    return net, in_wires, round_end, state_w, dig

def word_val(val, w): return sum(int(val[wi]) << i for i, wi in enumerate(w))

def run_pads(net, in_wires, in_bits, pad_ok, ndef, log=True):
    nw = len(net.lvl); val = np.zeros(nw, np.int8); val[1] = 1
    for w, bt in zip(in_wires, in_bits): val[w] = bt
    ga = np.array(net.ga); gb = np.array(net.gb); go = np.array(net.gout)
    lv = np.array([net.lvl[w] for w in go]); order = np.argsort(lv, kind="stable"); lvs = lv[order]
    bounds = np.searchsorted(lvs, np.arange(1, lvs.max() + 2))
    w0all, ball = pad_params(ndef); pad_of = np.zeros(len(go), int); off = 0; t0 = time.time()
    for L in range(lvs.max()):
        idx = order[bounds[L]:bounds[L + 1]]
        if len(idx) == 0: continue
        pads = pad_ok[(off + np.arange(len(idx))) % len(pad_ok)]; off += len(idx); pad_of[idx] = pads
        pk = nand_sim(val[ga[idx]], val[gb[idx]], w0all[pads], ball[pads])
        val[go[idx]] = (pk < THR).astype(np.int8)
        if log and L % 1000 == 0: print(f"  level {L}/{lvs.max()}  ({time.time()-t0:.0f}s)", flush=True)
    return val, pad_of, lv

# ------------------------------------------------------------------ animation
def animate(sites, tsd, bpd, ndef, E, bits, snaps, recpad, msg, ref, ok, traces, outs, net_res):
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
    digest = [word_val(val, w) for w in dig]; dhex = "".join(f"{w:08x}" for w in digest)

    NB, NR, NSF, NC, ND = len(snaps), 8, 8, 48, 12
    F1, F2, F3, F4 = NB, NB + NR, NB + NR + NSF, NB + NR + NSF + NC; NF = F4 + ND
    maxL = int(lv.max()); Lf = np.ceil(maxL * (np.arange(NC + 1)) / NC).astype(int)
    go_lv = lv; ids = np.arange(len(lv))
    minid = np.full(maxL + 2, len(lv)); permin = np.full(maxL + 2, len(lv))
    np.minimum.at(permin, go_lv, ids); minid = np.minimum.accumulate(permin[::-1])[::-1]   # min gate id with level >= L
    ranks = np.array(round_end)
    heat_f = []
    for c in range(NC):
        sel = (lv > Lf[c]) & (lv <= Lf[c + 1]); heat_f.append(np.bincount(pad_of[sel], minlength=NS))
    cum_f = np.cumsum([(lv <= Lf[c + 1]).sum() for c in range(NC)] and [((lv > Lf[c]) & (lv <= Lf[c + 1])).sum() for c in range(NC)])
    rdone_f = [int(np.searchsorted(ranks, minid[min(Lf[c + 1] + 1, maxL + 1)], side="left")) for c in range(NC)]
    t_ax = np.arange(NSTEP) * DT

    fig = plt.figure(figsize=(12.5, 7.8))
    axw = fig.add_axes([0.02, 0.28, 0.42, 0.63]); axr = fig.add_axes([0.47, 0.69, 0.51, 0.22])
    axm = fig.add_axes([0.52, 0.47, 0.46, 0.14]); axt = fig.add_axes([0.47, 0.265, 0.51, 0.17]); axt.axis("off")
    title = fig.text(0.5, 0.955, "", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.008, "Acoustic stage: real 2-D wave FDTD. Compute stage: every NAND is an RK4 integration of a defect-set Duffing pad; "
             "constants are assumptions; signals are regenerated and routed by a software scheduler.", fontsize=7, color="#555")
    axw.add_patch(Circle((0, 0), WAFER_R, fill=False, lw=3, color="#2c3e50", zorder=6))
    patches = []
    for k, (sx, sy, x0, y0) in enumerate(sites):
        p = Rectangle((x0, y0), SITE, SITE, fc="#f2f3f4", ec="#566573", lw=1.2, zorder=1); axw.add_patch(p); patches.append(p)
        axw.text(sx, sy, str(ndef[k]), ha="center", va="center", fontsize=6, color="#2c3e50", zorder=7)
    axw.scatter(*tsd.T, marker="x", c="#c0392b", s=9, lw=.8, zorder=5); axw.scatter(*bpd.T, marker="s", c="#e67e22", s=7, zorder=5)
    pr = np.array([[s_[0] + dx, s_[1] + dy] for s_ in sites for dx, dy in PROBES])
    probe_sc = axw.scatter(pr[:, 0], pr[:, 1], s=3, c="#2c3e50", zorder=8, visible=False)
    im = axw.imshow(np.zeros((NG, NG, 4)), extent=[-WAFER_R, WAFER_R, -WAFER_R, WAFER_R], origin="lower", zorder=3, interpolation="bilinear")
    axw.set_xlim(-WAFER_R * 1.05, WAFER_R * 1.05); axw.set_ylim(-WAFER_R * 1.05, WAFER_R * 1.05)
    axw.set_aspect("equal"); axw.set_xticks([]); axw.set_yticks([])
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
        Patch(fc="#2ecc71", ec="#1e8449", label="Pad passes NAND self-test"),
        Patch(fc="#bdc3c7", ec="#7f8c8d", hatch="///", label="Pad masked (defects break NAND)"),
        Patch(fc=heatcm(.85), label="Pad firing as NAND gate (brighter = more)"),
        Line2D([], [], c="#2980b9", label="Pad ODE |x|: inputs 00"), Line2D([], [], c="#27ae60", label="inputs 01"),
        Line2D([], [], c="#e67e22", label="inputs 10"), Line2D([], [], c="#c0392b", label="inputs 11"),
        Line2D([], [], c="k", ls="--", label="detector threshold"),
        Patch(fc="#ecf0f1", ec="#aab7b8", label="Input bit: empty"), Patch(fc="#ffffff", ec="#aab7b8", label="Input bit = 0"),
        Patch(fc="#00bcd4", ec="#aab7b8", label="Input bit = 1"), Patch(fc="#8e44ad", ec="#aab7b8", label="Padding bit = 1 (light = 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.8, title="Legend", title_fontsize=7.5,
               bbox_to_anchor=(0.5, 0.022), frameon=True)
    order_def = np.argsort(-ndef); tcol = ["#2980b9", "#27ae60", "#e67e22", "#c0392b"]
    wave_pads = [int(np.argmin(ndef)), int(np.argsort(ndef)[NS // 2]), int(np.argmax(ndef))]

    def update(f):
        axm.clear(); text = []; nbits = 0; im.set_visible(False); probe_sc.set_visible(False)
        for k, p in enumerate(patches): p.set_facecolor("#f2f3f4"); p.set_hatch(None); p.set_edgecolor("#566573")
        if f < F1:
            u = np.where(inside, snaps[f], 0) / amp; rgba = wavecm(0.5 + 0.5 * np.clip(u, -1, 1)); rgba[..., 3] = np.clip(np.abs(u) * 1.6, 0, .9)
            im.set_data(rgba); im.set_visible(True)
            title.set_text("1. Blast the patterned wafer with sound (ring transducer, real wave FDTD)")
            tmax = len(recpad); tcur = f * 14
            for wp, c in zip(wave_pads, ["#2980b9", "#27ae60", "#c0392b"]):
                axm.plot(recpad[:, wp], c=c, lw=.8, label=f"pad {wp}: {ndef[wp]} defects")
            axm.axvline(tcur, c="k", lw=1); axm.set_xlim(0, tmax); axm.legend(fontsize=6, loc="upper right")
            axm.set_title("displacement u(t) at pad centres (x = time step); defects scatter and reshape the echo", fontsize=8); axm.tick_params(labelsize=7)
            text = [f"time step {tcur}/{tmax}   tone burst 3 cycles, wavelength 8 cells = 4 mm",
                    "scalar wave eq.  u_tt = c^2 lap(u) - damping;  trenches slow c, defects scatter strongly"]
        elif f < F2:
            n = f - F1 + 1; nbits = min(440, 55 * n); npads = min(NS, int(np.ceil(NS * n / NR)))
            for k in range(NS): patches[k].set_facecolor(viri(Epn[k]) if k < npads else "#f2f3f4")
            probe_sc.set_visible(True)
            title.set_text("2. Pads listen: received acoustic energy at 5 probes per pad -> bits")
            order = np.argsort(E); axm.scatter(range(len(E)), E[order], s=5, c=np.where(bits[order] == 1, "#00bcd4", "#7f8c8d"))
            axm.axhline(med, c="k", ls="--", lw=1); axm.set_title("probe energies (sorted): above median -> 1", fontsize=8); axm.tick_params(labelsize=7)
            text = [f"bits so far: {nbits}/440   ones: {bits[:max(nbits,1)].mean()*100:.1f}%", "E = sum_t u(t)^2 at each probe;  bit = [E >= median]"]
        elif f < F3:
            nbits = 440; n = f - F2 + 1; nshow = min(NS, int(np.ceil(NS * n / NSF)))
            for k in range(NS):
                if k < nshow:
                    patches[k].set_facecolor("#2ecc71" if ok[k] else "#bdc3c7"); patches[k].set_hatch(None if ok[k] else "///")
            pp = order_def[(n - 1) % NS] if n < NSF else order_def[0]
            for c in range(4): axm.plot(t_ax, traces[pp, c], c=tcol[c], lw=1.1)
            axm.axhline(THR, c="k", ls="--", lw=1); axm.tick_params(labelsize=7)
            axm.set_title(f"self-test of pad {pp} ({ndef[pp]} defects): outputs for 00,01,10,11 = {outs[pp].tolist()}  (NAND = [1,1,1,0])", fontsize=8)
            title.set_text("3. Each defect pad self-tests as an acoustic NAND gate")
            text = [f"pads usable: {ok.sum()}/{NS}   masked: {(~ok).sum()}", f"pad ODE: x'' + (w0/Q)x' + w0^2 x + beta x^3 = F(b1+b2)sin(wt)",
                    "w0, beta set by the pad's defect count;  out = [peak|x| < THR]"]
        elif f < F4:
            nbits = 440; c = f - F3
            for k in range(NS):
                if ok[k]: patches[k].set_facecolor(heatcm(min(1, heat_f[c][k] / max(1, heat_f[c].max()) * .95)))
                else: patches[k].set_facecolor("#bdc3c7"); patches[k].set_hatch("///")
            pp = [p for p in np.argsort(-heat_f[c]) if ok[p]][0]
            for cc in range(4): axm.plot(t_ax, traces[pp, cc], c=tcol[cc], lw=1.1)
            axm.axhline(THR, c="k", ls="--", lw=1); axm.tick_params(labelsize=7)
            axm.set_title(f"busiest pad now: pad {pp} ({ndef[pp]} defects), fired {heat_f[c][pp]}x in this slice", fontsize=8)
            r = rdone_f[c]; title.set_text("4. Defect pads fire NAND gates: SHA-256 compression, 64 rounds")
            text = [f"gates evaluated: {int(cum_f[c]):,}/{ngates:,}   (each = one RK4 pad integration)",
                    f"rounds completed: {r:2d}/64   " + "#" * (r // 2) + "." * (32 - r // 2)]
            if r >= 1:
                st = [word_val(val, w) for w in state_w[r - 1]]
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
    anim.save("si_acoustic_sha256.gif", writer=PillowWriter(fps=11), dpi=62)
    update(8); fig.savefig("preview_blast.png", dpi=80); update(F3 + 24); fig.savefig("preview_compute.png", dpi=80)
    update(NF - 1); fig.savefig("preview_digest.png", dpi=80); print(f"saved si_acoustic_sha256.gif ({NF} frames)")

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    sites, tsd, bpd, ndef = make_wafer()
    if "--anim" in sys.argv:
        D = pickle.load(open("cache.pkl", "rb")); animate(sites, tsd, bpd, ndef, *D); sys.exit()
    print(f"pads {len(sites)}, defects/pad: max {ndef.max()}, mean {ndef.mean():.2f};  THR={THR:.3f} (p1={_p1:.3f}, p2={_p2:.3f})", flush=True)
    E, snaps, recpad = acoustic_blast(sites, tsd, bpd)
    bits = (E >= np.median(E)).astype(int); msg = bits_to_message(bits); ref = hashlib.sha256(msg).hexdigest()
    print(f"acoustic probe energy {E.min():.3e}..{E.max():.3e}, ones {bits.mean()*100:.1f}%\nmessage (55 B): {msg.hex()}\nhashlib SHA-256: {ref}", flush=True)
    ok, traces, outs = pad_selftest(ndef)
    print(f"pad self-test: {ok.sum()}/{len(ok)} pads pass NAND; masked pads defect counts: {ndef[~ok].tolist()}", flush=True)
    net, inw, rend, stw, dig = build_sha_net(440); ng = len(net.gout)
    print(f"netlist: {ng:,} NAND gates, depth {max(net.lvl)}", flush=True)
    val, pad_of, lv = run_pads(net, inw, bits, np.where(ok)[0], ndef)
    dhex = "".join(f"{word_val(val, w):08x}" for w in dig)
    print(f"pad digest    : {dhex}\nhashlib digest: {ref}\nRESULT: {'PASS' if dhex == ref else 'FAIL'}  (gates fired: {ng:,})", flush=True)
    D = (E, bits, snaps, recpad, msg, ref, ok, traces, outs, (val, pad_of, lv, rend, stw, dig, ng))
    D = (E, bits, snaps, recpad, msg, ref, ok, traces, outs, D[-1])
    pickle.dump(D, open("cache.pkl", "wb")); animate(sites, tsd, bpd, ndef, *D)
