"""
ir_phonon_sha256_all.py -- everything in one file (replaces si_acoustic_defect_sha256_v2.py, ir_phonon_speed_limit.py,
ir_phonon_full_sha256.py).

PIPELINE
  1. ACOUSTIC BLAST   real 2-D wave FDTD of the patterned wafer (pads, trenches, TSD/BPD scatterers) -> 440 bits/message, 16 messages.
                      Messages are generated automatically and cached in msgs.pkl (no manual --prep needed).
  2. PHONON LATTICE   every pad is a REAL diatomic Si-C lattice: a ring of NCELL=6 unit cells (12 atoms), masses 28.09/12.01 amu,
                      nearest-neighbour bond springs, quartic bond anharmonicity, uniform viscous damping (Q), IR drive through
                      opposite Born charges on Si and C.  A defect = a softened, more anharmonic bond at a random position in the pad.
                      The pad "output" is the total dipole  d = mean(u_Si - u_C)  (what an IR detector sees); lock-in at f0 = 23.9 THz.
                      State (u, v of all 12 atoms) PERSISTS between firings; pads idle between firings really decay (exact linear
                      propagator), nothing is reset by hand.
  3. SHA-256          the full compression as a threshold-logic netlist (60,928 firings, depth 2376) executed level by level on the pads.
  4. CHECKS           digests vs hashlib, raw gate errors, leftover ringing, simulated hash time.  --validate checks the lattice itself.

USAGE
  python ir_phonon_sha256_all.py                  # = --validate  then  --scen fast
  python ir_phonon_sha256_all.py --validate       # lattice physics checks (dispersion, sum rule, energy/momentum, regression)
  python ir_phonon_sha256_all.py --prep           # (re)generate the acoustic messages  (also happens automatically when missing)
  python ir_phonon_sha256_all.py --study          # speed-limit study: plain wait vs RZ reset, error rate vs Q / defects
  python ir_phonon_sha256_all.py --scen fast|q20|natural [--skip-micro] [--no-anim]
  python ir_phonon_sha256_all.py --all            # validate + study + fast + q20
Scenarios: fast = engineered Q=5 + per-pad calibrated anti-phase reset, 2-period cycle (84 fs)
           q20  = Q=20, plain wait, 32-period cycle      natural = SiC Q=150, plain wait, 256-period cycle  (slow)
"""
import os, sys, time, pickle, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================================================================
# 1. WAFER + ACOUSTIC FDTD
# =====================================================================================================================
WAFER_R, SITE, DX = 50.0, 8.0, 0.5            # mm
PROBES = [(0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2)]
NG = int(2 * WAFER_R / DX)                     # 200 x 200 cells
MSG_FILE = "msgs.pkl"

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

def acoustic_blast(sites, tsd, bpd, nsteps=1400, nsnap=40, snap_stride=14, P=16):
    xs = (np.arange(NG) + .5) * DX - WAFER_R
    X, Y = np.meshgrid(xs, xs, indexing="xy"); Rr = np.hypot(X, Y)
    c2 = np.full((NG, NG), 0.25)                              # (c dt/dx)^2, CFL-stable
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

def load_or_make_messages(sites, tsd, bpd, path=MSG_FILE, force=False):
    """Return the 16 acoustic messages (bits, bytes, hashlib refs, wave snapshots).  Generated + cached if the file is missing,
    unreadable, or was made for a different wafer (so a stale/foreign msgs.pkl can never be used silently)."""
    meta = dict(NG=int(NG), nsites=int(len(sites)), ndef=[int(x) for x in make_wafer()[3]], ver=3)
    if os.path.exists(path) and not force:
        try:
            with open(path, "rb") as fh: D = pickle.load(fh)
            if D.get("meta") == meta: return D
            print(f"{path}: made for a different wafer/version -> regenerating", flush=True)
        except Exception as e:
            print(f"{path}: unreadable ({e}) -> regenerating", flush=True)
    else:
        print(f"{path} not found -> running the acoustic blasts (16 x FDTD) ...", flush=True)
    t0 = time.time()
    E0, snaps, recpad = acoustic_blast(sites, tsd, bpd, P=16); bl = [(E0 >= np.median(E0)).astype(int)]
    for P_ in [10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        Ep, _, _ = acoustic_blast(sites, tsd, bpd, nsteps=1200, nsnap=0, P=P_); bl.append((Ep >= np.median(Ep)).astype(int))
    assert len(E0) == 440, f"expected 440 probe energies (88 pads x 5), got {len(E0)}"
    msgs = [bits_to_message(b) for b in bl]; refs = [hashlib.sha256(m).hexdigest() for m in msgs]
    D = dict(meta=meta, E0=E0, snaps=snaps, recpad=recpad, bits=bl, msgs=msgs, refs=refs)
    with open(path, "wb") as fh: pickle.dump(D, fh)
    print(f"acoustic messages ready in {time.time()-t0:.0f}s -> {path};  message 0: {msgs[0].hex()}", flush=True)
    return D

# =====================================================================================================================
# 2. SHA-256 THRESHOLD-LOGIC NETLIST
# =====================================================================================================================
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

# =====================================================================================================================
# 3. THE PHONON LATTICE (one pad = one Si-C ring lattice)
# =====================================================================================================================
M_SI, M_C = 28.0855, 12.011                    # amu
NCELL = 6; NA = 2 * NCELL; NB = NA             # 6 unit cells = 12 atoms in a ring (Born-von Karman cell) = 12 bonds
MASS = np.tile([M_SI, M_C], NCELL)             # atom j even = Si, odd = C ; bond j joins atom j and atom j+1 (periodic)
SGN = np.tile([1.0, -1.0], NCELL)              # sign of the Born charge (Si +, C -): neutral cell, so the field only drives the dipole
MU = M_SI * M_C / (M_SI + M_C)
K0 = 1.0 / (2.0 * (1.0 / M_SI + 1.0 / M_C))    # bond stiffness chosen so the zone-centre TO mode sits at omega = 1  (= 23.9 THz)
GDRV = SGN * MU / MASS                         # acceleration of atom j per unit effective drive (-> dipole drive = FD*s exactly)
IM = 1.0 / MASS
BETA0 = 0.05                                   # quartic bond anharmonicity  beta_b = BETA0*K0  (-> Duffing beta=0.05 on the dipole mode)
SOFT = 0.30                                    # a defective bond keeps 30% of its stiffness (core of a TSD/BPD)  [ASSUMPTION]
ANH_DEF = 2.0                                  # and is 2x more anharmonic                                         [ASSUMPTION]
SPP = 24; DT = 2 * np.pi / SPP; FD = 0.03      # RK4 steps per drive period, drive strength per unit input sum
F0_HZ = 23.9e12                                # 4H-SiC TO phonon (796 cm^-1)
HF = 6.626e-34 * F0_HZ
TNL_FRAC = 0.35                                # after a firing, integrate the weakly nonlinear ring-down for 0.35*Q periods, then
                                               # the (decayed) remainder with the exact linear propagator
SCEN = {"fast": dict(Q=5.0, cyc=(1, 1, 0), reset=True), "q20": dict(Q=20.0, cyc=(1, 0, 31), reset=False),
        "natural": dict(Q=150.0, cyc=(1, 0, 255), reset=False)}

def draw_mask(n, rng):
    """bitmask of n distinct defective bonds out of NB at random positions"""
    if n <= 0: return 0
    return int(sum(1 << int(b) for b in rng.choice(NB, min(int(n), NB), replace=False)))

def dip(u): return u[:, 0::2].mean(1) - u[:, 1::2].mean(1)        # dipole  d = <u_Si - u_C>

class Pads:
    """Table of distinct pad lattices (one per defect configuration) at a given Q, plus their per-class reset pulse (ra, rb)."""
    def __init__(self, masks, Q):
        self.masks = np.asarray(masks, np.int64); self.C = len(self.masks); self.Q = float(Q); self.gam = 1.0 / Q
        bits = ((self.masks[:, None] >> np.arange(NB)) & 1).astype(float)
        self.kb = K0 * (1 - (1 - SOFT) * bits); self.bb = BETA0 * K0 * (1 + (ANH_DEF - 1) * bits)
        self.ra = np.zeros(self.C); self.rb = np.zeros(self.C); self._pw = None
        self.tnl = int(np.ceil(TNL_FRAC * Q)); self.tcap = int(np.ceil(12 * Q))     # amplitude e^{-pi*12} < 1e-16 beyond tcap
    def stiffness(self):
        Km = np.zeros((self.C, NA, NA))
        for j in range(NB):
            a, b, k = j, (j + 1) % NA, self.kb[:, j]
            Km[:, a, a] += k; Km[:, b, b] += k; Km[:, a, b] -= k; Km[:, b, a] -= k
        return Km
    def powers(self):
        """Phi(2^j periods) for the LINEAR (harmonic + damping) lattice, j = 0..: exact free evolution of (u, v)."""
        if self._pw is None:
            A = np.zeros((self.C, 2 * NA, 2 * NA)); I = np.eye(NA)
            A[:, :NA, NA:] = I; A[:, NA:, :NA] = -self.stiffness() * IM[None, :, None]; A[:, NA:, NA:] = -self.gam * I
            Ah = A * DT; E = np.tile(np.eye(2 * NA), (self.C, 1, 1)); term = E.copy()
            for n in range(1, 30): term = term @ Ah / n; E = E + term          # exp(A*DT) to machine precision
            pw = [np.linalg.matrix_power(E, SPP)]
            for _ in range(int(self.tcap).bit_length() - 1): pw.append(pw[-1] @ pw[-1])
            self._pw = pw
        return self._pw

def idle(P, cls, X, V, T):
    """Free evolution for T periods (T: int or per-lane int array).  State beyond tcap periods has decayed < 1e-16 -> set to 0."""
    T = np.broadcast_to(np.asarray(T, np.int64), (len(cls),)).copy()
    if not T.any(): return X, V
    y = np.concatenate([X, V], 1); dead = T >= P.tcap; y[dead] = 0.0; T[dead] = 0
    for j, Mj in enumerate(P.powers()):
        m = ((T >> j) & 1).astype(bool)
        if m.any(): y[m] = np.matmul(Mj[cls[m]], y[m][:, :, None])[:, :, 0]
    return y[:, :NA].copy(), y[:, NA:].copy()

def fire(P, cls, X, V, s, cyc, trace=False, elapsed=None):
    """One firing of lanes (distinct pads) starting from their stored lattice state (X, V : (n, 12)).
    cyc = (W, R, gap): drive the IR field with weighted input sum s for W periods (lock-in of the dipole in the LAST drive period),
    then R periods of calibrated anti-phase reset drive, then `gap` idle periods.  `elapsed`: idle periods since the pad last fired.
    Equation of motion for atom j (bond j joins atoms j, j+1):
        m_j u_j'' = f_j - f_{j-1} - (m_j/Q) u_j' + q_j E(t),   f_j = k_j r_j + b_j r_j^3,  r_j = u_{j+1} - u_j."""
    W, R, gap = cyc; n = len(cls)
    if elapsed is not None: X, V = idle(P, cls, X, V, elapsed)
    kb, bb = P.kb[cls], P.bb[cls]; ra, rb = P.ra[cls][:, None], P.rb[cls][:, None]; F0 = (FD * np.asarray(s, float))[:, None]
    gam = P.gam; gm = GDRV[None, :]; im = IM[None, :]
    def acc(t, xx, vv, ms, mc):
        r = np.roll(xx, -1, 1) - xx; f = kb * r + bb * r**3
        return (f - np.roll(f, 1, 1)) * im + gm * F0 * (ms * np.sin(t) + mc * np.cos(t)) - gam * vv
    def rk(x, v, t, ms, mc, h=DT):
        k1x = v;               k1v = acc(t, x, k1x, ms, mc)
        k2x = v + .5*h*k1v;    k2v = acc(t + .5*h, x + .5*h*k1x, k2x, ms, mc)
        k3x = v + .5*h*k2v;    k3v = acc(t + .5*h, x + .5*h*k2x, k3x, ms, mc)
        k4x = v + h*k3v;       k4v = acc(t + h, x + h*k3x, k4x, ms, mc)
        return x + h/6*(k1x + 2*k2x + 2*k3x + k4x), v + h/6*(k1v + 2*k2v + 2*k3v + k4v)
    x = X.copy(); v = V.copy(); I = np.zeros(n); Qd = np.zeros(n); tr = []
    for st in range((W + R) * SPP):
        ms, mc = (1.0, 0.0) if st < W * SPP else (-ra, -rb); t = st * DT
        x, v = rk(x, v, t, ms, mc)
        if (W - 1) * SPP <= st < W * SPP:
            tt = (st + 1) * DT; d = dip(x); I += d * np.cos(tt); Qd += d * np.sin(tt)
        if trace: tr.append(dip(x))
    amp = (2.0 / SPP) * np.hypot(I, Qd)
    t0 = (W + R) * SPP * DT; gnl = min(gap, P.tnl)
    for st in range(gnl * SPP): x, v = rk(x, v, t0 + st * DT, 0.0, 0.0)           # nonlinear ring-down (undriven)
    if gap > gnl: x, v = idle(P, cls, x, v, gap - gnl)                             # decayed remainder: exact linear propagator
    return (amp, x, v, np.array(tr)) if trace else (amp, x, v)

def thresholds(Q):
    """Detector thresholds: midway between lock-in amplitudes of drive sums theta-1, theta on a perfect lattice from rest."""
    P0 = Pads([0], Q); z = np.zeros((6, NA)); a = fire(P0, np.zeros(6, int), z, z, np.arange(6.0), (1, 0, 0))[0]
    return np.array([0.0] + [0.5 * (a[t - 1] + a[t]) for t in (1, 2, 3)]), a

def calibrate_reset(P, W=1, R=1):
    """Per-pad cancellation pulse (a, b).  The reset drive is  -F0*(a sin t + b cos t)  during R periods.  The response is (nearly) linear
    in (a, b): three trial firings + a 2x2 solve null the DIPOLE-mode amplitude and phase.  Other (localised) lattice modes of a defective
    pad cannot be cancelled by a uniform pulse - their leftover is simulated, not assumed away."""
    C = P.C; z = np.zeros((C, NA)); cls = np.arange(C); one = np.ones(C); zero = np.zeros(C)
    def end(a, b):
        P.ra[:], P.rb[:] = a, b; _, x, v = fire(P, cls, z, z, one, (W, R, 0))
        return dip(x), v[:, 0::2].mean(1) - v[:, 1::2].mean(1)
    x0, v0 = end(zero, zero); xa, va = end(one, zero); xb, vb = end(zero, one)
    A = np.stack([np.stack([xa - x0, xb - x0], 1), np.stack([va - v0, vb - v0], 1)], 1)
    sol = np.linalg.solve(A, -np.stack([x0, v0], 1)[..., None])[..., 0]; P.ra[:], P.rb[:] = sol[:, 0], sol[:, 1]

def stress(P, Th, cyc, G=32, K=40, seed=2):
    """Self-test WITH history: random drive sequences back-to-back on G pads of every defect configuration; zero errors required."""
    rng = np.random.default_rng(seed); cls = np.repeat(np.arange(P.C), G); n = len(cls)
    X = np.zeros((n, NA)); V = np.zeros((n, NA)); bad = np.zeros(n)
    for k in range(K):
        s = rng.integers(0, 6, n).astype(float); amp, X, V = fire(P, cls, X, V, s, cyc)
        for th in (1, 2, 3): bad += ((amp >= Th[th]) != (s >= th))
    return bad.reshape(P.C, G).sum(1) == 0

# =====================================================================================================================
# 4. LATTICE VALIDATION
# =====================================================================================================================
def validate():
    print("=== PHONON LATTICE VALIDATION ===")
    P = Pads([0, 1, (1 << 0) | (1 << 5), (1 << 0) | (1 << 3) | (1 << 7)], Q=20.0)
    Km = P.stiffness(); D = Km / np.sqrt(MASS[:, None] * MASS[None, :])[None]; w2, vec = np.linalg.eigh(D)
    fth = lambda w2_: np.sqrt(np.maximum(w2_, 0)) * F0_HZ / 1e12
    # (1) dispersion of the perfect lattice vs the analytic diatomic-chain result
    q = 2 * np.pi * np.arange(NCELL) / NCELL; h = 1 / M_SI + 1 / M_C
    root = np.sqrt(h**2 - 4 * np.sin(q / 2)**2 / (M_SI * M_C)); an = np.sort(np.concatenate([K0 * (h + root), K0 * (h - root)]))
    err = np.abs(np.sort(w2[0]) - an).max(); wt = np.sort(fth(w2[0]))
    print(f"(1) perfect lattice: TO(Gamma) = {wt[-1]:.3f} THz (target {F0_HZ/1e12:.1f}), acoustic top {wt[wt < 18].max():.2f} THz, "
          f"optical bottom {wt[wt > 18].min():.2f} THz;  |omega^2 - analytic dispersion|max = {err:.1e}")
    # (2) oscillator-strength sum rule: dipole spectral weights of all 12 modes sum to exactly 1
    print("(2) dipole oscillator strength (sum rule: total must be 1), strongest mode:")
    for c in range(P.C):
        c_k = (SGN[None, :] / np.sqrt(MASS)[None, :] * vec[c].T).sum(1); wk = MU * c_k**2 / NCELL; i = wk.argmax()
        print(f"      {bin(int(P.masks[c])).count('1')} defective bond(s): sum = {wk.sum():.12f}   main line {fth(w2[c])[i]:6.2f} THz "
              f"weight {wk[i]:.3f}   ({(wk > 0.01).sum()} modes with >1% weight)")
    # (3) regression: perfect lattice == scalar Duffing oscillator x'' + x/Q + x + beta x^3 = FD*s*sin t   (uniform optical mode is exact)
    Q = 5.0; Pn = Pads([0], Q); z = np.zeros((1, NA)); s0 = 3.0
    tr = fire(Pn, np.zeros(1, int), z, z, np.array([s0]), (1, 1, 0), trace=True)[3][:, 0]
    x = v = 0.0; ref = []
    acc = lambda t, x_, v_: FD * s0 * (np.sin(t) if t < 2 * np.pi - 1e-9 else 0.0) - v_ / Q - x_ - BETA0 * x_**3
    for st in range(2 * SPP):
        t = st * DT; ms = 1.0 if st < SPP else 0.0
        f = lambda t_, x_, v_: FD * s0 * ms * np.sin(t_) - v_ / Q - x_ - BETA0 * x_**3
        k1x = v; k1v = f(t, x, v); k2x = v + .5*DT*k1v; k2v = f(t + .5*DT, x + .5*DT*k1x, k2x)
        k3x = v + .5*DT*k2v; k3v = f(t + .5*DT, x + .5*DT*k2x, k3x); k4x = v + DT*k3v; k4v = f(t + DT, x + DT*k3x, k4x)
        x += DT/6*(k1x + 2*k2x + 2*k3x + k4x); v += DT/6*(k1v + 2*k2v + 2*k3v + k4v); ref.append(x)
    print(f"(3) perfect lattice vs scalar Duffing oscillator (Q=5, s=3): max |d_lattice - x_scalar| = {np.abs(tr - np.array(ref)).max():.1e}"
          f"   (peak {np.abs(ref).max():.3f})")
    # (4) energy conservation (undriven, undamped, anharmonic, defective lattice) and (5) momentum conservation (driven, neutral cell)
    Pd = Pads([(1 << 0) | (1 << 3) | (1 << 7)], 20.0); Pd.gam = 0.0; cl = np.zeros(1, int); rng = np.random.default_rng(0)
    X0 = 0.05 * rng.standard_normal((1, NA)); V0 = 0.05 * rng.standard_normal((1, NA))
    En = lambda X_, V_: (0.5 * MASS * V_**2).sum() + (0.5 * Pd.kb[0] * (np.roll(X_, -1, 1) - X_)**2 + 0.25 * Pd.bb[0] * (np.roll(X_, -1, 1) - X_)**4).sum()
    _, X1, V1 = fire(Pd, cl, X0, V0, np.array([0.0]), (10, 0, 0))
    print(f"(4) energy drift, 10 periods undriven/undamped/anharmonic, 3-defect lattice: {abs(En(X1, V1) - En(X0, V0)) / En(X0, V0):.1e} (relative)")
    z1 = np.zeros((1, NA)); _, X2, V2 = fire(Pd, cl, z1, z1, np.array([5.0]), (3, 0, 0))
    print(f"(5) net momentum after 3 driven periods (neutral Si-C cell, should stay 0): {abs((MASS * V2).sum()):.1e}   (dipole velocity scale {np.abs(V2).max():.2f})")
    # (6) exact idle propagator vs RK4 ring-down (harmonic)
    Ph = Pads([(1 << 2) | (1 << 9)], 20.0); Ph.bb[:] = 0; Ph.tnl = 100; X0 = 0.1 * rng.standard_normal((1, NA)); V0 = 0.1 * rng.standard_normal((1, NA))
    _, Xr, Vr = fire(Ph, cl, X0, V0, np.array([0.0]), (1, 0, 9)); Ph.tnl = 0
    Xe, Ve = idle(Ph, cl, X0, V0, 10)
    print(f"(6) 10-period free evolution, exact propagator vs RK4 (24 steps/period): relative state difference {np.abs(Xr - Xe).max() / np.abs(Xe).max():.1e}")
    # figure: dispersion + oscillator strengths
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); qq = np.linspace(0, 2 * np.pi, 300); r_ = np.sqrt(h**2 - 4 * np.sin(qq / 2)**2 / (M_SI * M_C))
    for sgn in (1, -1): ax[0].plot(qq / (2 * np.pi), fth(K0 * (h + sgn * r_)), "k-", lw=1)
    ax[0].plot(q / (2 * np.pi), fth(K0 * (h + root)), "ro"); ax[0].plot(q / (2 * np.pi), fth(K0 * (h - root)), "bo")
    ax[0].set_xlabel("q (units of 2pi/a)"); ax[0].set_ylabel("frequency (THz)"); ax[0].set_title("Si-C chain dispersion: lattice modes (dots) vs analytic")
    for c, col in zip(range(P.C), ["k", "tab:blue", "tab:orange", "tab:red"]):
        c_k = (SGN[None, :] / np.sqrt(MASS)[None, :] * vec[c].T).sum(1); wk = MU * c_k**2 / NCELL
        ax[1].stem(fth(w2[c]) + 0.08 * c, np.maximum(wk, 1e-4), linefmt=col, markerfmt=col[0] if len(col) == 1 else "o", basefmt=" ", label=f"{bin(int(P.masks[c])).count('1')} defective bonds")
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, 1.5); ax[1].set_xlabel("mode frequency (THz)"); ax[1].set_ylabel("IR oscillator strength"); ax[1].legend(fontsize=8)
    ax[1].set_title("defects detune/split the IR-active TO line"); fig.tight_layout(); fig.savefig("lattice_validation.png", dpi=110); print("saved lattice_validation.png\n")

# =====================================================================================================================
# 5. SPEED-LIMIT STUDY (plain wait vs RZ anti-phase reset), real lattice dynamics
# =====================================================================================================================
def lane_pads(Q, nd, rng):
    masks = np.array([draw_mask(n, rng) for n in nd], np.int64); um, inv = np.unique(masks, return_inverse=True); return Pads(um, Q), inv

def error_rate(Q, cyc, G=600, K=14, nmax=5, seed=3, ndef=None, calibrated=False):
    rng = np.random.default_rng(seed); Th, a = thresholds(Q)
    nd = rng.integers(0, nmax + 1, G) if ndef is None else np.full(G, int(ndef)); P, cls = lane_pads(Q, nd, rng)
    if cyc[1] > 0:
        if calibrated: calibrate_reset(P)
        else: P.ra[:] = 1.0; P.rb[:] = 0.0                      # plain anti-phase copy of the drive
    X = np.zeros((G, NA)); V = np.zeros((G, NA)); nbad = 0
    for k in range(K):
        s = rng.integers(0, 6, G).astype(float); amp, X, V = fire(P, cls, X, V, s, cyc); good = np.ones(G, bool)
        for th in (1, 2, 3): good &= ((amp >= Th[th]) == (s >= th))
        nbad += int((~good).sum())
    return nbad / (G * K), a

def study():
    print(f"=== SPEED-LIMIT STUDY: 4H-SiC TO phonon {F0_HZ/1e12:.1f} THz, period {1e15/F0_HZ:.1f} fs, lattice pads (12 atoms) ===\n")
    gaps = [0, 1, 3, 7, 15, 31, 63, 127, 255]; res = {}; Qs = (2, 20, 150)
    print("A) PLAIN firing, drive 1 period then idle gap; random defects 0-5 per pad (random bond positions); error rate (fraction of firings with any wrong gate)")
    print("   cycle(periods)  " + "  ".join(f"Q={Q:<4}" for Q in Qs))
    for g in gaps:
        row = [error_rate(Q, (1, 0, g), G=240, K=12)[0] for Q in Qs]; res[g] = row
        print(f"   {1+g:>8} ({(1+g)*1e15/F0_HZ/1e3:7.2f} ps) " + "  ".join(f"{r:7.4f}" for r in row), flush=True)
    print("\nB) RZ anti-phase reset (plain copy of the drive in anti-phase): drive 1 + cancel 1 period => cycle = 2 periods (%.0f fs)" % (2e15 / F0_HZ))
    rz = {}
    for Q in (2, 5, 20, 150, 1000):
        e, a = error_rate(Q, (1, 1, 0), G=1200, K=20); rz[Q] = e
        print(f"   Q={Q:<5} error rate {e:.5f}   (amp ratio s=2..5 vs s=1: {np.round(a[2:]/a[1], 2).tolist()})", flush=True)
    print("\nB2) RZ with PER-PAD CALIBRATED cancellation pulse (amplitude+phase of the dipole mode nulled), cycle = 2 periods")
    rzc = {}
    for Q in (2, 5, 20, 150, 1000):
        e, _ = error_rate(Q, (1, 1, 0), G=1200, K=20, calibrated=True); rzc[Q] = e; print(f"   Q={Q:<5} error rate {e:.5f}", flush=True)
    print("\nC) calibrated RZ at Q=150 versus number of softened bonds in the pad:")
    for n in range(0, 10):
        e, _ = error_rate(150, (1, 1, 0), G=600, K=30, ndef=n, calibrated=True); print(f"   {n} defects: error rate {e:.5f}", flush=True)
    scen = [("natural SiC Q~150, plain wait", 256), ("engineered Q~5 + calibrated RZ reset", 2 if rzc[5] < 1e-3 else 256)]
    G_F, DEPTH = 60928, 2376; NQ, INP = 100, 3.0
    e_hash = G_F * INP * NQ * HF; area = np.pi * 5.0**2; rows = []
    print(f"\nSPEED  ({G_F:,} firings/hash, depth {DEPTH}, signal energy/hash {e_hash:.2e} J at {NQ} quanta/input; power cap = 100 W/cm2 of signal heat)")
    for sname, cyc in scen:
        tc = cyc / F0_HZ; print(f" [{sname}]  cycle {cyc} periods = {tc*1e15:.0f} fs, latency {DEPTH*tc*1e12:.0f} ps")
        for name, pitch_um in (("100 mm wafer, 10 um pads", 10), ("100 mm wafer, 2 um pads", 2), ("100 mm wafer, 1 um pads", 1)):
            Pn = int(np.pi * 50e3**2 / pitch_um**2); thr = Pn / (G_F * tc); cap = 100 * area / e_hash
            rows.append((sname, name, Pn, thr, min(thr, cap)))
            print(f"    {name:<26} pads {Pn:>14,}  speed-limit {thr:9.2e} H/s  power-cap {cap:9.2e}  -> {min(thr, cap):9.2e} H/s")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, Q in enumerate(Qs): ax[0].semilogy([1 + g for g in gaps], [max(res[g][i], 1e-4) for g in gaps], "o-", label=f"plain, Q={Q}")
    ax[0].axhline(1e-4, c="gray", ls=":"); ax[0].set_xscale("log"); ax[0].set_xlabel("cycle time (periods of 24 THz)"); ax[0].set_ylabel("error rate (floor 1e-4 = none seen)")
    ax[0].plot([2], [max(rz[150], 1e-4)], "kx", ms=10, label="uncalibrated RZ Q=150, 2 periods"); ax[0].plot([2], [max(rzc[150], 1e-4)], "r*", ms=14, label="calibrated RZ Q=150, 2 periods")
    ax[0].plot([2], [max(rzc[5], 1e-4)], "g^", ms=9, label="calibrated RZ Q=5, 2 periods"); ax[0].legend(fontsize=7); ax[0].set_title("Back-to-back firing: plain wait vs RZ reset (lattice pads)")
    lab = [f"{r[0].split(' ')[0]}/{r[1].split(', ')[1]}" for r in rows]
    ax[1].bar(range(len(rows)), [r[4] for r in rows], color=["#7f8c8d"] * 3 + ["#c0392b"] * 3); ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels(lab, fontsize=6, rotation=30); ax[1].set_ylabel("H/s (min of speed and power caps)")
    ax[1].set_title("grey: natural Q~150 | red: engineered Q~5 + RZ")
    fig.tight_layout(); fig.savefig("ir_phonon_speed_limit.png", dpi=110); print("saved ir_phonon_speed_limit.png\n")

# =====================================================================================================================
# 6. RUN THE NETLIST ON THE LATTICE PADS (state carried between firings, idle pads decay)
# =====================================================================================================================
def run_net_state(net, inw, bits_list, pad_ok, Np, cls_pad, P, Th, cyc, log=True):
    B = len(bits_list); nw = len(net.lvl); val = np.zeros((B, nw), np.int8); val[:, 1] = 1
    for b, bits in enumerate(bits_list): val[b, inw] = bits
    gi = np.array(net.gi); gw = np.array(net.gw, float); gth = np.array(net.gth); go = np.array(net.gout)
    lv = np.array([net.lvl[w] for w in go]); order = np.argsort(lv, kind="stable"); lvs = lv[order]; maxL = int(lvs.max())
    bounds = np.searchsorted(lvs, np.arange(1, maxL + 2)); al = {}
    for w, src in net.alias: al.setdefault(net.lvl[w], []).append((w, src))
    al = {L: (np.array([x[0] for x in v_]), np.array([x[1] for x in v_])) for L, v_ in al.items()}
    def do_alias(L):
        if L in al: val[:, al[L][0]] = 1 - val[:, al[L][1]]
    do_alias(0)
    cper = sum(cyc); X = np.zeros((Np, NA)); V = np.zeros((Np, NA)); last = np.full(Np, -1, np.int64); slot = 0
    pad_of = np.zeros(len(go), int); off = 0; wrong = 0; fired = 0; lo_full = 0.0; lo_dip = 0.0; t0 = time.time()
    for L in range(1, maxL + 1):
        idx = order[bounds[L - 1]:bounds[L]]
        if len(idx):
            n = len(idx); tot = B * n; Lc = len(pad_ok)
            sdr = (val[:, gi[idx]] * gw[idx][None]).sum(-1).reshape(-1); thr = np.tile(gth[idx], B)
            pads_all = pad_ok[(off + np.arange(tot)) % Lc]; off += tot; outs = np.zeros(tot, np.int8)
            for c0 in range(0, tot, Lc):                       # one chunk = one time slot of up to Lc distinct pads
                sl_ = slice(c0, min(tot, c0 + Lc)); pp = pads_all[sl_]
                el = np.where(last[pp] < 0, 0, (slot - last[pp] - 1) * cper)       # idle slots since this pad last fired
                amp, xn, vn = fire(P, cls_pad[pp], X[pp], V[pp], sdr[sl_], cyc, elapsed=el)
                X[pp] = xn; V[pp] = vn; last[pp] = slot; slot += 1; outs[sl_] = (amp >= Th[thr[sl_]])
                lo_full = max(lo_full, float(np.sqrt((xn**2 + vn**2).sum(1)).max()))
                lo_dip = max(lo_dip, float(np.hypot(dip(xn), vn[:, 0::2].mean(1) - vn[:, 1::2].mean(1)).max()))
            wrong += int((outs != (sdr >= thr)).sum()); fired += tot
            val[:, go[idx]] = outs.reshape(B, n); pad_of[idx] = pads_all[:n]
        do_alias(L)
        if log and L % 300 == 0: print(f"  level {L}/{maxL}  ({time.time()-t0:.0f}s)", flush=True)
    return val, pad_of, lv, dict(chunks=slot, wrong=wrong, fired=fired, leftover=lo_full, leftover_dip=lo_dip)

# =====================================================================================================================
# 7. ANIMATION
# =====================================================================================================================
def animate(sites, tsd, bpd, ndef, E, bits, snaps, recpad, msg, ref, ok, traces, net_res, tag="ir_phonon_full_sha256", cyc=(1, 1, 0)):
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
    NB_, NR, NSF, NC, ND = len(snaps), 8, 8, 48, 12
    F1, F2, F3, F4 = NB_, NB_ + NR, NB_ + NR + NSF, NB_ + NR + NSF + NC; NF = F4 + ND
    maxL = int(lv.max()); Lf = np.ceil(maxL * np.arange(NC + 1) / NC).astype(int)
    ids = np.arange(len(lv)); permin = np.full(maxL + 2, len(lv)); np.minimum.at(permin, lv, ids)
    minid = np.minimum.accumulate(permin[::-1])[::-1]; ranks = np.array(round_end)
    heat_f = []; ng_f = []
    for c in range(NC):
        sel = (lv > Lf[c]) & (lv <= Lf[c + 1]); heat_f.append(np.bincount(pad_of[sel], minlength=NS)); ng_f.append(int(sel.sum()))
    cum_f = np.cumsum(ng_f); rdone_f = [int(np.searchsorted(ranks, minid[min(Lf[c + 1] + 1, maxL + 1)], side="left")) for c in range(NC)]
    t_ax = (np.arange(traces.shape[2]) + 1) * DT / (2 * np.pi); t_drive = cyc[0]
    fig = plt.figure(figsize=(12.5, 7.8))
    axw = fig.add_axes([0.02, 0.28, 0.42, 0.63]); axr = fig.add_axes([0.47, 0.69, 0.51, 0.22])
    axm = fig.add_axes([0.52, 0.47, 0.46, 0.14]); axt = fig.add_axes([0.47, 0.265, 0.51, 0.17]); axt.axis("off")
    title = fig.text(0.5, 0.955, "", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.008, "Acoustic blast = real 2-D wave FDTD.  Pads = 12-atom Si-C lattices (masses, bond springs, anharmonicity, Born-charge IR drive, RK4) at "
             "the 23.9 THz TO phonon; defects = softened bonds; lattice state persists between firings.  Constants assumed; gain/readout/routing not modelled.", fontsize=6.4, color="#555")
    axw.add_patch(Circle((0, 0), WAFER_R, fill=False, lw=3, color="#2c3e50", zorder=6)); patches = []
    for k, (sx, sy, x0, y0) in enumerate(sites):
        p = Rectangle((x0, y0), SITE, SITE, fc="#f2f3f4", ec="#566573", lw=1.2, zorder=1); axw.add_patch(p); patches.append(p)
        axw.text(sx, sy, str(ndef[k]), ha="center", va="center", fontsize=6, color="#2c3e50", zorder=7)
    axw.scatter(*tsd.T, marker="x", c="#c0392b", s=9, lw=.8, zorder=5); axw.scatter(*bpd.T, marker="s", c="#e67e22", s=7, zorder=5)
    pr = np.array([[s_[0] + dx, s_[1] + dy] for s_ in sites for dx, dy in PROBES])
    probe_sc = axw.scatter(pr[:, 0], pr[:, 1], s=3, c="#2c3e50", zorder=8, visible=False)
    im = axw.imshow(np.zeros((NG, NG, 4)), extent=[-WAFER_R, WAFER_R, -WAFER_R, WAFER_R], origin="lower", zorder=3, interpolation="bilinear")
    axw.set_xlim(-WAFER_R * 1.05, WAFER_R * 1.05); axw.set_ylim(-WAFER_R * 1.05, WAFER_R * 1.05); axw.set_aspect("equal"); axw.set_xticks([]); axw.set_yticks([])
    axw.set_title("Patterned SiC wafer: pads (digit = defects in pad), TSD x, BPD \u25a0", fontsize=10)
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
        Patch(fc="#2ecc71", ec="#1e8449", label="Pad passes lattice self-test (with history)"),
        Patch(fc="#bdc3c7", ec="#7f8c8d", hatch="///", label="Pad masked (defects break logic)"),
        Patch(fc=heatcm(.85), label="Pad firing (brighter = more gates)"),
        Line2D([], [], c="#2980b9", label="Pad dipole d(t): drive sum 0"), Line2D([], [], c="#27ae60", label="drive sum 1"),
        Line2D([], [], c="#e67e22", label="drive sum 2"), Line2D([], [], c="#c0392b", label="drive sum 3"),
        Line2D([], [], c="k", ls="--", label="end of drive window (reset pulse follows)"),
        Patch(fc="#ecf0f1", ec="#aab7b8", label="Input bit: empty"), Patch(fc="#ffffff", ec="#aab7b8", label="Input bit = 0"),
        Patch(fc="#00bcd4", ec="#aab7b8", label="Input bit = 1"), Patch(fc="#8e44ad", ec="#aab7b8", label="Padding bit = 1 (light = 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.8, title="Legend", title_fontsize=7.5, bbox_to_anchor=(0.5, 0.022), frameon=True)
    order_def = np.argsort(-ndef); tcol = ["#2980b9", "#27ae60", "#e67e22", "#c0392b"]
    wave_pads = [int(np.argmin(ndef)), int(np.argsort(ndef)[NS // 2]), int(np.argmax(ndef))]
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
            axm.axvline(t_drive, c="k", ls="--", lw=.8); axm.tick_params(labelsize=7); axm.set_xlabel("time (fundamental periods)", fontsize=7)
            axm.set_title(f"self-test pad {pp} ({ndef[pp]} defects): lattice dipole d(t) for drive sums 0-3", fontsize=8)
            title.set_text("3. Each defect pad self-tests (with history) as a 24 THz phonon-lattice gate")
            text = [f"pads usable: {ok.sum()}/{NS}   masked: {(~ok).sum()}", "pad = 12-atom Si-C lattice, defects = softened bonds at random positions",
                    "out = [lock-in amplitude of the dipole >= theta threshold]; full adder = 2 pads"]
        elif f < F4:
            nbits = 440; c = f - F3
            for k in range(NS):
                if ok[k]: patches[k].set_facecolor(heatcm(min(1, heat_f[c][k] / max(1, heat_f[c].max()) * .95)))
                else: patches[k].set_facecolor("#bdc3c7"); patches[k].set_hatch("///")
            pp = [p for p in np.argsort(-heat_f[c]) if ok[p]][0]
            for cc in range(4): axm.plot(t_ax, traces[pp, cc], c=tcol[cc], lw=1.1)
            axm.axvline(t_drive, c="k", ls="--", lw=.8); axm.tick_params(labelsize=7)
            axm.set_title(f"busiest pad now: pad {pp} ({ndef[pp]} defects), fired {heat_f[c][pp]}x in this slice", fontsize=8)
            r = rdone_f[c]; title.set_text("4. 24 THz defect pads fire gates (lattice state carried over): SHA-256, 64 rounds")
            text = [f"gate firings: {int(cum_f[c]):,}/{ngates:,}  (1 gate per pad firing)", f"rounds completed: {r:2d}/64   " + "#" * (r // 2) + "." * (32 - r // 2)]
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
    anim.save(f"{tag}.gif", writer=PillowWriter(fps=11), dpi=62)
    update(8); fig.savefig("full_blast.png", dpi=80); update(F3 + 24); fig.savefig("full_compute.png", dpi=80); update(NF - 1); fig.savefig("full_digest.png", dpi=80)
    print(f"saved {tag}.gif ({NF} frames)", flush=True)

# =====================================================================================================================
# 8. FULL SCENARIO RUN
# =====================================================================================================================
def run_scenario(sc, skip_micro=False, anim=True):
    sites, tsd, bpd, ndef = make_wafer(); D = load_or_make_messages(sites, tsd, bpd); bl, refs = D["bits"], D["refs"]
    p = SCEN[sc]; Q, cyc = p["Q"], p["cyc"]; tc = sum(cyc) / F0_HZ; Th, _ = thresholds(Q)
    # --- pad populations: random defect POSITIONS inside every pad
    rngw = np.random.default_rng(11); wmask = np.array([draw_mask(n, rngw) for n in ndef], np.int64)
    Pm = int(np.pi * WAFER_R**2 / 0.1**2); rng = np.random.default_rng(5); ndm = np.zeros(Pm, int)
    for k in rng.integers(0, Pm, len(tsd) + len(bpd)): ndm[k] += 1
    mmask = np.zeros(Pm, np.int64)
    for k in np.nonzero(ndm)[0]: mmask[k] = draw_mask(ndm[k], rng)
    um = np.unique(np.concatenate([[0], wmask, mmask[ndm > 0]])); cls_w = np.searchsorted(um, wmask); cls_m = np.searchsorted(um, mmask)
    P = Pads(um, Q)
    if p["reset"]: calibrate_reset(P)
    print(f"[{sc}] Q={Q}, cycle {sum(cyc)} periods = {tc*1e15:.0f} fs, thresholds {np.round(Th[1:], 4).tolist()}, {P.C} distinct pad lattices; "
          f"reset(a,b) perfect pad: ({P.ra[0]:.3f},{P.rb[0]:.3f})", flush=True)
    t0 = time.time(); ok_c = stress(P, Th, cyc); nbd = np.array([bin(int(m)).count("1") for m in um])
    print(f"pad self-test with history ({time.time()-t0:.0f}s), lattices passing by number of defective bonds: " +
          "  ".join(f"{n}:{int(ok_c[nbd == n].sum())}/{int((nbd == n).sum())}" for n in range(int(nbd.max()) + 1) if (nbd == n).any()), flush=True)
    net, inw, rend, stw, dig = build_sha_net(440); ng = len(net.gout); depth = int(max(net.lvl)); print(f"netlist {ng:,} firings, depth {depth}", flush=True)
    ok = ok_c[cls_w]; ok_idx = np.where(ok)[0]
    if sc == "fast" or skip_micro:                             # --- the 88-pad wafer, one hash
        val, pad_of, lv, st = run_net_state(net, inw, [bl[0]], ok_idx, len(ndef), cls_w, P, Th, cyc)
        d0 = "".join(f"{word_val(val[0], w):08x}" for w in dig)
        print(f"[wafer, {ok.sum()}/88 usable pads] digest {d0}\n  hashlib {refs[0]} -> {'PASS' if d0 == refs[0] else 'FAIL'}; raw gate errors {st['wrong']}/{st['fired']:,}; "
              f"pad-time slots {st['chunks']:,} -> {st['chunks']*tc*1e9:.3f} ns/hash; leftover ringing after a firing: dipole {st['leftover_dip']:.2e}, whole lattice {st['leftover']:.2e}", flush=True)
    if not skip_micro:                                         # --- micro-pad array, 16 hashes pipelined
        okm = ok_c[cls_m]
        valm, _, lvm, st2 = run_net_state(net, inw, bl, np.where(okm)[0], Pm, cls_m, P, Th, cyc)
        dm = ["".join(f"{word_val(valm[b], w):08x}" for w in dig) for b in range(16)]; npass = sum(dm[b] == refs[b] for b in range(16))
        print(f"[micro-pad array {Pm:,} pads ({(~okm).sum()} masked), 16 pipelined hashes] {npass}/16 digests match hashlib; raw gate errors {st2['wrong']}/{st2['fired']:,}; "
              f"leftover ringing: dipole {st2['leftover_dip']:.2e}, lattice {st2['leftover']:.2e}; batch time {st2['chunks']*tc*1e12:.1f} ps (latency-bound)", flush=True)
        cap = Pm / (ng * tc); print(f"capacity of this array at this cycle: {cap:.2e} H/s (needs ~{cap*len(set(lvm.tolist()))*tc:,.0f} hashes in flight)", flush=True)
    if (sc == "fast" or skip_micro) and anim:
        tr = np.zeros((len(ndef), 4, (cyc[0] + cyc[1]) * SPP)); z = np.zeros((len(ndef), NA))
        for s_ in range(4): tr[:, s_, :] = fire(P, cls_w, z, z, np.full(len(ndef), float(s_)), cyc, trace=True)[3].T
        animate(sites, tsd, bpd, ndef, D["E0"], bl[0], D["snaps"], D["recpad"], D["msgs"][0], refs[0], ok, tr, (val, pad_of, lv, rend, stw, dig, ng), cyc=cyc)

if __name__ == "__main__":
    a = sys.argv[1:]
    if "--prep" in a:
        s_, t_, b_, _ = make_wafer(); load_or_make_messages(s_, t_, b_, force=True); sys.exit()
    todo = []
    if "--all" in a: todo = ["validate", "study", "fast", "q20"]
    if "--validate" in a: todo.append("validate")
    if "--study" in a: todo.append("study")
    if "--scen" in a: todo.append(a[a.index("--scen") + 1])
    if not todo: todo = ["validate", "fast"]
    for job in todo:
        if job == "validate": validate()
        elif job == "study": study()
        else: run_scenario(job, skip_micro="--skip-micro" in a, anim="--no-anim" not in a)
