
"""
SiC wafer: defects -> phonon measurements -> NAND-only macro dies running SHA-256.
(Single-file version: defect simulation, Callaway/Raman phonon model, gate library
and macrostructures, permutation nulls, and the SHA-256 macro-die simulation.)

Section order matters: every module-level name is defined before it is used.
  1. wafer + defect simulation        3. gate library, lattice, permutation nulls
  2. phonon model + phonon gates      4. SHA-256 fabric, fault injection, TMR, plots
"""
import csv
import hashlib
from collections import defaultdict
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree



# ####################################################################
# 1. WAFER + DEFECT SIMULATION
# ####################################################################
rng = np.random.default_rng(42)

# ---------------- wafer geometry ----------------
R = 75.0            # mm (150 mm wafer)
EDGE_EXCL = 3.0     # mm
DIE = 5.0           # mm square die
R_EFF = R - EDGE_EXCL


def uniform_in_disk(n, radius=R, rng=rng):
    r = radius * np.sqrt(rng.random(n))
    t = 2 * np.pi * rng.random(n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def in_wafer(p, radius=R):
    return np.hypot(p[:, 0], p[:, 1]) <= radius


# ---------------- defect generators ----------------
def gen_micropipes(density_cm2=0.1):
    area_cm2 = np.pi * (R / 10) ** 2
    return uniform_in_disk(rng.poisson(density_cm2 * area_cm2))


def gen_tsd(density_cm2=0.5):
    area_cm2 = np.pi * (R / 10) ** 2
    return uniform_in_disk(rng.poisson(density_cm2 * area_cm2))


def gen_bpd(n_parents=12, mean_children=25, sigma=2.5, edge_bias=2.0):
    """Thomas cluster process, parents biased toward the wafer edge."""
    u = rng.random(n_parents) ** (1 / edge_bias)          # push radius outward
    r = R * np.sqrt(u)
    t = 2 * np.pi * rng.random(n_parents)
    parents = np.column_stack([r * np.cos(t), r * np.sin(t)])
    pts = []
    for p in parents:
        k = rng.poisson(mean_children)
        pts.append(p + rng.normal(0, sigma, (k, 2)))
    pts = np.vstack(pts)
    return pts[in_wafer(pts)]


def gen_scratches(n_scratches=3, pts_per_mm=1.5):
    pts = []
    for _ in range(n_scratches):
        a, b = uniform_in_disk(2)
        L = np.linalg.norm(b - a)
        n = max(2, int(L * pts_per_mm))
        s = rng.random(n)[:, None]
        line = a + s * (b - a) + rng.normal(0, 0.1, (n, 2))
        pts.append(line)
    pts = np.vstack(pts)
    return pts[in_wafer(pts)]


def simulate_wafer():
    d = {
        "Micropipe": gen_micropipes(),
        "TSD": gen_tsd(),
        "BPD (clustered)": gen_bpd(),
        "Scratch": gen_scratches(),
    }
    allpts = np.vstack(list(d.values()))
    labels = np.concatenate([[k] * len(v) for k, v in d.items()])
    return d, allpts, labels


# ---------------- statistics ----------------
def mean_nn_distance(p):
    d, _ = cKDTree(p).query(p, k=2)
    return d[:, 1].mean()


def mean_radius(p):
    return np.hypot(p[:, 0], p[:, 1]).mean()


def permutation_test(pts, stat, n_perm=999, alternative="two-sided"):
    """Null = CSR: re-draw n points uniformly in the wafer each permutation."""
    obs = stat(pts)
    null = np.array([stat(uniform_in_disk(len(pts))) for _ in range(n_perm)])
    if alternative == "less":
        p = (1 + np.sum(null <= obs)) / (n_perm + 1)
    elif alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (n_perm + 1)
    return obs, null, p


# ---------------- die-level yield ----------------
def die_grid():
    xs = np.arange(-R, R, DIE)
    gx, gy = np.meshgrid(xs, xs)
    corners = [(gx, gy), (gx + DIE, gy), (gx, gy + DIE), (gx + DIE, gy + DIE)]
    inside = np.all([np.hypot(cx, cy) <= R_EFF for cx, cy in corners], axis=0)
    return gx[inside], gy[inside]


def die_counts(pts, gx, gy):
    ix = np.floor((pts[:, 0] - gx.min()) / DIE).astype(int)
    iy = np.floor((pts[:, 1] - gy.min()) / DIE).astype(int)
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(
        zip(np.floor((gx - gx.min()) / DIE + 1e-9), np.floor((gy - gy.min()) / DIE + 1e-9)))}
    counts = np.zeros(len(gx), int)
    for a, b in zip(ix, iy):
        if (a, b) in key:
            counts[key[(a, b)]] += 1
    return counts


def morans_i(counts, gx, gy):
    """Moran's I with inverse-distance weights for die defect counts."""
    xy = np.column_stack([gx, gy]) + DIE / 2
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    W = np.where(D > 0, 1 / np.where(D > 0, D, 1), 0)
    z = counts - counts.mean()
    return len(z) / W.sum() * (z @ W @ z) / (z @ z)


def die_permutation_test(counts, gx, gy, n_perm=999):
    """Permute die counts across die sites -> null of no spatial structure."""
    obs = morans_i(counts, gx, gy)
    null = np.array([morans_i(rng.permutation(counts), gx, gy) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, null, p



# ---------------- logic-gate detection ----------------
# Per-die boolean inputs from defect counts, combined through logic gates.
THRESH = {"Micropipe": 1, "TSD": 2, "BPD (clustered)": 3, "Scratch": 1}


def die_flags(by_type, gx, gy):
    """Boolean input vector per die: MP, TSD, BPD, SCR."""
    f = {}
    for k, v in by_type.items():
        f[k] = die_counts(v, gx, gy) >= THRESH[k]
    return f["Micropipe"], f["TSD"], f["BPD (clustered)"], f["Scratch"]


def gate_network(mp, tsd, bpd, scr):
    """
    G_FATAL = MP OR SCR              (OR gate  : immediate kill)
    G_RISK  = BPD AND TSD            (AND gate : co-located dislocations -> gate-oxide risk)
    G_MIXED = BPD XOR TSD            (XOR gate : exactly one dislocation type present)
    PASS    = NOT(G_FATAL OR G_RISK) (NOR gate : die ships)
    """
    g_fatal = np.logical_or(mp, scr)
    g_risk = np.logical_and(bpd, tsd)
    g_mixed = np.logical_xor(bpd, tsd)
    passed = np.logical_not(np.logical_or(g_fatal, g_risk))
    return g_fatal, g_risk, g_mixed, passed


def detect_gate_pattern(mp, tsd, bpd, scr):
    """Per-die: which gate(s) fired. Returns reason code array."""
    g_fatal, g_risk, g_mixed, passed = gate_network(mp, tsd, bpd, scr)
    code = np.full(len(mp), "PASS", dtype=object)
    code[g_mixed & passed] = "XOR-only (pass, watch)"
    code[g_risk] = "AND (BPD&TSD)"
    code[g_fatal] = "OR (MP|SCR)"
    return code


def truth_table_coverage(mp, tsd, bpd, scr):
    """Count dies for each of the 16 input combinations and the gate outputs."""
    rows = []
    idx = (mp * 8 + tsd * 4 + bpd * 2 + scr * 1).astype(int)
    for i in range(16):
        m, t, b, s = (i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1
        fatal, risk, mixed, ok = gate_network(*(np.array([x], bool) for x in (m, t, b, s)))
        rows.append((m, t, b, s, int(fatal[0]), int(risk[0]), int(mixed[0]), int(ok[0]), int((idx == i).sum())))
    return rows


def cooccurrence_permutation_test(bpd, tsd, n_perm=999):
    """
    Is the AND gate (BPD & TSD) firing more often than if the two flags were
    independent? Permute the TSD flag across die sites to break co-location.
    """
    obs = np.sum(bpd & tsd)
    null = np.array([np.sum(bpd & rng.permutation(tsd)) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, null, p


# ####################################################################
# 2. PHONON MODEL + PHONON GATES
# ####################################################################
# ---------------- physical constants / material parameters ----------------
kB, hbar = 1.380649e-23, 1.054572e-34
T = 300.0                  # K
V = 9000.0                 # m/s, average phonon group velocity
THETA = 1200.0             # K, Debye temperature (4H-SiC, approx.)
L_BND = 3.5e-4             # m, wafer thickness (boundary scattering)
A_PT = 1e-44               # s^3,   point-defect / isotope  (w^4)
B_UM = 4e-18               # s/K,   Umklapp
C_BPD = 5e-11              # strain-field scattering per m^-2 (~w)
C_TSD = 1.0e-10            # core scattering per m^-2 (~w^3)
W_REF = kB * T / hbar      # rad/s, reference frequency
ELL_HALO = 3e-7            # m, effective MFP inside a damage halo
KAPPA_PRISTINE = 370.0     # W/mK calibration target
DENS_SCALE = 100.0         # each simulated defect marker ~ 100 real defects / cm^2
DIE_AREA_CM2 = (DIE / 10) ** 2
DIE_AREA_MM2 = DIE ** 2
HALO_AREA_MM2 = {"Micropipe": 0.8, "Scratch": 0.13}   # damaged area per marker

# Raman E2(TO)
W0, G0 = 776.0, 2.0        # cm^-1 center, baseline FWHM
GB, GT, GH = 1.7e-4, 3.0e-4, 3.0     # linewidth broadening coefficients
SB, ST, SHL = 1.0e-4, 2.0e-4, 0.6    # strain-shift coefficients (cm^-1 downshift)


# ---------------- Callaway thermal conductivity ----------------
X = np.linspace(1e-3, THETA / T, 500)
OMEGA = X * kB * T / hbar


def scattering_rate(n_bpd, n_tsd, extra=0.0):
    """Total phonon scattering rate 1/tau  (shape: dies x freq). n_* in m^-2."""
    n_bpd = np.atleast_1d(n_bpd)[:, None]
    n_tsd = np.atleast_1d(n_tsd)[:, None]
    w = OMEGA[None, :]
    return (V / L_BND + A_PT * w ** 4
            + B_UM * w ** 2 * T * np.exp(-THETA / (3 * T))
            + C_BPD * n_bpd * w + C_TSD * n_tsd * w ** 3 / W_REF ** 2 + extra)


def spectral_kappa(n_bpd, n_tsd, extra=0.0):
    """Spectral kappa contribution per x (unnormalised)."""
    tau = 1.0 / scattering_rate(n_bpd, n_tsd, extra)
    return tau * X ** 4 * np.exp(X) / (np.exp(X) - 1) ** 2


_pref = kB / (2 * np.pi ** 2 * V) * (kB * T / hbar) ** 3
_cal = KAPPA_PRISTINE / (_pref * np.trapezoid(spectral_kappa(0, 0)[0], X))


def kappa(n_bpd, n_tsd, extra=0.0):
    return _cal * _pref * np.trapezoid(spectral_kappa(n_bpd, n_tsd, extra), X, axis=1)


# ---------------- per-die phonon observables ----------------
def die_phonon_maps(by_type, gx, gy):
    c = {k: die_counts(v, gx, gy) for k, v in by_type.items()}
    n_bpd = c["BPD (clustered)"] / DIE_AREA_CM2 * DENS_SCALE      # cm^-2
    n_tsd = c["TSD"] / DIE_AREA_CM2 * DENS_SCALE
    halo = (c["Micropipe"] * HALO_AREA_MM2["Micropipe"]
            + c["Scratch"] * HALO_AREA_MM2["Scratch"]) / DIE_AREA_MM2
    f = np.clip(halo, 0, 1)

    k_bulk = kappa(n_bpd * 1e4, n_tsd * 1e4)                       # dislocations only
    k_halo = kappa(n_bpd * 1e4, n_tsd * 1e4, extra=V / ELL_HALO)   # inside damage halo
    k_die = (1 - f) * k_bulk + f * k_halo                          # parallel mixing

    fwhm = G0 + GB * n_bpd + GT * n_tsd + GH * f + rng.normal(0, 0.03, len(f))
    shift = -(SB * n_bpd + ST * n_tsd + SHL * f) + rng.normal(0, 0.02, len(f))
    return dict(kappa=k_die, fwhm=fwhm, shift=shift, f=f, n_bpd=n_bpd, n_tsd=n_tsd)


# ---------------- phonon logic gates ----------------
def phonon_flags(m, th_drop=0.05, void_drop=0.20, d_fwhm=0.15, d_shift=0.15):
    drop = 1 - m["kappa"] / KAPPA_PRISTINE
    th = drop > th_drop
    void = drop > void_drop
    br = (m["fwhm"] - G0) > d_fwhm
    sh = np.abs(m["shift"]) > d_shift
    return th, br, sh, void


def phonon_gate_network(th, br, sh, void):
    fatal = np.logical_or(void, np.logical_and(th, br))     # OR( VOID, AND(TH,BR) )
    risk = np.logical_and(sh, br)                           # AND
    recheck = np.logical_xor(th, br)                        # XOR: probes disagree
    passed = np.logical_not(np.logical_or(fatal, risk))     # NOR
    return fatal, risk, recheck, passed


def reason_codes(fatal, risk, recheck, passed):
    code = np.full(len(fatal), "PASS", dtype=object)
    code[recheck & passed] = "XOR recheck (pass)"
    code[risk] = "AND (SH&BR) strain risk"
    code[fatal] = "FATAL (VOID | TH&BR)"
    return code


def probe_agreement_test(th, br, n_perm=999):
    """Permutation test: do thermal and Raman flags co-occur more than chance?"""
    obs = np.sum(th & br)
    null = np.array([np.sum(th & rng.permutation(br)) for _ in range(n_perm)])
    return obs, null, (1 + np.sum(null >= obs)) / (n_perm + 1)


def kappa_autocorr_test(m, gx, gy, n_perm=999):
    """Permute die kappa deficits across die sites -> spatial structure of kappa."""
    deficit = KAPPA_PRISTINE - m["kappa"]
    return die_permutation_test(deficit, gx, gy, n_perm)


def lorentzian(w, w0, gamma):
    return (gamma / 2) ** 2 / ((w - w0) ** 2 + (gamma / 2) ** 2)


# ####################################################################
# 3. GATE LIBRARY, LATTICE, PERMUTATION NULLS
# ####################################################################
# =====================================================================
# 1. NAND-only gate library (functional completeness)
# =====================================================================
def NAND(a, b): return ~(a & b)
def n_not(a): return NAND(a, a)
def n_and(a, b): return n_not(NAND(a, b))
def n_or(a, b): return NAND(n_not(a), n_not(b))
def n_nor(a, b): return n_not(n_or(a, b))
def n_xor(a, b):
    t = NAND(a, b)
    return NAND(NAND(a, t), NAND(b, t))
def n_xnor(a, b): return n_not(n_xor(a, b))
def n_buf(a, b): return n_not(n_not(a))

NAND_NET = {"BUF(A)": n_buf, "NOT(A)": lambda a, b: n_not(a), "AND": n_and, "OR": n_or,
            "NAND": NAND, "NOR": n_nor, "XOR": n_xor, "XNOR": n_xnor}
DIRECT = {"BUF(A)": lambda a, b: a.copy(), "NOT(A)": lambda a, b: ~a, "AND": lambda a, b: a & b,
          "OR": lambda a, b: a | b, "NAND": lambda a, b: ~(a & b), "NOR": lambda a, b: ~(a | b),
          "XOR": lambda a, b: a ^ b, "XNOR": lambda a, b: ~(a ^ b)}


def full_adder(a, b, c):
    x = n_xor(a, b)
    return n_xor(x, c), n_or(n_and(a, b), n_and(x, c))      # sum, carry


def half_adder(a, b):
    return n_xor(a, b), n_and(a, b)


def mux(sel, d0, d1):
    return n_or(n_and(n_not(sel), d0), n_and(sel, d1))


def ripple_carry(X, Y):
    """X, Y: (n_words, 4) bool, LSB first. Returns 5-bit value and carry-out."""
    carry = np.zeros(len(X), bool)
    val = np.zeros(len(X), int)
    for k in range(X.shape[1]):
        s, carry = full_adder(X[:, k], Y[:, k], carry)
        val += s.astype(int) << k
    val += carry.astype(int) << X.shape[1]
    return val, carry


def verify_completeness():
    A = np.array([0, 0, 1, 1], bool); B = np.array([0, 1, 0, 1], bool)
    ok = {k: bool(np.array_equal(NAND_NET[k](A, B), DIRECT[k](A, B))) for k in NAND_NET}
    # full adder exhaustive
    bits = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], bool)
    s, c = full_adder(bits[:, 0], bits[:, 1], bits[:, 2])
    tot = bits.sum(1)
    ok["FullAdder"] = bool(np.array_equal(s, tot % 2 == 1) and np.array_equal(c, tot >= 2))
    # ripple-carry vs integer addition, all 4-bit pairs
    xs, ys = np.meshgrid(np.arange(16), np.arange(16))
    xs, ys = xs.ravel(), ys.ravel()
    bx = ((xs[:, None] >> np.arange(4)) & 1).astype(bool)
    by = ((ys[:, None] >> np.arange(4)) & 1).astype(bool)
    val, _ = ripple_carry(bx, by)
    ok["RippleCarry4"] = bool(np.array_equal(val, xs + ys))
    return ok


# =====================================================================
# 2. Wafer lattice / macro geometry
# =====================================================================
by_type, pts, labels = simulate_wafer()
gx, gy = die_grid()
N_DIE = len(gx)
ix = np.rint((gx - gx.min()) / DIE).astype(int)
iy = np.rint((gy - gy.min()) / DIE).astype(int)
NX = ix.max() + 1 + (ix.max() + 1) % 2
NY = iy.max() + 1 + (iy.max() + 1) % 2
xc, yc = gx + DIE / 2, gy + DIE / 2

_valid = np.zeros((NY, NX), bool); _valid[iy, ix] = True
TILE_VALID = _valid.reshape(NY // 2, 2, NX // 2, 2).sum(axis=(1, 3)) >= 3
T_X = gx.min() + DIE * (2 * np.arange(NX // 2) + 1)
T_Y = gy.min() + DIE * (2 * np.arange(NY // 2) + 1)
TCX, TCY = np.meshgrid(T_X, T_Y)

# 4-die words along rows
WORDS = []
for r in np.unique(iy):
    row = np.where(iy == r)[0]
    row = row[np.argsort(ix[row])]
    for k in range(0, len(row) - 3, 4):
        WORDS.append(row[k:k + 4])
WORDS = np.array(WORDS)

# radial rings, rows, D4 symmetry maps
RING_W = 10.0
ring_id = (np.hypot(xc, yc) // RING_W).astype(int)
RINGS = [np.where(ring_id == k)[0] for k in np.unique(ring_id)]
ROWS = [np.where(iy == r)[0][np.argsort(ix[iy == r])] for r in np.unique(iy)]

_cell = {(a, b): i for i, (a, b) in enumerate(zip(ix, iy))}
_cx, _cy = (ix.max()) / 2, (iy.max()) / 2
def _d4_map(fn):
    m = np.arange(N_DIE)
    for i in range(N_DIE):
        u, v = ix[i] - _cx, iy[i] - _cy
        u2, v2 = fn(u, v)
        m[i] = _cell.get((int(round(u2 + _cx)), int(round(v2 + _cy))), i)
    return m
D4_MAPS = [_d4_map(f) for f in (
    lambda u, v: (-u, v), lambda u, v: (u, -v), lambda u, v: (-u, -v), lambda u, v: (v, u),
    lambda u, v: (-v, -u), lambda u, v: (-v, u), lambda u, v: (v, -u))]


def tile_any(flag):
    g = np.zeros((NY, NX), bool); g[iy, ix] = flag
    return g.reshape(NY // 2, 2, NX // 2, 2).any(axis=(1, 3))


# =====================================================================
# 3. Statistics: how many 1s each gate / macrostructure produces
# =====================================================================
STAT_NAMES = list(DIRECT) + ["FA sum (parity)", "FA carry (2-of-3)", "MUX(SH;TH,BR)",
                             "Ripple sum (4b words)", "Ripple overflow"]


def compute_stats(th, br, sh):
    A, B = tile_any(th), tile_any(br)
    out = [DIRECT[g](A, B)[TILE_VALID].sum() for g in DIRECT]
    s, c = full_adder(th, br, sh)
    out += [s.sum(), c.sum(), mux(sh, th, br).sum()]
    val, carry = ripple_carry(th[WORDS], br[WORDS])
    out += [val.sum(), carry.sum()]
    return np.array(out, float)


# =====================================================================
# 4. Permutation nulls
# =====================================================================
def null_ring_joint(th, br, sh):
    p = np.arange(N_DIE)
    for g in RINGS: p[g] = rng.permutation(g)
    return th[p], br[p], sh[p]


def null_ring_indep(th, br, sh):
    out = []
    for f in (th, br, sh):
        p = np.arange(N_DIE)
        for g in RINGS: p[g] = rng.permutation(g)
        out.append(f[p])
    return tuple(out)


def null_row_roll(th, br, sh):
    p = np.arange(N_DIE)
    for g in ROWS: p[g] = np.roll(g, rng.integers(len(g)))
    return th[p], br[p], sh[p]


def null_d4(th, br, sh):
    p = D4_MAPS[rng.integers(len(D4_MAPS))]
    return th[p], br[p], sh[p]


def null_probe_swap(th, br, sh):
    sw = rng.random(N_DIE) < 0.5
    return np.where(sw, br, th), np.where(sw, th, br), sh


def null_rotate(*_):
    a = rng.uniform(0, 2 * np.pi)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = {k: v @ Rm.T for k, v in by_type.items()}
    m = die_phonon_maps(rot, gx, gy)
    th, br, sh, _ = phonon_flags(m)
    return th, br, sh


NULLS = {"rotate": (null_rotate, 199), "ring-joint": (null_ring_joint, 499),
         "ring-indep": (null_ring_indep, 499), "row-roll": (null_row_roll, 499),
         "D4-symmetry": (null_d4, 499), "probe-swap": (null_probe_swap, 499)}


def bh_fdr(p, alpha=0.05):
    p = np.asarray(p); m = len(p); order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / m
    sig = np.zeros(m, bool)
    if ok.any(): sig[order[:np.max(np.where(ok)[0]) + 1]] = True
    return sig


# ####################################################################
# 4. SHA-256 MACRO DIES
# ####################################################################
# ====================================================================
# 1. SHA-256 constants (derived, then checked against hashlib)
# ====================================================================
def _primes(n):
    p, k = [], 2
    while len(p) < n:
        if all(k % q for q in p): p.append(k)
        k += 1
    return p

def _icbrt(n):
    lo, hi = 0, 1 << (n.bit_length() // 3 + 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** 3 <= n: lo = mid
        else: hi = mid - 1
    return lo

_P = _primes(64)
K = [_icbrt(p << 96) & 0xFFFFFFFF for p in _P]
H0 = [__import__("math").isqrt(p << 64) & 0xFFFFFFFF for p in _P[:8]]
assert K[0] == 0x428A2F98 and H0[0] == 0x6A09E667


# ====================================================================
# 2. NAND-only fabric, bit-parallel across macro dies (bit m = macro die m)
# ====================================================================
class Fabric:
    def __init__(self, M, sched=None, track=False):
        self.M, self.ALL = M, (1 << M) - 1
        self.c, self.sched, self.track = 0, sched or {}, track
        self.segs, self.cat, self.cats, self._last = [], "misc", defaultdict(int), None

    def stage(self, s):
        if self.track and s != self._last:
            self.segs.append((self.c, s)); self._last = s

    def nand(self, a, b):
        o = ~(a & b) & self.ALL
        f = self.sched.get(self.c)
        if f is not None:                      # fault injection on this physical gate
            s1, s0, fl = f
            o = ((o ^ fl) | s1) & ~s0 & self.ALL
        if self.track: self.cats[self.cat] += 1
        self.c += 1
        return o

    # derived gates (NAND-only)
    def not_(self, a): return self.nand(a, a)
    def and_(self, a, b): return self.not_(self.nand(a, b))
    def or_(self, a, b): return self.nand(self.not_(a), self.not_(b))
    def xor(self, a, b):
        t = self.nand(a, b)
        return self.nand(self.nand(a, t), self.nand(b, t))

    # word-level helpers (list of 32 ints, index = bit position, LSB = 0)
    def xorw(self, x, y): return [self.xor(a, b) for a, b in zip(x, y)]
    def xor3w(self, x, y, z): return self.xorw(self.xorw(x, y), z)

    def add(self, x, y):                       # 32-bit ripple-carry adder (full adders)
        carry, out = 0, []
        for a, b in zip(x, y):
            t = self.xor(a, b)
            out.append(self.xor(t, carry))
            carry = self.or_(self.and_(a, b), self.and_(t, carry))
        return out

    def const(self, v): return [self.ALL if (v >> k) & 1 else 0 for k in range(32)]


def rotr(x, n): return [x[(k + n) % 32] for k in range(32)]
def shr(x, n): return [x[k + n] if k + n < 32 else 0 for k in range(32)]


def sha256_block(f, Wblk):
    """Single-block SHA-256 compression. Wblk: 16 words (lists of 32 ints)."""
    W = list(Wblk)
    for t in range(16, 64):                                    # message schedule
        f.stage(min(3, t // 16)); f.cat = "schedule"
        s0 = f.xor3w(rotr(W[t - 15], 7), rotr(W[t - 15], 18), shr(W[t - 15], 3))
        s1 = f.xor3w(rotr(W[t - 2], 17), rotr(W[t - 2], 19), shr(W[t - 2], 10))
        W.append(f.add(f.add(f.add(W[t - 16], s0), W[t - 7]), s1))
    a, b, c, d, e, ff, g, h = [f.const(v) for v in H0]
    for t in range(64):
        f.stage(t // 16)
        f.cat = "sigma"
        S1 = f.xor3w(rotr(e, 6), rotr(e, 11), rotr(e, 25))
        S0 = f.xor3w(rotr(a, 2), rotr(a, 13), rotr(a, 22))
        f.cat = "ch/maj"
        ch = [f.or_(f.and_(e[k], ff[k]), f.and_(f.not_(e[k]), g[k])) for k in range(32)]
        mj = [f.or_(f.and_(a[k], b[k]), f.and_(c[k], f.or_(a[k], b[k]))) for k in range(32)]
        f.cat = "adders"
        T1 = f.add(f.add(f.add(f.add(h, S1), ch), f.const(K[t])), W[t])
        T2 = f.add(S0, mj)
        h, g, ff, e = g, ff, e, f.add(d, T1)
        d, c, b, a = c, b, a, f.add(T1, T2)
    f.stage(3); f.cat = "final adds"
    return [f.add(f.const(H0[i]), v) for i, v in enumerate([a, b, c, d, e, ff, g, h])]


def pack_blocks(msgs):
    """Pad each message to one 512-bit block and pack bit m of word t -> macro m."""
    blocks = []
    for s in msgs:
        b = s.encode(); assert len(b) <= 55
        blocks.append(b + b"\x80" + b"\x00" * (55 - len(b)) + (8 * len(b)).to_bytes(8, "big"))
    W = []
    for t in range(16):
        words = [int.from_bytes(bl[4 * t:4 * t + 4], "big") for bl in blocks]
        W.append([sum(((w >> k) & 1) << m for m, w in enumerate(words)) for k in range(32)])
    return W


def extract(words8, M):
    out = []
    for m in range(M):
        h = 0
        for w in range(8):
            h = (h << 32) | sum(((words8[w][k] >> m) & 1) << k for k in range(32))
        out.append(h)
    return out


# ====================================================================
# 3. Macro-die floorplan on the wafer (2x2 tiles) and phonon-driven fault rates
# ====================================================================
m_ = die_phonon_maps(by_type, gx, gy)
th, br, sh, void = phonon_flags(m_)
fatal, risk, recheck, passed = phonon_gate_network(th, br, sh, void)
drop_pct = 100 * (1 - m_["kappa"] / KAPPA_PRISTINE)
xc, yc = gx + DIE / 2, gy + DIE / 2

dgrid = np.full((NY, NX), -1); dgrid[iy, ix] = np.arange(len(gx))
tiles = [tuple(p) for p in np.argwhere(TILE_VALID)]          # row-major
M = len(tiles)
SNAKE = [(0, 0), (0, 1), (1, 1), (1, 0)]
SUB = np.zeros((M, 4), int)
for mi, (ti, tj) in enumerate(tiles):
    ds = [dgrid[2 * ti + a, 2 * tj + b] for a, b in SNAKE]
    good = [d for d in ds if d >= 0]
    SUB[mi] = [d if d >= 0 else good[k % len(good)] for k, d in enumerate(ds)]   # reroute missing

PF_PER_PCT, PT_RISK, PT_RECHECK = 2e-6, 1e-5, 2e-6
P_STUCK = np.where(fatal, PF_PER_PCT * np.clip(drop_pct, 0, None), 0.0)
P_FLIP = np.where(~fatal & risk, PT_RISK, np.where(~fatal & recheck, PT_RECHECK, 0.0))

# redundancy groups: 3 neighbouring macro dies in one tile row share a message
GROUP = np.zeros(M, int); TRIPLES = []
gid, rows = 0, {}
for mi, (ti, tj) in enumerate(tiles): rows.setdefault(ti, []).append(mi)
for ti, lst in rows.items():
    for k in range(0, len(lst), 3):
        chunk = lst[k:k + 3]
        GROUP[chunk] = gid; gid += 1
        if len(chunk) == 3: TRIPLES.append(chunk)
MSGS = [f"SiC-wafer macro group {GROUP[mi]:03d}" for mi in range(M)]
EXPECT = [int(hashlib.sha256(s.encode()).hexdigest(), 16) for s in MSGS]


def build_schedule(segs, N):
    sched = {}
    for (s, e, sub) in segs:
        L = e - s
        for mi in range(M):
            d = SUB[mi, sub]
            for kind, p in (("stuck", P_STUCK[d]), ("flip", P_FLIP[d])):
                if p <= 0: continue
                for idx in rng.integers(s, e, rng.binomial(L, p)):
                    ent = sched.get(int(idx), [0, 0, 0])
                    if kind == "stuck": ent[0 if rng.integers(2) else 1] |= 1 << mi
                    else: ent[2] |= 1 << mi
                    sched[int(idx)] = ent
    return {k: tuple(v) for k, v in sched.items()}


def tmr_vote(h1, h2, h3):
    """2-of-3 majority over 256 hash bits, built from NAND gates (assumed fault-free)."""
    v = Fabric(256)
    x = v.xor(h1, h2)
    return v.or_(v.and_(h1, h2), v.and_(x, h3))



"""
SHA-256 computed by PHONON-TRANSPORT (thermal) logic on the defective SiC wafer.

Signals are TEMPERATURES, not bits:  logic 0 = 300 K (cold rail), logic 1 = 360 K (hot rail).
Every NAND is a heat-flow network solved from energy balance at its output node:

      T_hot --[ G_load ]--- (T_out) ---[ switch A + SiC segment A ]---[ switch B + SiC segment B ]--- T_cold

   T_out = (G_load*T_hot + G_ch*T_cold) / (G_load + G_ch)

   * G_SiC segment = kappa(T) * A / L     <- REAL phonon physics: Callaway model, evaluated at the
                                              local temperature, with that die's measured defect
                                              scattering (BPD ~w, TSD ~w^3, point ~w^4, Umklapp)
   * switch conductance = thermal transistor / thermal switch (steep, thermally activated).
                                              SiC alone has kappa FALLING with T (Umklapp), so it
                                              cannot give gain; the switch is a PHENOMENOLOGICAL
                                              stand-in (VO2-like), the one non-first-principles part.
   * Each gate segment lands in one of three PHYSICAL states, sampled from the die's defect map:
        bulk   : dislocation-scattered Callaway kappa
        halo   : strain halo around micropipes/scratches (extra phonon scattering, MFP 0.3 um)
        core   : hollow/cracked core (phonon boundary scattering, MFP 10 nm)
     A segment with low kappa cannot pull the output node down -> the gate misreads. Faults are
     therefore EMERGENT from phonon transport; nothing is injected by hand.
   * Thermal noise on every node: sigma_T = sqrt(kB T^2 / C_node).

The SHA-256 netlist is the same NAND-only one as before (verified against hashlib); only the
gate physics changed. Needs sic_wafer_sha256_standalone.py in the same folder.
"""
import csv
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
"""
SHA-256 computed by PHONON-TRANSPORT (thermal) logic on the defective SiC wafer.

Signals are TEMPERATURES, not bits:  logic 0 = 300 K (cold rail), logic 1 = 360 K (hot rail).
Every NAND is a heat-flow network solved from energy balance at its output node:

      T_hot --[ G_load ]--- (T_out) ---[ switch A + SiC segment A ]---[ switch B + SiC segment B ]--- T_cold

   T_out = (G_load*T_hot + G_ch*T_cold) / (G_load + G_ch)

   * G_SiC segment = kappa(T) * A / L     <- REAL phonon physics: Callaway model, evaluated at the
                                              local temperature, with that die's measured defect
                                              scattering (BPD ~w, TSD ~w^3, point ~w^4, Umklapp)
   * switch conductance = thermal transistor / thermal switch (steep, thermally activated).
                                              SiC alone has kappa FALLING with T (Umklapp), so it
                                              cannot give gain; the switch is a PHENOMENOLOGICAL
                                              stand-in (VO2-like), the one non-first-principles part.
   * Each gate segment lands in one of three PHYSICAL states, sampled from the die's defect map:
        bulk   : dislocation-scattered Callaway kappa
        halo   : strain halo around micropipes/scratches (extra phonon scattering, MFP 0.3 um)
        core   : hollow/cracked core (phonon boundary scattering, MFP 10 nm)
     A segment with low kappa cannot pull the output node down -> the gate misreads. Faults are
     therefore EMERGENT from phonon transport; nothing is injected by hand.
   * Thermal noise on every node: sigma_T = sqrt(kB T^2 / C_node).

The SHA-256 netlist is the same NAND-only one as before (verified against hashlib); only the
gate physics changed. Needs sic_wafer_sha256_standalone.py in the same folder.
"""
import csv
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# ====================================================================
# 1. Phonon physics: Callaway kappa(T) with defect scattering
# ====================================================================
ELL_CORE = 1e-8                      # m, phonon MFP inside a hollow/cracked core
_pref0 = kB / (2 * np.pi ** 2 * V)


def kappa_at(Tv, n_bpd, n_tsd, extra=0.0):
    """Callaway kappa (W/mK) at temperature Tv; n_* in m^-2 arrays (one entry per die)."""
    x = np.linspace(1e-3, THETA / Tv, 400)
    w = x * kB * Tv / hbar
    rate = (V / L_BND + A_PT * w ** 4 + B_UM * w ** 2 * Tv * np.exp(-THETA / (3 * Tv))
            + C_BPD * n_bpd[:, None] * w + C_TSD * n_tsd[:, None] * w ** 3 / W_REF ** 2 + extra)
    integ = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2 / rate
    return _cal * _pref0 * (kB * Tv / hbar) ** 3 * np.trapezoid(integ, x, axis=1)


TGRID = np.linspace(290.0, 370.0, 17); DTG = TGRID[1] - TGRID[0]; NTG = len(TGRID)
m_ = m_
NB, NT_ = m_["n_bpd"] * 1e4, m_["n_tsd"] * 1e4
KT = np.zeros((len(gx), NTG, 3))                                     # die x T x state(bulk,halo,core)
for i, Tv in enumerate(TGRID):
    KT[:, i, 0] = kappa_at(Tv, NB, NT_)
    KT[:, i, 1] = kappa_at(Tv, NB, NT_, V / ELL_HALO)
    KT[:, i, 2] = kappa_at(Tv, NB, NT_, V / ELL_CORE)
RT = KT / KAPPA_PRISTINE                                             # relative conductance

# physical state probabilities per die (area fractions)
c_mp = die_counts(by_type["Micropipe"], gx, gy)
c_sc = die_counts(by_type["Scratch"], gx, gy)
A_CORE_MP = np.pi * 0.025 ** 2          # mm^2, 25 um radius micropipe core
A_CORE_SC = 0.010 * (1 / 1.5)           # mm^2, 10 um wide x 0.67 mm per scratch marker
P_CORE = np.clip((c_mp * A_CORE_MP + c_sc * A_CORE_SC) / DIE_AREA_MM2, 0, 1)
P_HALO = np.clip(m_["f"] - P_CORE, 0, 1)

# ====================================================================
# 2. Thermal NAND gate (heat-balance network)
# ====================================================================
TC, TH = 300.0, 360.0                   # K, logic 0 / logic 1 rails
# ---- FAST THERMAL NAND PHYSICS -------------------------------------------
# Rise time  ~ C_NODE / (LAMBDA * G_SEG0)
# Fall time  ~ C_NODE / (G_ON_eff * G_SEG0)
# Therefore: reduce C_NODE and increase G_SEG0 / G_ON.

T_TR, W_SW = 330.0, 1.5                 # K, sharper thermal-switch transition
A_CH, L_CH = (150e-9) ** 2, 2.5e-7      # m^2, m: 9x area, 4x shorter -> 36x G_SEG0
G_SEG0 = KAPPA_PRISTINE * A_CH / L_CH   # W/K, pristine SiC segment at 300 K
G_ON, G_OFF = 200 * G_SEG0, 1e-3 * G_SEG0
LN_RATIO = np.log(G_ON / G_OFF)
LAMBDA = 0.35                           # G_load / G_SEG0: faster rise, retains low-level margin
C_NODE = 2.2e6 * (40e-9) ** 3           # J/K, 40 nm SiC node: ~15.6x lower heat capacity
SIGMA_T = np.sqrt(kB * 330.0 ** 2 / C_NODE)

# Reported speed-up relative to the original design:
#   rise: ~(36 * 15.6) / (0.35 / 0.21) ~= 320x faster
#   fall: ~(36 * 15.6) * (200 / 50)    ~= 2250x faster
# Static heat while pulling low rises by roughly 36x because it scales with G_SEG0.
# --------------------------------------------------------------------------


def switch_G(T):
    s = 1.0 / (1.0 + np.exp(-(T - T_TR) / W_SW))
    return G_OFF * np.exp(s * LN_RATIO)                  # log-interpolated off->on conductance


def nand_T(Ta, Tb, rA, rB, lam=LAMBDA):
    """Static output temperature of the thermal NAND. rA/rB: callables T_seg -> rel. SiC conductance."""
    swA, swB = switch_G(Ta), switch_G(Tb)
    Tseg = TC + 6.0
    for _ in range(2):                                   # kappa(T) self-consistency
        gA = 1.0 / (1.0 / swA + 1.0 / (G_SEG0 * rA(Tseg)))
        gB = 1.0 / (1.0 / swB + 1.0 / (G_SEG0 * rB(Tseg)))
        Gch = 1.0 / (1.0 / gA + 1.0 / gB)
        To = (lam * G_SEG0 * TH + Gch * TC) / (lam * G_SEG0 + Gch)
        Tseg = TC + 0.5 * (To - TC)
    return To


class ThermalFabric(Fabric):
    """Same netlist interface as the Boolean fabric, but every gate is a thermal NAND."""
    B = 1024

    def __init__(self, SUB_all, real):
        self.Mt = len(real); self.SUB, self.real = SUB_all, real.astype(float)
        self.ar = np.arange(self.Mt); self.c = 0; self.cat = "misc"; self.s = None
        self.LO, self.HI = np.full(self.Mt, TC), np.full(self.Mt, TH)
        self.hit_core = np.zeros(self.Mt, int); self.hit_halo = np.zeros(self.Mt, int)
        self.ptr = self.B; self.track = False

    def stage(self, s):
        if s == self.s: return
        self.s = s; dd = self.SUB[:, s]
        self.Rs = RT[dd]
        self.pc, self.ph = P_CORE[dd] * self.real, P_HALO[dd] * self.real
        self.pch = self.pc + self.ph

    def _refill(self):
        self.U = rng.random((self.B, 2, self.Mt)); self.Z = rng.standard_normal((self.B, self.Mt)); self.ptr = 0

    def r_of(self, T, st):
        x = (T - TGRID[0]) / DTG
        i0 = np.clip(x.astype(int), 0, NTG - 2); w = x - i0
        a = self.Rs[self.ar, i0, st]; b = self.Rs[self.ar, i0 + 1, st]
        return a + (b - a) * w

    def _arr(self, x):
        return (self.HI if x else self.LO) if isinstance(x, (int, np.integer)) else x

    def nand(self, a, b):
        a, b = self._arr(a), self._arr(b)
        if self.ptr == self.B: self._refill()
        u, z = self.U[self.ptr], self.Z[self.ptr]; self.ptr += 1
        stA = np.where(u[0] < self.pc, 2, np.where(u[0] < self.pch, 1, 0))
        stB = np.where(u[1] < self.pc, 2, np.where(u[1] < self.pch, 1, 0))
        self.hit_core += (stA == 2) + (stB == 2); self.hit_halo += (stA == 1) + (stB == 1)
        To = nand_T(a, b, lambda T: self.r_of(T, stA), lambda T: self.r_of(T, stB))
        self.c += 1
        return To + SIGMA_T * z

    def const(self, v): return [self.HI if (v >> k) & 1 else self.LO for k in range(32)]

    def add(self, x, y):
        carry, out = self.LO, []
        for a, b in zip(x, y):
            t = self.xor(a, b); out.append(self.xor(t, carry))
            carry = self.or_(self.and_(a, b), self.and_(t, carry))
        return out


def to_digests(words8):
    Tm = np.array(words8)                                # (8, 32, Mt) temperatures
    bits = (Tm > T_TR).astype(np.uint64)
    wts = (np.uint64(1) << np.arange(32, dtype=np.uint64))[None, :, None]
    vals = (bits * wts).sum(1)                           # (8, Mt)
    return [int("".join(f"{int(vals[w, m]):08x}" for w in range(8)), 16) for m in range(Tm.shape[2])], Tm

# ====================================================================
# 1. Phonon physics: Callaway kappa(T) with defect scattering
# ====================================================================
ELL_CORE = 1e-8                      # m, phonon MFP inside a hollow/cracked core
_pref0 = kB / (2 * np.pi ** 2 * V)


def kappa_at(Tv, n_bpd, n_tsd, extra=0.0):
    """Callaway kappa (W/mK) at temperature Tv; n_* in m^-2 arrays (one entry per die)."""
    x = np.linspace(1e-3, THETA / Tv, 400)
    w = x * kB * Tv / hbar
    rate = (V / L_BND + A_PT * w ** 4 + B_UM * w ** 2 * Tv * np.exp(-THETA / (3 * Tv))
            + C_BPD * n_bpd[:, None] * w + C_TSD * n_tsd[:, None] * w ** 3 / W_REF ** 2 + extra)
    integ = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2 / rate
    return _cal * _pref0 * (kB * Tv / hbar) ** 3 * np.trapezoid(integ, x, axis=1)


TGRID = np.linspace(290.0, 370.0, 17); DTG = TGRID[1] - TGRID[0]; NTG = len(TGRID)
m_ = m_
NB, NT_ = m_["n_bpd"] * 1e4, m_["n_tsd"] * 1e4
KT = np.zeros((len(gx), NTG, 3))                                     # die x T x state(bulk,halo,core)
for i, Tv in enumerate(TGRID):
    KT[:, i, 0] = kappa_at(Tv, NB, NT_)
    KT[:, i, 1] = kappa_at(Tv, NB, NT_, V / ELL_HALO)
    KT[:, i, 2] = kappa_at(Tv, NB, NT_, V / ELL_CORE)
RT = KT / KAPPA_PRISTINE                                             # relative conductance

# physical state probabilities per die (area fractions)
c_mp = die_counts(by_type["Micropipe"], gx, gy)
c_sc = die_counts(by_type["Scratch"], gx, gy)
A_CORE_MP = np.pi * 0.025 ** 2          # mm^2, 25 um radius micropipe core
A_CORE_SC = 0.010 * (1 / 1.5)           # mm^2, 10 um wide x 0.67 mm per scratch marker
P_CORE = np.clip((c_mp * A_CORE_MP + c_sc * A_CORE_SC) / DIE_AREA_MM2, 0, 1)
P_HALO = np.clip(m_["f"] - P_CORE, 0, 1)

# ====================================================================
# 2. Thermal NAND gate (heat-balance network)
# ====================================================================
TC, TH = 300.0, 360.0                   # K, logic 0 / logic 1 rails
# ---- FAST THERMAL NAND PHYSICS -------------------------------------------
# Rise time  ~ C_NODE / (LAMBDA * G_SEG0)
# Fall time  ~ C_NODE / (G_ON_eff * G_SEG0)
# Therefore: reduce C_NODE and increase G_SEG0 / G_ON.

T_TR, W_SW = 330.0, 1.5                 # K, sharper thermal-switch transition
A_CH, L_CH = (150e-9) ** 2, 2.5e-7      # m^2, m: 9x area, 4x shorter -> 36x G_SEG0
G_SEG0 = KAPPA_PRISTINE * A_CH / L_CH   # W/K, pristine SiC segment at 300 K
G_ON, G_OFF = 200 * G_SEG0, 1e-3 * G_SEG0
LN_RATIO = np.log(G_ON / G_OFF)
LAMBDA = 0.35                           # G_load / G_SEG0: faster rise, retains low-level margin
C_NODE = 2.2e6 * (40e-9) ** 3           # J/K, 40 nm SiC node: ~15.6x lower heat capacity
SIGMA_T = np.sqrt(kB * 330.0 ** 2 / C_NODE)

# Reported speed-up relative to the original design:
#   rise: ~(36 * 15.6) / (0.35 / 0.21) ~= 320x faster
#   fall: ~(36 * 15.6) * (200 / 50)    ~= 2250x faster
# Static heat while pulling low rises by roughly 36x because it scales with G_SEG0.
# --------------------------------------------------------------------------


def switch_G(T):
    s = 1.0 / (1.0 + np.exp(-(T - T_TR) / W_SW))
    return G_OFF * np.exp(s * LN_RATIO)                  # log-interpolated off->on conductance


def nand_T(Ta, Tb, rA, rB, lam=LAMBDA):
    """Static output temperature of the thermal NAND. rA/rB: callables T_seg -> rel. SiC conductance."""
    swA, swB = switch_G(Ta), switch_G(Tb)
    Tseg = TC + 6.0
    for _ in range(2):                                   # kappa(T) self-consistency
        gA = 1.0 / (1.0 / swA + 1.0 / (G_SEG0 * rA(Tseg)))
        gB = 1.0 / (1.0 / swB + 1.0 / (G_SEG0 * rB(Tseg)))
        Gch = 1.0 / (1.0 / gA + 1.0 / gB)
        To = (lam * G_SEG0 * TH + Gch * TC) / (lam * G_SEG0 + Gch)
        Tseg = TC + 0.5 * (To - TC)
    return To


class ThermalFabric(Fabric):
    """Same netlist interface as the Boolean fabric, but every gate is a thermal NAND."""
    B = 1024

    def __init__(self, SUB_all, real):
        self.Mt = len(real); self.SUB, self.real = SUB_all, real.astype(float)
        self.ar = np.arange(self.Mt); self.c = 0; self.cat = "misc"; self.s = None
        self.LO, self.HI = np.full(self.Mt, TC), np.full(self.Mt, TH)
        self.hit_core = np.zeros(self.Mt, int); self.hit_halo = np.zeros(self.Mt, int)
        self.ptr = self.B; self.track = False

    def stage(self, s):
        if s == self.s: return
        self.s = s; dd = self.SUB[:, s]
        self.Rs = RT[dd]
        self.pc, self.ph = P_CORE[dd] * self.real, P_HALO[dd] * self.real
        self.pch = self.pc + self.ph

    def _refill(self):
        self.U = rng.random((self.B, 2, self.Mt)); self.Z = rng.standard_normal((self.B, self.Mt)); self.ptr = 0

    def r_of(self, T, st):
        x = (T - TGRID[0]) / DTG
        i0 = np.clip(x.astype(int), 0, NTG - 2); w = x - i0
        a = self.Rs[self.ar, i0, st]; b = self.Rs[self.ar, i0 + 1, st]
        return a + (b - a) * w

    def _arr(self, x):
        return (self.HI if x else self.LO) if isinstance(x, (int, np.integer)) else x

    def nand(self, a, b):
        a, b = self._arr(a), self._arr(b)
        if self.ptr == self.B: self._refill()
        u, z = self.U[self.ptr], self.Z[self.ptr]; self.ptr += 1
        stA = np.where(u[0] < self.pc, 2, np.where(u[0] < self.pch, 1, 0))
        stB = np.where(u[1] < self.pc, 2, np.where(u[1] < self.pch, 1, 0))
        self.hit_core += (stA == 2) + (stB == 2); self.hit_halo += (stA == 1) + (stB == 1)
        To = nand_T(a, b, lambda T: self.r_of(T, stA), lambda T: self.r_of(T, stB))
        self.c += 1
        return To + SIGMA_T * z

    def const(self, v): return [self.HI if (v >> k) & 1 else self.LO for k in range(32)]

    def add(self, x, y):
        carry, out = self.LO, []
        for a, b in zip(x, y):
            t = self.xor(a, b); out.append(self.xor(t, carry))
            carry = self.or_(self.and_(a, b), self.and_(t, carry))
        return out


def to_digests(words8):
    Tm = np.array(words8)                                # (8, 32, Mt) temperatures
    bits = (Tm > T_TR).astype(np.uint64)
    wts = (np.uint64(1) << np.arange(32, dtype=np.uint64))[None, :, None]
    vals = (bits * wts).sum(1)                           # (8, Mt)
    return [int("".join(f"{int(vals[w, m]):08x}" for w in range(8)), 16) for m in range(Tm.shape[2])], Tm


# ====================================================================
# 3. Main
# ====================================================================
if __name__ == "__main__":
    # ---- phonon / gate characterisation ----
    clean = int(np.argmin(NB + NT_ + P_CORE * 1e6 + P_HALO * 1e6))
    dense = int(np.argmax(NB + NT_))
    i300, i305 = 2, 3
    print("Callaway kappa at 300 K (W/mK):")
    print(f"   pristine bulk {KT[clean, i300, 0]:.0f} | dislocation-dense die {KT[dense, i300, 0]:.0f} | "
          f"strain halo {KT[clean, i300, 1]:.0f} | hollow core {KT[clean, i300, 2]:.0f}")
    print(f"   kappa(T) pristine: 300K {KT[clean,2,0]:.0f}  330K {KT[clean,8,0]:.0f}  360K {KT[clean,14,0]:.0f}  "
          f"-> falls with T (Umklapp), so SiC alone has no gain; thermal switch supplies it")
    rf = lambda st, d=clean: (lambda T: np.interp(T, TGRID, RT[d, :, st]))
    Tin = np.linspace(295, 365, 281)
    vtc = {n: nand_T(Tin, Tin, rf(st, d), rf(st, d)) for n, (st, d) in
           {"pristine": (0, clean), "dislocation-dense": (0, dense), "strain halo": (1, clean), "hollow core": (2, clean)}.items()}
    g = np.gradient(vtc["pristine"], Tin)
    steep = Tin[g < -1]
    VIL, VIH = steep.min(), steep.max()
    VOL, VOH = vtc["pristine"][-1], vtc["pristine"][0]
    print(f"\nThermal NAND (pristine SiC): V_OL={VOL:.1f} K  V_OH={VOH:.1f} K  max gain={abs(g.min()):.1f}")
    print(f"   noise margins: NM_low={VIL - VOL:.1f} K  NM_high={VOH - VIH:.1f} K   (swing {TH - TC:.0f} K)")
    for n in vtc: print(f"   NAND(1,1) low level, {n:<18}: {vtc[n][-1]:.1f} K  "
                        f"({'OK' if vtc[n][-1] < T_TR - W_SW else 'FAILS -> reads as 1'})")
    Gl = LAMBDA * G_SEG0
    Gon = 1 / (1 / Gl + 2 / (G_SEG0 * 0.96 * 0.5 * 2))
    tau_f, tau_r = C_NODE / (Gl + G_SEG0 * 0.48), C_NODE / Gl
    P_on = (TH - TC) ** 2 * (Gl * G_SEG0 * 0.48 / (Gl + G_SEG0 * 0.48))
    print(f"   FAST NAND geometry: A_CH={A_CH:.3e} m^2, L_CH={L_CH:.3e} m, node=40 nm")
    print(f"   G_SEG0={G_SEG0:.3e} W/K, G_ON={G_ON:.3e} W/K, LAMBDA={LAMBDA:.3f}")
    print(f"   delay: fall {tau_f*1e12:.1f} ps, rise {tau_r*1e12:.1f} ps | thermal noise sigma_T={SIGMA_T*1e3:.1f} mK")
    print(f"   static heat flow when output pulled low: {P_on*1e3:.2f} mW/gate")
    print("   NOTE: faster edges trade directly against higher static thermal power.")

    # ---- SHA-256 on the thermal fabric: wafer macro dies + 4 pristine reference macros ----
    n_ref = 4
    REF_MSGS = [f"reference thermal macro {i}" for i in range(n_ref)]
    msgs_all = MSGS + REF_MSGS
    exp_all = EXPECT + [int(hashlib.sha256(s.encode()).hexdigest(), 16) for s in REF_MSGS]
    Mt = M + n_ref
    SUB_all = np.vstack([SUB, np.full((n_ref, 4), clean)])
    real = np.arange(Mt) < M
    Wi = pack_blocks(msgs_all)
    Wt = [[np.where(np.array([(x >> m) & 1 for m in range(Mt)], bool), TH, TC) for x in word] for word in Wi]

    fab = ThermalFabric(SUB_all, real)
    print(f"\nRunning SHA-256 through {Mt} thermal macro dies ...")
    got, Tm = to_digests(sha256_block(fab, Wt))
    N = fab.c
    ok = np.array([g_ == e for g_, e in zip(got, exp_all)])
    print(f"Thermal NAND gates per macro die: {N:,}   (heat dissipated ~{N * 0.5 * P_on:.0f} W if half are pulling low)")
    print(f"Pristine reference macros correct: {ok[M:].sum()}/{n_ref}")
    assert ok[M:].all(), "thermal gates fail even on pristine SiC - adjust design (LAMBDA / W_SW)"
    correct = ok[:M]
    ham = np.array([bin(g_ ^ e).count("1") for g_, e in zip(got[:M], exp_all[:M])])
    wrong = ~correct

    hard = (P_CORE[SUB] > 0).any(1); soft = ((P_HALO[SUB] > 0).any(1) | (NB[SUB] + NT_[SUB] > 0).any(1)) & ~hard
    cat = np.where(hard, "hollow core / scratch", np.where(soft, "halo / dislocations only", "clean"))
    print(f"\nWafer macro dies: {correct.sum()}/{M} hashes correct ({100 * correct.mean():.1f}%)")
    for c in ("clean", "halo / dislocations only", "hollow core / scratch"):
        s_ = cat == c
        print(f"   {c:<26}: {correct[s_].sum():>3}/{s_.sum():<3} correct   "
              f"(avg gate segments in core state {fab.hit_core[:M][s_].mean():.1f}, halo state {fab.hit_halo[:M][s_].mean():.0f})")
    if wrong.any():
        print(f"   wrong digests differ by {ham[wrong].mean():.1f}/256 bits on average (avalanche ideal 128)")

    tmr_ok = np.array([tmr_vote(got[a], got[b], got[c]) == exp_all[a] for a, b, c in TRIPLES])
    raw = np.mean([correct[t].mean() for t in TRIPLES])
    print(f"TMR triples: raw replica yield {100 * raw:.1f}% -> voted {100 * tmr_ok.mean():.1f}% (ideal voter)")

    out_T = Tm.reshape(-1, Mt)                            # final digest-bit temperatures
    marg = np.abs(out_T - T_TR)
    print(f"Output-bit temperatures: pristine refs min margin {marg[:, M:].min():.1f} K | wafer min margin {marg[:, :M].min():.1f} K")

    with open("phonon_logic_macro_dies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["macro", "cx_mm", "cy_mm", "class", "core_state_segments", "halo_state_segments",
                    "correct", "hamming_bits"])
        for mi, (ti, tj) in enumerate(tiles):
            w.writerow([mi, f"{TCX[ti, tj]:.0f}", f"{TCY[ti, tj]:.0f}", cat[mi], fab.hit_core[mi],
                        fab.hit_halo[mi], int(correct[mi]), ham[mi]])

    # ---- plots ----
    fig, ax = plt.subplots(2, 3, figsize=(20, 12.5))
    a = ax[0, 0]
    for lab, d, st, c in (("pristine bulk", clean, 0, "k"), ("dislocation-dense die", dense, 0, "tab:blue"),
                          ("strain halo (MFP 0.3 um)", clean, 1, "tab:orange"), ("hollow core (MFP 10 nm)", clean, 2, "crimson")):
        a.plot(TGRID, KT[d, :, st], c=c, marker="o", ms=3, label=lab)
    a.axvspan(TC, TH, color="gold", alpha=0.15, label="logic temperature range")
    a.set_xlabel("temperature (K)"); a.set_ylabel("Callaway kappa (W/mK)"); a.set_yscale("log"); a.legend(fontsize=8)
    a.set_title("Phonon thermal conductivity of each physical state")

    a = ax[0, 1]
    for n, c in zip(vtc, ("k", "tab:blue", "tab:orange", "crimson")): a.plot(Tin, vtc[n], c=c, label=n)
    a.plot(Tin, Tin, "g:", lw=0.8); a.axvline(T_TR, c="gray", ls="--", lw=0.8); a.axhline(T_TR, c="gray", ls="--", lw=0.8)
    a.set_xlabel("input temperature (K)"); a.set_ylabel("output temperature (K)"); a.legend(fontsize=8)
    a.set_title(f"Thermal NAND transfer curve (inputs tied)\nNM_low={VIL - VOL:.0f} K, NM_high={VOH - VIH:.0f} K, gain={abs(g.min()):.0f}")

    a = ax[0, 2]
    combos = [(TC, TC), (TC, TH), (TH, TC), (TH, TH)]
    wd = 0.2
    for k_, (n, c) in enumerate(zip(vtc, ("k", "tab:blue", "tab:orange", "crimson"))):
        st, d = {"pristine": (0, clean), "dislocation-dense": (0, dense), "strain halo": (1, clean), "hollow core": (2, clean)}[n]
        ys = [float(nand_T(np.array(x), np.array(y_), rf(st, d), rf(st, d))) for x, y_ in combos]
        a.bar(np.arange(4) + (k_ - 1.5) * wd, ys, wd, color=c, label=n)
    a.axhline(T_TR, c="gray", ls="--"); a.set_xticks(range(4)); a.set_xticklabels(["00", "01", "10", "11"])
    a.set_ylim(290, 370); a.set_xlabel("inputs (A,B)"); a.set_ylabel("output temperature (K)"); a.legend(fontsize=8)
    a.set_title("NAND truth table in temperature (needs 11 -> cold)")

    th_ = np.linspace(0, 2 * np.pi, 300)
    a = ax[1, 0]
    a.plot(R * np.cos(th_), R * np.sin(th_), "k", lw=1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    xc, yc = gx + DIE / 2, gy + DIE / 2
    a.scatter(xc, yc, c=np.where(P_CORE > 0, "crimson", np.where(P_HALO > 0, "gold", "lightgray")), marker="s", s=34, zorder=2)
    for mi, (ti, tj) in enumerate(tiles):
        a.add_patch(Rectangle((TCX[ti, tj] - DIE, TCY[ti, tj] - DIE), 2 * DIE, 2 * DIE, fc="limegreen" if correct[mi] else "black",
                              alpha=0.30 if correct[mi] else 0.55, ec="green" if correct[mi] else "red", lw=1.1, zorder=3))
    a.set_title(f"Thermal-logic SHA-256 macro dies: {correct.sum()}/{M} correct\n"
                "sub-dies: gray bulk / gold halo / red hollow core", fontsize=10)

    a = ax[1, 1]
    bins = np.linspace(TC - 5, TH + 5, 90)
    a.hist(out_T[:, M:].ravel(), bins=bins, alpha=0.7, color="navy", density=True, label="pristine reference macros")
    if correct.any(): a.hist(out_T[:, :M][:, correct].ravel(), bins=bins, alpha=0.5, color="limegreen", density=True, label="wafer, correct")
    if wrong.any(): a.hist(out_T[:, :M][:, wrong].ravel(), bins=bins, alpha=0.5, color="crimson", density=True, label="wafer, wrong")
    a.axvline(T_TR, c="k", ls="--"); a.set_yscale("log"); a.set_xlabel("final digest-bit temperature (K)"); a.legend(fontsize=8)
    a.set_title("Analog output levels after ~380k thermal gates")

    a = ax[1, 2]
    lams = np.linspace(0.03, 1.0, 120)
    for n, c in zip(vtc, ("k", "tab:blue", "tab:orange", "crimson")):
        st, d = {"pristine": (0, clean), "dislocation-dense": (0, dense), "strain halo": (1, clean), "hollow core": (2, clean)}[n]
        a.plot(lams, [float(nand_T(np.array(TH), np.array(TH), rf(st, d), rf(st, d), lam=l_)) for l_ in lams], c=c, label=n)
    a.axhline(T_TR, c="gray", ls="--"); a.axvline(LAMBDA, c="g", ls=":", label=f"chosen load ratio {LAMBDA}")
    a.set_xlabel("load / segment conductance ratio"); a.set_ylabel("NAND(1,1) low level (K)"); a.legend(fontsize=8)
    a.set_title("Design window: gate fails once low level crosses the switch threshold")
    plt.tight_layout(); plt.savefig("sic_wafer_phonon_logic.png", dpi=120)
    print("\nSaved sic_wafer_phonon_logic.png, phonon_logic_macro_dies.csv")
