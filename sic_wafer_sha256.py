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


# ====================================================================
# 4. Run
# ====================================================================
if __name__ == "__main__":
    Wp = pack_blocks(MSGS)

    # ---- fault-free reference run (also counts gates + records stage segments) ----
    fab0 = Fabric(M, track=True)
    ref = extract(sha256_block(fab0, Wp), M)
    assert ref == EXPECT, "NAND-only SHA-256 netlist does not match hashlib!"
    N = fab0.c
    bounds = [s for s, _ in fab0.segs] + [N]
    segs = [(bounds[i], bounds[i + 1], fab0.segs[i][1]) for i in range(len(fab0.segs))]
    print(f"NAND-only SHA-256 verified against hashlib on {M} macro dies")
    print(f"Gate budget per macro die: {N:,} NAND2  (~{4 * N / 1e6:.1f} M transistors)")
    for k, v in fab0.cats.items(): print(f"   {k:<11}: {v:>8,}  ({100 * v / N:4.1f}%)")

    # ---- defective-wafer run ----
    sched = build_schedule(segs, N)
    print(f"\nInjected fault sites: {len(sched):,} gate instances across the wafer")
    fab = Fabric(M, sched=sched)
    got = extract(sha256_block(fab, Wp), M)
    correct = np.array([g == e for g, e in zip(got, EXPECT)])
    ham = np.array([bin(g ^ e).count("1") for g, e in zip(got, EXPECT)])

    n_fatal = (fatal[SUB]).sum(1); n_soft = ((~fatal) & (risk | recheck))[SUB].sum(1)
    cat = np.where(n_fatal > 0, "has FATAL sub-die", np.where(n_soft > 0, "soft faults only", "clean"))
    print(f"\nSingle macro die: {correct.sum()}/{M} hashes correct ({100 * correct.mean():.1f}%)")
    for c in ("clean", "soft faults only", "has FATAL sub-die"):
        sel = cat == c
        print(f"   {c:<18}: {correct[sel].sum():>3}/{sel.sum():<3} correct")
    assert correct[cat == "clean"].all(), "a clean macro die produced a wrong hash"
    wrong = ~correct
    print(f"   wrong hashes: mean Hamming distance {ham[wrong].mean():.1f}/256 bits "
          f"(avalanche ideal = 128)")

    # ---- TMR ----
    tmr_ok = []
    for tr in TRIPLES:
        voted = tmr_vote(got[tr[0]], got[tr[1]], got[tr[2]])
        tmr_ok.append(voted == EXPECT[tr[0]])
    tmr_ok = np.array(tmr_ok)
    raw_in_triples = np.mean([correct[tr].mean() for tr in TRIPLES])
    print(f"\nTMR triples: {len(TRIPLES)}   raw replica yield {100 * raw_in_triples:.1f}%  ->  "
          f"voted yield {100 * tmr_ok.mean():.1f}%")

    # ---- permutation test: does defect clustering help macro-die yield? ----
    def clean_frac(fat): return np.mean(~fat[SUB].any(1))
    obs = clean_frac(fatal)
    n_perm = 999
    null_ring, null_glob = [], []
    for _ in range(n_perm):
        p = np.arange(len(gx))
        for g in RINGS: p[g] = rng.permutation(g)
        null_ring.append(clean_frac(fatal[p]))
        null_glob.append(clean_frac(fatal[rng.permutation(len(gx))]))
    null_ring, null_glob = np.array(null_ring), np.array(null_glob)
    p_ring = (1 + np.sum(null_ring >= obs)) / (n_perm + 1)
    p_glob = (1 + np.sum(null_glob >= obs)) / (n_perm + 1)
    print(f"\nFault-free macro-die fraction: observed {100 * obs:.1f}%")
    print(f"   vs ring-shuffled defects : null {100 * null_ring.mean():.1f}%  p(obs>=null)={p_ring:.4f}")
    print(f"   vs fully scattered       : null {100 * null_glob.mean():.1f}%  p(obs>=null)={p_glob:.4f}")

    # ---- CSV ----
    with open("sha256_macro_dies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["macro", "cx_mm", "cy_mm", "group", "message", "n_fatal_subdies", "n_soft_subdies",
                    "expected_sha256", "computed_sha256", "correct", "hamming_bits"])
        for mi, (ti, tj) in enumerate(tiles):
            w.writerow([mi, f"{TCX[ti, tj]:.0f}", f"{TCY[ti, tj]:.0f}", GROUP[mi], MSGS[mi], n_fatal[mi],
                        n_soft[mi], f"{EXPECT[mi]:064x}", f"{got[mi]:064x}", int(correct[mi]), ham[mi]])

    # ---- plots ----
    th_ = np.linspace(0, 2 * np.pi, 300)
    def ring(a):
        a.plot(R * np.cos(th_), R * np.sin(th_), "k", lw=1)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])

    fig, ax = plt.subplots(2, 3, figsize=(20, 12.5))

    a = ax[0, 0]; ring(a)
    cls = np.where(fatal, "crimson", np.where(recheck | risk, "gold", "lightgray"))
    a.scatter(xc, yc, c=cls, marker="s", s=34, zorder=2)
    for mi, (ti, tj) in enumerate(tiles):
        a.add_patch(Rectangle((TCX[ti, tj] - DIE, TCY[ti, tj] - DIE), 2 * DIE, 2 * DIE,
                              fill=True, fc="limegreen" if correct[mi] else "black",
                              alpha=0.30 if correct[mi] else 0.55, ec="green" if correct[mi] else "red",
                              lw=1.1, zorder=3))
    a.set_title(f"SHA-256 macro dies: {correct.sum()}/{M} correct\n"
                "green = hash OK, black = wrong;  sub-dies: gray pass / gold soft / red fatal", fontsize=10)

    a = ax[0, 1]; ring(a)
    a.scatter(xc, yc, c="whitesmoke", marker="s", s=34)
    for k, tr in enumerate(TRIPLES):
        x0 = min(TCX[tiles[m][0], tiles[m][1]] for m in tr) - DIE
        y0 = TCY[tiles[tr[0]][0], tiles[tr[0]][1]] - DIE
        a.add_patch(Rectangle((x0, y0), 6 * DIE, 2 * DIE, fc="limegreen" if tmr_ok[k] else "black",
                              alpha=0.35 if tmr_ok[k] else 0.6, ec="k", lw=0.8))
        for m in tr:
            if not correct[m]:
                a.scatter(TCX[tiles[m][0], tiles[m][1]], TCY[tiles[m][0], tiles[m][1]], c="red", marker="x", s=40, zorder=4)
    a.set_title(f"TMR triples (3 macro dies + NAND voter): {tmr_ok.sum()}/{len(TRIPLES)} voted correct\n"
                "red x = replica with wrong hash", fontsize=10)

    a = ax[0, 2]
    a.hist(ham[wrong], bins=np.arange(90, 170, 2), color="steelblue", density=True, label="wrong hashes")
    xs = np.arange(90, 170)
    from math import comb
    a.plot(xs, [comb(256, int(k)) / 2 ** 256 * 2 for k in xs], "r-", label="Binomial(256, 0.5)")
    a.axvline(128, c="k", ls=":"); a.set_xlabel("bits differing from correct digest (of 256)")
    a.set_title("Avalanche: any surviving fault scrambles the digest"); a.legend()

    a = ax[1, 0]
    labels_ = ["all macro dies", "clean", "soft only", "has FATAL", "TMR (raw replicas)", "TMR voted"]
    vals = [correct.mean()] + [correct[cat == c].mean() if (cat == c).any() else 0
                               for c in ("clean", "soft faults only", "has FATAL sub-die")] \
           + [raw_in_triples, tmr_ok.mean()]
    bars = a.bar(labels_, np.array(vals) * 100, color=["gray", "limegreen", "gold", "crimson", "lightsteelblue", "navy"])
    for b_, v in zip(bars, vals): a.text(b_.get_x() + b_.get_width() / 2, v * 100 + 1, f"{100 * v:.0f}%", ha="center")
    a.set_ylim(0, 112); a.set_ylabel("correct SHA-256 (%)"); a.tick_params(axis="x", rotation=20)
    a.set_title("Hash yield by sub-die state and redundancy")

    a = ax[1, 1]
    names = list(fab0.cats); counts = [fab0.cats[n] for n in names]
    a.barh(names, counts, color="slategray")
    for i, c in enumerate(counts): a.text(c, i, f" {c:,}", va="center", fontsize=9)
    a.set_xlim(0, max(counts) * 1.25); a.invert_yaxis()
    a.set_title(f"NAND2 gate budget per macro die ({N:,} total)")

    a = ax[1, 2]
    a.hist(100 * null_glob, bins=25, alpha=0.6, color="gray", label=f"scattered defects (p={p_glob:.3f})")
    a.hist(100 * null_ring, bins=25, alpha=0.6, color="steelblue", label=f"ring-shuffled (p={p_ring:.3f})")
    a.axvline(100 * obs, c="r", lw=2, label=f"observed {100 * obs:.1f}%")
    a.set_xlabel("fault-free macro dies (%)"); a.legend(fontsize=8)
    a.set_title("Permutation test: does defect clustering raise macro-die yield?")

    plt.tight_layout(); plt.savefig("sic_wafer_sha256.png", dpi=120)

    ex_ok = int(np.where(correct)[0][0]); ex_bad = int(np.where(wrong)[0][0])
    print(f"\nExample good macro #{ex_ok}: {got[ex_ok]:064x}")
    print(f"Example bad  macro #{ex_bad}: {got[ex_bad]:064x}")
    print(f"      correct digest   : {EXPECT[ex_bad]:064x}   ({ham[ex_bad]} bits differ)")
    print("\nSaved sic_wafer_sha256.png, sha256_macro_dies.csv")
