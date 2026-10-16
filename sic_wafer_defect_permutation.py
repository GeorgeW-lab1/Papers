"""
Macrostructures + creative permutation nulls for the phonon-based SiC wafer.

Needs sic_wafer_defect_permutation.py and sic_wafer_phonon.py in the same folder.

MACROSTRUCTURES (all built from a NAND-only netlist -> functionally complete)
  * 2x2 gate tiles : A = any thermal anomaly in tile, B = any Raman broadening in tile
                     -> BUF, NOT, AND, OR, NAND, NOR, XOR, XNOR (all 8 two-input gates)
  * Full adder     : per die, inputs TH, BR, SH -> SUM (odd parity), CARRY (2-of-3 vote)
  * 2:1 MUX        : Y = SH ? BR : TH
  * Ripple-carry   : 4-die words along each wafer row, X = TH bits, Y = BR bits,
                     chained full adders -> 5-bit sum + overflow

PERMUTATION NULLS (each preserves a different structure)
  rotate     : rotate the whole defect map by a random angle (keeps clusters + radial profile)
  ring-joint : shuffle die probe-triples inside 10 mm radial rings (keeps radial profile + per-die logic)
  ring-indep : shuffle each probe independently inside rings (also breaks cross-probe association)
  row-roll   : cyclically roll every die row by a random offset (keeps row statistics)
  D4-symmetry: apply a random flip/rot90 of the die lattice (negative control, expect ~0)
  probe-swap : randomly swap TH<->BR per die (tests probe identity / gate asymmetry)

Output statistic = number of 1s produced by each gate/macrostructure.
Significance uses Benjamini-Hochberg FDR across all gate x null tests.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


"""
Phonon-based defect detection on a SiC wafer.

Reuses the 2D defect map from sic_wafer_defect_permutation.py, but the die-level
logic gates are now driven by two PHONON observables instead of raw defect counts:

  1. Thermal conductivity kappa(x,y)  -> Callaway model, phonon scattering
        1/tau = boundary + point(w^4) + Umklapp(w^2 T e^-th/3T)
                + BPD strain-field (~w) + TSD core (~w^3) + void/scratch halo
  2. Raman E2(TO) mode (~776 cm^-1)   -> linewidth (phonon lifetime) + shift (strain)

Gate network (phonon flags):
  TH  = kappa drop > 5 %          BR = Raman linewidth broadening
  SH  = Raman strain shift         VOID = kappa drop > 20 %

  FATAL = VOID OR (TH AND BR)     RISK = SH AND BR
  RECHECK = TH XOR BR             PASS = NOT(FATAL OR RISK)

NOTE: coefficients are illustrative (calibrated so pristine 4H-SiC kappa ~ 370 W/mK
and Raman FWHM ~ 2 cm^-1), not fitted to measured data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
2D defect simulation + permutation testing for a SiC wafer.

1. Simulates a 150 mm 4H-SiC wafer defect map:
   - micropipes (MP)               : sparse, uniform (Poisson)
   - threading screw disloc. (TSD) : uniform (Poisson)
   - basal plane disloc. (BPD)     : clustered (Thomas process), edge-enriched
   - scratches                     : random line segments
2. Permutation (Monte Carlo) tests: is the observed defect map different
   from complete spatial randomness (CSR)?
   - mean nearest-neighbour distance  (clustering)
   - edge-enrichment (mean radius)    (radial non-uniformity)
3. Die-level yield with a permutation test on die-defect counts.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

rng = np.random.default_rng(42)

# ---------------- wafer geometry ----------------
R = 75.0            # mm (150 mm wafer)
EDGE_EXCL = 3.0     # mm
DIE = 5.0           # mm square die
R_EFF = R - EDGE_EXCL


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


# =====================================================================
# 5. Main
# =====================================================================
if __name__ == "__main__":
    print("Functional-completeness check (NAND-only netlists vs direct logic):")
    for k, v in verify_completeness().items():
        print(f"  {k:<13}: {'OK' if v else 'FAIL'}")

    m = die_phonon_maps(by_type, gx, gy)
    th, br, sh, _ = phonon_flags(m)
    obs = compute_stats(th, br, sh)
    A, B = tile_any(th), tile_any(br)
    print(f"\nDies={N_DIE}  gate tiles={TILE_VALID.sum()}  ripple words={len(WORDS)}")
    print(f"Probe flags: TH={th.sum()} BR={br.sum()} SH={sh.sum()}")
    print("\nGate outputs (tiles firing / valid tiles):")
    for name, v in zip(STAT_NAMES[:8], obs[:8]):
        print(f"  {name:<8}: {int(v):>3} / {TILE_VALID.sum()}")
    print(f"  FA sum={int(obs[8])}  FA carry={int(obs[9])}  MUX={int(obs[10])}  "
          f"ripple Σ={int(obs[11])}  overflow={int(obs[12])}")

    # ---- permutation nulls ----
    null_store, Z, P = {}, np.full((len(STAT_NAMES), len(NULLS)), np.nan), np.ones((len(STAT_NAMES), len(NULLS)))
    for j, (nm, (fn, n)) in enumerate(NULLS.items()):
        arr = np.array([compute_stats(*fn(th, br, sh)) for _ in range(n)])
        null_store[nm] = arr
        mu, sd = arr.mean(0), arr.std(0)
        for i in range(len(STAT_NAMES)):
            if sd[i] > 0:
                Z[i, j] = (obs[i] - mu[i]) / sd[i]
                P[i, j] = (1 + np.sum(np.abs(arr[:, i] - mu[i]) >= abs(obs[i] - mu[i]))) / (n + 1)
        print(f"  null '{nm}' done ({n} perms)")

    finite = np.isfinite(Z)
    SIG = np.zeros_like(finite)
    SIG[finite] = bh_fdr(P[finite])

    print("\nz-scores (obs vs null), * = significant after BH-FDR 5%:")
    print(f"{'':<22}" + "".join(f"{n:>13}" for n in NULLS))
    for i, nm in enumerate(STAT_NAMES):
        row = "".join(f"{('%+.1f%s' % (Z[i, j], '*' if SIG[i, j] else '')) if finite[i, j] else 'invariant':>13}"
                      for j in range(len(NULLS)))
        print(f"{nm:<22}{row}")

    # ---------------- Figure 1: macrostructure maps ----------------
    fig, ax = plt.subplots(3, 4, figsize=(19, 14))
    th_ = np.linspace(0, 2 * np.pi, 300)
    def ring(a): a.plot(R * np.cos(th_), R * np.sin(th_), "k", lw=1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])

    for k, g in enumerate(DIRECT):
        a = ax[k // 4, k % 4]; ring(a)
        out = NAND_NET[g](A, B)
        a.scatter(TCX[TILE_VALID], TCY[TILE_VALID], c=np.where(out[TILE_VALID], "crimson", "lightgray"),
                  marker="s", s=330)
        a.set_title(f"{g}  (NAND-only)  {int(out[TILE_VALID].sum())} on", fontsize=10)

    s_, c_ = full_adder(th, br, sh)
    y_ = mux(sh, th, br)
    for a, f, ttl in ((ax[2, 0], s_, "Full adder SUM (odd parity)"), (ax[2, 1], c_, "Full adder CARRY (2-of-3 vote)"),
                      (ax[2, 2], y_, "MUX  Y = SH ? BR : TH")):
        ring(a)
        a.scatter(xc, yc, c=np.where(f, "darkorange", "lightgray"), marker="s", s=38)
        a.set_title(f"{ttl}  {int(f.sum())} on", fontsize=10)
    a = ax[2, 3]; ring(a)
    a.scatter(xc, yc, c="whitesmoke", marker="s", s=38)
    val, carry = ripple_carry(th[WORDS], br[WORDS])
    sc = a.scatter(xc[WORDS].ravel(), yc[WORDS].ravel(), c=np.repeat(val, 4), cmap="plasma",
                   marker="s", s=38, vmin=0, vmax=max(8, val.max()))
    ov = np.where(carry)[0]
    if len(ov): a.scatter(xc[WORDS[ov]].ravel(), yc[WORDS[ov]].ravel(), facecolors="none", edgecolors="cyan", marker="s", s=60)
    plt.colorbar(sc, ax=a, fraction=0.04, label="5-bit word sum")
    a.set_title("Ripple-carry words (TH + BR), overflow = cyan", fontsize=10)
    plt.tight_layout(); plt.savefig("sic_wafer_macro_gates.png", dpi=120); plt.close(fig)

    # ---------------- Figure 2: permutation heatmap ----------------
    fig = plt.figure(figsize=(17, 9))
    gs = GridSpec(2, 2, width_ratios=[1.7, 1], figure=fig)
    a = fig.add_subplot(gs[:, 0])
    im = a.imshow(np.nan_to_num(Z), cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            txt = ("%+.1f%s" % (Z[i, j], "*" if SIG[i, j] else "")) if finite[i, j] else "inv."
            a.text(j, i, txt, ha="center", va="center", fontsize=9,
                   fontweight="bold" if SIG[i, j] else "normal", color="k" if abs(np.nan_to_num(Z[i, j])) < 4 else "w")
    a.set_xticks(range(len(NULLS))); a.set_xticklabels(NULLS, rotation=25)
    a.set_yticks(range(len(STAT_NAMES))); a.set_yticklabels(STAT_NAMES)
    plt.colorbar(im, ax=a, label="z-score vs permutation null"); a.set_title("Gate / macrostructure output vs creative nulls (* = BH-FDR 5%)")

    for pos, (nm, st) in enumerate((("ring-indep", "AND"), ("row-roll", "Ripple sum (4b words)"))):
        b = fig.add_subplot(gs[pos, 1]); i = STAT_NAMES.index(st)
        b.hist(null_store[nm][:, i], bins=30, color="lightgray")
        b.axvline(obs[i], c="r", label=f"observed={int(obs[i])}  p={P[i, list(NULLS).index(nm)]:.3f}")
        b.set_title(f"{st} under '{nm}' null"); b.legend()
    plt.tight_layout(); plt.savefig("sic_wafer_permutations.png", dpi=120); plt.close(fig)
    print("\nSaved sic_wafer_macro_gates.png, sic_wafer_permutations.png")
