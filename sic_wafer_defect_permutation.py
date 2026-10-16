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


# ---------------- main ----------------
if __name__ == "__main__":
    by_type, pts, labels = simulate_wafer()
    print(f"Total defects: {len(pts)}")
    for k, v in by_type.items():
        print(f"  {k:<16}: {len(v)}")

    nn_obs, nn_null, nn_p = permutation_test(pts, mean_nn_distance, alternative="less")
    r_obs, r_null, r_p = permutation_test(pts, mean_radius)
    print(f"\nClustering (mean NN dist): obs={nn_obs:.3f} mm, null={nn_null.mean():.3f} mm, p={nn_p:.4f}")
    print(f"Radial (mean radius)     : obs={r_obs:.2f} mm, null={r_null.mean():.2f} mm, p={r_p:.4f}")

    gx, gy = die_grid()
    counts = die_counts(pts, gx, gy)
    good = counts == 0
    print(f"\nDies: {len(gx)}, defect-free: {good.sum()}, yield = {100*good.mean():.1f}%")
    mi, mi_null, mi_p = die_permutation_test(counts, gx, gy)
    print(f"Moran's I (die counts)   : obs={mi:.3f}, null mean={mi_null.mean():.3f}, p={mi_p:.4f}")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    theta = np.linspace(0, 2 * np.pi, 400)

    a = ax[0, 0]
    a.plot(R * np.cos(theta), R * np.sin(theta), "k")
    a.plot(R_EFF * np.cos(theta), R_EFF * np.sin(theta), "k--", lw=0.6)
    styles = {"Micropipe": ("r", "x", 40), "TSD": ("tab:blue", ".", 8),
              "BPD (clustered)": ("tab:green", "o", 6), "Scratch": ("tab:orange", "-", 6)}
    for k, v in by_type.items():
        c, m, s = styles[k]
        a.scatter(v[:, 0], v[:, 1], c=c, marker="o" if m == "-" else m, s=s, label=f"{k} ({len(v)})")
    a.set_aspect("equal"); a.legend(loc="upper right", fontsize=7); a.set_title("SiC wafer defect map")

    a = ax[0, 1]
    a.plot(R * np.cos(theta), R * np.sin(theta), "k")
    cm = a.scatter(gx + DIE / 2, gy + DIE / 2, c=counts, cmap="YlOrRd", marker="s", s=38)
    a.scatter((gx + DIE / 2)[good], (gy + DIE / 2)[good], facecolors="none", edgecolors="g", marker="s", s=38)
    plt.colorbar(cm, ax=a, label="defects / die")
    a.set_aspect("equal"); a.set_title(f"Die map (green = good), yield {100*good.mean():.1f}%")

    a = ax[1, 0]
    a.hist(nn_null, bins=40, color="lightgray")
    a.axvline(nn_obs, c="r", label=f"observed (p={nn_p:.3f})")
    a.set_xlabel("mean NN distance (mm)"); a.legend(); a.set_title("Permutation test: clustering")

    a = ax[1, 1]
    a.hist(mi_null, bins=40, color="lightgray")
    a.axvline(mi, c="r", label=f"observed (p={mi_p:.3f})")
    a.set_xlabel("Moran's I"); a.legend(); a.set_title("Permutation test: die-level autocorrelation")

    plt.tight_layout()
    plt.savefig("sic_wafer_defects.png", dpi=130)
    print("\nSaved sic_wafer_defects.png")
