"""
sic_wafer_phonon_animation_sha256.py
--------------------------------------------------------------------------------
Complete end-to-end pipeline:
  1. SiC Wafer generation with TSD & BPD defects.
  2. IR transmission probe mapping with 0.001% optical noise (noise=1e-5).
  3. Real anharmonic lattice phonon dynamics (velocity-Verlet FPUT chain, cubic + quartic).
  4. Mode-energy quantization -> bit extraction -> SHA-256 cryptographic hashing.
  5. 4-Panel animated visualization (Wafer, IR Heatmap, Phonon Modes, Live SHA-256)
     saved as 'sic_wafer_phonon_sha256.gif'.

Requires: numpy, matplotlib
Run: python sic_wafer_phonon_animation_sha256.py
"""

import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. WAFER & OPTICS CONSTANTS
# ==========================================
WAFER_R, SITE = 50.0, 8.0
ALPHA_TSD, ALPHA_BPD = 0.30, 0.12
SIGMA, TAU0 = 1.5, 0.05
PROBES = [(0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2)]

# ==========================================
# 2. LATTICE & MD PARAMETERS
# ==========================================
N, ALPHA, BETA = 32, 0.25, 0.10
K1, K2 = 2, 3; K3 = K1 + K2
A0, DT, T_END = 0.9, 0.05, 1200.0
W = 2 * np.sin(np.pi * np.arange(N) / N)


def make_wafer(seed=42):
    """Generates SiC wafer geometry with Threading Screw Dislocations (TSD) and Basal Plane Dislocations (BPD)."""
    R = WAFER_R; rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(120)); th = 2 * np.pi * rng.random(120)
    tsd = np.column_stack((r * np.cos(th), r * np.sin(th)))
    s = rng.uniform(-R * .8, R * .8, 45)
    x = s * np.cos(np.pi / 4) + rng.normal(0, 2, 45)
    y = s * np.sin(np.pi / 4) + rng.normal(0, 8, 45)
    bpd = np.column_stack((x, y))[x**2 + y**2 <= R**2]
    
    sites = []; n = int(R // SITE) + 1
    for i in range(-n, n):
        for j in range(-n, n):
            x0, y0 = i * SITE, j * SITE
            if all(cx**2 + cy**2 <= (R - 2)**2 for cx, cy in
                   [(x0, y0), (x0 + SITE, y0), (x0, y0 + SITE), (x0 + SITE, y0 + SITE)]):
                sites.append((x0 + SITE / 2, y0 + SITE / 2, x0, y0))
    sites.sort(key=lambda s_: (-s_[3], s_[2]))
    return np.array(sites), tsd, bpd


def probe_T(sites, tsd, bpd, noise=1e-5, rng=None):
    """Calculates IR transmission T across probe sites with 0.001% noise (noise=1e-5)."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.zeros((len(sites), len(PROBES)))
    for i, s in enumerate(sites):
        for m, (dx, dy) in enumerate(PROBES):
            p = np.array([s[0] + dx, s[1] + dy])
            tau = ALPHA_TSD * np.exp(-((tsd - p)**2).sum(1) / (2 * SIGMA**2)).sum() \
                + ALPHA_BPD * np.exp(-((bpd - p)**2).sum(1) / (2 * SIGMA**2)).sum()
            out[i, m] = np.exp(-tau)
    if noise > 0:
        out = np.clip(out * (1 + rng.normal(0, noise, out.shape)), 1e-6, 1.0)
    return out


# ==========================================
# 3. ANHARMONIC LATTICE DYNAMICS (FPUT)
# ==========================================
def force(u, alpha, beta):
    d = np.roll(u, -1, axis=1) - u
    f = d + alpha * d**2 + beta * d**3
    return f - np.roll(f, 1, axis=1)

def mode_energy(u, v):
    U = np.fft.fft(u, axis=1) / np.sqrt(N)
    V = np.fft.fft(v, axis=1) / np.sqrt(N)
    E = 0.5 * (np.abs(V)**2 + (W**2) * np.abs(U)**2)
    out = E[:, :N // 2 + 1].copy()
    out[:, 1:N // 2] += E[:, N - 1:N // 2:-1]
    return out

def initial_state(Qa, Qb, phia, phib):
    n = np.arange(N)
    u = np.sqrt(2 / N) * (A0 * Qa[:, None] * np.cos(2 * np.pi * K1 * n / N + phia[:, None])
                          + A0 * Qb[:, None] * np.cos(2 * np.pi * K2 * n / N + phib[:, None]))
    return u, np.zeros_like(u)

def run_simulation_with_history(T):
    tau = -np.log(T); phi = 2 * np.pi * tau / TAU0
    Qa, Qb = T.ravel(), np.roll(T, -1, axis=0).ravel()
    pa, pb = phi.ravel(), np.roll(phi, -1, axis=0).ravel()
    u, v = initial_state(Qa, Qb, pa, pb)
    steps = int(T_END / DT)
    a = force(u, ALPHA, BETA)
    hist_Ek3 = []
    acc = np.zeros(len(Qa)); nacc = 0
    for s in range(steps):
        v += 0.5 * DT * a; u += DT * v
        a = force(u, ALPHA, BETA); v += 0.5 * DT * a
        if s % 15 == 0:
            Ek = mode_energy(u, v)
            hist_Ek3.append(Ek[:, K3].copy())
            if s >= 0.5 * steps:
                acc += Ek[:, K3]; nacc += 1
    return np.array(hist_Ek3), acc / nacc


# ==========================================
# 4. MAIN PIPELINE & ANIMATION
# ==========================================
if __name__ == "__main__":
    print("[1/4] Generating SiC Wafer & Defect Topologies...")
    rng = np.random.default_rng(42)
    sites, tsd, bpd = make_wafer(seed=42)

    print("[2/4] Computing IR Transmission with 0.001% Noise (1e-5)...")
    noise_level = 1e-5  # Exactly 0.001%
    T = probe_T(sites, tsd, bpd, noise=noise_level, rng=rng)

    print("[3/4] Running Anharmonic Molecular Dynamics & SHA-256 Conversion...")
    hist_Ek3, E3_mean = run_simulation_with_history(T)
    bits = (E3_mean >= np.median(E3_mean)).astype(int)
    bits_bytes = bytes(int("".join(map(str, bits[k:k+8])), 2) for k in range(0, len(bits)//8*8, 8))
    sha_hash = hashlib.sha256(bits_bytes).hexdigest()
    print(f"      -> SHA-256 Digest: {sha_hash}")

    print("[4/4] Rendering 4-Panel Animation with Legend ('sic_wafer_phonon_sha256.gif')...")
    fig = plt.figure(figsize=(15, 8), facecolor="#0f172a")
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3, left=0.06, right=0.94, top=0.90, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, 0]) # Wafer & Defects
    ax2 = fig.add_subplot(gs[0, 1]) # IR Transmission Heatmap
    ax3 = fig.add_subplot(gs[1, 0]) # Phonon Mode Dynamics
    ax4 = fig.add_subplot(gs[1, 1]) # SHA-256 Representation

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#475569")

    # Panel 1: Wafer & Defects Legend
    ax1.set_title("1. SiC Wafer & Defect Topology", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.scatter(tsd[:, 0], tsd[:, 1], c="#ef4444", s=18, label="TSD Defects", alpha=0.8)
    ax1.scatter(bpd[:, 0], bpd[:, 1], c="#38bdf8", s=12, label="BPD Defects", alpha=0.6)
    ax1.scatter([s[0] for s in sites], [s[1] for s in sites], c="#22c55e", s=25, marker="s", label="Probe Sites")
    ax1.set_xlim(-WAFER_R-5, WAFER_R+5); ax1.set_ylim(-WAFER_R-5, WAFER_R+5)
    ax1.tick_params(colors="#94a3b8", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8)

    # Panel 2: IR Transmission Heatmap (0.001% Noise)
    ax2.set_title(f"2. IR Transmission (Noise = {noise_level*100:.3f}%)", color="#f8fafc", fontsize=11, fontweight="bold")
    T_mean_site = T.mean(axis=1)
    sc = ax2.scatter([s[0] for s in sites], [s[1] for s in sites], c=T_mean_site, cmap="plasma", s=40, edgecolors="#ffffff", linewidths=0.5)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_color("#94a3b8")
        label.set_fontsize(8)
    ax2.set_xlim(-WAFER_R-5, WAFER_R+5); ax2.set_ylim(-WAFER_R-5, WAFER_R+5)
    ax2.tick_params(colors="#94a3b8", labelsize=8)

    # Panel 3: Phonon Mode Dynamics Legend
    ax3.set_title(f"3. Anharmonic Lattice Phonon Transfer ({K1}+{K2} -> {K3})", color="#f8fafc", fontsize=11, fontweight="bold")
    time_steps = np.arange(len(hist_Ek3)) * 15 * 0.05
    lines = []
    colors_line = ["#38bdf8", "#a855f7", "#ec4899", "#22c55e", "#eab308"]
    for i in range(min(5, hist_Ek3.shape[1])):
        l, = ax3.plot([], [], lw=1.5, color=colors_line[i % len(colors_line)], label=f"Probe {i}")
        lines.append(l)
    ax3.set_xlim(0, time_steps[-1])
    ax3.set_ylim(0, hist_Ek3.max() * 1.1)
    ax3.set_xlabel("MD Time", color="#f8fafc", fontsize=9)
    ax3.set_ylabel("Mode Energy $E_{k3}$", color="#f8fafc", fontsize=9)
    ax3.tick_params(colors="#94a3b8", labelsize=8)
    ax3.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=7)

    # Panel 4: SHA-256 Representation Dashboard
    ax4.set_title("4. Bit Extraction & SHA-256 Hash Representation", color="#f8fafc", fontsize=11, fontweight="bold")
    ax4.axis("off")
    bit_text = ax4.text(0.05, 0.75, "", color="#38bdf8", fontsize=9, family="monospace", transform=ax4.transAxes)
    hash_text = ax4.text(0.05, 0.35, "", color="#22c55e", fontsize=10, family="monospace", fontweight="bold", transform=ax4.transAxes)
    info_text = ax4.text(0.05, 0.10, f"Final Target Hash: {sha_hash[:32]}...", color="#94a3b8", fontsize=8, family="monospace", transform=ax4.transAxes)

    fig.suptitle("SiC Wafer IR Transmission -> Anharmonic Phonon MD -> SHA-256 Pipeline", color="#f8fafc", fontsize=14, fontweight="bold", y=0.95)

    def update(frame):
        current_t = time_steps[:frame+1]
        current_data = hist_Ek3[:frame+1]
        for idx, l in enumerate(lines):
            if idx < current_data.shape[1]:
                l.set_data(current_t, current_data[:, idx])
        
        bits_str = "".join(map(str, bits[:32]))
        bit_text.set_text("Extracted Bits (sample):\n[" + bits_str[:16] + "...]")
        
        progress_hash = hashlib.sha256(bits_bytes[:max(1, frame * len(bits_bytes) // len(time_steps))]).hexdigest()
        hash_text.set_text("Live SHA-256 Digest:\n" + progress_hash)
        return lines + [bit_text, hash_text]

    ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=50, blit=False)
    ani.save("sic_wafer_phonon_sha256.gif", writer="pillow", fps=20, dpi=100)
    print("SUCCESS: Animation saved as 'sic_wafer_phonon_sha256.gif'")
