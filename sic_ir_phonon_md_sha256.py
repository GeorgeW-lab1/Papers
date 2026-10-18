"""
sic_ir_phonon_md_sha256.py   --  IR transmission -> REAL anharmonic-lattice phonon dynamics -> bits -> SHA-256

No hand-written "I = g*Qi*Qj*cos(dphi)" shortcut.  The phonon-phonon interaction here is obtained by
integrating Newton's equations (velocity-Verlet) for a periodic FPUT chain with cubic + quartic bond anharmonicity:

    H = sum_n  p_n^2/2  +  V(u_{n+1}-u_n),    V(d) = d^2/2 + alpha d^3/3 + beta d^4/4

Linear modes:  w_k = 2 sin(pi k / N).  The cubic term is the genuine three-phonon vertex: two launched modes
(k1,k2) scatter into k3 = k1+k2 with an efficiency set by amplitudes, relative phase, and the dispersion detuning
w1+w2-w3 -- all of that emerges from the dynamics, nothing is prescribed.

Pipeline
  1. SiC wafer with TSD/BPD defects; IR transmission T at 5 probe points per site (Gaussian defect footprint).
  2. Each probe launches phonon pair into one lattice:   mode k1 from this probe, mode k2 from the same probe
     on the NEXT site (raster ring).  amplitude = A0*T,   phase = 2*pi*tau/TAU0,  tau = -ln T.
  3. Integrate all probe-lattices in parallel for t_end.  Readout = time-averaged energy that appeared in mode k3.
  4. Bit = 1 if E_k3 >= median(E_k3).  Bits -> bytes -> SHA-256 (standard hashlib; SHA-256 itself is not physics).
  5. Diagnostics: energy conservation, mode-energy transfer plot, harmonic (alpha=beta=0) control, noise avalanche.

Needs numpy, matplotlib.      Run: python sic_ir_phonon_md_sha256.py
"""
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- assumed wafer / optics constants (assumptions, not measurements)
WAFER_R, SITE = 50.0, 8.0
ALPHA_TSD, ALPHA_BPD = 0.30, 0.12
SIGMA, TAU0 = 1.5, 0.05
PROBES = [(0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2)]

# ---------------- lattice parameters
N, ALPHA, BETA = 32, 0.25, 0.10
K1, K2 = 2, 3; K3 = K1 + K2
A0, DT, T_END = 0.9, 0.05, 2500.0
W = 2 * np.sin(np.pi * np.arange(N) / N)


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
    sites.sort(key=lambda s_: (-s_[3], s_[2]))
    return np.array(sites), tsd, bpd


def probe_T(sites, tsd, bpd, noise=0.0, rng=None):
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


# ---------------- real lattice dynamics (batched over probes)
def force(u, alpha, beta):
    d = np.roll(u, -1, axis=1) - u                   # bond stretch d_n = u_{n+1}-u_n
    f = d + alpha * d**2 + beta * d**3               # V'(d)
    return f - np.roll(f, 1, axis=1)

def energy(u, v, alpha, beta):
    d = np.roll(u, -1, axis=1) - u
    return (0.5 * v**2).sum(1) + (0.5 * d**2 + alpha * d**3 / 3 + beta * d**4 / 4).sum(1)

def mode_energy(u, v):
    """Harmonic mode energies E_k (k and N-k merged), shape (B, N//2+1)."""
    U = np.fft.fft(u, axis=1) / np.sqrt(N); V = np.fft.fft(v, axis=1) / np.sqrt(N)
    E = 0.5 * (np.abs(V)**2 + (W**2) * np.abs(U)**2)
    out = E[:, :N // 2 + 1].copy(); out[:, 1:N // 2] += E[:, N - 1:N // 2:-1]
    return out

def initial_state(Qa, Qb, phia, phib):
    n = np.arange(N)
    u = np.sqrt(2 / N) * (A0 * Qa[:, None] * np.cos(2 * np.pi * K1 * n / N + phia[:, None])
                          + A0 * Qb[:, None] * np.cos(2 * np.pi * K2 * n / N + phib[:, None]))
    return u, np.zeros_like(u)

def run_lattices(T, alpha=ALPHA, beta=BETA, record=False):
    """T: (n_sites,5).  Probe (i,m) launches mode K1; probe (i+1,m) launches mode K2."""
    tau = -np.log(T); phi = 2 * np.pi * tau / TAU0
    Qa, Qb = T.ravel(), np.roll(T, -1, axis=0).ravel()
    pa, pb = phi.ravel(), np.roll(phi, -1, axis=0).ravel()
    u, v = initial_state(Qa, Qb, pa, pb)
    E0 = energy(u, v, alpha, beta)
    steps = int(T_END / DT); acc = np.zeros(len(Qa)); nacc = 0; hist = []
    a = force(u, alpha, beta)
    for s in range(steps):                            # velocity Verlet (symplectic)
        v += 0.5 * DT * a; u += DT * v
        a = force(u, alpha, beta); v += 0.5 * DT * a
        if s % 20 == 0:
            Ek = mode_energy(u, v)
            if s >= 0.5 * steps: acc += Ek[:, K3]; nacc += 1
            if record and s % 100 == 0: hist.append(Ek[:, :K3 + 2].copy())
    drift = np.abs(energy(u, v, alpha, beta) - E0).max() / E0.mean()
    return acc / nacc, drift, np.array(hist)

def bits_to_message(bits):
    bits = bits[: len(bits) // 8 * 8]
    return bytes(int("".join(map(str, bits[k:k + 8])), 2) for k in range(0, len(bits), 8))

def to_bits(E):
    return (E >= np.median(E)).astype(int)



# ---------------- SHA-256 reference trace (for the animation)
MASK = 0xFFFFFFFF
K = [int(x, 16) for x in """428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5 d807aa98 12835b01 243185be 550c7dc3
72be5d74 80deb1fe 9bdc06a7 c19bf174 e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da
983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967 27b70a85 2e1b2138 4d2c6dfc 53380d13
650a7354 766a0abb 81c2c92e 92722c85 a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070
19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3 748f82ee 78a5636f 84c87814 8cc70208
90befffa a4506ceb bef9a3f7 c67178f2""".split()]
H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
def rotr(x, n): return ((x >> n) | (x << (32 - n))) & MASK

def sha_trace(block):
    Wd = [int.from_bytes(block[4 * t:4 * t + 4], "big") for t in range(16)]
    for t in range(16, 64):
        s0 = rotr(Wd[t-15], 7) ^ rotr(Wd[t-15], 18) ^ (Wd[t-15] >> 3)
        s1 = rotr(Wd[t-2], 17) ^ rotr(Wd[t-2], 19) ^ (Wd[t-2] >> 10)
        Wd.append((Wd[t-16] + s0 + Wd[t-7] + s1) & MASK)
    a, b, c, d, e, f, g, h = H0; states = []
    for t in range(64):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25); ch = (e & f) ^ (~e & MASK & g)
        T1 = (h + S1 + ch + K[t] + Wd[t]) & MASK
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22); maj = (a & b) ^ (a & c) ^ (b & c)
        h, g, f, e, d, c, b, a = g, f, e, (d + T1) & MASK, c, b, a, (T1 + S0 + maj) & MASK
        states.append((a, b, c, d, e, f, g, h))
    return states, [(H0[i] + v) & MASK for i, v in enumerate((a, b, c, d, e, f, g, h))]


def animate(sites, tsd, bpd, T, E3, bits, hist, msg, ref):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Circle, Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap, Normalize
    T_MIN = 0.8
    NS = len(sites); Tsite = T.mean(1)
    tt = np.arange(hist.shape[0]) * 100 * DT
    block = msg + b"\x80" + b"\x00" * (55 - len(msg)) + (8 * len(msg)).to_bytes(8, "big")
    block_bits = np.array([(by >> (7 - k)) & 1 for by in block for k in range(8)])
    states, final = sha_trace(block)
    trace_digest = "".join(f"{w:08x}" for w in final)
    assert trace_digest == ref, "reference trace != hashlib"
    med = np.median(E3)
    cmapT = matplotlib.colormaps['YlGnBu_r']; normT = Normalize(0.3, 1.0)
    P1, P2, P3, P4, P5 = 44, 44, 8, 32, 16
    F1, F2, F3, F4 = P1, P1 + P2, P1 + P2 + P3, P1 + P2 + P3 + P4
    NF = F4 + P5
    pcol = ['#c0392b', '#e67e22', '#27ae60', '#2980b9', '#8e44ad']

    fig = plt.figure(figsize=(12.5, 7.6))
    axw = fig.add_axes([0.02, 0.27, 0.42, 0.64])
    axr = fig.add_axes([0.47, 0.67, 0.51, 0.24])
    axm = fig.add_axes([0.52, 0.47, 0.46, 0.14])
    axt = fig.add_axes([0.47, 0.25, 0.51, 0.17]); axt.axis('off')
    title = fig.text(0.5, 0.955, "", ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.008, "Phonon stage = real velocity-Verlet MD of an anharmonic (cubic+quartic) chain; IR attenuation constants are "
             "assumptions; SHA-256 itself is the standard algorithm (verified vs hashlib).", fontsize=7.2, color='#555')

    axw.add_patch(Circle((0, 0), WAFER_R, fill=False, lw=3, color='#2c3e50'))
    patches = []
    for sx, sy, x0, y0 in sites:
        p = Rectangle((x0, y0), SITE, SITE, fc='#f2f3f4', ec='#99a3a4', lw=.6); axw.add_patch(p); patches.append(p)
    axw.scatter(*tsd.T, marker='x', c='#c0392b', s=8, lw=.7, zorder=4, alpha=.7)
    axw.scatter(*bpd.T, marker='s', c='#e67e22', s=7, zorder=4, alpha=.7)
    cur = Rectangle((0, 0), SITE, SITE, fill=False, ec='#e74c3c', lw=3, zorder=8, visible=False)
    nxt = Rectangle((0, 0), SITE, SITE, fill=False, ec='#8e44ad', lw=3, zorder=8, visible=False)
    axw.add_patch(cur); axw.add_patch(nxt)
    links = [axw.plot([], [], '-', c='#8e44ad', lw=1.6, zorder=9)[0] for _ in PROBES]
    pts = axw.scatter([], [], s=28, c='#e74c3c', zorder=10)
    axw.set_xlim(-WAFER_R * 1.05, WAFER_R * 1.05); axw.set_ylim(-WAFER_R * 1.05, WAFER_R * 1.05)
    axw.set_aspect('equal'); axw.set_xticks([]); axw.set_yticks([])
    axw.set_title("4H-SiC wafer: IR transmission per site", fontsize=10)

    regcmap = ListedColormap(['#ecf0f1', '#ffffff', '#00bcd4', '#e8daef', '#8e44ad'])
    reg = axr.imshow(np.zeros((16, 32)), cmap=regcmap, vmin=0, vmax=4, aspect='auto')
    axr.set_xticks(np.arange(-.5, 32, 1), minor=True); axr.set_yticks(np.arange(-.5, 16, 1), minor=True)
    axr.grid(which='minor', color='#aab7b8', lw=.4); axr.tick_params(which='both', length=0)
    axr.set_xticks([0, 31]); axr.set_xticklabels(["b31", "b0"], fontsize=7)
    axr.set_yticks([0, 15]); axr.set_yticklabels(["W0", "W15"], fontsize=7)
    axr.set_title("SHA-256 512-bit input block (16 words x 32 bits)", fontsize=9)

    handles = [
        Patch(fc=cmapT(normT(0.95)), ec='#566573', label='Site: high IR transmission (T \u2265 80%)'),
        Patch(fc=cmapT(normT(0.5)), ec='#566573', hatch='xxx', label='Site: IR blocked (T < 80%)'),
        Patch(fc='#f2f3f4', ec='#99a3a4', label='Site not yet scanned'),
        Line2D([], [], marker='x', ls='', c='#c0392b', label='TSD defect'),
        Line2D([], [], marker='s', ls='', c='#e67e22', label='BPD defect'),
        Patch(fc='none', ec='#e74c3c', lw=2.5, label='Current site / probe'),
        Patch(fc='none', ec='#8e44ad', lw=2.5, label='Neighbour site (phonon partner)'),
        Line2D([], [], marker='o', ls='-', c='#8e44ad', mfc='#e74c3c', label='Probe pair launches modes k=%d,%d' % (K1, K2)),
        Line2D([], [], ls='-', c=pcol[0], label='MD: energy in k=%d (3-phonon product), probes 1-5' % K3),
        Line2D([], [], ls='--', c='k', label='median of readout (bit threshold)'),
        Patch(fc='#ecf0f1', ec='#aab7b8', label='Input bit: empty'),
        Patch(fc='#ffffff', ec='#aab7b8', label='Input bit = 0'),
        Patch(fc='#00bcd4', ec='#aab7b8', label='Input bit = 1'),
        Patch(fc='#8e44ad', ec='#aab7b8', label='Padding bit = 1 (light = 0)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7.3, title='Legend', title_fontsize=8,
               bbox_to_anchor=(0.5, 0.025), frameon=True)

    def phase(f):
        return "scan" if f < F1 else "phonon" if f < F2 else "pad" if f < F3 else "rounds" if f < F4 else "digest"

    def update(f):
        ph = phase(f)
        nrev = NS if f >= F1 else min(NS, 2 * (f + 1))
        for k, p in enumerate(patches):
            if k < nrev:
                p.set_facecolor(cmapT(normT(Tsite[k]))); p.set_hatch('xxx' if Tsite[k] < T_MIN else None)
            else:
                p.set_facecolor('#f2f3f4'); p.set_hatch(None)
        cur.set_visible(False); nxt.set_visible(False); pts.set_offsets(np.empty((0, 2)))
        for l in links: l.set_data([], [])
        axm.clear(); nbits = 0; text = []

        if ph == "scan":
            k = nrev - 1; sx, sy, x0, y0 = sites[k]
            cur.set_xy((x0, y0)); cur.set_visible(True)
            title.set_text("1. IR beam scans the SiC wafer, site by site")
            axm.bar(range(1, 6), T[k], color=pcol); axm.set_ylim(0, 1.05); axm.axhline(T_MIN, c='k', ls=':', lw=1)
            axm.set_title("IR transmission at the 5 probe points of this site", fontsize=8)
            axm.set_xlabel("probe", fontsize=7); axm.tick_params(labelsize=7)
            text = [f"IR probe: site {k+1}/{NS}   mean T = {Tsite[k]*100:5.1f}%   tau = {-np.log(Tsite[k]):.3f}",
                    "T(p) = exp(-sum_d alpha_d * gaussian(|p-d|)),  5 probe points per site -> 440 samples",
                    "each probe will launch a phonon: amplitude ~ T, phase = 2*pi*tau/%.2f" % TAU0]
        elif ph == "phonon":
            ns = min(NS, 2 * (f - F1 + 1)); k = ns - 1; j = (k + 1) % NS; nbits = 5 * ns
            sk, sj = sites[k], sites[j]
            cur.set_xy((sk[2], sk[3])); nxt.set_xy((sj[2], sj[3])); cur.set_visible(True); nxt.set_visible(True)
            for m, (dx, dy) in enumerate(PROBES):
                links[m].set_data([sk[0] + dx, sj[0] + dx], [sk[1] + dy, sj[1] + dy])
            pts.set_offsets(np.array([[sk[0] + dx, sk[1] + dy] for dx, dy in PROBES] +
                                     [[sj[0] + dx, sj[1] + dy] for dx, dy in PROBES]))
            for m in range(5):
                axm.plot(tt, hist[:, 5 * k + m, K3], c=pcol[m], lw=1.1)
            axm.axvspan(T_END / 2, T_END, color='#d5d8dc', alpha=.5, zorder=0)
            axm.axhline(med, c='k', ls='--', lw=1)
            axm.set_title(f"MD, site {k+1} (5 lattices): energy in k={K3} from {K1}+{K3-K1}; x = time, grey = readout window", fontsize=8)
            axm.tick_params(labelsize=7); axm.set_xlim(0, T_END)
            title.set_text("2. Two probes' phonons interact anharmonically (real MD) -> one bit per probe")
            text = [f"{'site %d<->%d' % (k+1, j+1):<14}probe   T_i    T_j    <E_k{K3}>     bit"]
            for m in range(5):
                text.append(f"{'':<14}  {m+1}   {T[k,m]:.3f}  {T[j,m]:.3f}  {E3[5*k+m]:.3e}   {bits[5*k+m]}")
            text.append(f"bit = [<E_k{K3}> >= median = {med:.3e}]")
        else:
            nbits = 440
            order = np.argsort(E3)
            axm.scatter(range(len(E3)), E3[order], s=5, c=np.where(bits[order] == 1, '#00bcd4', '#7f8c8d'))
            axm.axhline(med, c='k', ls='--', lw=1)
            axm.set_title("All 440 MD readouts (sorted): above median -> bit 1 (cyan)", fontsize=8)
            axm.set_xlabel("probe rank", fontsize=7); axm.tick_params(labelsize=7)
            if ph == "pad":
                title.set_text("3. 440 bits = 55 bytes -> pad to one 512-bit SHA-256 block")
                text = [f"message: {msg.hex()[:60]}...",
                        "append 0x80, zero fill, 64-bit length (440)" if f >= F2 + 2 else "appending padding..."]
            elif ph == "rounds":
                r = min(64, 2 * (f - F3 + 1)); a, b, c, d, e, ff, g, h = states[r - 1]
                title.set_text("4. SHA-256 compression: 64 rounds")
                text = [f"round {r:2d}/64   " + "#" * (r // 2) + "." * (32 - r // 2),
                        f"a={a:08x} b={b:08x} c={c:08x} d={d:08x}", f"e={e:08x} f={ff:08x} g={g:08x} h={h:08x}"]
            else:
                nw = min(8, f - F4 + 1)
                title.set_text("5. Digest out - checked against hashlib")
                text = [" ".join(f"{final[i]:08x}" if i < nw else "........" for i in range(4)),
                        " ".join(f"{final[i]:08x}" if i < nw else "........" for i in range(4, 8))]
                if nw == 8: text.append("trace == hashlib:  " + ("PASS" if trace_digest == ref else "FAIL"))
        data = np.zeros(512, dtype=int); data[:nbits] = 1 + bits[:nbits]
        if ph in ("rounds", "digest") or (ph == "pad" and f >= F2 + 2): data[440:] = 3 + block_bits[440:]
        reg.set_data(data.reshape(16, 32))
        axt.clear(); axt.axis('off')
        axt.text(0.0, 1.0, "\n".join(text), va='top', ha='left', family='monospace', fontsize=8.5)
        return []

    anim = FuncAnimation(fig, update, frames=NF, interval=90, blit=False)
    anim.save("ir_phonon_sha256.gif", writer=PillowWriter(fps=11), dpi=62)
    update(F1 + 20); fig.savefig("preview_phonon.png", dpi=80)
    update(NF - 1); fig.savefig("preview_digest.png", dpi=80)
    print(f"saved ir_phonon_sha256.gif ({NF} frames)")


if __name__ == "__main__":
    import sys, os
    sites, tsd, bpd = make_wafer()
    T = probe_T(sites, tsd, bpd)
    if "--anim" in sys.argv:
        E3, _, hist = run_lattices(T, record=True)
        bits = to_bits(E3); msg = bits_to_message(bits)
        animate(sites, tsd, bpd, T, E3, bits, hist, msg, hashlib.sha256(msg).hexdigest()); sys.exit()
    print(f"sites {len(sites)}, probes {T.size}, T range {T.min():.4f}..{T.max():.4f}")
    E3, drift, hist = run_lattices(T, record=True)
    bits = to_bits(E3); msg = bits_to_message(bits)
    ref = hashlib.sha256(msg).hexdigest()
    print(f"max relative energy drift (symplectic check): {drift:.2e}")
    print(f"E_k{K3} spread: {E3.min():.3e} .. {E3.max():.3e}   ones: {bits.mean()*100:.1f}%")
    print(f"message ({len(msg)} B): {msg.hex()}\nSHA-256: {ref}")

    Eh, _, _ = run_lattices(T[:6], alpha=0.0, beta=0.0); En, _, _ = run_lattices(T[:6])
    print(f"control, mean E_k{K3}: harmonic {Eh.mean():.2e}  vs anharmonic {En.mean():.2e}")

    rng = np.random.default_rng(0)
    print("\nNoise sensitivity of the full IR->phonon->SHA chain (multiplicative IR noise):")
    for noise in (1e-5, 1e-4, 1e-3, 1e-2):
        Tn = probe_T(sites, tsd, bpd, noise=noise, rng=rng)
        b2 = to_bits(run_lattices(Tn)[0]); d2 = hashlib.sha256(bits_to_message(b2)).digest()
        dd = bin(int.from_bytes(d2, "big") ^ int(ref, 16)).count("1")
        print(f"  {noise*100:6.3f}% IR noise: {(b2 != bits).sum():3d}/{len(bits)} bits flip -> {dd:3d}/256 digest bits change")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    t = np.arange(hist.shape[0]) * 100 * DT
    for a_, pi in zip(ax, [0, int(np.argmin(T.ravel())), 7]):
        for k in range(1, K3 + 1): a_.plot(t, hist[:, pi, k], label=f"k={k}")
        a_.set_title(f"probe {pi}: T={T.ravel()[pi]:.3f}"); a_.set_xlabel("time")
    ax[0].set_ylabel("mode energy"); ax[0].legend(fontsize=7)
    fig.suptitle(f"Three-phonon transfer {K1}+{K2}->{K3} in the anharmonic chain (real MD)")
    fig.tight_layout(); fig.savefig("phonon_mode_transfer.png", dpi=110)
    animate(sites, tsd, bpd, T, E3, bits, hist, msg, ref)
