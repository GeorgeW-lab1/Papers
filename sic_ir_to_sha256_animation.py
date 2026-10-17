"""
sic_ir_to_sha256_animation.py

IR transmission  ->  phonon-phonon interaction  ->  SHA-256 input  ->  digest  (+ animated GIF)

ALGORITHM (a defined mapping; the constants are assumptions, not physics measurements)
  1. Wafer: 4H-SiC, 8 mm sites, TSD/BPD defects (same map as the IR-blocking figure).
  2. IR probe: every site is probed at 5 points. At probe p the modeled transmission is
         T(p) = exp( - sum_d alpha_d * exp(-|p-d|^2 / (2 sigma^2)) )       (Gaussian defect footprint)
  3. Phonon mode per probe:   amplitude Q = T,   optical depth tau = -ln T,   phase phi = 2*pi*tau/TAU0
  4. Three-phonon (anharmonic) interaction of probe m on site i with probe m on the next site j (raster ring):
         I = g * Q_i * Q_j * cos(phi_i - phi_j)          <- same coupling term the phonon-NAND uses, with phases
  5. Bit = 1 if I >= median(I over all probes) else 0   -> 88 sites x 5 probes = 440 bits = 55 bytes
  6. Pack the 55 bytes as a SHA-256 block (0x80 pad + length), hash on the PhononPhononScatteringFabric,
     verify against hashlib, and animate.

Needs: numpy, matplotlib, Pillow.   Run: python sic_ir_to_sha256_animation.py
"""
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap, Normalize

from sic_phonon_sha256_all_in_one import (phonon_sha256_batch, K, H0, ALPHA_TSD, ALPHA_BPD,
                                          T_MIN, WAFER_R, SITE)

G_COUPLING, TAU0, SIGMA = 1.0, 0.05, 1.5
PROBES = [(0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2)]     # mm offsets from site centre
MASK = 0xFFFFFFFF


# ---------------------------------------------------------------- wafer + IR probing
def make_wafer(seed=123):
    R = WAFER_R
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(120)); th = 2 * np.pi * rng.random(120)
    tsd = np.column_stack((r * np.cos(th), r * np.sin(th)))
    s = rng.uniform(-R * .8, R * .8, 45)
    x = s * np.cos(np.pi / 4) + rng.normal(0, 2, 45); y = s * np.sin(np.pi / 4) + rng.normal(0, 8, 45)
    m = x**2 + y**2 <= R**2
    bpd = np.column_stack((x[m], y[m]))
    sites = []
    n = int(R // SITE) + 1
    for i in range(-n, n):
        for j in range(-n, n):
            x0, y0 = i * SITE, j * SITE
            if all(cx**2 + cy**2 <= (R - 2)**2 for cx, cy in
                   [(x0, y0), (x0 + SITE, y0), (x0, y0 + SITE), (x0 + SITE, y0 + SITE)]):
                cnt = lambda p: int(((p[:, 0] >= x0) & (p[:, 0] < x0 + SITE) &
                                     (p[:, 1] >= y0) & (p[:, 1] < y0 + SITE)).sum())
                T = float(np.exp(-(cnt(tsd) * ALPHA_TSD + cnt(bpd) * ALPHA_BPD)))
                sites.append(dict(x0=x0, y0=y0, cx=x0 + SITE / 2, cy=y0 + SITE / 2, T=T))
    sites.sort(key=lambda s_: (-s_['y0'], s_['x0']))        # raster order
    return sites, tsd, bpd


def probe_T(sites, tsd, bpd, noise=0.0, rng=None):
    """T at every probe point -> array (n_sites, 5)."""
    out = np.zeros((len(sites), len(PROBES)))
    for i, s in enumerate(sites):
        for m, (dx, dy) in enumerate(PROBES):
            p = np.array([s['cx'] + dx, s['cy'] + dy])
            tau = ALPHA_TSD * np.exp(-((tsd - p)**2).sum(1) / (2 * SIGMA**2)).sum() \
                + ALPHA_BPD * np.exp(-((bpd - p)**2).sum(1) / (2 * SIGMA**2)).sum()
            out[i, m] = np.exp(-tau)
    if noise > 0:
        out = np.clip(out * (1 + rng.normal(0, noise, out.shape)), 1e-6, 1.0)
    return out


def phonon_interaction_bits(T):
    """T: (n_sites, 5) -> interaction I, bits (n_sites*5,), phases."""
    tau = -np.log(T)
    phi = 2 * np.pi * tau / TAU0
    Tn, phin = np.roll(T, -1, axis=0), np.roll(phi, -1, axis=0)       # next site in raster ring
    I = G_COUPLING * T * Tn * np.cos(phi - phin)
    bits = (I >= np.median(I)).astype(int).ravel()                    # site-major, probe-minor
    return I, bits, phi


def bits_to_message(bits):
    assert len(bits) % 8 == 0
    return bytes(int("".join(map(str, bits[k:k + 8])), 2) for k in range(0, len(bits), 8))


# ---------------------------------------------------------------- SHA-256 reference trace (for the animation)
def rotr(x, n): return ((x >> n) | (x << (32 - n))) & MASK

def sha_trace(block):
    W = [int.from_bytes(block[4 * t:4 * t + 4], "big") for t in range(16)]
    for t in range(16, 64):
        s0 = rotr(W[t-15], 7) ^ rotr(W[t-15], 18) ^ (W[t-15] >> 3)
        s1 = rotr(W[t-2], 17) ^ rotr(W[t-2], 19) ^ (W[t-2] >> 10)
        W.append((W[t-16] + s0 + W[t-7] + s1) & MASK)
    a, b, c, d, e, f, g, h = H0
    states = []
    for t in range(64):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25); ch = (e & f) ^ (~e & MASK & g)
        T1 = (h + S1 + ch + K[t] + W[t]) & MASK
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22); maj = (a & b) ^ (a & c) ^ (b & c)
        T2 = (S0 + maj) & MASK
        h, g, f, e, d, c, b, a = g, f, e, (d + T1) & MASK, c, b, a, (T1 + T2) & MASK
        states.append((a, b, c, d, e, f, g, h))
    final = [(H0[i] + v) & MASK for i, v in enumerate((a, b, c, d, e, f, g, h))]
    return states, final


# ---------------------------------------------------------------- pipeline
sites, tsd, bpd = make_wafer()
T_probe = probe_T(sites, tsd, bpd)
I, bits, phi = phonon_interaction_bits(T_probe)
message = bits_to_message(bits)
assert len(message) == 55
block = message + b"\x80" + b"\x00" * (55 - len(message)) + (8 * len(message)).to_bytes(8, "big")
block_bits = np.array([(byte >> (7 - k)) & 1 for byte in block for k in range(8)])
states, final_words = sha_trace(block)

(fabric_digest,), fab = phonon_sha256_batch([message], defect_scattering=0.0)
ref_digest = hashlib.sha256(message).hexdigest()
trace_digest = "".join(f"{w:08x}" for w in final_words)

print("=" * 72)
print("IR TRANSMISSION -> PHONON INTERACTION -> SHA-256")
print("=" * 72)
print(f"sites: {len(sites)}   probes/site: {len(PROBES)}   bits: {len(bits)}  (ones: {bits.mean()*100:.1f}%)")
print(f"IR probe T range: {T_probe.min():.4f} .. {T_probe.max():.4f}")
print(f"message (55 B): {message.hex()}")
print(f"fabric digest : {fabric_digest}")
print(f"hashlib digest: {ref_digest}")
print(f"trace digest  : {trace_digest}")
print("RESULT:", "PASS (phonon-NAND fabric == hashlib == reference trace)"
      if fabric_digest == ref_digest == trace_digest else "FAIL")

rng = np.random.default_rng(0)
print("\nSensitivity of the IR->bits conversion to measurement noise (avalanche):")
for noise in (0.0001, 0.001, 0.01):
    Tn = probe_T(sites, tsd, bpd, noise=noise, rng=rng)
    _, b2, _ = phonon_interaction_bits(Tn)
    flips = int((b2 != bits).sum())
    d2 = hashlib.sha256(bits_to_message(b2)).digest()
    dd = bin(int.from_bytes(d2, "big") ^ int(ref_digest, 16)).count("1")
    print(f"  {noise*100:5.2f}% multiplicative IR noise: {flips:3d}/440 input bits flip -> {dd:3d}/256 digest bits change")


# ---------------------------------------------------------------- animation
cmapT = matplotlib.colormaps['YlGnBu_r']; normT = Normalize(0.3, 1.0)
N_SITES = len(sites)
P1, P2, P3, P4, P5 = 44, 44, 8, 32, 16            # scan, phonon, pad, rounds, digest frames
F1, F2, F3, F4 = P1, P1 + P2, P1 + P2 + P3, P1 + P2 + P3 + P4
NFRAMES = F4 + P5

fig = plt.figure(figsize=(12.5, 7.2))
axw = fig.add_axes([0.02, 0.07, 0.42, 0.80])
axr = fig.add_axes([0.47, 0.50, 0.51, 0.37])
axt = fig.add_axes([0.47, 0.06, 0.51, 0.38]); axt.axis('off')
title = fig.text(0.5, 0.955, "", ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.012, "Conceptual simulation: IR->bit mapping and attenuation values are assumptions; "
         "SHA-256 is computed on the software phonon-NAND model.", fontsize=7.5, color='#555')

axw.add_patch(Circle((0, 0), WAFER_R, fill=False, lw=3, color='#2c3e50'))
patches = []
for s in sites:
    p = Rectangle((s['x0'], s['y0']), SITE, SITE, fc='#f2f3f4', ec='#99a3a4', lw=.6)
    axw.add_patch(p); patches.append(p)
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
axr.set_title("SHA-256 512-bit input block  (cyan = 1, white = 0, purple = padding)", fontsize=9)


def frame_state(f):
    if f < F1: return "scan"
    if f < F2: return "phonon"
    if f < F3: return "pad"
    if f < F4: return "rounds"
    return "digest"


def update(f):
    ph = frame_state(f)
    nrev = N_SITES if f >= F1 else min(N_SITES, 2 * (f + 1))
    for k, p in enumerate(patches):
        if k < nrev:
            T = sites[k]['T']; p.set_facecolor(cmapT(normT(T)))
            p.set_hatch('xxx' if T < T_MIN else None)
        else:
            p.set_facecolor('#f2f3f4'); p.set_hatch(None)

    cur.set_visible(False); nxt.set_visible(False); pts.set_offsets(np.empty((0, 2)))
    for l in links: l.set_data([], [])
    nbits = 0; text = []

    if ph == "scan":
        k = nrev - 1; s = sites[k]
        cur.set_xy((s['x0'], s['y0'])); cur.set_visible(True)
        title.set_text("1. IR beam scans the SiC wafer, site by site")
        text = [f"IR probe: site {k+1}/{N_SITES}", f"  transmission T = {s['T']*100:5.1f}%",
                f"  optical depth tau = {-np.log(s['T']):.3f}", "",
                "T(p) = exp(-sum alpha_d * gaussian(|p-d|))", "5 probe points per site -> 440 samples"]
    elif ph == "phonon":
        ns = min(N_SITES, 2 * (f - F1 + 1)); k = ns - 1; j = (k + 1) % N_SITES
        nbits = 5 * ns
        sk, sj = sites[k], sites[j]
        cur.set_xy((sk['x0'], sk['y0'])); nxt.set_xy((sj['x0'], sj['y0']))
        cur.set_visible(True); nxt.set_visible(True)
        for m, (dx, dy) in enumerate(PROBES):
            links[m].set_data([sk['cx'] + dx, sj['cx'] + dx], [sk['cy'] + dy, sj['cy'] + dy])
        pts.set_offsets(np.array([[sk['cx'] + dx, sk['cy'] + dy] for dx, dy in PROBES] +
                                 [[sj['cx'] + dx, sj['cy'] + dy] for dx, dy in PROBES]))
        title.set_text("2. Three-phonon interaction between neighbouring probes -> one bit each")
        text = [f"site {k+1} <-> site {j+1}   (Q = T, phi = 2*pi*tau/{TAU0})",
                "probe   Q_i     Q_j    dphi(rad)      I      bit"]
        for m in range(5):
            dphi = phi[k, m] - phi[j, m]
            text.append(f"  {m+1}    {T_probe[k,m]:.3f}  {T_probe[j,m]:.3f}  {dphi:9.2f}  {I[k,m]:7.3f}   {bits[5*k+m]}")
        text += ["", "I = g*Q_i*Q_j*cos(phi_i - phi_j);  bit = [I >= median(I)]"]
    elif ph == "pad":
        nbits = 440
        title.set_text("3. 440 bits = 55 bytes  ->  pad to one 512-bit SHA-256 block")
        text = [f"message: {message.hex()[:44]}...", "",
                "append 0x80, zero fill, 64-bit length (440)" if f >= F2 + 2 else "appending padding...",
                "", f"ones fraction in message bits: {bits.mean()*100:.1f}%"]
    elif ph == "rounds":
        nbits = 440
        r = min(64, 2 * (f - F3 + 1)); a, b, c, d, e, ff, g, h = states[r - 1]
        title.set_text("4. SHA-256 compression on the phonon-NAND fabric (64 rounds)")
        text = [f"round {r:2d}/64   " + "#" * (r // 2) + "." * (32 - r // 2), "",
                f"  a = {a:08x}   e = {e:08x}", f"  b = {b:08x}   f = {ff:08x}",
                f"  c = {c:08x}   g = {g:08x}", f"  d = {d:08x}   h = {h:08x}", "",
                f"fabric gates in full run: {fab.gate_count:,} NAND evaluations"]
    else:
        nbits = 440
        nw = min(8, f - F4 + 1)
        title.set_text("5. Digest out - checked against hashlib")
        text = [f"H{i} = {final_words[i]:08x}" if i < nw else f"H{i} = ........" for i in range(8)]
        if nw == 8:
            text += ["", "fabric == hashlib == reference trace:  " +
                     ("PASS" if fabric_digest == ref_digest == trace_digest else "FAIL")]

    data = np.zeros(512, dtype=int)
    data[:nbits] = 1 + bits[:nbits]
    if ph in ("rounds", "digest") or (ph == "pad" and f >= F2 + 2):
        data[440:] = 3 + block_bits[440:]
    reg.set_data(data.reshape(16, 32))

    axt.clear(); axt.axis('off')
    axt.text(0.0, 1.0, "\n".join(text), va='top', ha='left', family='monospace', fontsize=9.5)
    return []


anim = FuncAnimation(fig, update, frames=NFRAMES, interval=90, blit=False)
if __name__ == "__main__":
    update(30);  fig.savefig("preview_scan.png", dpi=70)
    update(F1 + 20); fig.savefig("preview_phonon.png", dpi=70)
    update(F3 + 10); fig.savefig("preview_rounds.png", dpi=70)
    update(NFRAMES - 1); fig.savefig("preview_digest.png", dpi=70)
    anim.save("ir_to_sha256.gif", writer=PillowWriter(fps=11), dpi=62)
    print(f"\nSaved ir_to_sha256.gif ({NFRAMES} frames)")
