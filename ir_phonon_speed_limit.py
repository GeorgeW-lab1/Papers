"""
ir_phonon_speed_limit.py -- how fast can SiC infrared-phonon (24 THz) defect pads fire, back-to-back?

REAL dynamics: a pad is a damped, defect-detuned oscillator driven at f0 = 23.9 THz (4H-SiC TO phonon).  A firing drives it with
weighted input sum s for W periods; the lock-in amplitude over that window is compared with thresholds (theta = 1,2,3).
The pad is then fired AGAIN immediately with a new random input, state carried over (no artificial reset), so ring-down
contamination is simulated, not assumed.  Two schemes are compared:
   plain      : drive W=1 period, wait `gap` idle periods
   RZ reset   : drive W=1 period, then drive R=1 period in ANTI-phase with the same inputs (cancels the stored amplitude;
                for W << ring-down time the response is proportional to integrated drive, so equal anti-phase drive nulls it)
Then speed = pads * modes * f0 / (firings per hash * periods per firing); power cap from quanta per signal.
Run: python ir_phonon_speed_limit.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPP = 24; DT = 2 * np.pi / SPP; FDRV = 0.03; BETA0 = 0.05
F0_THZ = 23.9e12                     # 4H-SiC TO phonon (796 cm^-1)
HF = 6.626e-34 * F0_THZ              # 15.8 aJ per quantum

def run_seq(Q, W, R, gap, s_seq, ndef, ab=None):
    """s_seq: (G,K) drive sums.  Reset-phase drive is  -F0*(a sin t + b cos t)  (a=1,b=0 = plain anti-phase copy).
    Returns lock-in amplitudes (G,K), final |state| (G,), final (x, v)."""
    G, K = s_seq.shape; w0 = 1 - 0.03 * ndef; beta = BETA0 * (1 + 0.15 * ndef)
    gam = w0 / Q; w02 = w0 * w0; cyc = (W + R + gap) * SPP
    ra, rb = (np.ones(G), np.zeros(G)) if ab is None else ab
    x = np.zeros(G); v = np.zeros(G); amps = np.zeros((G, K))
    for k in range(K):
        F0 = FDRV * s_seq[:, k]; I = np.zeros(G); Qd = np.zeros(G)
        for st in range(cyc):
            if st < W * SPP: ms, mc = 1.0, 0.0
            elif st < (W + R) * SPP: ms, mc = -ra, -rb
            else: ms, mc = 0.0, 0.0
            t = (k * cyc + st) * DT
            acc = lambda tt, xx, vv: F0 * (ms * np.sin(tt) + mc * np.cos(tt)) - gam * vv - w02 * xx - beta * xx**3
            k1x = v;               k1v = acc(t, x, v)
            k2x = v + .5*DT*k1v;   k2v = acc(t + .5*DT, x + .5*DT*k1x, v + .5*DT*k1v)
            k3x = v + .5*DT*k2v;   k3v = acc(t + .5*DT, x + .5*DT*k2x, v + .5*DT*k2v)
            k4x = v + DT*k3v;      k4v = acc(t + DT, x + DT*k3x, v + DT*k3v)
            x = x + DT/6*(k1x + 2*k2x + 2*k3x + k4x); v = v + DT/6*(k1v + 2*k2v + 2*k3v + k4v)
            if (W - 1) * SPP <= st < W * SPP:
                tt = (k * cyc + st + 1) * DT; I += x * np.cos(tt); Qd += x * np.sin(tt)
        amps[:, k] = (2.0 / SPP) * np.hypot(I, Qd)
    return amps, np.hypot(x, v), x, v

def calibrate_reset(Q, ndefs):
    """Per-pad cancellation pulse (a,b): linear in the state, so 3 trial runs + a 2x2 solve null the leftover amplitude."""
    nd = np.asarray(ndefs, float); one = np.ones((len(nd), 1))
    z = np.zeros(len(nd)); o = np.ones(len(nd))
    _, _, x0, v0 = run_seq(Q, 1, 1, 0, one, nd, (z, z))
    _, _, xa, va = run_seq(Q, 1, 1, 0, one, nd, (o, z)); _, _, xb, vb = run_seq(Q, 1, 1, 0, one, nd, (z, o))
    A = np.stack([np.stack([xa - x0, xb - x0], 1), np.stack([va - v0, vb - v0], 1)], 1)       # (n,2,2)
    sol = np.linalg.solve(A, -np.stack([x0, v0], 1)[..., None])[..., 0]; return sol[:, 0], sol[:, 1]

def thresholds(Q, W, R, gap):
    s = np.arange(6, dtype=float)[:, None]; a = run_seq(Q, W, R, gap, s, np.zeros(6))[0][:, 0]
    return {th: 0.5 * (a[th - 1] + a[th]) for th in (1, 2, 3)}, a

def error_rate(Q, W, R, gap, G=600, K=14, nmax=5, seed=3, ndef=None, calibrated=False):
    rng = np.random.default_rng(seed); Th, a = thresholds(Q, W, R, gap)
    s = rng.integers(0, 6, (G, K)).astype(float); nd = rng.integers(0, nmax + 1, G).astype(float) if ndef is None else np.full(G, float(ndef))
    ab = None
    if calibrated:
        u = np.unique(nd); ca, cb = calibrate_reset(Q, u); idx = np.searchsorted(u, nd); ab = (ca[idx], cb[idx])
    amp, res, _, _ = run_seq(Q, W, R, gap, s, nd, ab); good = np.ones((G, K), bool)
    for th in (1, 2, 3): good &= ((amp >= Th[th]) == (s >= th))
    return 1 - good.mean(), a

if __name__ == "__main__":
    print(f"mode: 4H-SiC TO phonon {F0_THZ/1e12:.1f} THz, period {1e15/F0_THZ:.1f} fs, hf = {HF*1e20:.1f}e-20 J\n")
    gaps = [0, 1, 3, 7, 15, 31, 63, 127, 255]; res = {}
    print("A) PLAIN firing, drive 1 period, then idle gap; random defects 0-5 per pad; error rate over back-to-back firings")
    print("   cycle(periods)  " + "  ".join(f"Q={Q:<4}" for Q in (2, 20, 150)))
    for g in gaps:
        row = [error_rate(Q, 1, 0, g)[0] for Q in (2, 20, 150)]; res[g] = row
        print(f"   {1+g:>8} ({(1+g)*1e15/F0_THZ/1e3:7.2f} ps) " + "  ".join(f"{r:7.4f}" for r in row), flush=True)
    print("\nB) RZ anti-phase reset: drive 1 period + cancel 1 period  => cycle = 2 periods ( %.0f fs )" % (2e15 / F0_THZ))
    rz = {}
    for Q in (2, 5, 20, 150, 1000):
        e, a = error_rate(Q, 1, 1, 0, G=1500, K=20); rz[Q] = e
        print(f"   Q={Q:<5} error rate {e:.5f}   (amp ratio s=2..5 vs s=1: {np.round(a[2:]/a[1], 2).tolist()})", flush=True)
    print("\nB2) RZ with PER-PAD CALIBRATED cancellation pulse (amplitude+phase solved per defect count), cycle = 2 periods")
    rzc = {}
    for Q in (2, 5, 20, 150, 1000):
        e, _ = error_rate(Q, 1, 1, 0, G=1500, K=20, calibrated=True); rzc[Q] = e
        print(f"   Q={Q:<5} error rate {e:.5f}", flush=True)
    print("\nC) calibrated RZ at Q=150 versus defects in the pad (each defect shifts f0 by 3%):")
    byn = {}
    for n in range(0, 10):
        e, _ = error_rate(150, 1, 1, 0, G=800, K=40, ndef=n, calibrated=True); byn[n] = e; print(f"   {n} defects: error rate {e:.5f}", flush=True)
    scen = [("natural SiC Q~150, plain wait", 256), ("engineered Q~5 + calibrated RZ reset", 2 if rzc[5] < 1e-3 else 256)]
    G_F, DEPTH = 60928, 2376; NQ, INP = 100, 3.0
    e_hash = G_F * INP * NQ * HF; area = np.pi * 5.0**2; rows = []
    print(f"\nSPEED  ({G_F:,} firings/hash, depth {DEPTH}, signal energy/hash {e_hash:.2e} J at {NQ} quanta/input; power cap = 100 W/cm2 of signal heat)")
    for sname, cyc in scen:
        tc = cyc / F0_THZ; print(f" [{sname}]  cycle {cyc} periods = {tc*1e15:.0f} fs, latency {DEPTH*tc*1e12:.0f} ps")
        for name, pitch_um in (("100 mm wafer, 10 um pads", 10), ("100 mm wafer, 2 um pads", 2), ("100 mm wafer, 1 um pads", 1)):
            P = int(np.pi * 50e3**2 / pitch_um**2); thr = P / (G_F * tc); cap = 100 * area / e_hash
            rows.append((sname, name, P, thr, min(thr, cap)))
            print(f"    {name:<26} pads {P:>14,}  speed-limit {thr:9.2e} H/s  power-cap {cap:9.2e}  -> {min(thr, cap):9.2e} H/s")
    # ---------------- plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, Q in enumerate((2, 20, 150)): ax[0].semilogy([1 + g for g in gaps], [max(res[g][i], 1e-4) for g in gaps], "o-", label=f"plain, Q={Q}")
    ax[0].axhline(1e-4, c="gray", ls=":"); ax[0].set_xscale("log"); ax[0].set_xlabel("cycle time (periods of 24 THz)"); ax[0].set_ylabel("error rate (floor 1e-4 = none seen)")
    ax[0].plot([2], [max(rz[150], 1e-4)], "kx", ms=10, label="uncalibrated RZ Q=150, 2 periods"); ax[0].plot([2], [max(rzc[150], 1e-4)], "r*", ms=14, label="calibrated RZ Q=150, 2 periods"); ax[0].legend(fontsize=7); ax[0].set_title("Back-to-back firing: plain wait vs RZ reset")
    lab = [f"{r[0].split(' ')[0]}/{r[1].split(', ')[1]}" for r in rows]
    ax[1].bar(range(len(rows)), [r[4] for r in rows], color=["#7f8c8d"] * 3 + ["#c0392b"] * 3); ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels(lab, fontsize=6, rotation=30); ax[1].set_ylabel("H/s (min of speed and power caps)")
    ax[1].set_title("grey: natural Q~150 | red: engineered Q~5 + RZ")
    fig.tight_layout(); fig.savefig("ir_phonon_speed_limit.png", dpi=110); print("saved ir_phonon_speed_limit.png")
