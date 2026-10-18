"""
ir_phonon_full_sha256.py -- the WHOLE simulation, end to end, with stateful 24 THz SiC infrared-phonon defect pads.

  1. acoustic blast of the patterned wafer (2-D wave FDTD, from v2) -> 440 bits per message, 16 messages
  2. every pad is a defect-detuned damped oscillator (4H-SiC TO phonon, f0 = 23.9 THz) whose (x, v) state PERSISTS between
     firings; a firing = drive 1 period with the weighted input sum, lock-in readout, then reset / wait; the next gate fired
     on that pad starts from whatever ringing is left.  Nothing is reset by hand.
  3. the full SHA-256 compression (threshold-logic netlist from v2, 60,928 firings, depth 2376) is executed level by level on the pads
  4. digests are checked against hashlib; raw gate errors, leftover ringing and simulated hash time are reported
Scenarios:  fast  = engineered Q=5 + per-pad calibrated anti-phase reset, 2-period cycle (84 fs)
            q20   = Q=20, plain wait, 32-period cycle      natural = SiC Q=150, plain wait, 256-period cycle
Usage:  python ir_phonon_full_sha256.py --prep ;  python ir_phonon_full_sha256.py --scen fast|q20|natural
"""
import sys, time, pickle, inspect, hashlib
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("v2", "si_acoustic_defect_sha256_v2.py"); v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
spec = importlib.util.spec_from_file_location("sl", "ir_phonon_speed_limit.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)

SPP = 24; DT = 2 * np.pi / SPP; FD = 0.03; F0_HZ = 23.9e12
SCEN = {"fast": dict(Q=5.0, cyc=(1, 1, 0), reset=True), "q20": dict(Q=20.0, cyc=(1, 0, 31), reset=False),
        "natural": dict(Q=150.0, cyc=(1, 0, 255), reset=False)}

def fire_state(X, V, s, w0, be, ra, rb, Q, cyc, trace=False):
    """One firing on a batch of DISTINCT pads, starting from their stored state.  Returns lock-in amp and new state."""
    W, R, gap = cyc; gam = w0 / Q; w02 = w0 * w0; F0 = FD * s
    x = X.copy(); v = V.copy(); I = np.zeros_like(x); Qd = np.zeros_like(x); tr = []
    def rk(x, v, t, h, ms, mc):
        acc = lambda tt, xx, vv: F0 * (ms * np.sin(tt) + mc * np.cos(tt)) - gam * vv - w02 * xx - be * xx**3
        k1x = v;               k1v = acc(t, x, v)
        k2x = v + .5*h*k1v;    k2v = acc(t + .5*h, x + .5*h*k1x, v + .5*h*k1v)
        k3x = v + .5*h*k2v;    k3v = acc(t + .5*h, x + .5*h*k2x, v + .5*h*k2v)
        k4x = v + h*k3v;       k4v = acc(t + h, x + h*k3x, v + h*k3v)
        return x + h/6*(k1x + 2*k2x + 2*k3x + k4x), v + h/6*(k1v + 2*k2v + 2*k3v + k4v)
    for st in range((W + R) * SPP):
        ms, mc = (1.0, 0.0) if st < W * SPP else (-ra, -rb); t = st * DT
        x, v = rk(x, v, t, DT, ms, mc)
        if st >= (W - 1) * SPP and st < W * SPP:
            tt = (st + 1) * DT; I += x * np.cos(tt); Qd += x * np.sin(tt)
        if trace: tr.append(x.copy())
    h2 = 2 * DT; t0 = (W + R) * SPP * DT
    for st in range(gap * SPP // 2): x, v = rk(x, v, t0 + st * h2, h2, 0.0, 0.0)     # idle wait: 12 steps / period
    amp = (2.0 / SPP) * np.hypot(I, Qd)
    return (amp, x, v, np.array(tr)) if trace else (amp, x, v)

def setup(sc):
    p = SCEN[sc]; Q, cyc = p["Q"], p["cyc"]
    s = np.arange(6, dtype=float); z = np.zeros(6)
    a = fire_state(z, z, s, np.ones(6), np.full(6, sl.BETA0), z, z, Q, (1, 0, 0))[0]
    Th = np.array([0.0] + [0.5 * (a[t - 1] + a[t]) for t in (1, 2, 3)])
    ra = np.zeros(10); rb = np.zeros(10)
    if p["reset"]: ra, rb = sl.calibrate_reset(Q, np.arange(10))
    return Q, cyc, Th, ra, rb

def pad_stress(Q, cyc, Th, ra, rb, G=300, K=40, seed=2):
    """Self-test WITH history: random drive sequences back-to-back on pads of each defect count; zero errors required."""
    rng = np.random.default_rng(seed); ok = np.zeros(10, bool)
    for n in range(10):
        w0, be = sl.__dict__.get("x", None) or (1 - 0.03 * n, sl.BETA0 * (1 + 0.15 * n))
        X = np.zeros(G); V = np.zeros(G); bad = 0
        for k in range(K):
            s = rng.integers(0, 6, G).astype(float)
            amp, X, V = fire_state(X, V, s, np.full(G, w0), np.full(G, be), np.full(G, ra[n]), np.full(G, rb[n]), Q, cyc)
            for th in (1, 2, 3): bad += int(((amp >= Th[th]) != (s >= th)).sum())
        ok[n] = bad == 0
    return ok

def run_net_state(net, inw, bits_list, pad_ok, P, ndef_p, Th, ra_n, rb_n, Q, cyc, log=True):
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
    w0_p = 1 - 0.03 * ndef_p; be_p = sl.BETA0 * (1 + 0.15 * ndef_p); ra_p = ra_n[ndef_p]; rb_p = rb_n[ndef_p]
    X = np.zeros(P); V = np.zeros(P); pad_of = np.zeros(len(go), int); off = 0; chunks = 0; wrong = 0; fired = 0; t0 = time.time()
    for L in range(1, maxL + 1):
        idx = order[bounds[L - 1]:bounds[L]]
        if len(idx):
            n = len(idx); tot = B * n; Lc = len(pad_ok)
            sdr = (val[:, gi[idx]] * gw[idx][None]).sum(-1).reshape(-1); thr = np.tile(gth[idx], B)
            pads_all = pad_ok[(off + np.arange(tot)) % Lc]; off += tot; outs = np.zeros(tot, np.int8)
            for c0 in range(0, tot, Lc):
                sl_ = slice(c0, min(tot, c0 + Lc)); pp = pads_all[sl_]
                amp, xn, vn = fire_state(X[pp], V[pp], sdr[sl_], w0_p[pp], be_p[pp], ra_p[pp], rb_p[pp], Q, cyc)
                X[pp] = xn; V[pp] = vn; outs[sl_] = (amp >= Th[thr[sl_]]); chunks += 1
            wrong += int((outs != (sdr >= thr)).sum()); fired += tot
            val[:, go[idx]] = outs.reshape(B, n); pad_of[idx] = pads_all[:n]
        do_alias(L)
        if log and L % 300 == 0: print(f"  level {L}/{maxL}  ({time.time()-t0:.0f}s)", flush=True)
    return val, pad_of, lv, dict(chunks=chunks, wrong=wrong, fired=fired, leftover=float(np.hypot(X, V).max()))

if __name__ == "__main__":
    sites, tsd, bpd, ndef = v2.make_wafer()
    if "--prep" in sys.argv:
        E0, snaps, recpad = v2.acoustic_blast(sites, tsd, bpd, P=16); bl = [(E0 >= np.median(E0)).astype(int)]
        for P_ in [10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
            Ep, _, _ = v2.acoustic_blast(sites, tsd, bpd, nsteps=1200, nsnap=0, P=P_); bl.append((Ep >= np.median(Ep)).astype(int))
        msgs = [v2.bits_to_message(b) for b in bl]; refs = [hashlib.sha256(m).hexdigest() for m in msgs]
        pickle.dump(dict(E0=E0, snaps=snaps, recpad=recpad, bits=bl, msgs=msgs, refs=refs), open("msgs.pkl", "wb")); print("prep done; message0", msgs[0].hex()); sys.exit()
    sc = sys.argv[sys.argv.index("--scen") + 1]; D = pickle.load(open("msgs.pkl", "rb")); bl, refs = D["bits"], D["refs"]
    Q, cyc, Th, ra, rb = setup(sc); tc = sum(cyc) / F0_HZ
    print(f"[{sc}] Q={Q}, cycle {sum(cyc)} periods = {tc*1e15:.0f} fs, thresholds {np.round(Th[1:],4).tolist()}, reset(a,b) n=0: ({ra[0]:.3f},{rb[0]:.3f})", flush=True)
    ok_n = pad_stress(Q, cyc, Th, ra, rb); print(f"pad self-test with history, defects 0..9: {ok_n.astype(int).tolist()}", flush=True)
    net, inw, rend, stw, dig = v2.build_sha_net(440); ng = len(net.gout); print(f"netlist {ng:,} firings, depth {int(max(net.lvl))}", flush=True)
    results = {}
    if sc == "fast":                                           # --- the 88-pad wafer, one hash, animated
        ok = ok_n[ndef]; val, pad_of, lv, st = run_net_state(net, inw, [bl[0]], np.where(ok)[0], len(ndef), ndef, Th, ra, rb, Q, cyc)
        d0 = "".join(f"{v2.word_val(val[0], w):08x}" for w in dig)
        print(f"[wafer, {ok.sum()}/88 usable pads] digest {d0}\n  hashlib {refs[0]} -> {'PASS' if d0 == refs[0] else 'FAIL'}; raw gate errors {st['wrong']}/{st['fired']:,}; "
              f"pad-time slots {st['chunks']:,} -> {st['chunks']*tc*1e9:.3f} ns/hash; leftover ringing max {st['leftover']:.2e}", flush=True)
        results["wafer"] = (d0 == refs[0], st)
    # --- micro-pad array, 16 hashes pipelined
    Pm = int(np.pi * v2.WAFER_R**2 / 0.1**2); rng = np.random.default_rng(5); ndm = np.zeros(Pm, int)
    for k in rng.integers(0, Pm, len(tsd) + len(bpd)): ndm[k] += 1
    okm = ok_n[ndm]; valm, _, lvm, st = run_net_state(net, inw, bl, np.where(okm)[0], Pm, ndm, Th, ra, rb, Q, cyc)
    dm = ["".join(f"{v2.word_val(valm[b], w):08x}" for w in dig) for b in range(16)]; npass = sum(dm[b] == refs[b] for b in range(16))
    print(f"[micro-pad array {Pm:,} pads ({(~okm).sum()} masked), 16 pipelined hashes] {npass}/16 digests match hashlib; raw gate errors {st['wrong']}/{st['fired']:,}; "
          f"leftover ringing max {st['leftover']:.2e}; batch time {st['chunks']*tc*1e12:.1f} ps (latency-bound)", flush=True)
    G_F = ng; cap = Pm / (G_F * tc)
    print(f"capacity of this array at this cycle: {cap:.2e} H/s (needs ~{cap*len(set(lvm.tolist()))*tc:,.0f} hashes in flight)", flush=True)
    if sc == "fast":                                           # --- animation
        tr = np.zeros((len(ndef), 4, 2 * SPP)); w0 = 1 - 0.03 * ndef; be = sl.BETA0 * (1 + 0.15 * ndef)
        for s_ in range(4):
            tr[:, s_, :] = fire_state(np.zeros(len(ndef)), np.zeros(len(ndef)), np.full(len(ndef), float(s_)), w0, be, ra[ndef], rb[ndef], Q, cyc, trace=True)[3].T
        src = inspect.getsource(v2.animate)
        rep = [('"v2: acoustic stage = real 2-D wave FDTD; each pad firing = RK4 of a %d-mode defect-set resonator with lock-in readout; "\n             "constants assumed; signals regenerated/routed by a software scheduler." % M',
                '"Acoustic blast = real 2-D wave FDTD. Pads = 23.9 THz SiC TO-phonon oscillators with persistent state, calibrated anti-phase reset, RK4 per firing; constants assumed; gain/readout/routing not modelled."'),
               ('f"each pad carries {M} acoustic modes (f0, 2f0, 3f0) = {M} gates per firing"', '"pad = SiC TO phonon, 23.9 THz; one gate per 84 fs firing; ringing carried to next firing"'),
               ('({M} gates per pad firing)', '(1 gate per 84 fs pad firing)'),
               ('3. Each defect pad self-tests as a multi-mode threshold-logic gate', '3. Each defect pad self-tests (with history) as a 24 THz phonon gate'),
               ('4. Defect pads fire gates: SHA-256 compression, 64 rounds', '4. 24 THz defect pads fire gates (state carried over): SHA-256, 64 rounds'),
               ('for th in (1, 2, 3): ax.axhline(Th[0, th], c="k", ls="--", lw=.8)', 'ax.axvline(1.0, c="k", ls="--", lw=.8)'),
               ('"detector thresholds (theta=1,2,3)"', '"end of drive window (reset pulse follows)"'),
               ('"si_acoustic_sha256_v2.gif"', '"ir_phonon_full_sha256.gif"'), ('"v2_blast.png"', '"full_blast.png"'), ('"v2_compute.png"', '"full_compute.png"'), ('"v2_digest.png"', '"full_digest.png"'),
               ('saved si_acoustic_sha256_v2.gif', 'saved ir_phonon_full_sha256.gif')]
        for a_, b_ in rep:
            assert a_ in src, a_[:40]; src = src.replace(a_, b_)
        exec(src, v2.__dict__)
        v2.animate(sites, tsd, bpd, ndef, D["E0"], bl[0], D["snaps"], D["recpad"], D["msgs"][0], refs[0], ok, tr, Th.reshape(1, 4), (val, pad_of, lv, rend, stw, dig, ng), 1)
