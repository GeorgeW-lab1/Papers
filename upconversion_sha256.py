#!/usr/bin/env python3
"""Defect-dependent NaYF4:Yb,Er phenomenological SHA-256 simulator.
Requires Python 3.10+ and numpy. No external files or original script required.
The semiconductor lattice has been removed; the original threshold netlist is retained.
NOT an atomistic model, measured device, all-optical computer, or mining-speed claim.
"""
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np

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

def build_compression_net(nbits=512):
    net = Net(); inw = [net.inp() for _ in range(nbits)]
    hwords = [[net.inp() for _ in range(32)] for _ in range(8)]
    blk = inw
    W = [[blk[32 * t + 31 - i] for i in range(32)] for t in range(16)]
    a, b, c, d, e, f, g, h = hwords; rend, stw = [], []
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
    dig = [net.add(hwords[i], w) for i, w in enumerate([a, b, c, d, e, f, g, h])]
    return net, inw, hwords, rend, stw, dig

def word_val(val, w): return sum(int(val[wi]) << i for i, wi in enumerate(w))


class Coating:
    """Two effective normalized populations, independently parameterized per pad.
    dx/dt = alpha*I*(1-x) - gx*x
    dy/dt = k*x*x*(1-y) - gy*y
    gy = grad + gnr; detected rate = N_eff*collection*grad*y.
    x is an effective reservoir, NOT a literal Yb-ion number balance.
    Energy accounting, reabsorption, temperature and interpad scattering are absent.
    """
    def __init__(self, args):
        self.a = args
        self.rng = np.random.default_rng(args.seed)
        n = args.pads
        self.defects = self.rng.poisson(args.defects, n)
        d = self.defects.astype(float)
        self.alpha = 180.0*np.exp(self.rng.normal(0, .08, n))
        self.gx = 1000.0*(1+.18*d)
        self.k = 1200.0*np.exp(self.rng.normal(0, .10, n))/(1+.08*d)
        self.grad = np.full(n, 2500.0)
        self.gnr = 800.0+900.0*d
        self.gy = self.grad+self.gnr
        self.collection = .025*np.exp(self.rng.normal(0, .12, n))
        self.neff = args.emitters
        self.x = np.zeros(n); self.y = np.zeros(n)
        self.last = np.zeros(n, dtype=np.int64)
        self.slot = 0; self.offset = 0
        self.cycle = args.drive_ms*.001+args.dark_ms*.001
        self.threshold = np.zeros((n, 4))
        self.good = np.zeros(n, bool)
        self.raw_errors = 0; self.firings = 0; self.photons = 0.0
        self.peak_x = 0.0; self.peak_y = 0.0

    def evolve(self, ids, x, y, irradiance, duration, collect=True):
        # Exact reservoir step and exponential emitter update with midpoint feed.
        # Integrate emission analytically within each frozen-feed substep.
        duration = np.broadcast_to(np.asarray(duration, float), x.shape)
        h = duration/self.a.steps
        pump = self.alpha[ids]*np.asarray(irradiance)
        lam = pump+self.gx[ids]; eq = pump/lam
        half = np.exp(-lam*h*.5); full = half*half
        photons = np.zeros_like(x)
        for _ in range(self.a.steps):
            xm = eq+(x-eq)*half
            feed = self.k[ids]*xm*xm
            rate = feed+self.gy[ids]; yeq = feed/rate
            one_minus = -np.expm1(-rate*h)
            if collect:
                integ = yeq*h+(y-yeq)*one_minus/rate
                photons += self.neff*self.collection[ids]*self.grad[ids]*integ
            y = y+(yeq-y)*one_minus
            x = eq+(x-eq)*full
        return x, y, np.maximum(photons, 0)

    def calibrate(self):
        ids = np.arange(self.a.pads); z = np.zeros(self.a.pads)
        signals = []
        for s in range(7):
            _, _, ph = self.evolve(ids, z.copy(), z.copy(), s*self.a.irradiance,
                                    self.a.drive_ms*.001)
            signals.append(ph+self.a.background*self.a.drive_ms*.001)
        self.signals = np.stack(signals, axis=1)
        for th in (1, 2, 3):
            self.threshold[:, th] = .5*(self.signals[:, th-1]+self.signals[:, th])
        sigma = np.sqrt(self.signals+self.a.read_noise**2)
        margins = (np.diff(self.signals[:, :4], axis=1)/2)/np.maximum(sigma[:, 1:4], 1e-20)
        self.good = np.all(np.diff(self.signals[:, :4], axis=1)>0, axis=1)
        self.good &= np.min(margins, axis=1)>=self.a.min_snr
        self.usable = np.flatnonzero(self.good)
        if not len(self.usable):
            raise RuntimeError('No pads passed calibration. Increase --emitters or reduce --min-snr.')
        return float(margins[self.good].min())

    def reset_state(self):
        # Explicit experiment initialization only; never called between gates/blocks.
        self.x.fill(0); self.y.fill(0); self.last.fill(0)
        self.slot = 0; self.offset = 0
        self.raw_errors = 0; self.firings = 0; self.photons = 0
        self.peak_x = 0; self.peak_y = 0

    def gates(self, sums, thresholds):
        out = np.empty(len(sums), np.int8)
        for start in range(0, len(sums), len(self.usable)):
            stop = min(start+len(self.usable), len(sums))
            ids = self.usable[(self.offset+np.arange(stop-start))%len(self.usable)]
            self.offset = (self.offset+stop-start)%len(self.usable)
            elapsed = (self.slot-self.last[ids])*self.cycle
            x, y = self.x[ids].copy(), self.y[ids].copy()
            if np.any(elapsed):
                x, y, _ = self.evolve(ids, x, y, 0., elapsed, False)
            x, y, ph = self.evolve(ids, x, y,
                sums[start:stop]*self.a.irradiance, self.a.drive_ms*.001)
            self.peak_x = max(self.peak_x, float(x.max()))
            self.peak_y = max(self.peak_y, float(y.max()))
            means = ph+self.a.background*self.a.drive_ms*.001
            detected = means.copy()
            if self.a.noise:
                detected = self.rng.poisson(means).astype(float)
                detected += self.rng.normal(0, self.a.read_noise, len(ids))
            bits = detected>=self.threshold[ids, thresholds[start:stop]]
            out[start:stop] = bits
            self.raw_errors += int(np.count_nonzero(bits != (sums[start:stop]>=thresholds[start:stop])))
            self.firings += len(ids); self.photons += float(ph.sum())
            x, y, _ = self.evolve(ids, x, y, 0., self.a.dark_ms*.001, False)
            self.x[ids] = x; self.y[ids] = y
            self.slot += 1; self.last[ids] = self.slot
        return out

    def write_pads(self, path):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pad','defects','alpha_per_W_cm2_s','reservoir_decay_per_s',
                        'transfer_per_s','radiative_decay_per_s','nonradiative_decay_per_s',
                        'collection','usable','threshold_1_photons','threshold_2_photons',
                        'threshold_3_photons'])
            for i in range(self.a.pads):
                w.writerow([i, self.defects[i], self.alpha[i], self.gx[i], self.k[i],
                    self.grad[i], self.gnr[i], self.collection[i], int(self.good[i]),
                    *self.threshold[i, 1:]])

class Engine:
    def __init__(self, coating=None):
        self.coating = coating
        self.net, self.inw, self.hw, _, _, self.dig = build_compression_net()
        n = self.net
        self.gi = np.asarray(n.gi); self.gw = np.asarray(n.gw)
        self.th = np.asarray(n.gth); self.go = np.asarray(n.gout)
        levels = np.asarray(n.lvl)[self.go]
        self.depth = int(levels.max())
        order = np.argsort(levels, kind='stable')
        bounds = np.searchsorted(levels[order], np.arange(1, self.depth+2))
        self.layers = [order[bounds[i]:bounds[i+1]] for i in range(self.depth)]
        self.alias = {}
        for dst, source in n.alias:
            self.alias.setdefault(n.lvl[dst], []).append((dst, source))
        self.alias = {l: np.asarray(v).T for l,v in self.alias.items()}

    def compress(self, block, state):
        v = np.zeros(len(self.net.lvl), np.int8); v[1] = 1
        v[self.inw] = np.unpackbits(np.frombuffer(block, np.uint8))
        for wires, word in zip(self.hw, state):
            v[wires] = [(word>>i)&1 for i in range(32)]
        if 0 in self.alias:
            dst, source = self.alias[0]; v[dst] = 1-v[source]
        for level, idx in enumerate(self.layers, 1):
            if len(idx):
                sums = (v[self.gi[idx]]*self.gw[idx]).sum(axis=1)
                v[self.go[idx]] = (sums>=self.th[idx] if self.coating is None else
                                      self.coating.gates(sums, self.th[idx]))
            if level in self.alias:
                dst, source = self.alias[level]; v[dst] = 1-v[source]
        return [word_val(v, wires) for wires in self.dig]

    def digest(self, message):
        size = len(message)*8
        if size>=2**64: raise ValueError('SHA-256 length limit exceeded')
        padded = message+b'\x80'
        padded += b'\x00'*((56-len(padded)%64)%64)+size.to_bytes(8, 'big')
        state = H0.copy()
        for i in range(0, len(padded), 64):
            state = self.compress(padded[i:i+64], state)
        return ''.join(f'{x:08x}' for x in state), len(padded)//64

def selftest():
    eng = Engine()
    vectors = [b'', b'abc', b'a'*55, b'a'*56, b'a'*64, bytes(range(256))]
    results = []
    for message in vectors:
        digest, blocks = eng.digest(message)
        ok = digest == hashlib.sha256(message).hexdigest()
        results.append(dict(bytes=len(message), blocks=blocks, match=ok))
        print(f'Boolean netlist: {len(message):3d} bytes, {blocks} blocks: {"PASS" if ok else "FAIL"}')
        if not ok: raise AssertionError('Netlist SHA-256 self-test failed')
    return results

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--message', default='abc')
    p.add_argument('--hex', dest='hex_message', help='Input bytes as hexadecimal, overrides --message')
    p.add_argument('--input-file', type=Path, help='Hash arbitrary file bytes, overrides text/hex')
    p.add_argument('--pads', type=int, default=256)
    p.add_argument('--defects', type=float, default=2., help='Poisson mean; hypothetical defects')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--drive-ms', type=float, default=2.)
    p.add_argument('--dark-ms', type=float, default=12.)
    p.add_argument('--irradiance', type=float, default=1., help='W/cm^2 per unit weighted input')
    p.add_argument('--emitters', type=float, default=1e8, help='Effective emitting population scale')
    p.add_argument('--steps', type=int, default=32, help='Integration substeps per phase; check convergence')
    p.add_argument('--noise', action='store_true', help='Enable Poisson shot + Gaussian read noise')
    p.add_argument('--read-noise', type=float, default=3., help='RMS detected photons per read window')
    p.add_argument('--background', type=float, default=100., help='Detected background photons/s')
    p.add_argument('--min-snr', type=float, default=6., help='Minimum static threshold margin / sigma')
    p.add_argument('--self-test', action='store_true', help='Run Boolean known-answer tests before optical run')
    p.add_argument('--boolean-only', action='store_true', help='Run exact netlist without coating physics')
    p.add_argument('--out', type=Path, default=Path('upconversion_results'))
    args = p.parse_args()
    if args.pads<1 or args.steps<1 or args.defects<0 or args.drive_ms<=0 or args.dark_ms<0:
        p.error('pads/steps/drive must be positive; defects/dark must be nonnegative')
    if args.irradiance<=0 or args.emitters<=0 or min(args.read_noise,args.background,args.min_snr)<0:
        p.error('invalid intensity, population, or noise settings')
    tests = selftest() if args.self_test else []
    message = args.input_file.read_bytes() if args.input_file else (
        bytes.fromhex(args.hex_message) if args.hex_message is not None else args.message.encode())
    coating = None if args.boolean_only else Coating(args)
    args.out.mkdir(parents=True, exist_ok=True)
    margin = None
    if coating is not None:
        margin = coating.calibrate()
        coating.write_pads(args.out/'pads.csv')
        print(f'Calibrated {len(coating.usable)}/{args.pads} pads; minimum static SNR margin {margin:.2f}')
    eng = Engine(coating)
    print(f'Compression netlist: {len(eng.go):,} firings/block, depth {eng.depth:,}')
    t0 = time.perf_counter(); digest, blocks = eng.digest(message); elapsed = time.perf_counter()-t0
    reference = hashlib.sha256(message).hexdigest(); match = digest==reference
    result = dict(model='hypothetical reduced population model, not measured hardware',
        message_bytes=len(message), blocks=blocks, digest=digest, reference=reference,
        match=match, wall_seconds=elapsed, gates_per_block=len(eng.go), depth=eng.depth,
        selftests=tests, parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if coating is not None:
        result.update(usable_pads=len(coating.usable), minimum_static_margin_snr=margin,
            gate_firings=coating.firings, raw_gate_errors=coating.raw_errors,
            simulated_slots=coating.slot, simulated_schedule_seconds=coating.slot*coating.cycle,
            expected_detected_signal_photons=coating.photons,
            peak_reservoir=coating.peak_x, peak_emitter=coating.peak_y,
            remaining_reservoir=float(coating.x.max()),remaining_emitter=float(coating.y.max()))
        print(f'Raw gate errors: {coating.raw_errors}/{coating.firings}; slots: {coating.slot:,}')
        print(f'Model schedule: {result["simulated_schedule_seconds"]:.6f} s (ideal electronic routing)')
    print(f'Digest:    {digest}\nReference: {reference}\nResult: {"PASS" if match else "FAIL"}\nCPU simulation: {elapsed:.3f} s')
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    if not match: raise SystemExit(1)

if __name__=='__main__':
    main()
