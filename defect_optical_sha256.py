#!/usr/bin/env python3
"""Hypothetical defect-programmed optical SHA-256 steady-state simulator.

NOT validated NaYF4 physics. NOT an experimentally demonstrated device.
Continuous dual-rail intensities connect all internal gates. Only final digest
ports are decoded. Uses a phenomenological pumped bistable defect-switch model,
not atomic defects in powder or an atomistic/coupled-Maxwell solver.
Python 3.10+, numpy. Optional output directory contains physical-target netlist.
"""
import argparse, hashlib, json, time
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


class DefectCircuit:
    """One engineered defect switch per threshold gate, not a time-shared pad.

    Assumed local bistable material state:
        tau*dq/dt = q - q^3 + gain*(sum_i c_i I_i - bias)
    Assumed pumped dual-rail transduction:
        I_true = .5*(1+tanh(contrast*q))
        I_false = .5*(1-tanh(contrast*q))

    This rail transfer is an ASSUMED device law, NOT inferred from NaYF4 data.
    It includes ideal regeneration, complementary outputs and power supply.
    q is a dimensionless order parameter, not a literal optical field amplitude.
    Defect geometry is represented by static bias/coupling/tau perturbations.
    No realizable fabrication map from these parameters is asserted.

    A feedforward steady-state solver advances topological layers while already
    solved upstream outputs remain continuous. This is NOT a simultaneous
    transient simulation and does not predict physical latency or clock speed.
    """
    def __init__(self, args):
        self.args=args
        self.net,self.inw,self.hw,_,_,self.digest_wires=build_compression_net()
        self.gi=np.asarray(self.net.gi,dtype=np.int32)
        self.weights=np.asarray(self.net.gw,float)
        self.go=np.asarray(self.net.gout,dtype=np.int32)
        self.theta=np.asarray(self.net.gth,float)
        self.level=np.asarray(self.net.lvl)[self.go]
        self.depth=int(self.level.max()); self.n=len(self.go)
        rng=np.random.default_rng(args.seed)
        # Static fabrication disorder, drawn once and retained for all blocks.
        self.bias=self.theta-.5+rng.normal(0,args.detuning_std,self.n)
        self.coupling=self.weights*np.exp(rng.normal(0,args.coupling_std,self.weights.shape))
        self.tau=np.exp(rng.normal(0,args.tau_std,self.n))
        self.initial_q=rng.uniform(-1,1,self.n)
        order=np.argsort(self.level,kind='stable')
        edges=np.searchsorted(self.level[order],np.arange(1,self.depth+2))
        self.layers=[order[edges[l]:edges[l+1]] for l in range(self.depth)]
        self.alias={}
        for dst,src in self.net.alias:
            self.alias.setdefault(self.net.lvl[dst],[]).append((dst,src))
        self.alias={l:np.asarray(a,dtype=np.int32).T for l,a in self.alias.items()}
        self.worst_residual=0.; self.maximum_settle=0.; self.blocks=0
        self.last_ports=None

    def solve_switches(self, idx, drive):
        q=self.initial_q[idx].copy()
        tau=self.tau[idx]
        dt=self.args.dt
        def f(x): return (x-x*x*x+drive)/tau
        elapsed=0.
        # Normalized numerical time only, not an experimental device timescale.
        for step in range(int(np.ceil(self.args.max_settle/dt))):
            k1=f(q); k2=f(q+.5*dt*k1); k3=f(q+.5*dt*k2); k4=f(q+dt*k3)
            q=q+(dt/6)*(k1+2*k2+2*k3+k4); elapsed+=dt
            if not np.all(np.isfinite(q)):
                raise RuntimeError('Unstable integration; reduce --dt or fabrication disorder')
            if step%8==7 and elapsed>=self.args.min_settle:
                if np.max(np.abs(f(q)))<self.args.tolerance: break
        residual=float(np.max(np.abs(f(q))))
        self.worst_residual=max(self.worst_residual,residual)
        self.maximum_settle=max(self.maximum_settle,elapsed)
        true=.5*(1+np.tanh(self.args.contrast*q))
        # Both rails are the explicitly assumed physical transduction law.
        return np.stack((true,1-true),axis=-1)

    def compress(self, block, chaining_ports):
        ports=np.zeros((len(self.net.lvl),2))
        ports[0]=[0,1]; ports[1]=[1,0]
        inputs=np.unpackbits(np.frombuffer(block,np.uint8)).astype(float)
        ports[self.inw]=np.stack((inputs,1-inputs),axis=-1)
        for wires,values in zip(self.hw,chaining_ports): ports[wires]=values
        def swap_aliases(level):
            if level in self.alias:
                dst,src=self.alias[level]
                ports[dst]=ports[src,::-1]
        swap_aliases(0)
        for level,idx in enumerate(self.layers,1):
            if len(idx):
                # Continuous optical intensities only. No intermediate bit decode.
                incident=np.sum(self.coupling[idx]*ports[self.gi[idx],0],axis=1)
                drive=self.args.gain*(incident-self.bias[idx])
                ports[self.go[idx]]=self.solve_switches(idx,drive)
            swap_aliases(level)
        self.blocks+=1
        self.last_ports=ports
        # Preserve analog rails between compression blocks, no digest read here.
        return np.asarray([ports[w] for w in self.digest_wires])

    def run(self,message):
        bitlen=8*len(message)
        if bitlen>=2**64: raise ValueError('Message too long for SHA-256')
        padded=message+b'\x80'
        padded+=b'\x00'*((56-len(padded)%64)%64)+bitlen.to_bytes(8,'big')
        chaining=np.asarray([[[float((h>>i)&1),float(1-((h>>i)&1))]
                                for i in range(32)] for h in H0])
        for offset in range(0,len(padded),64):
            chaining=self.compress(padded[offset:offset+64],chaining)
        return chaining

    def export(self,directory,blocks):
        # Logical optical-port image. No fabricated nanometre geometry is invented.
        image=dict(format='hypothetical-defect-optical-netlist-v1',
            status='design target; no validated NaYF4 transfer or fabrication model',
            pads_per_block=self.n,depth=self.depth,unrolled_blocks=blocks,
            total_switch_instances=self.n*blocks,
            semantics='waveguide routes carry continuous dual-rail intensities',
            constants={'0':[0,1],'1':[1,0]},message_ports=self.inw,
            chaining_ports=self.hw,output_ports=self.digest_wires,
            rail_swap_routes=[list(x) for x in self.net.alias],
            global_gain=self.args.gain,global_contrast=self.args.contrast,
            pads=[dict(pad=int(i),output_port=int(self.go[i]),level=int(self.level[i]),
                inputs=self.gi[i].tolist(),couplings=self.coupling[i].tolist(),
                defect_bias=float(self.bias[i]),relative_tau=float(self.tau[i]))
                for i in range(self.n)])
        (directory/'defect_pad_image.json').write_text(json.dumps(image,separators=(',',':')))
        # A real fully unrolled multiblock circuit requires separate instances.
        manifest=dict(blocks=blocks,one_block_image='defect_pad_image.json',
            connections='block j final analog rails -> block j+1 chaining input rails',
            each_block_has_separate_input_ports_and_defect_switches=True,
            disorder='same instance pattern repeated for each block in this model',
            note='CPU reuses arrays to solve separate physical block instances; no intermediate decode')
        (directory/'block_connections.json').write_text(json.dumps(manifest,indent=2))

def decode_final(rails):
    # The ONLY analog-to-bit classification in the optical execution path.
    bits=rails[:,:,0]>rails[:,:,1]
    words=[sum(int(bit)<<i for i,bit in enumerate(word)) for word in bits]
    return ''.join(f'{word:08x}' for word in words)

def simulate(args,message):
    circuit=DefectCircuit(args)
    start=time.perf_counter(); rails=circuit.run(message); wall=time.perf_counter()-start
    digest=decode_final(rails)
    reference=hashlib.sha256(message).hexdigest()
    result=dict(model='hypothetical defect-bistability / ideal rail-regenerator model',
        validated_NaYF4_device=False,solver='feedforward steady state, not full transient',
        message_bytes=len(message),blocks=circuit.blocks,
        pads_per_block=circuit.n,total_switch_instances=circuit.n*circuit.blocks,
        depth_per_block=circuit.depth,simulated_digest=digest,reference_digest=reference,
        match=digest==reference,final_min_rail_separation=float(np.min(np.abs(rails[:,:,0]-rails[:,:,1]))),
        maximum_ode_residual=circuit.worst_residual,
        converged=circuit.worst_residual<args.tolerance,
        maximum_local_normalized_settle=circuit.maximum_settle,
        cpu_seconds=wall,physical_latency_seconds=None,
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    return circuit,rails,result

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--message',default='abc')
    p.add_argument('--hex',dest='hex_message')
    p.add_argument('--input-file',type=Path)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--gain',type=float,default=2.)
    p.add_argument('--contrast',type=float,default=4.)
    p.add_argument('--detuning-std',type=float,default=.015)
    p.add_argument('--coupling-std',type=float,default=.005)
    p.add_argument('--tau-std',type=float,default=.03)
    p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--min-settle',type=float,default=2.)
    p.add_argument('--max-settle',type=float,default=40.)
    p.add_argument('--tolerance',type=float,default=1e-7)
    p.add_argument('--export-image',action='store_true',help='Export abstract pad configuration, not fabrication geometry')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--out',type=Path,default=Path('defect_results'))
    return p

def main():
    p=parser(); a=p.parse_args()
    if min(a.gain,a.contrast,a.dt,a.max_settle,a.tolerance)<=0:
        p.error('gain, contrast, dt, max-settle and tolerance must be positive')
    if min(a.detuning_std,a.coupling_std,a.tau_std,a.min_settle)<0 or a.min_settle>a.max_settle:
        p.error('invalid disorder or settling settings')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        checks=[]
        for message in [b'',b'abc',b'a'*55,b'a'*56,b'a'*64]:
            _,_,r=simulate(a,message); checks.append(r)
            print(f'Test {len(message)} bytes: match={r["match"]}, converged={r["converged"]}')
        (a.out/'selftests.json').write_text(json.dumps(checks,indent=2))
        if not all(x['match'] and x['converged'] for x in checks):
            raise SystemExit('Self-test failed; see selftests.json')
    message=a.input_file.read_bytes() if a.input_file else (
        bytes.fromhex(a.hex_message) if a.hex_message is not None else a.message.encode())
    circuit,rails,r=simulate(a,message)
    (a.out/'report.json').write_text(json.dumps(r,indent=2))
    np.savetxt(a.out/'final_optical_rails.csv',
        np.asarray([[word,bit,*rails[word,bit]] for word in range(8) for bit in range(32)]),
        delimiter=',',header='word,bit_lsb_first,true_intensity,false_intensity',comments='')
    if a.export_image: circuit.export(a.out,circuit.blocks)
    print(f'Physical target: {r["total_switch_instances"]:,} hypothetical switches; {r["blocks"]} blocks')
    print(f'Digest:    {r["simulated_digest"]}\nReference: {r["reference_digest"]}')
    print(f'Match: {r["match"]}; ODE converged: {r["converged"]}; CPU seconds: {r["cpu_seconds"]:.3f}')
    print('No physical speed claim. No measured NaYF4 device or fabrication parameters.')
    if not(r['match'] and r['converged']): raise SystemExit(1)

if __name__=='__main__':
    main()
