#!/usr/bin/env python3
"""SHA-256 PoW simulation using the exact thermal SiC NAND equations from the supplied wafer model.

The SHA netlist is NAND-only. Each NAND returns a physical output-node temperature
from the supplied heat-balance equation, using Callaway-model relative SiC conductance.
This is a numerical model, not a fabricated or experimentally validated device.
"""
import argparse, json, math, random, time
from pathlib import Path

# Constants and parameters copied from the supplied SiC thermal-NAND model.
kB, hbar = 1.380649e-23, 1.054572e-34
V, THETA, L_BND = 9000.0, 1200.0, 3.5e-4
A_PT, B_UM, C_BPD, C_TSD = 1e-44, 4e-18, 5e-11, 1e-10
TC, TH = 300.0, 360.0
T_TR, W_SW = 330.0, 3.0
KAPPA_PRISTINE = 370.0
A_CH, L_CH = (50e-9)**2, 1e-6
G_SEG0 = KAPPA_PRISTINE * A_CH / L_CH
G_ON, G_OFF = 50*G_SEG0, 1e-3*G_SEG0
LN_RATIO = math.log(G_ON/G_OFF)
LAMBDA = 0.21
C_NODE = 2.2e6 * (100e-9)**3
SIGMA_T = math.sqrt(kB * 330.0**2 / C_NODE)
ELL_HALO, ELL_CORE = 3e-7, 1e-8
TGRID = [290.0 + 5.0*i for i in range(17)]
MASK32 = 0xffffffff

# Standard SHA-256 constants and initial state.
K = (0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,
0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,
0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2)
H0 = (0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19)

# Callaway conductivity, using the same spectral integral and calibration as source.
def _kappa(Tv, extra, n_bpd=0.0, n_tsd=0.0):
    n = 400
    xmax = THETA / Tv
    dx = (xmax - 1e-3)/(n-1)
    # Trapezoidal integration, with the same scattering terms as the source model.
    vals = []
    wref = kB * 300.0 / hbar
    for i in range(n):
        x = 1e-3 + i*dx
        w = x*kB*Tv/hbar
        rate = (V/L_BND + A_PT*w**4 + B_UM*w**2*Tv*math.exp(-THETA/(3*Tv))
                + C_BPD*n_bpd*w + C_TSD*n_tsd*w**3/(kB*300.0/hbar)**2 + extra)
        ex = math.exp(x)
        vals.append(x**4*ex/(ex-1)**2/rate)
    integ = dx*(0.5*vals[0] + sum(vals[1:-1]) + 0.5*vals[-1])
    pref0 = kB/(2*math.pi**2*V) * (kB*Tv/hbar)**3
    return integ*pref0

# Source calibration: 500-point spectrum at 300 K; gate conductivities use 400 points.
def _source_calibration():
    n=500; Tv=300.0; xmax=THETA/Tv; dx=(xmax-1e-3)/(n-1); vals=[]
    for i in range(n):
        x=1e-3+i*dx; w=x*kB*Tv/hbar
        rate=V/L_BND+A_PT*w**4+B_UM*w**2*Tv*math.exp(-THETA/(3*Tv))
        ex=math.exp(x); vals.append(x**4*ex/(ex-1)**2/rate)
    integ=dx*(0.5*vals[0]+sum(vals[1:-1])+0.5*vals[-1])
    pref=kB/(2*math.pi**2*V)*(kB*Tv/hbar)**3
    return KAPPA_PRISTINE/(pref*integ)
_CAL = _source_calibration()

def kappa_at(Tv, state, n_bpd=0.0, n_tsd=0.0):
    extra = 0.0 if state == 'bulk' else V/(ELL_HALO if state == 'halo' else ELL_CORE)
    return _CAL * _kappa(Tv, extra, n_bpd, n_tsd)

# Relative conductance tables RT[state][temperature-grid index], as in the source.
RT = {state: [kappa_at(t, state)/KAPPA_PRISTINE for t in TGRID] for state in ('bulk','halo','core')}

def rel_kappa(Tv, state):
    if Tv <= TGRID[0]: return RT[state][0]
    if Tv >= TGRID[-1]: return RT[state][-1]
    q = (Tv-TGRID[0])/5.0
    i = int(q); w = q-i
    a,b = RT[state][i],RT[state][i+1]
    return a+(b-a)*w

def switch_G(Tin):
    # Numerically stable logistic, identical functional form to the source.
    z = (Tin-T_TR)/W_SW
    s = 1/(1+math.exp(-max(-700.0,min(700.0,z))))
    return G_OFF*math.exp(s*LN_RATIO)

def nand_temperature(Ta, Tb, state_a='bulk', state_b='bulk', lam=LAMBDA):
    """Exact source thermal NAND: two switched SiC segments + load, two kappa iterations."""
    swA, swB = switch_G(Ta), switch_G(Tb)
    Tseg = TC+6.0
    for _ in range(2):
        gA = 1.0/(1.0/swA + 1.0/(G_SEG0*rel_kappa(Tseg,state_a)))
        gB = 1.0/(1.0/swB + 1.0/(G_SEG0*rel_kappa(Tseg,state_b)))
        Gch = 1.0/(1.0/gA + 1.0/gB)
        To = (lam*G_SEG0*TH + Gch*TC)/(lam*G_SEG0+Gch)
        Tseg = TC+0.5*(To-TC)
    return To

class ThermalNANDFabric:
    """NAND-only gate library; each gate output remains an analog temperature."""
    def __init__(self, seed=7, noise=True, p_halo=0.0, p_core=0.0):
        if p_halo < 0 or p_core < 0 or p_halo+p_core > 1: raise ValueError('p_halo + p_core must be <= 1')
        self.rng=random.Random(seed); self.noise=noise; self.p_halo=p_halo; self.p_core=p_core
        self.gates=0; self.counts={'bulk':0,'halo':0,'core':0}; self.errors=0
        self.LO,self.HI=TC,TH
    def _state(self):
        r=self.rng.random()
        if r < self.p_core: return 'core'
        if r < self.p_core+self.p_halo: return 'halo'
        return 'bulk'
    def nand(self,a,b):
        sa,sb=self._state(),self._state()
        out=nand_temperature(a,b,sa,sb)
        if self.noise: out += SIGMA_T*self.rng.gauss(0.0,1.0)
        self.gates+=1; self.counts[sa]+=1; self.counts[sb]+=1
        return out
    def not_(self,a): return self.nand(a,a)
    def and_(self,a,b): return self.not_(self.nand(a,b))
    def or_(self,a,b): return self.nand(self.not_(a),self.not_(b))
    def xor(self,a,b):
        t=self.nand(a,b)
        return self.nand(self.nand(a,t),self.nand(b,t))
    def const(self,v): return [self.HI if (v>>k)&1 else self.LO for k in range(32)]
    def add(self,x,y):
        carry=self.LO; out=[]
        for a,b in zip(x,y):
            t=self.xor(a,b); out.append(self.xor(t,carry))
            carry=self.or_(self.and_(a,b),self.and_(t,carry))
        return out

def rotr(x,n): return [x[(k+n)%32] for k in range(32)]
def shr(x,n,lo): return [x[k+n] if k+n<32 else lo for k in range(32)]
def xor3w(f,x,y,z): return [f.xor(f.xor(a,b),c) for a,b,c in zip(x,y,z)]
def sha256_block(f, Wblk):
    W=list(Wblk)
    for t in range(16,64):
        s0=xor3w(f,rotr(W[t-15],7),rotr(W[t-15],18),shr(W[t-15],3,f.LO))
        s1=xor3w(f,rotr(W[t-2],17),rotr(W[t-2],19),shr(W[t-2],10,f.LO))
        W.append(f.add(f.add(f.add(W[t-16],s0),W[t-7]),s1))
    a,b,c,d,e,ff,g,h=[f.const(v) for v in H0]
    for t in range(64):
        S1=xor3w(f,rotr(e,6),rotr(e,11),rotr(e,25))
        S0=xor3w(f,rotr(a,2),rotr(a,13),rotr(a,22))
        ch=[f.or_(f.and_(e[k],ff[k]),f.and_(f.not_(e[k]),g[k])) for k in range(32)]
        mj=[f.or_(f.and_(a[k],b[k]),f.and_(c[k],f.or_(a[k],b[k]))) for k in range(32)]
        T1=f.add(f.add(f.add(f.add(h,S1),ch),f.const(K[t])),W[t])
        T2=f.add(S0,mj)
        h,g,ff,e=g,ff,e,f.add(d,T1)
        d,c,b,a=c,b,a,f.add(T1,T2)
    return [f.add(f.const(H0[i]),v) for i,v in enumerate([a,b,c,d,e,ff,g,h])]

def sha256_thermal(f,message):
    bitlen=len(message)*8
    data=message+b'\x80'
    data+=b'\x00'*((56-len(data)%64)%64)+bitlen.to_bytes(8,'big')
    state=[f.const(v) for v in H0]
    for off in range(0,len(data),64):
        block=data[off:off+64]
        words=[]
        for i in range(0,64,4):
            v=int.from_bytes(block[i:i+4],'big'); words.append(f.const(v))
        # f.const returns bits LSB-first; input bytes are interpreted big-endian as words.
        state=sha256_block_state(f, words, state)
    digest=[]
    for w in state:
        # Decode each bit at the output threshold; index 0 is LSB.
        value=sum((1<<i) for i,t in enumerate(w) if t>T_TR)
        digest.extend(value.to_bytes(4,'big'))
    return bytes(digest)

def sha256_block_state(f,Wblk,state):
    W=list(Wblk)
    for t in range(16,64):
        s0=xor3w(f,rotr(W[t-15],7),rotr(W[t-15],18),shr(W[t-15],3,f.LO))
        s1=xor3w(f,rotr(W[t-2],17),rotr(W[t-2],19),shr(W[t-2],10,f.LO))
        W.append(f.add(f.add(f.add(W[t-16],s0),W[t-7]),s1))
    a,b,c,d,e,ff,g,h=state
    for t in range(64):
        S1=xor3w(f,rotr(e,6),rotr(e,11),rotr(e,25))
        S0=xor3w(f,rotr(a,2),rotr(a,13),rotr(a,22))
        ch=[f.or_(f.and_(e[k],ff[k]),f.and_(f.not_(e[k]),g[k])) for k in range(32)]
        mj=[f.or_(f.and_(a[k],b[k]),f.and_(c[k],f.or_(a[k],b[k]))) for k in range(32)]
        T1=f.add(f.add(f.add(f.add(h,S1),ch),f.const(K[t])),W[t])
        T2=f.add(S0,mj)
        h,g,ff,e=g,ff,e,f.add(d,T1)
        d,c,b,a=c,b,a,f.add(T1,T2)
    return [f.add(state[i],v) for i,v in enumerate([a,b,c,d,e,ff,g,h])]

def mine(args):
    global RT
    RT = {state: [kappa_at(t, state, args.bpd_density_m2, args.tsd_density_m2)/KAPPA_PRISTINE for t in TGRID] for state in ('bulk','halo','core')}
    payload=args.payload.encode('utf-8'); target='0'*args.difficulty
    start=time.perf_counter(); attempts=0; total_gates=0; total_segments={'bulk':0,'halo':0,'core':0}
    for nonce in range(args.max_nonce+1):
        f=ThermalNANDFabric(args.seed+nonce,not args.no_noise,args.p_halo,args.p_core)
        digest=sha256_thermal(f,payload+str(nonce).encode())
        attempts+=1; total_gates += f.gates
        for st in total_segments: total_segments[st] += f.counts[st]
        # Only accept if output bits from the thermal NAND network meet target.
        if digest.hex().startswith(target):
            return {'payload':args.payload,'nonce':nonce,'digest':digest.hex(),'difficulty_hex_zeros':args.difficulty,
                    'attempts':attempts,'elapsed_seconds':time.perf_counter()-start,
                    'physical_nand_gate_evaluations_final_hash':f.gates,'physical_nand_gate_evaluations_all_attempts':total_gates,
                    'segment_state_counts_final_hash':f.counts,'segment_state_counts_all_attempts':total_segments,
                    'noise_enabled':not args.no_noise,'p_halo_per_segment':args.p_halo,'p_core_per_segment':args.p_core,'bpd_density_m-2':args.bpd_density_m2,'tsd_density_m-2':args.tsd_density_m2}
    raise RuntimeError('No solution found in requested nonce range. Increase --max-nonce or lower --difficulty.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload',default='SiC phononic SHA-256 PoW')
    p.add_argument('--difficulty',type=int,default=1,help='leading hexadecimal zeroes; gate-level thermal model is slow')
    p.add_argument('--max-nonce',type=int,default=0,help='default 0 means test one nonce; increase to mine')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--no-noise',action='store_true',help='disable node thermal noise (noise is enabled by default)')
    p.add_argument('--p-halo',type=float,default=0.0,help='illustrative probability each segment is a halo segment')
    p.add_argument('--p-core',type=float,default=0.0,help='illustrative probability each segment is a cracked-core segment')
    p.add_argument('--bpd-density-m2',type=float,default=0.0,help='BPD density in m^-2 for Callaway scattering')
    p.add_argument('--tsd-density-m2',type=float,default=0.0,help='TSD density in m^-2 for Callaway scattering')
    p.add_argument('--output',default='sic_exact_phonon_nand_sha256_results.json')
    a=p.parse_args()
    if not 1<=a.difficulty<=4: p.error('--difficulty must be 1..4')
    try: result=mine(a)
    except RuntimeError as e: p.error(str(e))
    report={'model':'source-matched SiC thermal NAND network + NAND-only SHA-256 netlist','result':result,
      'physical_parameters':{'T_logic0_K':TC,'T_logic1_K':TH,'switch_threshold_K':T_TR,'switch_width_K':W_SW,
        'channel_area_m2':A_CH,'channel_length_m':L_CH,'pristine_kappa_W_mK':KAPPA_PRISTINE,
        'load_ratio_lambda':LAMBDA,'node_heat_capacity_J_K':C_NODE,'thermal_noise_sigma_K':SIGMA_T,
        'callaway_relative_conductance_states':RT,'bpd_density_m-2':a.bpd_density_m2,'tsd_density_m-2':a.tsd_density_m2},
      'caveats':['This is a simulation of the supplied thermal-NAND equations, not a physical SiC device.',
        'The wafer defect-map probabilities are not reconstructed here; halo/core probabilities and BPD/TSD densities are explicit user inputs and default to zero.',
        'The model uses two fixed-point iterations for each gate as in the supplied source.',
        'The returned digest is decoded from thermal gate output temperatures at the source threshold; noise or defects can make it differ from standard SHA-256.']}
    Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('=== Exact-equation SiC thermal NAND SHA-256 simulation ===')
    print('No hashlib used to compute the digest; every logic primitive is built from thermal NANDs.')
    print(json.dumps(result,indent=2)); print('Saved:',a.output)
if __name__=='__main__': main()
