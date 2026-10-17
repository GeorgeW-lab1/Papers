"""
sic_wafer_phonon_logic_fast.py

Coherent Anharmonic Phonon-to-Phonon Mode-Resolved SHA-256 Simulator
with Multi-Type Wafer Defect Spatial Mapping and Image Generation.

Dependency: numpy, matplotlib
Run: python sic_wafer_phonon_logic_fast.py
"""

import hashlib
import platform
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

MASK32 = 0xFFFFFFFF

K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
]

H0 = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
]


def sha256_reference(message):
    return hashlib.sha256(message).hexdigest()


class SiCWaferDefectMapper:
    """
    Generates a spatial 2D coordinate map of a 4-inch (100mm) SiC wafer 
    populated with multiple defect types:
    - Type 1: Threading Screw Dislocations (TSD) [Point scatter]
    - Type 2: Basal Plane Stacking Faults (BPD) [Slip bands]
    - Type 3: Laser-Patterned Phononic Lattice Cavities [Grid arrays]
    """
    def __init__(self, wafer_diameter_mm=100.0, seed=42):
        self.radius = wafer_diameter_mm / 2.0
        self.rng = np.random.default_rng(seed)

    def generate_defects(self, n_tsd=120, n_bpd=45, n_cavities=200):
        defects = {}
        
        # 1. Threading Screw Dislocations (TSDs) - Uniform random distribution across wafer disk
        r_tsd = self.radius * np.sqrt(self.rng.random(n_tsd))
        theta_tsd = 2 * np.pi * self.rng.random(n_tsd)
        defects['TSD'] = np.column_stack((r_tsd * np.cos(theta_tsd), r_tsd * np.sin(theta_tsd)))

        # 2. Basal Plane Dislocations (BPDs) - Concentrated along crystallographic slip lines
        slip_angle = np.pi / 4.0
        d_bpd = self.rng.uniform(-self.radius * 0.8, self.radius * 0.8, n_bpd)
        x_bpd = d_bpd * np.cos(slip_angle) + self.rng.normal(0, 2, n_bpd)
        y_bpd = d_bpd * np.sin(slip_angle) + self.rng.normal(0, 8, n_bpd)
        # Clip to wafer boundary
        mask = (x_bpd**2 + y_bpd**2) <= self.radius**2
        defects['BPD'] = np.column_stack((x_bpd[mask], y_bpd[mask]))

        # 3. Laser-Patterned Phononic Cavities - Periodic central computational array
        grid_pts = np.linspace(-self.radius * 0.5, self.radius * 0.5, int(np.sqrt(n_cavities)))
        gx, gy = np.meshgrid(grid_pts, grid_pts)
        cav_coords = np.column_stack((gx.ravel(), gy.ravel()))
        cav_mask = (cav_coords[:, 0]**2 + cav_coords[:, 1]**2) <= (self.radius * 0.6)**2
        defects['Phononic_Cavity'] = cav_coords[cav_mask]

        return defects

    def render_wafer_map(self, filename="sic_wafer_defect_map.png"):
        defects = self.generate_defects()
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111)
        
        # Draw SiC Wafer boundary (4-inch disk with primary flat at bottom)
        wafer_circle = plt.Circle((0, 0), self.radius, color='#2c3e50', fill=False, linewidth=3, label='SiC Wafer Edge (100mm)')
        ax.add_patch(wafer_circle)
        
        # Plot defects with distinct markers & colors
        ax.scatter(defects['TSD'][:, 0], defects['TSD'][:, 1], c='#e74c3c', marker='x', s=35, label='Thread Screw Dislocation (TSD)')
        ax.scatter(defects['BPD'][:, 0], defects['BPD'][:, 1], c='#f39c12', marker='s', s=25, label='Basal Plane Fault (BPD)')
        ax.scatter(defects['Phononic_Cavity'][:, 0], defects['Phononic_Cavity'][:, 1], c='#3498db', marker='o', s=15, label='Phononic Lattice Cavity')

        ax.set_xlim(-self.radius * 1.15, self.radius * 1.15)
        ax.set_ylim(-self.radius * 1.15, self.radius * 1.15)
        ax.set_aspect('equal')
        ax.set_title("4-inch 4H-SiC Wafer Phonon Logic & Defect Topography", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Wafer X-Coordinate (mm)")
        ax.set_ylabel("Wafer Y-Coordinate (mm)")
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Wafer defect topography map successfully saved to: {filename}")


class PhononPhononScatteringFabric:
    """
    Coherent Phonon-to-Phonon Anharmonic Scattering Fabric.
    """

    def __init__(
        self,
        n_lanes,
        seed=7,
        anharmonic_coupling=1.0,
        damping_gamma=0.0,
        defect_scattering=0.01,
        thermal_noise_sigma=0.0,
    ):
        if n_lanes < 1:
            raise ValueError("n_lanes must be positive")
        
        self.n = int(n_lanes)
        self.rng = np.random.default_rng(seed)
        self.g_abc = float(anharmonic_coupling)
        self.gamma = float(damping_gamma)
        self.defect_scattering = float(defect_scattering)
        self.noise_sigma = float(thermal_noise_sigma)
        
        self.gate_count = 0
        self.scattering_events = 0
        self.stage_name = "initialization"

    def stage(self, name):
        self.stage_name = str(name)

    def _vec(self, x):
        if np.isscalar(x):
            return np.full(self.n, float(x), dtype=np.float64)
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (self.n,):
            raise ValueError(f"Expected vector shape {(self.n,)}, got {arr.shape}")
        return arr

    def phonon_nand(self, a, b):
        Qa = self._vec(a)
        Qb = self._vec(b)
        
        phi_a = np.zeros_like(Qa)
        phi_b = np.zeros_like(Qb)
        
        interaction = self.g_abc * Qa * Qb * np.cos(phi_a - phi_b)
        p_out = 1.0 - (interaction / (1.0 + self.gamma))
        
        if self.defect_scattering > 0:
            defect_hit = self.rng.random(self.n) < self.defect_scattering
            if np.any(defect_hit):
                self.scattering_events += int(defect_hit.sum())
                p_out[defect_hit] = 1.0 - p_out[defect_hit]
            
        if self.noise_sigma > 0:
            p_out += self.rng.normal(0.0, self.noise_sigma, self.n)
            
        self.gate_count += 1
        return np.clip(p_out, 0.0, 1.0)

    def not_(self, a):
        return self.phonon_nand(a, a)

    def and_(self, a, b):
        return self.not_(self.phonon_nand(a, b))

    def or_(self, a, b):
        return self.phonon_nand(self.not_(a), self.not_(b))

    def xor(self, a, b):
        t = self.phonon_nand(a, b)
        return self.phonon_nand(self.phonon_nand(a, t), self.phonon_nand(b, t))

    def xorw(self, x, y):
        return [self.xor(a, b) for a, b in zip(x, y)]

    def xor3w(self, x, y, z):
        return self.xorw(self.xorw(x, y), z)

    def add(self, x, y):
        carry = np.zeros(self.n, dtype=np.float64)
        out = []
        for a, b in zip(x, y):
            t = self.xor(a, b)
            out.append(self.xor(t, carry))
            carry = self.or_(
                self.and_(a, b),
                self.and_(t, carry),
            )
        return out

    def const(self, value):
        return [
            np.full(self.n, float((value >> k) & 1))
            for k in range(32)
        ]


def rotr_word(x, n):
    return [x[(k + n) % 32] for k in range(32)]


def shr_word(x, n):
    zero = np.zeros_like(x[0])
    return [x[k + n] if k + n < 32 else zero for k in range(32)]


def sha256_compress(fabric, block_words):
    W = list(block_words)

    for t in range(16, 64):
        fabric.stage("message schedule")
        x = W[t - 15]
        s0 = fabric.xor3w(rotr_word(x, 7), rotr_word(x, 18), shr_word(x, 3))
        x = W[t - 2]
        s1 = fabric.xor3w(rotr_word(x, 17), rotr_word(x, 19), shr_word(x, 10))
        W.append(fabric.add(fabric.add(fabric.add(W[t - 16], s0), W[t - 7]), s1))

    a, b, c, d, e, ff, g, h = [fabric.const(v) for v in H0]

    for t in range(64):
        fabric.stage(f"round {t + 1}")
        S1 = fabric.xor3w(rotr_word(e, 6), rotr_word(e, 11), rotr_word(e, 25))
        S0 = fabric.xor3w(rotr_word(a, 2), rotr_word(a, 13), rotr_word(a, 22))

        ch = [
            fabric.or_(
                fabric.and_(e[k], ff[k]),
                fabric.and_(fabric.not_(e[k]), g[k]),
            )
            for k in range(32)
        ]

        maj = [
            fabric.or_(
                fabric.and_(a[k], b[k]),
                fabric.and_(c[k], fabric.or_(a[k], b[k])),
            )
            for k in range(32)
        ]

        T1 = fabric.add(fabric.add(fabric.add(fabric.add(h, S1), ch), fabric.const(K[t])), W[t])
        T2 = fabric.add(S0, maj)

        h, g, ff, e = g, ff, e, fabric.add(d, T1)
        d, c, b, a = c, b, a, fabric.add(T1, T2)

    fabric.stage("final addition")
    return [
        fabric.add(fabric.const(H0[i]), word)
        for i, word in enumerate([a, b, c, d, e, ff, g, h])
    ]


def pack_messages(messages):
    blocks = []
    for message in messages:
        data = message if isinstance(message, bytes) else str(message).encode()
        if len(data) > 55:
            raise ValueError("This implementation accepts messages up to 55 bytes.")
        padded = (
            data
            + b"\x80"
            + b"\x00" * (55 - len(data))
            + (8 * len(data)).to_bytes(8, "big")
        )
        blocks.append(padded)

    words = []
    for t in range(16):
        values = [int.from_bytes(block[4 * t:4 * t + 4], "big") for block in blocks]
        words.append([
            np.array([(value >> bit) & 1 for value in values], dtype=np.float64)
            for bit in range(32)
        ])
    return words


def unpack_digest(words8, n_lanes):
    results = []
    for lane in range(n_lanes):
        parts = []
        for word in words8:
            value = 0
            for bit in range(32):
                logical_bit = int(word[bit][lane] >= 0.5)
                value |= logical_bit << bit
            parts.append(f"{value:08x}")
        results.append("".join(parts))
    return results


def run_benchmark():
    # 1. Generate and save the multi-type wafer defect map image
    mapper = SiCWaferDefectMapper(wafer_diameter_mm=100.0, seed=123)
    mapper.render_wafer_map("sic_wafer_defect_map.png")

    messages = [
        "SiC phonon scattering 000",
        "SiC phonon scattering 001",
        "anharmonic lattice SHA256",
        "reference phonon macro",
    ]

    references = [sha256_reference(m.encode()) for m in messages]

    fabric = PhononPhononScatteringFabric(
        n_lanes=len(messages),
        seed=123,
        anharmonic_coupling=1.0,
        damping_gamma=0.0,
        defect_scattering=0.0,
        thermal_noise_sigma=0.0,
    )

    print("=" * 72)
    print("COHERENT ANHARMONIC PHONON-TO-PHONON SHA-256 SIMULATION")
    print("=" * 72)
    print(f"Python version:           {platform.python_version()}")
    print(f"Platform:                 {platform.platform()}")
    print(f"NumPy version:            {np.__version__}")
    print(f"Message lanes:            {len(messages)}")
    print(f"Anharmonic coupling (g):  {fabric.g_abc}")
    print(f"Acoustic damping (gamma): {fabric.gamma}")
    print(f"Defect scattering rate:   {fabric.defect_scattering}")
    print("-" * 72)

    tracemalloc.start()
    start = time.perf_counter()

    block_words = pack_messages(messages)
    digest_words = sha256_compress(fabric, block_words)
    actual = unpack_digest(digest_words, len(messages))

    elapsed = time.perf_counter() - start
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    passed = sum(r == ref for r, ref in zip(actual, references))
    failed = len(messages) - passed

    print(f"Elapsed time:             {elapsed:.6f} seconds")
    print(f"Effective NAND calls:     {fabric.gate_count:,}")
    print(f"Current traced memory:    {current_memory / 2**20:.3f} MiB")
    print(f"Peak traced memory:       {peak_memory / 2**20:.3f} MiB")
    print("-" * 72)

    for index, (message, result, reference) in enumerate(zip(messages, actual, references)):
        status = "PASS" if result == reference else "FAIL"
        print(f"[{status}] Lane {index}: {message}")
        if result != reference:
            print(f"  Simulator: {result}")
            print(f"  hashlib:   {reference}")

    accuracy = 100.0 * passed / len(messages)
    print("-" * 72)
    print(f"Correct digests:          {passed}/{len(messages)}")
    print(f"Digest accuracy:          {accuracy:.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
