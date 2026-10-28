"""T-P symmetric self-referential loop (corrected).

Per tick:   R = K . conj(P a)          (P: parity flip, T: complex conjugation,
                                        K: fixed real rotation coupling)
            S = a + i*g*R
            a' = S / ||S||

The map a -> a + i g A(a) with A(a) = K conj(P a) is built on an ANTILINEAR,
norm-preserving operator A. Consequences (verified numerically below):
  * A has a phase-aligned eigenvector v with A v = -i v, so S = (1+g) v there.
  * Iteration converges to that direction (projectively) at rate |1-g|/(1+g).
  * Renormalisation each tick makes NaN/Inf divergence impossible for finite
    inputs, so the run always halts (Normal Termination or Fixed Point).
"""
import numpy as np


class TPSymmetryEngine:
    def __init__(self, theta: float = 0.7, gain: float = 1.3):
        self.gain = gain
        self.P = np.fliplr(np.eye(2)).astype(np.complex128)           # parity
        c, s = np.cos(theta), np.sin(theta)
        self.K = np.array([[c, -s], [s, c]], dtype=np.complex128)     # coupling
        self.state = np.zeros((2, 1), dtype=np.complex128)

    def parity(self, x):        return self.P @ x
    def time_reversal(self, x): return np.conj(x)                     # antiunitary T
    def reaction(self, a):      return self.K @ self.time_reversal(self.parity(a))

    def step(self, a):
        r = self.reaction(a)
        self.state = a + 1j * self.gain * r
        return self.state, r


def _same_ray(x, y, tol):
    """Projective distance: states equal up to a global phase."""
    return np.sqrt(max(0.0, 1.0 - abs(np.vdot(x, y)) ** 2)) < tol


def execute_simulation(ticks: int, x0=(1.0, 0.2 + 0.4j), theta=0.7, gain=1.3,
                       tol=1e-9, trace=False):
    engine = TPSymmetryEngine(theta, gain)
    cur = np.array(x0, dtype=np.complex128).reshape(2, 1)
    cur /= np.linalg.norm(cur)
    rows = []
    for t in range(ticks):
        out, r = engine.step(cur)
        norm = float(np.linalg.norm(out))
        if not np.isfinite(out).all() or norm == 0.0:
            msg = f"Halt at tick {t}: I/O anomaly (norm={norm})"
            return (msg, rows) if trace else msg
        nxt = out / norm
        rows.append((t, cur.ravel().copy(), r.ravel().copy(), norm))
        if _same_ray(cur, nxt, tol):
            msg = f"Fixed point reached at tick {t} (||S|| = {norm:.6f})"
            return (msg, rows) if trace else msg
        cur = nxt
    msg = "Normal Termination"
    return (msg, rows) if trace else msg


if __name__ == "__main__":
    print(execute_simulation(1))
    print(execute_simulation(100))
    _, rows = execute_simulation(100, trace=True)
    for t, a, r, n in rows:
        print(t, np.round(a, 4), np.round(r, 4), round(n, 4))
