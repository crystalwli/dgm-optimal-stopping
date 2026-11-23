"""Shared setup for the jump-intensity control problem (1-D)."""
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded

T, r = 1.0, 0.1
delta = 0.05                       # Gaussian smoothing width of (.)_+
x_lo, x_hi = 0.01, 20.0
z_lo, z_hi = np.log(x_lo), np.log(x_hi)
PHIS = (1.0, 0.1, 0.01, 0.001)

def splus(u, d=delta):             # E[(u + d Z)_+]
    return u * norm.cdf(u / d) + d * norm.pdf(u / d)

def H_np(t, x):                    # smoothed stopping reward H(t,x) = (e^{-rt} - x)_+
    return splus(np.exp(-r * t) - x)

def fd_1d(phi, dz=0.01, nt=4000, store=True):
    """Backward-in-time FD in z = log x for
         h_t - 1/2 h_z + 1/2 h_zz + (1/(2 phi)) ((H-h)^+)^2 = 0,  h(T)=H(T).
       Operator splitting per step: (i) implicit (backward Euler) diffusion,
       (ii) exact pointwise solve of the penalty ODE  d' = -d^2/(2 phi), d = H - h.
       phi = 'hard'  -> projection u = max(u, H)   (hard optimal stopping)
       phi = 'euro'  -> no stopping term            (lambda = 0)
       Boundaries: right end u = H (~0).  Left end: the diffusion is switched off
       (h = a(t) - x is an exact solution there), so the node only feels step (ii).
       Returns x grid, t grid (increasing), U[it, ix]."""
    z = np.arange(z_lo, z_hi + dz / 2, dz); x = np.exp(z); n = len(z)
    dt = T / nt; ts = np.linspace(0, T, nt + 1)
    a = 0.5 / dz**2; b = 0.5 / (2 * dz)            # 1/2 u_zz - 1/2 u_z
    lo = -dt * (a + b); di = 1 + 2 * dt * a; up = -dt * (a - b)
    ab = np.zeros((3, n)); ab[0, 1:] = up; ab[1, :] = di; ab[2, :-1] = lo
    ab[1, 0] = 1; ab[0, 1] = 0; ab[1, -1] = 1; ab[2, -2] = 0
    u = H_np(T, x); U = np.empty((nt + 1, n)) if store else None
    if store: U[nt] = u
    for k in range(nt - 1, -1, -1):
        Hk = H_np(ts[k], x)
        rhs = u.copy(); rhs[-1] = Hk[-1]
        u = solve_banded((1, 1), ab, rhs)
        if phi == 'hard':
            u = np.maximum(u, Hk)
        elif phi != 'euro':
            d = np.maximum(Hk - u, 0.0)
            u = u + d - d / (1 + d * dt / (2 * phi))
        if store: U[k] = u
    return x, ts, (U if store else u)
