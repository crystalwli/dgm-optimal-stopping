"""Finite-difference solution for each phi, plus the European (no stopping) and hard-stopping references."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
res = {}
for p in ('euro',) + PHIS + ('hard',):
    x, ts, U = fd_1d(p); res[p] = U
np.savez_compressed("fd1d.npz", x=x, ts=ts, **{str(k): v for k, v in res.items()})
at = lambda U, it=0, xv=1.0: np.interp(xv, x, U[it])
print("h(0,1):  European g = %.4f | hard stopping h_stop = %.4f" % (at(res['euro']), at(res['hard'])))
for p in PHIS: print("  phi = %-6g h_phi(0,1) = %.4f   max lambda* = %.1f" % (p, at(res[p]), (np.maximum(H_np(ts[:, None], x[None]) - res[p], 0) / p).max()))
# checks
chain = [res['euro']] + [res[p] for p in PHIS] + [res['hard']]
print("ordering g <= h_1 <= h_0.1 <= h_0.01 <= h_0.001 <= h_stop holds everywhere:", all((b - a).min() > -1e-6 for a, b in zip(chain, chain[1:])))
for p in PHIS: print("  phi = %-6g max(h_stop - h_phi) = %.4f   sqrt(2 phi r) = %.4f" % (p, (res['hard'] - res[p]).max(), np.sqrt(2 * p * r)))
# hard boundary x*(0)
Hg = H_np(ts[:, None], x[None]); stop = (res['hard'] - Hg < 1e-6) & (Hg > 1e-3)
xb = np.array([x[s].max() if s.any() else np.nan for s in stop]); print("hard boundary x*(0) = %.3f" % xb[0])
# convergence in grid for the stiffest case
for p in (0.001, 'hard'):
    x2, _, u2 = fd_1d(p, dz=0.005, nt=16000, store=False); print("  refinement check phi=%s: h(0,1) = %.4f (fine grid)" % (p, np.interp(1.0, x2, u2)))
# figure
xs = np.linspace(0.02, 2.5, 500); fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
cols = plt.cm.viridis(np.linspace(0, 0.85, 4))
for c, tv in enumerate((0.0, 0.5, 0.9)):
    it = int(round(tv / T * (len(ts) - 1)))
    ax[c].plot(xs, np.interp(xs, x, res['hard'][it]), "k-", lw=2.5, label="hard stopping $h_{stop}$")
    for p, col in zip(PHIS, cols): ax[c].plot(xs, np.interp(xs, x, res[p][it]), color=col, label=rf"$h_\varphi$, $\varphi$={p:g}")
    ax[c].plot(xs, np.interp(xs, x, res['euro'][it]), "r--", label=r"no stopping $g$ ($\lambda=0$)")
    ax[c].plot(xs, H_np(tv, xs), ":", color="0.4", label="reward $H$")
    ax[c].set_title(f"FD value functions, t = {tv}"); ax[c].set_xlabel("x"); ax[c].set_ylabel("value"); ax[c].grid(True); ax[c].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig1_fd_values.png", dpi=130)
