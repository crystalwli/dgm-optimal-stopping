"""Heat map of the optimal intensity lambda* and Monte Carlo check of the FD values (run step1_fd.py first)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
D = np.load("fd1d.npz"); x, ts = D["x"], D["ts"]; z = np.log(x); dz = z[1] - z[0]; nt = len(ts) - 1; dt = T / nt
Hg = H_np(ts[:, None], x[None])
stop = (D["hard"] - Hg < 1e-6) & (Hg > 1e-3); xb = np.array([x[s].max() if s.any() else np.nan for s in stop])
np.save("xb_hard.npy", xb)
def simulate(lam, phi, n=40000, seed=0):
    """Paths of Z = log X on the FD time grid; stop at the first jump of the counting process with intensity lam(t, X_t)."""
    rng = np.random.default_rng(seed); Z = np.zeros(n); alive = np.ones(n, bool)
    tau = np.full(n, T); cost = np.zeros(n); pay = np.zeros(n); xs_ = np.full(n, np.nan); gap = np.full(n, np.nan)
    for k in range(nt):
        i = np.flatnonzero(alive); w = np.clip((Z[i] - z[0]) / dz, 0, len(z) - 1.001); j = w.astype(int); w -= j
        l = (1 - w) * lam[k, j] + w * lam[k, j + 1]
        jump = rng.random(len(i)) < 1 - np.exp(-l * dt)
        s = i[jump]; X = np.exp(Z[s]); tau[s] = ts[k]; xs_[s] = X; pay[s] = H_np(ts[k], X); gap[s] = l[jump] * phi; alive[s] = False
        c = i[~jump]; cost[c] += 0.5 * phi * l[~jump] ** 2 * dt
        Z[c] += -0.5 * dt + np.sqrt(dt) * rng.standard_normal(len(c))
    pay[alive] = H_np(T, np.exp(Z[alive])); J = pay - cost
    return tau, xs_, ~alive, J.mean(), J.std() / np.sqrt(n), np.nanmin(gap)
fig, ax = plt.subplots(1, 4, figsize=(22, 5)); it = slice(0, nt + 1, 8); ix = x <= 1.25
print("phi     h_FD(0,1)  MC value (s.e.)    stopped before T   mean tau | mean X_tau (stopped)   min (H-h) at stops")
for a, p in zip(ax, PHIS):
    h = D[str(p)]; lam = np.maximum(Hg - h, 0) / p
    tau, xs_, st, m, se, gmin = simulate(lam, p)
    print("%-7g %.4f     %.4f (%.4f)    %5.1f%%             %.3f    | %.3f                 %.1e" % (p, np.interp(1.0, x, h[0]), m, se, 100 * st.mean(), tau[st].mean(), np.nanmean(xs_), gmin))
    pm = a.pcolormesh(ts[it], x[ix], lam[it][:, ix].T, cmap="magma", shading="auto"); plt.colorbar(pm, ax=a, label=r"$\lambda^*$")
    k = np.flatnonzero(st)[:1500]; a.scatter(tau[k], xs_[k], s=5, c="cyan", alpha=0.6, lw=0, label="simulated stopping points")
    a.plot(ts, xb, "w--", lw=2, label=r"hard stopping boundary $x^*(t)$")
    a.set_ylim(0, 1.25); a.set_xlabel("t"); a.set_ylabel("x"); a.set_title(rf"$\lambda^*(t,x)=(H-h)^+/\varphi$,  $\varphi$ = {p:g}"); a.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("fig2_lambda_heatmap.png", dpi=120)
