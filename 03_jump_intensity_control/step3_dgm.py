"""1-D DGM for  h_t - 1/2 h_z + 1/2 h_zz + (1/(2 phi)) ((H-h)^+)^2 = 0  with continuation in phi.
usage: python3 step3_dgm.py N_ITER   (iterations per phi; 8000 takes about 7 minutes on CPU; run step1_fd.py first)"""
import sys, time, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 500
torch.manual_seed(0); torch.set_default_dtype(torch.float32)
Nd = torch.distributions.Normal(0.0, 1.0)
def splus_t(u, d=delta): return u * Nd.cdf(u / d) + d * torch.exp(Nd.log_prob(u / d))
H_t = lambda t, z: splus_t(torch.exp(-r * t) - torch.exp(z))

class Net(nn.Module):            # h = H(T, z) + (T - t) N_theta(t, z): terminal condition holds exactly
    def __init__(s, hidden=64):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(s, t, z):
        return H_t(torch.full_like(t, T), z) + (T - t) * s.net(torch.cat([t, (z - z_lo) / (z_hi - z_lo) * 2 - 1], 1))

def left_bc(phi, n=4000):        # a(t) with a' + ((e^{-rt}-a)^+)^2/(2 phi) = 0, a(T)=e^{-rT};  h(t,x) = a(t) - x for small x
    ts = np.linspace(0, T, n + 1); a = np.empty(n + 1); a[n] = np.exp(-r * T); dt = T / n
    for k in range(n - 1, -1, -1):
        d = max(np.exp(-r * ts[k]) - a[k + 1], 0.0); a[k] = np.exp(-r * ts[k]) - d / (1 + d * dt / (2 * phi))
    return ts, a - x_lo

def residual(m, phi, t, z):
    t.requires_grad_(True); z.requires_grad_(True); h = m(t, z); o = torch.ones_like(h)
    ht = torch.autograd.grad(h, t, o, create_graph=True)[0]; hz = torch.autograd.grad(h, z, o, create_graph=True)[0]
    hzz = torch.autograd.grad(hz, z, o, create_graph=True)[0]
    return ht - 0.5 * hz + 0.5 * hzz + torch.relu(H_t(t, z) - h) ** 2 / (2 * phi)

def train(m, phi, n_iter, batch=2048, nb=256, w_bc=10.0, lr=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr); sch = torch.optim.lr_scheduler.StepLR(opt, max(1, n_iter // 4), 0.4)
    tb_np, ab_np = left_bc(phi); hist = []
    for it in range(n_iter):
        t = T * torch.rand(batch, 1); z = z_lo + (z_hi - z_lo) * torch.rand(batch, 1)
        loss_pde = (residual(m, phi, t, z) ** 2).mean()
        tb = T * torch.rand(nb, 1); tgt = torch.tensor(np.interp(tb.numpy().ravel(), tb_np, ab_np), dtype=torch.float32).reshape(-1, 1)
        loss_bc = ((m(tb, torch.full_like(tb, z_lo)) - tgt) ** 2).mean() + (m(tb, torch.full_like(tb, z_hi)) ** 2).mean()
        loss = loss_pde + w_bc * loss_bc
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); hist.append((loss_pde.item(), loss_bc.item()))
    return np.array(hist)

def ev(m, t, x):
    with torch.no_grad():
        x = np.asarray(x, float); t = np.broadcast_to(np.asarray(t, float), x.shape)
        return m(torch.tensor(t.reshape(-1, 1), dtype=torch.float32), torch.tensor(np.log(x).reshape(-1, 1), dtype=torch.float32)).numpy().reshape(x.shape)

if __name__ == "__main__":
    D = np.load("fd1d.npz"); xg, ts = D["x"], D["ts"]; xs = np.linspace(0.05, 2.5, 400); tshow = (0.0, 0.5, 0.9)
    m = Net(); hists = {}; t0 = time.time()
    fig, ax = plt.subplots(4, 3, figsize=(16, 15))
    print("phi      PDE loss  BC loss | h(0,1): DGM    FD   | max|DGM-FD| on x in [0.05,2.5] at t=0, 0.5, 0.9")
    for i, p in enumerate(PHIS):                     # continuation: the net for phi_k starts from the trained net for phi_{k-1}
        hists[p] = train(m, p, n_iter); torch.save(m.state_dict(), f"dgm_phi{p:g}.pt")
        errs = []
        for c, tv in enumerate(tshow):
            fd = np.interp(xs, xg, D[str(p)][int(round(tv * (len(ts) - 1)))]); nn_ = ev(m, tv, xs); errs.append(np.abs(nn_ - fd).max())
            a = ax[i, c]; a.plot(xs, fd, "k-", lw=2.5, label="FD"); a.plot(xs, nn_, "r--", label="DGM"); a.plot(xs, H_np(tv, xs), ":", color="0.4", label="reward $H$")
            a.set_title(rf"$h_\varphi(t,x)$, $\varphi$ = {p:g}, t = {tv}"); a.set_xlabel("x"); a.set_ylabel("value"); a.grid(True); a.legend(fontsize=8)
        lp, lb = hists[p][-100:].mean(0)
        print("%-7g  %.1e   %.1e |          %.4f %.4f |  %.4f  %.4f  %.4f   [%ds]" % (p, lp, lb, ev(m, 0.0, np.array([1.0]))[0], np.interp(1.0, xg, D[str(p)][0]), *errs, time.time() - t0), flush=True)
    plt.tight_layout(); plt.savefig("fig3_dgm_vs_fd.png", dpi=110)
    fig, a = plt.subplots(figsize=(8, 4.5))
    for p in PHIS: a.semilogy(np.convolve(hists[p][:, 0], np.ones(50) / 50, "valid"), label=rf"$\varphi$ = {p:g}")
    a.set_xlabel("training iteration (per $\\varphi$ stage)"); a.set_ylabel("PDE loss (moving average)"); a.grid(True); a.legend(); plt.tight_layout(); plt.savefig("fig3_dgm_loss.png", dpi=110)
    np.savez("dgm_hist.npz", **{str(p): h for p, h in hists.items()})
