"""Deep Image Prior 장난감 재현: 학습되지 않은 CNN 을 잡음 영상 한 장에 맞추면, 잡음보다 영상 구조를 먼저 맞춘다."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, matplotlib, time, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(8)
N = 96
yy, xx = np.mgrid[:N, :N] / N
clean = np.zeros((N, N)) + .15
clean += ((xx-.35)**2 + (yy-.4)**2 < .05) * .6
clean += ((np.abs(xx-.72) < .13) & (np.abs(yy-.3) < .1)) * .4
clean += (np.sin(2*np.pi*(5*xx+3*yy)) > 0) * ((yy > .62) & (xx > .1) & (xx < .9)) * .25
clean = gaussian_filter(clean, .7); clean = np.clip(clean, 0, 1)
sigma = 0.15
noisy = np.clip(clean + sigma * np.random.randn(N, N), 0, 1.3)   # 클리핑 대신 넉넉한 범위
noisy = clean + sigma * np.random.randn(N, N)
psnr = lambda a, b: 10*np.log10(1/np.mean((a-b)**2))
print("noisy PSNR", psnr(noisy, clean))

class Hourglass(nn.Module):
    def __init__(s, cin=32, ch=64, depth=4):
        super().__init__()
        def blk(i, o, stride=1): return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, padding_mode="reflect"), nn.BatchNorm2d(o), nn.LeakyReLU(.2))
        s.down = nn.ModuleList([blk(cin if i == 0 else ch, ch, 2) for i in range(depth)])
        s.up = nn.ModuleList([nn.Sequential(blk(ch, ch), blk(ch, ch, 1)) for _ in range(depth)])
        s.out = nn.Conv2d(ch, 1, 1)
    def forward(s, z):
        x = z
        for d in s.down: x = d(x)
        for u in s.up: x = u(Fn.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return s.out(x)

def run(iters=3000, seed=0, lr=0.01, snap=(25, 100, 300, 1000, 3000)):
    torch.manual_seed(seed)
    net = Hourglass(); z = torch.rand(1, 32, N, N) * 0.1
    y = torch.tensor(noisy, dtype=torch.float32)[None, None]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []; snaps = {}; best = (-1, None, 0); ema = None
    for it in range(1, iters+1):
        opt.zero_grad(); out = net(z + torch.randn_like(z) * 0.03)   # 입력 잡음 섭동(원 논문과 동일)
        loss = Fn.mse_loss(out, y); loss.backward(); opt.step()
        if it % 10 == 0 or it in snap:
            with torch.no_grad(): o = net(z)[0, 0].numpy()
            hist.append((it, loss.item(), psnr(o, clean), psnr(o, noisy)))
            if hist[-1][2] > best[0]: best = (hist[-1][2], o.copy(), it)
            if it in snap: snaps[it] = o.copy()
    return hist, snaps, best
t0 = time.time()
hist, snaps, best = run()
print("time %.0fs" % (time.time()-t0))
H = np.array(hist)
print("best PSNR vs clean %.2f at iter %d; final(3000) %.2f; fit-to-noisy PSNR final %.2f" % (best[0], best[2], H[-1, 2], H[-1, 3]))
np.save("hist.npy", H)

# baseline
gb = gaussian_filter(noisy, 1.2); md = median_filter(noisy, 5)
print("gauss 1.2: %.2f, median5: %.2f" % (psnr(gb, clean), psnr(md, clean)))
# 주파수 대역별 오차(출력 - 정답) 이 반복마다 어떻게 변하나: 저/고주파 성분의 정답 대비 비율
def band_err(o):
    F = np.fft.fft2(o - clean); C = np.fft.fft2(clean)
    fy = np.fft.fftfreq(N)[:, None]; fx = np.fft.fftfreq(N)[None, :]; r = np.hypot(fx, fy)
    out = []
    for lo, hi in [(0, .1), (.1, .25), (.25, .8)]:
        m = (r >= lo) & (r < hi); out.append(np.sqrt((np.abs(F[m])**2).sum() / (np.abs(C[m])**2).sum() if (np.abs(C[m])**2).sum() > 0 else np.nan))
    return out
def band_noise(o):  # 출력이 '잡음 성분'을 얼마나 따라갔나: 출력-정답 vs 잡음 의 상관
    return np.corrcoef((o-clean).ravel(), (noisy-clean).ravel())[0, 1]
print("corr(out-clean, noise) at snaps:", {k: round(band_noise(v), 3) for k, v in snaps.items()})
print("band rel err at 100/300/3000:", {k: np.round(band_err(snaps[k]), 3).tolist() for k in (100, 300, 3000)})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(H[:, 0], H[:, 2], "r", label="PSNR(출력, 깨끗한 영상)  ← 우리가 원하는 것")
ax[0].plot(H[:, 0], H[:, 3], "b", label="PSNR(출력, 잡음 영상)  ← 학습이 최소화하는 것")
ax[0].axhline(psnr(noisy, clean), c="gray", ls=":", label=f"잡음 영상 자체 ({psnr(noisy, clean):.1f} dB)")
ax[0].axvline(best[2], c="r", ls="--", alpha=.5); ax[0].set_xscale("log"); ax[0].set_xlabel("반복 횟수"); ax[0].set_ylabel("PSNR (dB)"); ax[0].legend(fontsize=8); ax[0].set_title(f"조기 종료가 핵심: 최고 {best[0]:.1f} dB @ {best[2]}회 → 이후 잡음에 과적합")
ax[1].plot(H[:, 0], H[:, 1], "k"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("반복 횟수"); ax[1].set_title("학습 손실(MSE to 잡음 영상): 계속 감소")
plt.tight_layout(); plt.savefig("figures/dip_curves.png", dpi=125); plt.close()

show = [("정답", clean), ("잡음 입력", noisy), ("Gaussian 블러", gb), ("median 5×5", md)]
show += [(f"DIP {k}회", snaps[k]) for k in (100, 300, 1000, 3000)]
show.insert(4, (f"DIP 최적 {best[2]}회", best[1]))
fig, ax = plt.subplots(2, 5, figsize=(16, 6.8))
ax.ravel()[-1].axis('off')
for a, (nm, im) in zip(ax.ravel(), show):
    a.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1); a.set_title(nm + ("" if nm in ("정답",) else f"\n{psnr(im, clean):.1f} dB"), fontsize=10); a.axis("off")
plt.tight_layout(); plt.savefig("figures/dip_gallery.png", dpi=120); plt.close()
json.dump({"best": [best[0], best[2]], "final": H[-1, 2], "noisy": psnr(noisy, clean), "gauss": psnr(gb, clean), "median": psnr(md, clean)}, open("results.json", "w"))
