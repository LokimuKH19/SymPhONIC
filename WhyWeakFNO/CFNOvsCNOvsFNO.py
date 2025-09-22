import torch
import torch.nn as nn
import torch.fft
import math
import numpy as np
import time
from matplotlib import pyplot as plt
import random
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


def seed_everything(seed):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=16)
    np.set_printoptions(precision=16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=114514)


# -------------------------
# DCT/IDCT (DCT-II / DCT-III) 1D and separable 2D implementations, this aims to use fft in chebyshev expansion
# thus we applied the 1st-type Chebyshev polynomials, Tk(x) = cos(k arccos x),
# and it can be writtin in the Discrete Cosine Transform(DCT).
# -------------------------
def dct_1d(x):
    # x: (..., N), real
    N = x.shape[-1]
    v = torch.cat([x, x.flip(-1)], dim=-1)  # (..., 2N)s
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    exp_factor = torch.exp(-1j * math.pi * k / (2 * N))
    X = (V[..., :N] * exp_factor).real
    X[..., 0] *= 0.5
    return X


def idct_1d(X):
    # inverse of dct_1d (DCT-III), X: (..., N)
    N = X.shape[-1]
    c = X.clone()
    c[..., 0] = c[..., 0] * 2.0
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    exp_factor = torch.exp(1j * math.pi * k / (2 * N))
    V = torch.zeros(X.shape[:-1] + (2 * N,), dtype=torch.cfloat, device=X.device)
    V[..., :N] = (c * exp_factor)
    if N > 1:
        V[..., N + 1:] = torch.conj(V[..., 1:N].flip(-1))
    V[..., N] = torch.tensor(0.0 + 0.0j)
    v = torch.fft.ifft(V, dim=-1)
    x = v[..., :N].real
    return x


def dct_2d(x):
    # x: (..., H, W)
    # apply dct along last dim then along -2
    orig_shape = x.shape
    # last dim
    x_resh = x.reshape(-1, orig_shape[-1])
    y = dct_1d(x_resh).reshape(*orig_shape)
    # swap last two and apply again
    y_perm = y.permute(*range(y.dim() - 2), y.dim() - 1, y.dim() - 2)
    shp = y_perm.shape
    y2 = dct_1d(y_perm.reshape(-1, shp[-1])).reshape(shp)
    return y2.permute(*range(y2.dim() - 2), y2.dim() - 1, y2.dim() - 2)


def idct_2d(X):
    # inverse 2D: apply idct along -2 then -1 (reverse order)
    X_perm = X.permute(*range(X.dim() - 2), X.dim() - 1, X.dim() - 2)
    shp = X_perm.shape
    y = idct_1d(X_perm.reshape(-1, shp[-1])).reshape(shp)
    y = y.permute(*range(y.dim() - 2), y.dim() - 1, y.dim() - 2)
    z = idct_1d(y.reshape(-1, y.shape[-1])).reshape(y.shape)
    return z


# -------------------------
# Chebyshev / Cosine spectral conv (real coefficients)
# -------------------------
class ChebSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_h, modes_w):
        """
        in_channels, out_channels: channels
        modes_h, modes_w: number of retained modes in each dim (use <= H, W)
        The weight shape: (in_channels, out_channels, modes_h, modes_w) real
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m_h = modes_h
        self.m_w = modes_w
        # real coefficients for Chebyshev/DCT space
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_h, modes_w) * (1.0 / (in_channels * out_channels) ** 0.5))

    def forward(self, x):
        # x: [B, C, H, W] real
        B, C, H, W = x.shape
        # compute DCT2 on each channel
        # reshape to (..., H, W) to operate with dct_2d
        x_dct = dct_2d(x)  # shape [B, C, H, W]
        # crop modes (take top-left modes_h x modes_w)
        x_modes = x_dct[:, :, :self.m_h, :self.m_w]  # [B, C, m_h, m_w]
        # multiply by real weights: einsum over in_channel
        # out_modes[b, o, i, j] = sum_c x_modes[b, c, i, j] * weight[c, o, i, j]
        out_modes = torch.einsum("b c i j, c o i j -> b o i j", x_modes, self.weight)
        # create full spectral tensor with zeros then place modes back
        out_dct = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        out_dct[:, :, :self.m_h, :self.m_w] = out_modes
        # inverse DCT2
        out = idct_2d(out_dct)
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        in_channels, out_channels: number of channels
        modes: modes that reserved, Assume that H, W >= modes!!!!!
        weights are in complex，symmetric can be recovered by conjugate mirror
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # einsum over in_channel
        # input: [B, in, H, W], weights: [in, out, mh, mw]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        x: [B, C, H, W] (实数)
        """
        B, C, H, W = x.shape
        # 2D FFT (use complex)
        x_ft = torch.fft.rfft2(x, norm="forward")  # [B, C, H, W//2+1]

        # Output a frequency tensor
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        # Low frequency modes × modes
        mh, mw = self.modes, self.modes
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(x_ft[:, :, :mh, :mw], self.weights)

        # IFFT
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="forward")
        return x_out


# -------------------------
# CFNO block: combine Fourier spectral conv and Chebyshev spectral conv per layer
# -------------------------
class CFNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, cheb_modes):
        super().__init__()
        self.fourier = SpectralConv2d(in_channels, out_channels, modes)
        mh, mw = cheb_modes
        self.cheb = ChebSpectralConv2d(in_channels, out_channels, mh, mw)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y_f = self.fourier(x)
        y_c = self.cheb(x)
        a = torch.sigmoid(self.alpha)
        y_blend = a * y_f + (1.0 - a) * y_c
        y_cat = torch.cat([y_f, y_c], dim=1)
        y_fused = self.fuse(y_cat)
        return y_blend + y_fused


# -------------------------
# CFNO network (example stack)
# -------------------------
class CFNO2d(nn.Module):
    def __init__(self, modes=12, cheb_modes=(12, 12), width=32, depth=4):
        super().__init__()
        self.width = width
        self.depth = depth
        # input lifting (like your FNO fc0)
        self.fc0 = nn.Linear(2, width)
        # create layer stacks of CFNOBlock with 1x1 conv residuals (similar to FNO architecture)
        self.blocks = nn.ModuleList()
        self.w_convs = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(CFNOBlock(width, width, modes, cheb_modes))
            self.w_convs.append(nn.Conv2d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x: [B, 2, H, W]
        B, C, H, W = x.shape
        # lift
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 2]
        x = self.fc0(x)  # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]
        # stack
        for block, w_conv in zip(self.blocks, self.w_convs):
            y = block(x)
            x = y + w_conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # [B, H, W, 2]
        x = x.permute(0, 3, 1, 2)  # [B, 2, H, W]
        return x


# -------------------- Networks: FNO, CNO, CFNO --------------------
class FNO2d_small(nn.Module):
    def __init__(self, modes=8, width=16, depth=3):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.blocks = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):  # x: [B,1,H,W] source f
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,H,W,1]
        x = self.fc0(x)  # [B,H,W,width]
        x = x.permute(0, 3, 1, 2)  # [B,width,H,W]
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# CNO model: use ChebSpectralConv2d blocks instead of Fourier
class CNO2d_small(nn.Module):
    def __init__(self, cheb_modes=(8, 8), width=16, depth=3):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.blocks = nn.ModuleList(
            [ChebSpectralConv2d(width, width, cheb_modes[0], cheb_modes[1]) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# CFNO combining both
class CFNO2d_small(nn.Module):
    def __init__(self, modes=8, cheb_modes=(8, 8), width=16, depth=3):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.blocks = nn.ModuleList([CFNOBlock(width, width, modes, cheb_modes) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# -------------------- Poisson dataset generation using Jacobi solver --------------------
def poisson(f, iters=500, tol=1e-6):
    # Solve -Δ u = f on unit square with zero Dirichlet BC using Jacobi for interior points
    # f: [H, W] tensor (including boundary rows/cols, but boundary f ignored)
    H, W = f.shape
    u = torch.zeros_like(f)
    # keep boundary zero
    dx = 1.0 / (H - 1)
    dy = 1.0 / (W - 1)
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2 * (dx2 + dy2)
    # use Jacobi iteration
    for _ in range(iters):
        u_old = u.clone()
        # update interior, 5 points center differential for 2D cases
        u[1:-1, 1:-1] = ((u_old[2:, 1:-1] + u_old[:-2, 1:-1]) * dy2 +
                         (u_old[1:-1, 2:] + u_old[1:-1, :-2]) * dx2 -
                         f[1:-1, 1:-1] * dx2 * dy2) / denom
        # boundaries remain zero (Dirichlet)
        if torch.max(torch.abs(u - u_old)) < tol:
            break
    return u


def make_dataset(n_samples=200, H=32, W=32):
    # create random RHS f with localized sources (sum of gaussians bumps)
    X = []
    Y = []
    for _ in range(n_samples):
        f = torch.zeros(H, W)
        # add a few random gaussian bumps
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            sx = np.random.uniform(0.03, 0.12)
            sy = np.random.uniform(0.03, 0.12)
            xv = torch.linspace(0, 1, H)
            yv = torch.linspace(0, 1, W)
            Xg, Yg = torch.meshgrid(xv, yv, indexing='ij')
            g = torch.exp(-((Xg - cx) ** 2) / (2 * sx ** 2) - ((Yg - cy) ** 2) / (2 * sy ** 2))
            amp = np.random.uniform(-5, 5)
            f += amp * g
        # solve Poisson
        u = poisson(f, iters=2000, tol=1e-6)
        X.append(f.unsqueeze(0))  # channel dim
        Y.append(u.unsqueeze(0))
    X = torch.stack(X)  # [N,1,H,W]
    Y = torch.stack(Y)  # [N,1,H,W]
    return X, Y


# generate dataset
H = 32
W = 32
n_train = 120
n_val = 40
n_test = 40
X_all, Y_all = make_dataset(n_train + n_val + n_test, H=H, W=W)
X_train = X_all[:n_train].to(device)
Y_train = Y_all[:n_train].to(device)
X_val = X_all[n_train:n_train + n_val].to(device)
Y_val = Y_all[n_train:n_train + n_val].to(device)
X_test = X_all[n_train + n_val:].to(device)
Y_test = Y_all[n_train + n_val:].to(device)

print("Dataset shapes:", X_train.shape, Y_train.shape, X_val.shape, X_test.shape)

# mask for interior points (for loss computation) to ignore boundaries (Dirichlet enforced)
mask = torch.ones(1, 1, H, W)
mask[:, :, 0, :] = 0
mask[:, :, -1, :] = 0
mask[:, :, :, 0] = 0
mask[:, :, :, -1] = 0
mask = mask.to(device)


# -------------------- Training utilities --------------------
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=60, batch_size=8, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X_train.shape[0]
    logs = {'train': [], 'val': []}
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        train_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = Y_train[idx]
            pred = model(xb)
            # enforce Dirichlet BC by zeroing boundaries of pred before loss, hard constraint used
            pred = pred * mask
            yb = yb * mask
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= n
        # val
        model.eval()
        with torch.no_grad():
            predv = model(X_val) * mask
            val_loss = loss_fn(predv, Y_val * mask).item()
        logs['train'].append(train_loss)
        logs['val'].append(val_loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep + 1}/{epochs} train {train_loss:.6e} val {val_loss:.6e}")
    return model, logs


if __name__ == '__main__':
    # -------------------- Instantiate and train three models --------------------
    fno = FNO2d_small(modes=6, width=16, depth=3)
    cno = CNO2d_small(cheb_modes=(8, 8), width=16, depth=3)
    cfno = CFNO2d_small(modes=6, cheb_modes=(8, 8), width=16, depth=3)

    print("Training FNO:")
    start = time.time()
    fno, logs_fno = train_model(fno, X_train, Y_train, X_val, Y_val, epochs=60, batch_size=8, lr=1e-3)
    t_fno = time.time() - start
    print("Training CNO:")
    start = time.time()
    cno, logs_cno = train_model(cno, X_train, Y_train, X_val, Y_val, epochs=60, batch_size=8, lr=1e-3)
    t_cno = time.time() - start
    print("Training CFNO:")
    start = time.time()
    cfno, logs_cfno = train_model(cfno, X_train, Y_train, X_val, Y_val, epochs=60, batch_size=8, lr=1e-3)
    t_cfno = time.time() - start


    # -------------------- Evaluate on test set --------------------
    def evaluate(model, X, Y):
        model.eval()
        with torch.no_grad():
            pred = model(X) * mask
            mse = torch.mean((pred - Y * mask) ** 2).item()
        return mse, pred


    mse_fno, pred_fno = evaluate(fno, X_test, Y_test)
    mse_cno, pred_cno = evaluate(cno, X_test, Y_test)
    mse_cfno, pred_cfno = evaluate(cfno, X_test, Y_test)

    print("\nTest MSEs (interior):")
    print(f"FNO: {mse_fno:.6e}  time {t_fno:.1f}s")
    print(f"CNO: {mse_cno:.6e}  time {t_cno:.1f}s")
    print(f"CFNO: {mse_cfno:.6e} time {t_cfno:.1f}s")

    # -------------------- Plot losses and a sample comparison --------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(logs_fno['train'], label='FNO train')
    plt.plot(logs_fno['val'], linestyle='--', label='FNO val')
    plt.plot(logs_cno['train'], label='CNO train')
    plt.plot(logs_cno['val'], linestyle='--', label='CNO val')
    plt.plot(logs_cfno['train'], label='CFNO train')
    plt.plot(logs_cfno['val'], linestyle='--', label='CFNO val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()

    # sample index
    idx = 0
    gt = Y_test[idx, 0].cpu().numpy()
    pf = pred_fno[idx, 0].cpu().numpy()
    pc = pred_cno[idx, 0].cpu().numpy()
    pcf = pred_cfno[idx, 0].cpu().numpy()
    f_sample = X_test[idx, 0].cpu().numpy()

    plt.subplot(1, 2, 2)
    plt.suptitle("Sample solution comparison (interior masked BC applied)")
    # show 4 panels
    plt.imshow(gt, origin='lower')
    plt.title('GT')
    plt.colorbar(fraction=0.046, pad=0.01)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(f_sample, origin='lower')
    plt.title('f (Source Term)')
    plt.colorbar(fraction=0.046, pad=0.01)
    plt.subplot(2, 3, 2)
    plt.imshow(gt, origin='lower')
    plt.title('Ground Truth u')
    plt.colorbar(fraction=0.046, pad=0.01)
    plt.subplot(2, 3, 3)
    plt.imshow(pf, origin='lower')
    plt.title('FNO pred')
    plt.colorbar(fraction=0.046, pad=0.01)
    plt.subplot(2, 3, 4)
    plt.imshow(pc, origin='lower')
    plt.title('CNO pred')
    plt.colorbar(fraction=0.046, pad=0.01)
    plt.subplot(2, 3, 5)
    plt.imshow(pcf, origin='lower')
    plt.title('CFNO pred')
    plt.colorbar(fraction=0.046, pad=0.01)
    plt.tight_layout()
    plt.show()

    # Report MSEs in a small dict
    results = {'model': ['FNO', 'CNO', 'CFNO'], 'mse': [mse_fno, mse_cno, mse_cfno], 'time': [t_fno, t_cno, t_cfno]}
    print(pd.DataFrame(results))

# next plan: 1d case (linear, nonlinear) 2d case (nonlinear)
# corresponding weak form training strategy
# (8 cases in total)
