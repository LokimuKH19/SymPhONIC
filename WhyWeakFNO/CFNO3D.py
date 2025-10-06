import torch
import torch.nn as nn
import torch.fft
import numpy as np
import math

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


# -------------------------
# DCT helpers (1D and separable 3D)
# Implement DCT-II and its inverse via FFT and apply separably for 3D
# -------------------------
def dct_1d(x):
    """DCT-II along last dimension for real input tensor x."""
    N = x.shape[-1]
    v = torch.cat([x, x.flip(-1)], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    exp_factor = torch.exp(-1j * math.pi * k / (2 * N))
    X = (V[..., :N] * exp_factor).real
    X[..., 0] *= 0.5
    return X


def idct_1d(X):
    """Inverse DCT (DCT-III) along last dimension for real input X."""
    N = X.shape[-1]
    c = X.clone()
    c[..., 0] = c[..., 0] * 2.0
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    exp_factor = torch.exp(1j * math.pi * k / (2 * N))
    V = torch.zeros(X.shape[:-1] + (2 * N,), dtype=torch.cfloat, device=X.device)
    V[..., :N] = (c * exp_factor)
    if N > 1:
        V[..., N + 1:] = torch.conj(V[..., 1:N].flip(-1))
    V[..., N] = torch.tensor(0.0 + 0.0j, device=X.device)
    v = torch.fft.ifft(V, dim=-1)
    x = v[..., :N].real
    return x


def dct_3d(x):
    """
    Separable DCT-II for 3D tensors. Input shape (..., D, H, W) and returns same shape.
    The transform is applied along W, then H, then D.
    """
    orig_shape = x.shape
    # W
    y = dct_1d(x.reshape(-1, orig_shape[-1])).reshape(*orig_shape)
    # H
    y = y.permute(*range(y.dim() - 3), y.dim() - 1, y.dim() - 3, y.dim() - 2)
    shp = y.shape
    y = dct_1d(y.reshape(-1, shp[-1])).reshape(shp)
    y = y.permute(*range(y.dim() - 3), y.dim() - 2, y.dim() - 1, y.dim() - 3)
    # D
    y = y.permute(*range(y.dim() - 3), y.dim() - 2, y.dim() - 3, y.dim() - 1)
    shp = y.shape
    y = dct_1d(y.reshape(-1, shp[-1])).reshape(shp)
    out = y.permute(*range(y.dim() - 3), y.dim() - 1, y.dim() - 3, y.dim() - 2)
    return out


def idct_3d(X):
    """Inverse separable DCT for 3D inputs. Reverse order of dct_3d."""
    X_perm = X.permute(*range(X.dim() - 3), X.dim() - 1, X.dim() - 3, X.dim() - 2)
    shp = X_perm.shape
    y = idct_1d(X_perm.reshape(-1, shp[-1])).reshape(shp)
    y = y.permute(*range(y.dim() - 3), y.dim() - 2, y.dim() - 1, y.dim() - 3)
    y_perm2 = y.permute(*range(y.dim() - 3), y.dim() - 1, y.dim() - 3, y.dim() - 2)
    shp2 = y_perm2.shape
    z = idct_1d(y_perm2.reshape(-1, shp2[-1])).reshape(shp2)
    z = z.permute(*range(z.dim() - 3), z.dim() - 2, z.dim() - 1, z.dim() - 3)
    z_perm3 = z.permute(*range(z.dim() - 3), z.dim() - 1, z.dim() - 3, z.dim() - 2)
    shp3 = z_perm3.shape
    out = idct_1d(z_perm3.reshape(-1, shp3[-1])).reshape(shp3)
    out = out.permute(*range(out.dim() - 3), out.dim() - 2, out.dim() - 1, out.dim() - 3)
    return out


# -------------------------
# Spectral convolution blocks (Fourier & Chebyshev) for 3D
# -------------------------
class ChebSpectralConv3d(nn.Module):
    """Real-valued Chebyshev (cosine) spectral convolution using separable DCT."""
    def __init__(self, in_channels, out_channels, modes_d, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m_d = modes_d
        self.m_h = modes_h
        self.m_w = modes_w
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_d, modes_h, modes_w) * (1.0 / (in_channels * out_channels) ** 0.5)
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x_dct = dct_3d(x)
        x_modes = x_dct[:, :, :self.m_d, :self.m_h, :self.m_w]
        out_modes = torch.einsum('b c d h w, c o d h w -> b o d h w', x_modes, self.weight)
        out_dct = torch.zeros(B, self.out_channels, D, H, W, device=x.device, dtype=x.dtype)
        out_dct[:, :, :self.m_d, :self.m_h, :self.m_w] = out_modes
        out = idct_3d(out_dct)
        return out


class SpectralConv3d(nn.Module):
    """Complex Fourier spectral convolution using rfftn/irfftn."""
    def __init__(self, in_channels, out_channels, modes_d, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_d = modes_d
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes_d, modes_h, modes_w, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum('b i d h w, i o d h w -> b o d h w', input, weights)

    def forward(self, x):
        # x: [B, C, D, H, W] real
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='forward')  # [B, C, D, H, W//2+1]
        out_ft = torch.zeros(B, self.out_channels, D, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
        md = min(self.modes_d, D)
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W // 2 + 1)
        out_ft[:, :, :md, :mh, :mw] = self.compl_mul3d(x_ft[:, :, :md, :mh, :mw], self.weights[:, :, :md, :mh, :mw])
        x_out = torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1), norm='forward')
        return x_out


# -------------------------
# CFNOBlock3d: fuse Fourier and Chebyshev outputs
# -------------------------
class CFNOBlock3d(nn.Module):
    """A block that computes Fourier conv and Chebyshev conv and fuses them."""
    def __init__(self, in_channels, out_channels, modes_dhw, cheb_modes, alpha_init=0.5):
        super().__init__()
        md, mh, mw = modes_dhw
        ch_md, ch_mh, ch_mw = cheb_modes
        self.fourier = SpectralConv3d(in_channels, out_channels, md, mh, mw)
        self.cheb = ChebSpectralConv3d(in_channels, out_channels, ch_md, ch_mh, ch_mw)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.fuse = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y_f = self.fourier(x)
        y_c = self.cheb(x)
        a = torch.sigmoid(self.alpha)
        y_blend = a * y_f + (1.0 - a) * y_c
        y_cat = torch.cat([y_f, y_c], dim=1)
        y_fused = self.fuse(y_cat)
        return y_blend + y_fused


# -------------------------
# Model variants: FNO3d_small, CNO3d_small, CFNO3d_small
# All follow the same input/output convention: [B, C, D, H, W]
# -------------------------
class FNO3d_small(nn.Module):
    """A small 3D Fourier Neural Operator style model for scalar fields."""
    def __init__(self, in_channels=1, modes=(6,6,6), width=16, depth=3):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([SpectralConv3d(width, width, *modes) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, in_channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, width, D, H, W]
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class CNO3d_small(nn.Module):
    """A small 3D Chebyshev Neural Operator style model using DCT-based convs."""
    def __init__(self, cheb_modes=(6,6,6), in_channels=1, width=16, depth=3):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([ChebSpectralConv3d(width, width, *cheb_modes) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, in_channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class CFNO3d_small(nn.Module):
    """Compact CFNO3d combining Fourier and Chebyshev operators per block."""
    def __init__(self, in_channels=1, modes=(6,6,6), cheb_modes=(6,6,6), width=16, depth=3, alpha_init=0.5):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList()
        self.wconvs = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(CFNOBlock3d(width, width, modes, cheb_modes, alpha_init=alpha_init))
            self.wconvs.append(nn.Conv3d(width, width, 1))
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, in_channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for blk, w_conv in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w_conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x
