import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import numpy as np
import random


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
    def __init__(self, in_channels, out_channels, modes, cheb_modes, alpha_init=0.5):
        # alpha\in[0,1], 0.5 is the default for initialization and self-adaptive fitting
        super().__init__()
        self.fourier = SpectralConv2d(in_channels, out_channels, modes)
        mh, mw = cheb_modes
        self.cheb = ChebSpectralConv2d(in_channels, out_channels, mh, mw)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y_f = self.fourier(x)
        y_c = self.cheb(x)
        a = torch.sigmoid(self.alpha)
        y_blend = a * y_f + (1.0 - a) * y_c
        y_cat = torch.cat([y_f, y_c], dim=1)
        y_fused = self.fuse(y_cat)
        return y_blend + y_fused


def mixed_boundary_pad2d(x, pad_h, pad_w):
    if pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="replicate")
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="circular")
    return x


class MultiBandSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, low_modes, high_modes=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.low_modes = int(low_modes)
        self.high_modes = int(max(high_modes, 0))
        scale = 1 / max(in_channels * out_channels, 1)
        self.weights_low_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.low_modes, self.low_modes, dtype=torch.cfloat)
        )
        self.weights_low_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.low_modes, self.low_modes, dtype=torch.cfloat)
        )
        if self.high_modes > 0:
            self.weights_high_pos = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.low_modes, self.high_modes, dtype=torch.cfloat)
            )
            self.weights_high_neg = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.low_modes, self.high_modes, dtype=torch.cfloat)
            )
        else:
            self.register_parameter("weights_high_pos", None)
            self.register_parameter("weights_high_neg", None)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="forward")
        freq_w = W // 2 + 1
        out_ft = torch.zeros(B, self.out_channels, H, freq_w, device=x.device, dtype=torch.cfloat)

        mh = min(self.low_modes, H)
        mw = min(self.low_modes, freq_w)
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(
            x_ft[:, :, :mh, :mw],
            self.weights_low_pos[:, :, :mh, :mw],
        )
        if mh > 0:
            out_ft[:, :, -mh:, :mw] = self.compl_mul2d(
                x_ft[:, :, -mh:, :mw],
                self.weights_low_neg[:, :, :mh, :mw],
            )

        high_available = max(freq_w - mw, 0)
        hw = min(self.high_modes, high_available)
        if hw > 0:
            out_ft[:, :, :mh, -hw:] = out_ft[:, :, :mh, -hw:] + self.compl_mul2d(
                x_ft[:, :, :mh, -hw:],
                self.weights_high_pos[:, :, :mh, :hw],
            )
            out_ft[:, :, -mh:, -hw:] = out_ft[:, :, -mh:, -hw:] + self.compl_mul2d(
                x_ft[:, :, -mh:, -hw:],
                self.weights_high_neg[:, :, :mh, :hw],
            )

        return torch.fft.irfft2(out_ft, s=(H, W), norm="forward")


class FourierFeatureGrid2d(nn.Module):
    def __init__(self, bands=(1, 2, 4, 8)):
        super().__init__()
        self.register_buffer("bands", torch.tensor(list(bands), dtype=torch.float32), persistent=False)

    @property
    def extra_channels(self):
        return int(self.bands.numel()) * 4

    def forward(self, x):
        if self.bands.numel() == 0:
            return x
        B, _, H, W = x.shape
        yy = torch.linspace(0.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        xx = torch.linspace(0.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        bands = self.bands.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        phase_y = 2.0 * math.pi * bands * yy
        phase_x = 2.0 * math.pi * bands * xx
        y_features = torch.cat([torch.sin(phase_y), torch.cos(phase_y)], dim=1).expand(B, -1, H, W)
        x_features = torch.cat([torch.sin(phase_x), torch.cos(phase_x)], dim=1).expand(B, -1, H, W)
        return torch.cat([x, y_features, x_features], dim=1)


class LocalHighPassBlock2d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.pad = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=0)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        smooth = F.avg_pool2d(mixed_boundary_pad2d(x, 1, 1), kernel_size=3, stride=1)
        high = x - smooth
        y = self.depthwise(mixed_boundary_pad2d(high, self.pad, self.pad))
        y = F.gelu(self.pointwise(y))
        return self.mix(y)


class HFCFNOBlock(nn.Module):
    def __init__(
        self,
        channels,
        modes,
        cheb_modes,
        high_modes=4,
        alpha_init=0.5,
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        self.low_cfno = CFNOBlock(channels, channels, modes, cheb_modes, alpha_init=alpha_init)
        self.band_spectral = MultiBandSpectralConv2d(channels, channels, modes, high_modes=high_modes)
        self.use_local_highpass = bool(use_local_highpass)
        self.local_high = LocalHighPassBlock2d(channels) if self.use_local_highpass else None
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(float(high_gate_init)))

    def forward(self, x):
        low = self.low_cfno(x)
        band = self.band_spectral(x)
        local = self.local_high(x) if self.local_high is not None else torch.zeros_like(band)
        gate = torch.sigmoid(self.high_gate)
        fused = self.fuse(torch.cat([low, band, local], dim=1))
        return low + gate * (band + local) + fused


class HFFNOBlock(nn.Module):
    def __init__(
        self,
        channels,
        modes,
        high_modes=4,
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        self.low_fno = SpectralConv2d(channels, channels, modes)
        self.band_spectral = MultiBandSpectralConv2d(channels, channels, modes, high_modes=high_modes)
        self.use_local_highpass = bool(use_local_highpass)
        self.local_high = LocalHighPassBlock2d(channels) if self.use_local_highpass else None
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(float(high_gate_init)))

    def forward(self, x):
        low = self.low_fno(x)
        band = self.band_spectral(x)
        local = self.local_high(x) if self.local_high is not None else torch.zeros_like(band)
        gate = torch.sigmoid(self.high_gate)
        fused = self.fuse(torch.cat([low, band, local], dim=1))
        return low + gate * (band + local) + fused


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
    def __init__(self, modes=8, width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(depth)])     # fourier transform
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])    # weights
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):  # x: [B,1,H,W] source f
        # B, C, H, W = x.shape
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
    def __init__(self, cheb_modes=(8, 8), width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList(
            [ChebSpectralConv2d(width, width, cheb_modes[0], cheb_modes[1]) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        # B, C, H, W = x.shape
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
    def __init__(self, modes=8, cheb_modes=(8, 8), width=16, depth=3, alpha_init=0.5, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([CFNOBlock(width, width, modes, cheb_modes, alpha_init=alpha_init) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        # B, C, H, W = x.shape
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


class HF_CFNO2d_small(nn.Module):
    def __init__(
        self,
        modes=16,
        cheb_modes=(16, 16),
        high_modes=None,
        width=24,
        depth=5,
        alpha_init=0.5,
        input_features=1,
        output_features=1,
        fourier_feature_bands=(1, 2, 4, 8),
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        if high_modes is None:
            high_modes = max(2, int(modes) // 2)
        self.feature_grid = FourierFeatureGrid2d(fourier_feature_bands)
        lifted_features = input_features + self.feature_grid.extra_channels
        self.fc0 = nn.Linear(lifted_features, width)
        self.blocks = nn.ModuleList(
            [
                HFCFNOBlock(
                    width,
                    modes=modes,
                    cheb_modes=cheb_modes,
                    high_modes=high_modes,
                    alpha_init=alpha_init,
                    high_gate_init=high_gate_init,
                    use_local_highpass=use_local_highpass,
                )
                for _ in range(depth)
            ]
        )
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        hidden = max(64, width * 2)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, output_features)

    def forward(self, x):
        x = self.feature_grid(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = F.gelu(y + w(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


class HF_FNO2d_small(nn.Module):
    def __init__(
        self,
        modes=16,
        high_modes=None,
        width=24,
        depth=5,
        input_features=1,
        output_features=1,
        fourier_feature_bands=(1, 2, 4, 8),
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        if high_modes is None:
            high_modes = max(2, int(modes) // 2)
        self.feature_grid = FourierFeatureGrid2d(fourier_feature_bands)
        lifted_features = input_features + self.feature_grid.extra_channels
        self.fc0 = nn.Linear(lifted_features, width)
        self.blocks = nn.ModuleList(
            [
                HFFNOBlock(
                    width,
                    modes=modes,
                    high_modes=high_modes,
                    high_gate_init=high_gate_init,
                    use_local_highpass=use_local_highpass,
                )
                for _ in range(depth)
            ]
        )
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        hidden = max(64, width * 2)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, output_features)

    def forward(self, x):
        x = self.feature_grid(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = F.gelu(y + w(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
