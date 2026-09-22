from __future__ import annotations

import argparse
import bisect
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Keep CUDA's host-side module/workspace commitment small on memory-constrained
# Windows sessions. These must be set before importing torch.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuroOperators import apply_activation


# This file is intentionally dedicated to native steady 1D operators. It does
# not add another execution path to the already large 2D/transient suite.
MODEL_ORDER = ("FNO", "CFNO", "HF_FNO", "HF_CFNO")
MODE_ORDER = ("pde", "hybrid", "data")
MODE_DIR = {"pde": "PhysicsOnly", "hybrid": "Hybrid", "data": "DataOnly"}
PRIMARY = ("#3c7fb1", "#e64532", "#44a05c", "#b53289")
SECONDARY = ("#90bfd5", "#f5a65b", "#a0d292", "#f296ac")
MODEL_COLOR = dict(zip(MODEL_ORDER, PRIMARY))
MODE_COLOR = dict(zip(MODE_ORDER, PRIMARY[:3]))
SERIES_COLOR = {"train": PRIMARY[0], "validation": PRIMARY[1], "test": PRIMARY[2]}
SERIES_STYLE = {"train": "-", "validation": "--", "test": ":"}
EPS = np.finfo(np.float64).tiny
plt = None
LogNorm = None


def configure_plotting() -> None:
    global plt, LogNorm
    if plt is not None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.colors import LogNorm as MatplotlibLogNorm

    plt = pyplot
    LogNorm = MatplotlibLogNorm
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.titlesize": 13,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


REFERENCES = [
    {
        "title": "Fourier Neural Operator for Parametric Partial Differential Equations",
        "url": "https://arxiv.org/abs/2010.08895",
        "used_for": "Source-to-solution operator-learning formulation and FNO baseline.",
    },
    {
        "title": "PDEBench: An Extensive Benchmark for Scientific Machine Learning",
        "url": "https://arxiv.org/abs/2210.07182",
        "used_for": "Burgers, reaction-diffusion, and Allen-Cahn PDE families.",
    },
    {
        "title": "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations",
        "url": "https://www.sciencedirect.com/science/article/pii/S0021999118307125",
        "used_for": "Canonical nonlinear PDE forms and physics-informed optimization context.",
    },
]


@dataclass(frozen=True)
class Steady1DCase:
    pde: str
    case: str
    profile: str
    domain_x: float
    nx: int
    modes: int
    high_modes: int
    depth: int
    target_params: int
    batch_size: int
    lr: float
    input_channels: int
    params: dict[str, float | int | str]

    @property
    def slug(self) -> str:
        return f"{self.pde}_{self.case}"

    @property
    def dx(self) -> float:
        return self.domain_x / max(self.nx - 1, 1)

    @property
    def high_nonlinear(self) -> bool:
        return self.profile == "high_nonlinear"


@dataclass(frozen=True)
class FieldScaling1D:
    source_mean: float
    source_std: float
    solution_shift: float = 0.0
    solution_scale: float = 1.0

    def normalize_source(self, source: torch.Tensor) -> torch.Tensor:
        return (source - self.source_mean) / max(self.source_std, 1e-12)

    def denormalize_solution(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.solution_shift + self.solution_scale * normalized


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # On the current Windows driver, cuDNN's Conv1d backward-plan selection can
    # report a false 128 MiB OOM on an otherwise empty 8 GiB GPU. Native CUDA
    # convolution is stable here and leaves FFT execution on the GPU.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() * (2 if torch.is_complex(parameter) else 1) for parameter in model.parameters())


def project_self_conjugate_bins(spectrum: torch.Tensor, signal_length: int) -> torch.Tensor:
    imaginary_scale = torch.ones(spectrum.shape[-1], device=spectrum.device, dtype=spectrum.real.dtype)
    imaginary_scale[0] = 0.0
    if signal_length % 2 == 0:
        imaginary_scale[-1] = 0.0
    return torch.complex(spectrum.real, spectrum.imag * imaginary_scale.view(1, 1, -1))


class RealFourierTransform1d(nn.Module):
    """Exact one-sided DFT pair for small real 1D grids without a cuFFT plan."""

    def __init__(self, length: int):
        super().__init__()
        self.length = int(length)
        frequencies = self.length // 2 + 1
        k = torch.arange(frequencies, dtype=torch.float64).view(-1, 1)
        n = torch.arange(self.length, dtype=torch.float64).view(1, -1)
        phase = 2.0 * math.pi * k * n / self.length
        cosine = torch.cos(phase).to(torch.float32)
        sine = torch.sin(phase).to(torch.float32)
        synthesis_scale = torch.full((frequencies,), 2.0, dtype=torch.float32)
        synthesis_scale[0] = 1.0
        if self.length % 2 == 0:
            synthesis_scale[-1] = 1.0
        self.register_buffer("cosine", cosine, persistent=False)
        self.register_buffer("sine", sine, persistent=False)
        self.register_buffer("synthesis_scale", synthesis_scale, persistent=False)

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.length:
            raise ValueError(f"Fourier transform was built for length {self.length}, received {x.shape[-1]}")
        real = torch.einsum("bin,kn->bik", x, self.cosine) / self.length
        imaginary = -torch.einsum("bin,kn->bik", x, self.sine) / self.length
        return torch.complex(real, imaginary)

    def synthesize(self, spectrum: torch.Tensor) -> torch.Tensor:
        weighted_real = spectrum.real * self.synthesis_scale.view(1, 1, -1)
        weighted_imaginary = spectrum.imag * self.synthesis_scale.view(1, 1, -1)
        return torch.einsum("bik,kn->bin", weighted_real, self.cosine) - torch.einsum(
            "bik,kn->bin", weighted_imaginary, self.sine
        )


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, length: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        self.transform = RealFourierTransform1d(length)
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    @staticmethod
    def multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bik,iok->bok", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        x_ft = self.transform.analyze(x)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        retained = min(self.modes, x_ft.shape[-1])
        if retained > 0:
            out_ft[..., :retained] = self.multiply(x_ft[..., :retained], self.weights[..., :retained])
        # A real 1D field has a Hermitian spectrum. rFFT stores the independent
        # non-negative branch only; DC and Nyquist are self-conjugate and must
        # therefore be real. The synthesis basis reconstructs the negative
        # branch uniquely through the conjugate contribution.
        out_ft = project_self_conjugate_bins(out_ft, length)
        return self.transform.synthesize(out_ft)


class ChebSpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, length: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        self.length = int(length)
        scale = 1.0 / math.sqrt(max(in_channels * out_channels, 1))
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))
        sample = torch.arange(self.length, dtype=torch.float64)
        frequency = torch.arange(self.length, dtype=torch.float64).view(-1, 1)
        transform = 2.0 * torch.cos(math.pi * frequency * (sample + 0.5) / self.length)
        transform[0] *= 0.5
        inverse = torch.linalg.inv(transform)
        self.register_buffer("dct_matrix", transform.to(torch.float32), persistent=False)
        self.register_buffer("idct_matrix", inverse.to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        if length != self.length:
            raise ValueError(f"Chebyshev layer was built for length {self.length}, received {length}")
        x_dct = torch.einsum("bin,kn->bik", x, self.dct_matrix)
        out_dct = torch.zeros(batch, self.out_channels, length, device=x.device, dtype=x.dtype)
        retained = min(self.modes, length)
        if retained > 0:
            out_dct[..., :retained] = torch.einsum(
                "bik,iok->bok",
                x_dct[..., :retained],
                self.weights[..., :retained],
            )
        return torch.einsum("bok,nk->bon", out_dct, self.idct_matrix)


class CFNOBlock1d(nn.Module):
    def __init__(self, channels: int, modes: int, length: int, alpha_init: float = 0.5):
        super().__init__()
        self.fourier = SpectralConv1d(channels, channels, modes, length)
        self.chebyshev = ChebSpectralConv1d(channels, channels, modes, length)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.fuse = nn.Conv1d(2 * channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fourier = self.fourier(x)
        chebyshev = self.chebyshev(x)
        alpha = torch.sigmoid(self.alpha)
        blended = alpha * fourier + (1.0 - alpha) * chebyshev
        return blended + self.fuse(torch.cat([fourier, chebyshev], dim=1))


class MultiBandSpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, low_modes: int, high_modes: int, length: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.low_modes = int(low_modes)
        self.high_modes = int(max(high_modes, 0))
        self.transform = RealFourierTransform1d(length)
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weights_low = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.low_modes, dtype=torch.cfloat)
        )
        if self.high_modes > 0:
            self.weights_high = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.high_modes, dtype=torch.cfloat)
            )
        else:
            self.register_parameter("weights_high", None)

    @staticmethod
    def multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bik,iok->bok", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        x_ft = self.transform.analyze(x)
        frequencies = x_ft.shape[-1]
        out_ft = torch.zeros(batch, self.out_channels, frequencies, device=x.device, dtype=torch.cfloat)
        low = min(self.low_modes, frequencies)
        if low > 0:
            out_ft[..., :low] = self.multiply(x_ft[..., :low], self.weights_low[..., :low])
        high = min(self.high_modes, max(frequencies - low, 0))
        if high > 0 and self.weights_high is not None:
            out_ft[..., -high:] += self.multiply(x_ft[..., -high:], self.weights_high[..., :high])
        out_ft = project_self_conjugate_bins(out_ft, length)
        return self.transform.synthesize(out_ft)


class LocalHighPassBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, activation: str = "gelu"):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.activation = activation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=0)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.pad, self.pad), mode="replicate")
        smooth = F.avg_pool1d(padded, kernel_size=2 * self.pad + 1, stride=1)
        high = x - smooth
        filtered = self.depthwise(F.pad(high, (self.pad, self.pad), mode="replicate"))
        return self.mix(apply_activation(self.pointwise(filtered), self.activation))


class FourierFeatureGrid1d(nn.Module):
    def __init__(self, bands: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.register_buffer("bands", torch.tensor(bands, dtype=torch.float32), persistent=False)

    @property
    def extra_channels(self) -> int:
        return 2 * int(self.bands.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bands.numel() == 0:
            return x
        batch, _, length = x.shape
        coordinate = torch.linspace(0.0, 1.0, length, device=x.device, dtype=x.dtype).view(1, 1, length)
        bands = self.bands.to(device=x.device, dtype=x.dtype).view(1, -1, 1)
        phase = 2.0 * math.pi * bands * coordinate
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1).expand(batch, -1, -1)
        return torch.cat([x, features], dim=1)


class HFFNOBlock1d(nn.Module):
    def __init__(self, channels: int, modes: int, high_modes: int, length: int, local_activation: str = "gelu"):
        super().__init__()
        self.low = SpectralConv1d(channels, channels, modes, length)
        self.band = MultiBandSpectralConv1d(channels, channels, modes, high_modes, length)
        self.local_high = LocalHighPassBlock1d(channels, activation=local_activation)
        self.fuse = nn.Conv1d(3 * channels, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(-0.5))
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low(x)
        band = self.band(x)
        local = self.local_high(x)
        gate = torch.sigmoid(self.high_gate)
        self.last_gate = gate.detach()
        return low + gate * (band + local) + self.fuse(torch.cat([low, band, local], dim=1))


class HFCFNOBlock1d(nn.Module):
    def __init__(self, channels: int, modes: int, high_modes: int, length: int, local_activation: str = "gelu"):
        super().__init__()
        self.low = CFNOBlock1d(channels, modes, length)
        self.band = MultiBandSpectralConv1d(channels, channels, modes, high_modes, length)
        self.local_high = LocalHighPassBlock1d(channels, activation=local_activation)
        self.fuse = nn.Conv1d(3 * channels, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(-0.5))
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.low(x)
        band = self.band(x)
        local = self.local_high(x)
        gate = torch.sigmoid(self.high_gate)
        self.last_gate = gate.detach()
        return low + gate * (band + local) + self.fuse(torch.cat([low, band, local], dim=1))


class NeuralOperator1d(nn.Module):
    def __init__(
        self,
        variant: str,
        input_channels: int,
        width: int,
        modes: int,
        high_modes: int,
        depth: int,
        head_hidden: int,
        length: int,
        local_activation: str = "gelu",
    ):
        super().__init__()
        self.variant = variant
        self.feature_grid = FourierFeatureGrid1d() if variant.startswith("HF_") else None
        lifted_channels = input_channels + (self.feature_grid.extra_channels if self.feature_grid else 0)
        self.lift = nn.Conv1d(lifted_channels, width, 1)
        blocks: list[nn.Module] = []
        for _ in range(depth):
            if variant == "FNO":
                blocks.append(SpectralConv1d(width, width, modes, length))
            elif variant == "CFNO":
                blocks.append(CFNOBlock1d(width, modes, length))
            elif variant == "HF_FNO":
                blocks.append(HFFNOBlock1d(width, modes, high_modes, length, local_activation))
            elif variant == "HF_CFNO":
                blocks.append(HFCFNOBlock1d(width, modes, high_modes, length, local_activation))
            else:
                raise ValueError(f"Unknown variant: {variant}")
        self.blocks = nn.ModuleList(blocks)
        self.local_paths = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        hidden = int(head_hidden)
        if hidden < 1:
            raise ValueError("head_hidden must be positive")
        self.head1 = nn.Conv1d(width, hidden, 1)
        self.head2 = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Native 1D operator expects [B,C,N], received {tuple(x.shape)}")
        if self.feature_grid is not None:
            x = self.feature_grid(x)
        x = self.lift(x)
        for block, local in zip(self.blocks, self.local_paths):
            x = F.gelu(block(x) + local(x))
        return self.head2(F.gelu(self.head1(x)))

    def high_pass_gate_summary(self) -> dict[str, float] | None:
        gates = [block.last_gate for block in self.blocks if getattr(block, "last_gate", None) is not None]
        if not gates:
            return None
        values = torch.stack(gates)
        return {
            "mean": float(values.mean().cpu()),
            "min": float(values.min().cpu()),
            "max": float(values.max().cpu()),
        }


class HardConstraint1d(nn.Module):
    def __init__(self, base: NeuralOperator1d, spec: Steady1DCase, scaling: FieldScaling1D):
        super().__init__()
        self.base = base
        self.spec = spec
        self.scaling = scaling
        coordinate = torch.linspace(0.0, 1.0, spec.nx, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("coordinate", coordinate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.scaling.denormalize_solution(self.base(x))
        xi = self.coordinate
        value_left = x[:, 1:2, :1]
        if self.spec.pde == "Poisson":
            value_right = x[:, 2:3, :1]
            lift = (1.0 - xi) * value_left + xi * value_right
            return lift + 4.0 * xi * (1.0 - xi) * correction
        slope_left_scaled = x[:, 2:3, :1]
        lift = value_left + slope_left_scaled * xi
        if self.spec.pde == "KdV":
            curvature_left_scaled = x[:, 3:4, :1]
            lift = lift + 0.5 * curvature_left_scaled * xi.square()
            return lift + xi.pow(3) * correction
        return lift + xi.square() * correction


def make_model(
    variant: str,
    width: int,
    head_hidden: int,
    spec: Steady1DCase,
    scaling: FieldScaling1D | None = None,
    local_activation: str = "gelu",
) -> nn.Module:
    base = NeuralOperator1d(
        variant=variant,
        input_channels=spec.input_channels,
        width=width,
        modes=spec.modes,
        high_modes=spec.high_modes,
        depth=spec.depth,
        head_hidden=head_hidden,
        length=spec.nx,
        local_activation=local_activation,
    )
    if scaling is None:
        return base
    return HardConstraint1d(base, spec, scaling)


def parameter_count_for_configuration(variant: str, width: int, head_hidden: int, spec: Steady1DCase) -> int:
    """Exact real-scalar parameter count without constructing temporary models."""
    width = int(width)
    hidden = int(head_hidden)
    depth = int(spec.depth)
    modes = int(spec.modes)
    high_modes = int(spec.high_modes)
    lifted_inputs = int(spec.input_channels) + (8 if variant.startswith("HF_") else 0)
    total = width * lifted_inputs + width
    total += depth * (width * width + width)  # Outer pointwise residual paths.
    total += hidden * width + hidden
    total += hidden + 1
    if variant == "FNO":
        block = 2 * width * width * modes
    elif variant == "CFNO":
        block = (3 * modes + 2) * width * width + width + 1
    elif variant == "HF_FNO":
        block = (4 * modes + 2 * high_modes + 5) * width * width + 7 * width + 1
    elif variant == "HF_CFNO":
        block = (5 * modes + 2 * high_modes + 7) * width * width + 8 * width + 2
    else:
        raise ValueError(variant)
    return int(total + depth * block)


def match_widths(spec: Steady1DCase, variants: tuple[str, ...]) -> dict[str, dict[str, int | str]]:
    def configurations(variant: str) -> list[tuple[int, int, int]]:
        by_parameters: dict[int, tuple[int, int, int]] = {}
        for width in range(1, 33):
            for head_hidden in range(1, 513):
                parameters = parameter_count_for_configuration(variant, width, head_hidden, spec)
                candidate = (parameters, width, head_hidden)
                previous = by_parameters.get(parameters)
                if previous is None or (width, -head_hidden) > (previous[1], -previous[2]):
                    by_parameters[parameters] = candidate
        return sorted(by_parameters.values())

    all_candidates = {variant: configurations(variant) for variant in variants}

    def closest_independent(variant: str) -> tuple[int, int, int]:
        return min(
            all_candidates[variant],
            key=lambda item: (abs(item[0] - spec.target_params), -item[1], item[2]),
        )

    def baseline_for(
        enhanced: tuple[int, int, int],
        baseline_candidates: list[tuple[int, int, int]],
        parameter_values: list[int],
    ) -> tuple[int, int, int] | None:
        enhanced_params = enhanced[0]
        target_index = bisect.bisect_left(parameter_values, max(enhanced_params, spec.target_params))
        lower_index = bisect.bisect_right(parameter_values, spec.target_params) - 1
        indices = {target_index, target_index - 1, lower_index, lower_index + 1}
        feasible = [
            baseline_candidates[index]
            for index in indices
            if 0 <= index < len(baseline_candidates) and baseline_candidates[index][0] >= enhanced_params
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda item: (abs(item[0] - spec.target_params), item[0] - enhanced_params, -item[1], item[2]))

    matched: dict[str, dict[str, int | str]] = {}
    paired_variants: set[str] = set()
    for baseline, enhanced in (("FNO", "HF_FNO"), ("CFNO", "HF_CFNO")):
        if baseline not in all_candidates or enhanced not in all_candidates:
            continue
        baseline_parameter_values = [item[0] for item in all_candidates[baseline]]
        pair_candidates = []
        for enhanced_configuration in all_candidates[enhanced]:
            baseline_configuration = baseline_for(
                enhanced_configuration,
                all_candidates[baseline],
                baseline_parameter_values,
            )
            if baseline_configuration is None:
                continue
            baseline_error = abs(baseline_configuration[0] - spec.target_params)
            enhanced_error = abs(enhanced_configuration[0] - spec.target_params)
            maximum_error = max(baseline_error, enhanced_error)
            tolerance = max(4, int(round(0.02 * spec.target_params)))
            score = (
                0 if maximum_error <= tolerance else 1,
                -min(baseline_configuration[1], enhanced_configuration[1]),
                -(baseline_configuration[1] + enhanced_configuration[1]),
                maximum_error,
                baseline_error + enhanced_error,
                baseline_configuration[0] - enhanced_configuration[0],
            )
            pair_candidates.append((score, baseline_configuration, enhanced_configuration))
        if not pair_candidates:
            raise RuntimeError(f"No feasible parameter-matched pair for {baseline}/{enhanced}")
        _, baseline_configuration, enhanced_configuration = min(pair_candidates, key=lambda item: item[0])
        for variant, configuration, partner in (
            (baseline, baseline_configuration, enhanced),
            (enhanced, enhanced_configuration, baseline),
        ):
            parameters, width, head_hidden = configuration
            matched[variant] = {
                "width": width,
                "head_hidden": head_hidden,
                "params": parameters,
                "matched_to": partner,
            }
            paired_variants.add(variant)
    for variant in variants:
        if variant in paired_variants:
            continue
        parameters, width, head_hidden = closest_independent(variant)
        matched[variant] = {"width": width, "head_hidden": head_hidden, "params": parameters}
    return matched


def build_cases(args: argparse.Namespace) -> list[Steady1DCase]:
    common = {
        "depth": args.depth,
        "target_params": args.target_params,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    baseline = [
        Steady1DCase("Poisson", "Steady1D", "baseline", 1.0, args.grid, args.modes, args.high_modes, input_channels=4, params={}, **common),
        Steady1DCase("KdV", "Steady1D", "baseline", 2.0 * math.pi, args.grid, args.modes, args.high_modes, input_channels=5, params={"nonlinear_coeff": 6.0}, **common),
        Steady1DCase("AllenCahn", "Steady1D", "baseline", 1.0, args.grid, args.modes, args.high_modes, input_channels=4, params={"epsilon": 0.03, "reaction": 1.0}, **common),
        Steady1DCase("Burgers", "Steady1D", "baseline", 1.0, args.grid, args.modes, args.high_modes, input_channels=4, params={"nu": 0.01, "convective_coeff": 1.0}, **common),
        Steady1DCase("ReactionDiffusion", "Steady1D", "baseline", 1.0, args.grid, args.modes, args.high_modes, input_channels=4, params={"diffusion": 0.01, "rate": 1.0}, **common),
    ]
    stress_common = {
        "mode_min": 4,
        "mode_max": 15,
        "mode_count": 5,
        "amp_scale": 0.80,
        "boundary_amplitude": 0.25,
    }

    def stress(**extra: float) -> dict[str, float | int | str]:
        result = dict(stress_common)
        result.update(extra)
        return result

    high = [
        Steady1DCase("KdV", "HighNonlinear_Steady1D", "high_nonlinear", 2.0 * math.pi, args.high_grid, args.high_profile_modes, args.high_profile_high_modes, input_channels=5, params=stress(nonlinear_coeff=12.0, boundary_amplitude=0.35), **common),
        Steady1DCase("AllenCahn", "HighNonlinear_Steady1D", "high_nonlinear", 1.0, args.high_grid, args.high_profile_modes, args.high_profile_high_modes, input_channels=4, params=stress(epsilon=0.015, reaction=2.0), **common),
        Steady1DCase("Burgers", "HighNonlinear_Steady1D", "high_nonlinear", 1.0, args.high_grid, args.high_profile_modes, args.high_profile_high_modes, input_channels=4, params=stress(nu=0.0025, convective_coeff=2.0), **common),
        Steady1DCase("ReactionDiffusion", "HighNonlinear_Steady1D", "high_nonlinear", 1.0, args.high_grid, args.high_profile_modes, args.high_profile_high_modes, input_channels=4, params=stress(diffusion=0.0025, rate=4.0), **common),
    ]
    cases = baseline + high
    if args.cases:
        wanted = {value.strip().lower() for value in args.cases.split(",") if value.strip()}
        cases = [
            spec
            for spec in cases
            if spec.case.lower() in wanted
            or spec.slug.lower() in wanted
            or f"{spec.pde}/{spec.case}".lower() in wanted
        ]
    return cases


def well_posedness(spec: Steady1DCase) -> dict[str, str | list[str]]:
    if spec.pde == "Poisson":
        return {
            "problem_type": "two-point Dirichlet boundary-value problem",
            "input_channels": ["source", "u_left", "u_right", "x/domain_x"],
            "hard_constraint": "linear endpoint lift + 4*xi*(1-xi)*network_correction",
            "uniqueness": "The difference of two solutions has zero second derivative and zero endpoint values, hence is identically zero.",
        }
    if spec.pde == "KdV":
        return {
            "problem_type": "third-order spatial Cauchy problem",
            "input_channels": ["source", "u_left", "domain_x*u_x_left", "domain_x^2*u_xx_left", "x/domain_x"],
            "hard_constraint": "left quadratic Taylor jet + xi^3*network_correction",
            "uniqueness": "The third-order ODE is a smooth first-order system after prescribing the signed left value, slope, and curvature; the manufactured global solution fixes its unique branch on the benchmark interval.",
        }
    return {
        "problem_type": "second-order spatial Cauchy problem",
        "input_channels": ["source", "u_left", "domain_x*u_x_left", "x/domain_x"],
        "hard_constraint": "left linear Taylor jet + xi^2*network_correction",
        "uniqueness": "The nonlinear second-order ODE is a smooth first-order system after prescribing the signed left value and slope; the manufactured global solution fixes its unique branch on the benchmark interval.",
    }


def pde_lhs_numpy(spec: Steady1DCase, u: np.ndarray, ux: np.ndarray, uxx: np.ndarray, uxxx: np.ndarray) -> np.ndarray:
    if spec.pde == "Poisson":
        return -uxx
    if spec.pde == "KdV":
        return uxxx + float(spec.params["nonlinear_coeff"]) * u * ux
    if spec.pde == "AllenCahn":
        epsilon = float(spec.params["epsilon"])
        reaction = float(spec.params["reaction"])
        return -(epsilon**2) * uxx - reaction * u + reaction * u**3
    if spec.pde == "Burgers":
        viscosity = float(spec.params["nu"])
        coefficient = float(spec.params["convective_coeff"])
        return coefficient * u * ux - viscosity * uxx
    if spec.pde == "ReactionDiffusion":
        diffusion = float(spec.params["diffusion"])
        rate = float(spec.params["rate"])
        return -diffusion * uxx - rate * u * (1.0 - u)
    raise ValueError(spec.pde)


def generate_dataset(spec: Steady1DCase, samples: int, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, spec.domain_x, spec.nx, dtype=np.float64)
    xhat = x / max(spec.domain_x, 1e-12)
    input_rows = []
    exact_rows = []
    source_rows = []
    for _ in range(samples):
        if spec.high_nonlinear:
            available = np.arange(int(spec.params["mode_min"]), int(spec.params["mode_max"]) + 1)
            modes = rng.choice(available, size=int(spec.params["mode_count"]), replace=False)
            amplitudes = rng.uniform(-float(spec.params["amp_scale"]), float(spec.params["amp_scale"]), size=len(modes))
            boundary_amplitude = float(spec.params["boundary_amplitude"])
            left, right = rng.uniform(-boundary_amplitude, boundary_amplitude, size=2)
            lift = left * (1.0 - xhat) + right * xhat
            lift_x = np.full_like(x, (right - left) / spec.domain_x)
        else:
            modes = rng.choice(np.arange(1, 7), size=3, replace=False)
            amplitudes = rng.uniform(-0.8, 0.8, size=3)
            lift = np.zeros_like(x)
            lift_x = np.zeros_like(x)
        u = lift.copy()
        ux = lift_x.copy()
        uxx = np.zeros_like(x)
        uxxx = np.zeros_like(x)
        for amplitude, mode in zip(amplitudes, modes):
            k = mode * math.pi / spec.domain_x
            sine = np.sin(k * x)
            cosine = np.cos(k * x)
            u += amplitude * sine
            ux += amplitude * k * cosine
            uxx += -amplitude * k**2 * sine
            uxxx += -amplitude * k**3 * cosine
        source = pde_lhs_numpy(spec, u, ux, uxx, uxxx)
        constant = lambda value: np.full_like(x, float(value))
        if spec.pde == "Poisson":
            channels = [source, constant(u[0]), constant(u[-1]), xhat]
        elif spec.pde == "KdV":
            channels = [
                source,
                constant(u[0]),
                constant(spec.domain_x * ux[0]),
                constant(spec.domain_x**2 * uxx[0]),
                xhat,
            ]
        else:
            channels = [source, constant(u[0]), constant(spec.domain_x * ux[0]), xhat]
        if len(channels) != spec.input_channels:
            raise AssertionError(f"{spec.slug}: expected {spec.input_channels} input channels, got {len(channels)}")
        input_rows.append(np.stack(channels, axis=0).astype(np.float32))
        exact_rows.append(u[None, :].astype(np.float32))
        source_rows.append(source[None, :].astype(np.float32))
    return {
        "x": torch.from_numpy(np.stack(input_rows)),
        "y": torch.from_numpy(np.stack(exact_rows)),
        "source": torch.from_numpy(np.stack(source_rows)),
    }


def split_and_scale_data(spec: Steady1DCase, args: argparse.Namespace, seed: int) -> tuple[dict[str, torch.Tensor], FieldScaling1D]:
    total = args.train_samples + args.val_samples + args.test_samples
    raw = generate_dataset(spec, total, seed)
    n_train = args.train_samples
    n_val = args.val_samples
    source_mean = float(raw["source"][:n_train].mean())
    source_std = float(raw["source"][:n_train].std().clamp_min(1e-6))
    scaling = FieldScaling1D(source_mean=source_mean, source_std=source_std)
    inputs = raw["x"].clone()
    inputs[:, 0:1] = scaling.normalize_source(inputs[:, 0:1])
    return {
        "x_train": inputs[:n_train],
        "y_train": raw["y"][:n_train],
        "source_train": raw["source"][:n_train],
        "x_val": inputs[n_train : n_train + n_val],
        "y_val": raw["y"][n_train : n_train + n_val],
        "source_val": raw["source"][n_train : n_train + n_val],
        "x_test": inputs[n_train + n_val :],
        "y_test": raw["y"][n_train + n_val :],
        "source_test": raw["source"][n_train + n_val :],
    }, scaling


def first_derivative(values: torch.Tensor, spacing: float) -> torch.Tensor:
    result = torch.zeros_like(values)
    result[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / (2.0 * spacing)
    result[..., 0] = (values[..., 1] - values[..., 0]) / spacing
    result[..., -1] = (values[..., -1] - values[..., -2]) / spacing
    return result


def second_derivative(values: torch.Tensor, spacing: float) -> torch.Tensor:
    result = torch.zeros_like(values)
    result[..., 1:-1] = (values[..., 2:] - 2.0 * values[..., 1:-1] + values[..., :-2]) / spacing**2
    result[..., 0] = result[..., 1]
    result[..., -1] = result[..., -2]
    return result


def third_derivative(values: torch.Tensor, spacing: float) -> torch.Tensor:
    return first_derivative(second_derivative(values, spacing), spacing)


def pde_lhs_torch(spec: Steady1DCase, u: torch.Tensor) -> torch.Tensor:
    ux = first_derivative(u, spec.dx)
    uxx = second_derivative(u, spec.dx)
    uxxx = third_derivative(u, spec.dx)
    if spec.pde == "Poisson":
        return -uxx
    if spec.pde == "KdV":
        return uxxx + float(spec.params["nonlinear_coeff"]) * u * ux
    if spec.pde == "AllenCahn":
        epsilon = float(spec.params["epsilon"])
        reaction = float(spec.params["reaction"])
        return -(epsilon**2) * uxx - reaction * u + reaction * u**3
    if spec.pde == "Burgers":
        return float(spec.params["convective_coeff"]) * u * ux - float(spec.params["nu"]) * uxx
    if spec.pde == "ReactionDiffusion":
        return -float(spec.params["diffusion"]) * uxx - float(spec.params["rate"]) * u * (1.0 - u)
    raise ValueError(spec.pde)


def cropped_residual(spec: Steady1DCase, prediction: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    residual = pde_lhs_torch(spec, prediction) - source
    padding = 3 if spec.pde == "KdV" else 2
    return residual[..., padding:-padding]


def residual_losses(
    spec: Steady1DCase,
    prediction: torch.Tensor,
    source: torch.Tensor,
    source_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = cropped_residual(spec, prediction, source)
    absolute = torch.mean(residual**2)
    normalized = torch.mean((residual / source_scale.clamp_min(1.0)) ** 2)
    return normalized, absolute


def relative_l2(prediction: torch.Tensor, theory: torch.Tensor) -> torch.Tensor:
    numerator = torch.sqrt(torch.sum((prediction - theory).flatten(1).square(), dim=1))
    denominator = torch.sqrt(torch.sum(theory.flatten(1).square(), dim=1)).clamp_min(1e-12)
    return torch.mean(numerator / denominator)


def compose_loss(mode: str, data_mse: torch.Tensor, pde_mse: torch.Tensor, physics_weight: float) -> torch.Tensor:
    if mode == "data":
        return data_mse
    if mode == "pde":
        return pde_mse
    if mode == "hybrid":
        return data_mse + physics_weight * pde_mse
    raise ValueError(mode)


def selection_key(mode: str) -> str:
    return {"data": "val_mse", "pde": "val_pde_mse", "hybrid": "val_loss"}[mode]


def metric_summary(array: np.ndarray) -> dict[str, float]:
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "mean_abs": float(np.mean(np.abs(flat))),
        "rms": float(np.sqrt(np.mean(flat**2))),
        "p95_abs": float(np.percentile(np.abs(flat), 95)),
        "max_abs": float(np.max(np.abs(flat))),
    }


def numerical_residual(spec: Steady1DCase, field: np.ndarray, source: np.ndarray) -> np.ndarray:
    ux = np.gradient(field, spec.dx, edge_order=2)
    uxx = np.gradient(ux, spec.dx, edge_order=2)
    uxxx = np.gradient(uxx, spec.dx, edge_order=2)
    return pde_lhs_numpy(spec, field, ux, uxx, uxxx) - source


def style_axis(axis: plt.Axes, legend: bool = False) -> None:
    axis.grid(True, which="both", color="#c8c8c8", alpha=0.45, linewidth=0.7)
    axis.tick_params(axis="both", labelsize=10.5)
    if legend:
        axis.legend(fontsize=10.5, frameon=True, edgecolor="#c0c0c0")


def close_figure(figure: plt.Figure) -> None:
    plt.close(figure)
    gc.collect()


def safe_log(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=np.float64))
    positive = values[np.isfinite(values) & (values > 0.0)]
    floor = max(float(positive.min()) * 0.1, 1e-16) if positive.size else 1e-16
    return np.where(np.isfinite(values), np.maximum(values, floor), np.nan)


def first_five_scale(history: list[dict]) -> float:
    rows = history[: min(5, len(history))]
    values = []
    for row in rows:
        for key, value in row.items():
            if key != "epoch" and ("loss" in key or "mse" in key):
                value = float(value)
                if math.isfinite(value):
                    values.append(abs(value))
    scale = max(values, default=1.0)
    return scale if scale > 0.0 else 1.0


HISTORY_PANELS = (
    ("objective", "train_loss", "val_loss", "test_loss", "total objective"),
    ("data", "train_mse", "val_mse", "test_mse", "data MSE"),
    ("physics", "train_pde_mse", "val_pde_mse", "test_pde_mse", "normalized PDE residual MSE"),
    (
        "absolute",
        "train_pde_mse_absolute",
        "val_pde_mse_absolute",
        "test_pde_mse_absolute",
        "absolute PDE residual MSE",
    ),
)


def draw_history_axis(axis: plt.Axes, history: list[dict], panel: tuple[str, str, str, str, str], scale: float | None) -> None:
    _, train_key, val_key, test_key, ylabel = panel
    epochs = np.asarray([row["epoch"] for row in history])
    for label, key in (("train", train_key), ("validation", val_key), ("test", test_key)):
        values = np.asarray([row[key] for row in history], dtype=np.float64)
        if scale is not None:
            values = values / scale
        axis.semilogy(
            epochs,
            safe_log(values),
            label=label,
            color=SERIES_COLOR[label],
            linestyle=SERIES_STYLE[label],
        )
    axis.set_xlabel("epoch", fontsize=12)
    axis.set_ylabel(("normalized " if scale is not None else "") + ylabel, fontsize=12)
    style_axis(axis, legend=True)


def plot_training_history(history: list[dict], output_dir: Path, title: str) -> None:
    configure_plotting()
    scale = first_five_scale(history)
    for normalized, suffix in ((True, "normalized"), (False, "absolute")):
        fig, axes = plt.subplots(1, len(HISTORY_PANELS), figsize=(4.25 * len(HISTORY_PANELS), 4.6), constrained_layout=True)
        for axis, panel in zip(np.atleast_1d(axes), HISTORY_PANELS):
            axis.set_title(panel[0])
            draw_history_axis(axis, history, panel, scale if normalized else None)
        fig.suptitle(f"{title}; first-five scale = {scale:.3e}" if normalized else title)
        fig.savefig(output_dir / f"training_convergence_{suffix}.png")
        if normalized:
            fig.savefig(output_dir / "training_convergence.png")
        close_figure(fig)
    for panel in HISTORY_PANELS:
        for normalized, suffix in ((True, "normalized"), (False, "absolute")):
            fig, axis = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
            draw_history_axis(axis, history, panel, scale if normalized else None)
            axis.set_title(f"{title} / {panel[0]} ({suffix})")
            fig.savefig(output_dir / f"training_convergence_{panel[0]}_{suffix}.png")
            if normalized and panel[0] in ("objective", "data", "physics"):
                fig.savefig(output_dir / f"training_convergence_{panel[0]}.png")
            close_figure(fig)
    (output_dir / "plot_normalization.json").write_text(
        json.dumps(
            {
                "definition": "Largest absolute loss/MSE component among the first five epochs.",
                "first_epochs_used": min(5, len(history)),
                "model_first_five_scale": scale,
                "font": "Times New Roman",
                "axis_label_size": 12,
                "legend_size": 10.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_fields(
    spec: Steady1DCase,
    variant: str,
    prediction: np.ndarray,
    theory: np.ndarray,
    source: np.ndarray,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    configure_plotting()
    plt.close("all")
    gc.collect()
    x = np.linspace(0.0, spec.domain_x, spec.nx)
    signed_error = prediction - theory
    absolute_error = np.abs(signed_error)
    prediction_residual = numerical_residual(spec, prediction, source)
    theory_residual = numerical_residual(spec, theory, source)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    axes[0].plot(x, theory, label="Theory", color=PRIMARY[0])
    axes[0].plot(x, prediction, label="Prediction", color=PRIMARY[1], linestyle="--")
    axes[0].set_ylabel("u", fontsize=12)
    axes[0].set_title("Prediction and theory")
    style_axis(axes[0], legend=True)
    axes[1].plot(x, signed_error, color=PRIMARY[2])
    axes[1].axhline(0.0, color="#606060", linewidth=0.8)
    axes[1].set_ylabel(r"$u_\theta-u_{\mathrm{th}}$", fontsize=12)
    axes[1].set_title("Signed error")
    style_axis(axes[1])
    axes[2].plot(x, absolute_error, color=PRIMARY[3])
    axes[2].fill_between(x, 0.0, absolute_error, color=SECONDARY[3], alpha=0.35)
    axes[2].set_ylabel(r"$|u_\theta-u_{\mathrm{th}}|$", fontsize=12)
    axes[2].set_title("Absolute error")
    style_axis(axes[2])
    for axis in axes:
        axis.set_xlabel("x", fontsize=12)
    fig.suptitle(f"{spec.pde} / {spec.case} / {variant}")
    fig.savefig(output_dir / "prediction_exact_error.png")
    fig.savefig(output_dir / "prediction_exact_error_line.png")
    close_figure(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axes[0].plot(x, prediction_residual, color=PRIMARY[1])
    axes[0].set_title("Prediction residual")
    axes[1].plot(x, theory_residual, color=PRIMARY[0])
    axes[1].set_title("Theory residual")
    for axis in axes:
        axis.axhline(0.0, color="#606060", linewidth=0.8)
        axis.set_xlabel("x", fontsize=12)
        axis.set_ylabel("PDE residual", fontsize=12)
        style_axis(axis)
    fig.suptitle(f"{spec.pde} / {spec.case} / {variant}")
    fig.savefig(output_dir / "residual_distribution.png")
    close_figure(fig)

    prediction_ft = np.fft.rfft(prediction, norm="forward")
    theory_ft = np.fft.rfft(theory, norm="forward")
    coefficient_error = np.abs(prediction_ft - theory_ft)
    wavenumber = np.arange(len(prediction_ft))
    fig, axis = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    axis.semilogy(wavenumber, safe_log(np.abs(theory_ft)), label="Theory magnitude", color=PRIMARY[0])
    axis.semilogy(wavenumber, safe_log(np.abs(prediction_ft)), label="Prediction magnitude", color=PRIMARY[1], linestyle="--")
    axis.semilogy(wavenumber, safe_log(coefficient_error), label="Absolute coefficient error", color=PRIMARY[2], linestyle=":")
    axis.set_xlabel("spatial wavenumber", fontsize=12)
    axis.set_ylabel("Fourier coefficient magnitude", fontsize=12)
    axis.set_title(f"{spec.pde} / {spec.case} / {variant} spectrum")
    style_axis(axis, legend=True)
    fig.savefig(output_dir / "spectrum_analysis.png")
    close_figure(fig)

    fig, axis = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    axis.semilogy(wavenumber, safe_log(coefficient_error), color=PRIMARY[3], label=r"$|\hat{u}_\theta-\hat{u}_{\mathrm{th}}|$")
    axis.set_xlabel("spatial wavenumber", fontsize=12)
    axis.set_ylabel("absolute Fourier coefficient error", fontsize=12)
    axis.set_title(f"{spec.pde} / {spec.case} / {variant} spectral error")
    style_axis(axis, legend=True)
    fig.savefig(output_dir / "frequency_component_abs_error.png")
    close_figure(fig)
    with (output_dir / "frequency_component_abs_error.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("spatial_wavenumber", "prediction_magnitude", "theory_magnitude", "absolute_coefficient_error"))
        writer.writerows(zip(wavenumber, np.abs(prediction_ft), np.abs(theory_ft), coefficient_error))
    return {
        "prediction_residual": prediction_residual,
        "theory_residual": theory_residual,
        "coefficient_error": coefficient_error,
    }


def write_frequency_error_csv(prediction: np.ndarray, theory: np.ndarray, output_dir: Path) -> None:
    prediction_ft = np.fft.rfft(prediction, norm="forward")
    theory_ft = np.fft.rfft(theory, norm="forward")
    coefficient_error = np.abs(prediction_ft - theory_ft)
    wavenumber = np.arange(len(prediction_ft))
    with (output_dir / "frequency_component_abs_error.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("spatial_wavenumber", "prediction_magnitude", "theory_magnitude", "absolute_coefficient_error"))
        writer.writerows(zip(wavenumber, np.abs(prediction_ft), np.abs(theory_ft), coefficient_error))


def evaluate_model(
    spec: Steady1DCase,
    variant: str,
    model: HardConstraint1d,
    data: dict[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
    make_plots: bool,
) -> dict:
    model.eval()
    x_test = data["x_test"].to(device)
    theory = data["y_test"].to(device)
    source = data["source_test"].to(device)
    source_scale = data["source_train"].to(device).std().clamp_min(1.0)
    with torch.no_grad():
        prediction = model(x_test)
        pde_mse, pde_mse_absolute = residual_losses(spec, prediction, source, source_scale)
        test_mse = F.mse_loss(prediction, theory)
        test_rel_l2 = relative_l2(prediction, theory)
    sample_prediction = prediction[0, 0].cpu().numpy()
    sample_theory = theory[0, 0].cpu().numpy()
    sample_source = source[0, 0].cpu().numpy()
    signed_error = sample_prediction - sample_theory
    prediction_residual = numerical_residual(spec, sample_prediction, sample_source)
    theory_residual = numerical_residual(spec, sample_theory, sample_source)
    np.savez_compressed(
        output_dir / "sample_fields.npz",
        prediction=sample_prediction,
        exact=sample_theory,
        abs_error=np.abs(signed_error),
        signed_error=signed_error,
        residual_prediction=prediction_residual,
        residual_exact=theory_residual,
        source=sample_source,
        x=np.linspace(0.0, spec.domain_x, spec.nx),
    )
    write_frequency_error_csv(sample_prediction, sample_theory, output_dir)
    deferred_plot_error = "deferred by --skip-plots" if not make_plots else None
    if make_plots:
        try:
            plot_fields(spec, variant, sample_prediction, sample_theory, sample_source, output_dir)
        except MemoryError as error:
            plt.close("all")
            gc.collect()
            deferred_plot_error = repr(error)
            (output_dir / "PLOTTING_DEFERRED.json").write_text(
                json.dumps({"reason": deferred_plot_error}, indent=2),
                encoding="utf-8",
            )
            print(f"Deferred plots for {spec.slug}/{variant}: {error}", flush=True)
    return {
        "test_pde_mse": float(pde_mse.cpu()),
        "test_pde_mse_absolute": float(pde_mse_absolute.cpu()),
        "test_mse": float(test_mse.cpu()),
        "test_rel_l2": float(test_rel_l2.cpu()),
        "abs_error": metric_summary(np.abs(signed_error)),
        "signed_error": metric_summary(signed_error),
        "residual_prediction": metric_summary(prediction_residual),
        "residual_exact": metric_summary(theory_residual),
        "high_pass_gate": model.base.high_pass_gate_summary(),
        "plotting_deferred_error": deferred_plot_error,
    }


def train_one(
    spec: Steady1DCase,
    variant: str,
    mode: str,
    width: int,
    head_hidden: int,
    data: dict[str, torch.Tensor],
    scaling: FieldScaling1D,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(seed)
    model = make_model(variant, width, head_hidden, spec, scaling, args.local_activation).to(device)
    assert isinstance(model, HardConstraint1d)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.lr,
        weight_decay=1e-5,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=spec.lr * 0.05,
    )
    x_train = data["x_train"].to(device)
    y_train = data["y_train"].to(device)
    source_train = data["source_train"].to(device)
    x_val = data["x_val"].to(device)
    y_val = data["y_val"].to(device)
    source_val = data["source_val"].to(device)
    x_test = data["x_test"].to(device)
    y_test = data["y_test"].to(device)
    source_test = data["source_test"].to(device)
    source_scale = source_train.std().clamp_min(1.0)
    key = selection_key(mode)
    best = {"epoch": -1, key: float("inf"), "selection_metric": key}
    best_state = None
    history: list[dict] = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(x_train.shape[0], device=device)
        sums = {"loss": 0.0, "mse": 0.0, "pde": 0.0, "pde_absolute": 0.0, "relative": 0.0}
        seen = 0
        for start in range(0, x_train.shape[0], spec.batch_size):
            batch_indices = permutation[start : start + spec.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(batch_indices), args.micro_batch_size):
                indices = batch_indices[micro_start : micro_start + args.micro_batch_size]
                prediction = model(x_train[indices])
                # Theory participates in the graph only for data and hybrid modes.
                if mode == "pde":
                    data_mse_for_loss = prediction.new_zeros(())
                else:
                    data_mse_for_loss = F.mse_loss(prediction, y_train[indices]) / scaling.solution_scale**2
                pde_mse, pde_mse_absolute = residual_losses(
                    spec,
                    prediction,
                    source_train[indices],
                    source_scale,
                )
                loss = compose_loss(mode, data_mse_for_loss, pde_mse, args.physics_weight)
                micro_weight = len(indices) / max(len(batch_indices), 1)
                (micro_weight * loss).backward()
                with torch.no_grad():
                    diagnostic_mse = F.mse_loss(prediction, y_train[indices])
                    diagnostic_relative = relative_l2(prediction.detach(), y_train[indices])
                micro_size = len(indices)
                sums["loss"] += float(loss.detach().cpu()) * micro_size
                sums["mse"] += float(diagnostic_mse.cpu()) * micro_size
                sums["pde"] += float(pde_mse.detach().cpu()) * micro_size
                sums["pde_absolute"] += float(pde_mse_absolute.detach().cpu()) * micro_size
                sums["relative"] += float(diagnostic_relative.cpu()) * micro_size
                seen += micro_size
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val)
            val_data_mse = F.mse_loss(val_prediction, y_val)
            val_pde, val_pde_absolute = residual_losses(spec, val_prediction, source_val, source_scale)
            val_loss = compose_loss(mode, val_data_mse, val_pde, args.physics_weight)
            test_prediction = model(x_test)
            test_data_mse = F.mse_loss(test_prediction, y_test)
            test_pde, test_pde_absolute = residual_losses(spec, test_prediction, source_test, source_scale)
            test_loss = compose_loss(mode, test_data_mse, test_pde, args.physics_weight)
            val_relative = relative_l2(val_prediction, y_val)
            test_relative = relative_l2(test_prediction, y_test)
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / seen,
            "train_mse": sums["mse"] / seen,
            "train_pde_mse": sums["pde"] / seen,
            "train_pde_mse_absolute": sums["pde_absolute"] / seen,
            "train_state_rel_l2": sums["relative"] / seen,
            "val_loss": float(val_loss.cpu()),
            "val_mse": float(val_data_mse.cpu()),
            "val_pde_mse": float(val_pde.cpu()),
            "val_pde_mse_absolute": float(val_pde_absolute.cpu()),
            "val_state_rel_l2": float(val_relative.cpu()),
            "test_loss": float(test_loss.cpu()),
            "test_mse": float(test_data_mse.cpu()),
            "test_pde_mse": float(test_pde.cpu()),
            "test_pde_mse_absolute": float(test_pde_absolute.cpu()),
            "test_state_rel_l2": float(test_relative.cpu()),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if row[key] < best[key]:
            best = dict(row)
            best["selection_metric"] = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"[{MODE_DIR[mode]}] {spec.slug}/{variant} epoch {epoch:04d}/{args.epochs} "
                f"loss={row['train_loss']:.3e}/{row['val_loss']:.3e} "
                f"data={row['val_mse']:.3e} pde={row['val_pde_mse']:.3e} "
                f"relL2={row['val_state_rel_l2']:.3e}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    elapsed = time.perf_counter() - start_time
    checkpoint_state = best_state if best_state is not None else {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "variant": variant,
            "training_mode": mode,
            "spec": asdict(spec),
            "normalization": asdict(scaling),
            "width": width,
            "head_hidden": head_hidden,
            "params": parameters,
            "best": best,
            "state_dict": checkpoint_state,
        },
        output_dir / "best_checkpoint.pt",
    )
    del optimizer, scheduler, best_state, checkpoint_state
    del x_train, y_train, source_train, x_val, y_val, source_val, x_test, y_test, source_test, source_scale
    gc.collect()
    torch.cuda.empty_cache()
    metrics = evaluate_model(spec, variant, model, data, output_dir, device, not args.skip_plots)
    metrics.update(
        {
            "variant": variant,
            "training_mode": mode,
            "width": width,
            "head_hidden": head_hidden,
            "params": parameters,
            "best": best,
            "final": history[-1],
            "elapsed_s": elapsed,
            "physics_weight": args.physics_weight,
            "tensor_contract": "[B,C,N]",
            "native_dimension": 1,
            "logical_batch_size": spec.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "normalization": asdict(scaling),
            "pure_physics_exact_solution_in_objective": False if mode == "pde" else None,
        }
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.skip_plots:
        try:
            plot_training_history(history, output_dir, f"{spec.pde} / {spec.case} / {variant} / {MODE_DIR[mode]}")
        except MemoryError as error:
            plt.close("all")
            gc.collect()
            (output_dir / "PLOTTING_DEFERRED.json").write_text(
                json.dumps({"reason": repr(error)}, indent=2),
                encoding="utf-8",
            )
            print(f"Deferred convergence plots for {spec.slug}/{variant}: {error}", flush=True)
        plt.close("all")
    gc.collect()
    return metrics


def history_from_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: int(value) if key == "epoch" else float(value) for key, value in row.items()} for row in rows]


def plot_case_comparison(case_dir: Path, records: list[dict]) -> None:
    configure_plotting()
    histories = {record["variant"]: history_from_csv(case_dir / record["variant"] / "history.csv") for record in records}
    shared_scale = max(first_five_scale(history) for history in histories.values())
    for normalized, suffix in ((True, "normalized"), (False, "absolute")):
        fig, axes = plt.subplots(1, len(HISTORY_PANELS), figsize=(4.3 * len(HISTORY_PANELS), 4.7), constrained_layout=True)
        for axis, panel in zip(np.atleast_1d(axes), HISTORY_PANELS):
            _, _, val_key, _, ylabel = panel
            for variant in MODEL_ORDER:
                if variant not in histories:
                    continue
                history = histories[variant]
                epochs = np.asarray([row["epoch"] for row in history])
                values = np.asarray([row[val_key] for row in history])
                if normalized:
                    values = values / shared_scale
                parameters = next(record["params"] for record in records if record["variant"] == variant)
                axis.semilogy(epochs, safe_log(values), color=MODEL_COLOR[variant], label=f"{variant} ({parameters:,})")
            axis.set_title(panel[0])
            axis.set_xlabel("epoch", fontsize=12)
            axis.set_ylabel(("normalized " if normalized else "") + f"validation {ylabel}", fontsize=12)
            style_axis(axis, legend=True)
        fig.suptitle(
            f"{case_dir.parent.name} / {case_dir.name}; shared first-five scale = {shared_scale:.3e}"
            if normalized
            else f"{case_dir.parent.name} / {case_dir.name}"
        )
        fig.savefig(case_dir / f"comparison_convergence_{suffix}.png")
        if normalized:
            fig.savefig(case_dir / "comparison_convergence.png")
        close_figure(fig)

    for panel in HISTORY_PANELS[:3]:
        _, _, val_key, _, ylabel = panel
        fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for variant in MODEL_ORDER:
            if variant not in histories:
                continue
            history = histories[variant]
            epochs = np.asarray([row["epoch"] for row in history])
            values = np.asarray([row[val_key] for row in history], dtype=np.float64) / shared_scale
            parameters = next(record["params"] for record in records if record["variant"] == variant)
            axis.semilogy(
                epochs,
                safe_log(values),
                color=MODEL_COLOR[variant],
                label=f"{variant} ({parameters:,})",
            )
        axis.set_xlabel("epoch", fontsize=12)
        axis.set_ylabel(f"normalized validation {ylabel}", fontsize=12)
        axis.set_title(f"{case_dir.parent.name} / {case_dir.name} / {panel[0]}")
        style_axis(axis, legend=True)
        fig.savefig(case_dir / f"comparison_convergence_{panel[0]}.png")
        close_figure(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for record in records:
        spectrum_path = case_dir / record["variant"] / "frequency_component_abs_error.csv"
        spectrum = np.genfromtxt(spectrum_path, delimiter=",", names=True)
        axis.semilogy(
            spectrum["spatial_wavenumber"],
            safe_log(spectrum["absolute_coefficient_error"]),
            color=MODEL_COLOR[record["variant"]],
            label=f"{record['variant']} ({record['params']:,})",
        )
    axis.set_xlabel("spatial wavenumber", fontsize=12)
    axis.set_ylabel("absolute Fourier coefficient error", fontsize=12)
    axis.set_title(f"{case_dir.parent.name} / {case_dir.name} spectral error comparison")
    style_axis(axis, legend=True)
    fig.savefig(case_dir / "comparison_frequency_component_abs_error.png")
    close_figure(fig)

    fig, axis = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    axis.bar(
        [record["variant"] for record in records],
        [max(float(record["test_rel_l2"]), EPS) for record in records],
        color=[MODEL_COLOR[record["variant"]] for record in records],
        edgecolor="#404040",
        linewidth=0.6,
    )
    axis.set_yscale("log")
    axis.set_xlabel("model", fontsize=12)
    axis.set_ylabel("test relative L2", fontsize=12)
    axis.set_title(f"{case_dir.parent.name} / {case_dir.name} model comparison")
    style_axis(axis)
    fig.savefig(case_dir / "comparison_test_rel_l2.png")
    close_figure(fig)
    (case_dir / "comparison_normalization.json").write_text(
        json.dumps(
            {
                "definition": "Case comparison curves use the largest first-five all-loss scale among the four models.",
                "shared_case_scale": shared_scale,
                "model_scales": {variant: first_five_scale(history) for variant, history in histories.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_pde_frequency_comparisons(mode_root: Path, mode_rows: list[dict]) -> None:
    configure_plotting()
    pdes = list(dict.fromkeys(row["pde"] for row in mode_rows))
    for pde in pdes:
        pde_rows = [row for row in mode_rows if row["pde"] == pde]
        cases = list(dict.fromkeys(row["case"] for row in pde_rows))
        fig, axes = plt.subplots(
            1,
            len(cases),
            figsize=(8.2 * len(cases), 4.8),
            constrained_layout=True,
            squeeze=False,
        )
        for axis, case in zip(axes.ravel(), cases):
            case_rows = [row for row in pde_rows if row["case"] == case]
            for variant in MODEL_ORDER:
                record = next((row for row in case_rows if row["variant"] == variant), None)
                if record is None:
                    continue
                spectrum_path = mode_root / pde / case / variant / "frequency_component_abs_error.csv"
                spectrum = np.genfromtxt(spectrum_path, delimiter=",", names=True)
                axis.semilogy(
                    spectrum["spatial_wavenumber"],
                    safe_log(spectrum["absolute_coefficient_error"]),
                    color=MODEL_COLOR[variant],
                    label=f"{variant} ({record['params']:,})",
                )
            axis.set_xlabel("spatial wavenumber", fontsize=12)
            axis.set_ylabel("absolute Fourier coefficient error", fontsize=12)
            axis.set_title(case)
            style_axis(axis, legend=True)
        fig.suptitle(f"{pde}: frequency-component error comparison")
        fig.savefig(mode_root / pde / "frequency_component_error_comparison.png")
        close_figure(fig)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def summary_row(spec: Steady1DCase, mode: str, metrics: dict) -> dict:
    return {
        "training_mode": mode,
        "mode_directory": MODE_DIR[mode],
        "pde": spec.pde,
        "case": spec.case,
        "profile": spec.profile,
        "variant": metrics["variant"],
        "params": metrics["params"],
        "width": metrics["width"],
        "head_hidden": metrics["head_hidden"],
        "best_epoch": metrics["best"]["epoch"],
        "selection_metric": metrics["best"]["selection_metric"],
        "test_mse": metrics["test_mse"],
        "test_rel_l2": metrics["test_rel_l2"],
        "test_pde_mse": metrics["test_pde_mse"],
        "test_pde_mse_absolute": metrics["test_pde_mse_absolute"],
        "error_mean_abs": metrics["abs_error"]["mean_abs"],
        "error_max_abs": metrics["abs_error"]["max_abs"],
        "residual_mean_abs": metrics["residual_prediction"]["mean_abs"],
        "residual_max_abs": metrics["residual_prediction"]["max_abs"],
        "elapsed_s": metrics["elapsed_s"],
    }


def plot_global_heatmap(root: Path, rows: list[dict]) -> None:
    configure_plotting()
    case_names = []
    column_names = []
    for row in rows:
        case_name = f"{row['pde']}/{row['case']}"
        column_name = f"{MODE_DIR[row['training_mode']]}/{row['variant']}"
        if case_name not in case_names:
            case_names.append(case_name)
        if column_name not in column_names:
            column_names.append(column_name)
    matrix = np.full((len(case_names), len(column_names)), np.nan)
    for row in rows:
        matrix[case_names.index(f"{row['pde']}/{row['case']}"), column_names.index(f"{MODE_DIR[row['training_mode']]}/{row['variant']}")] = float(row["test_rel_l2"])
    finite = matrix[np.isfinite(matrix)]
    vmin = max(float(finite.min()), EPS)
    vmax = max(float(finite.max()), vmin * (1.0 + 1e-12))
    fig, axis = plt.subplots(figsize=(15.5, max(6.0, 0.46 * len(case_names) + 2.0)), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
    axis.set_xticks(np.arange(len(column_names)), labels=column_names, rotation=40, ha="right")
    axis.set_yticks(np.arange(len(case_names)), labels=case_names)
    axis.set_xlabel("training mode / native 1D operator", fontsize=12)
    axis.set_ylabel("PDE / case", fontsize=12)
    axis.set_title("Native steady 1D global test relative L2 comparison")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86, pad=0.025)
    colorbar.ax.tick_params(labelsize=10.5)
    colorbar.set_label("test relative L2", fontsize=10.5)
    fig.savefig(root / "global_test_rel_l2_heatmap.png")
    close_figure(fig)


def plot_cross_mode_comparison(root: Path, rows: list[dict]) -> None:
    configure_plotting()
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["pde"], row["case"]), []).append(row)
    for (pde, case), group in grouped.items():
        output_dir = root / "CrossModeComparisons" / pde / case
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4.1 * len(MODEL_ORDER), 4.6), constrained_layout=True)
        for axis, variant in zip(np.atleast_1d(axes), MODEL_ORDER):
            values = []
            labels = []
            colors = []
            for mode in MODE_ORDER:
                record = next((row for row in group if row["variant"] == variant and row["training_mode"] == mode), None)
                if record is None:
                    continue
                labels.append(MODE_DIR[mode])
                values.append(max(float(record["test_rel_l2"]), EPS))
                colors.append(MODE_COLOR[mode])
            axis.bar(labels, values, color=colors, edgecolor="#404040", linewidth=0.6)
            axis.set_yscale("log")
            axis.set_title(variant)
            axis.set_xlabel("training mode", fontsize=12)
            axis.set_ylabel("test relative L2", fontsize=12)
            axis.tick_params(axis="x", rotation=25)
            style_axis(axis)
        fig.suptitle(f"{pde} / {case}: training-mode comparison")
        fig.savefig(output_dir / "training_mode_test_rel_l2_comparison.png")
        close_figure(fig)


def validate_plot_inventory(
    root: Path,
    cases: list[Steady1DCase],
    modes: tuple[str, ...],
    variants: tuple[str, ...],
) -> dict:
    common_model_plots = {
        "prediction_exact_error.png",
        "prediction_exact_error_line.png",
        "residual_distribution.png",
        "spectrum_analysis.png",
        "frequency_component_abs_error.png",
        "training_convergence.png",
        "training_convergence_normalized.png",
        "training_convergence_absolute.png",
        "training_convergence_physics.png",
    }
    common_case_plots = {
        "comparison_convergence.png",
        "comparison_convergence_normalized.png",
        "comparison_convergence_absolute.png",
        "comparison_convergence_physics.png",
        "comparison_frequency_component_abs_error.png",
        "comparison_test_rel_l2.png",
    }
    data_model_plots = {"training_convergence_objective.png", "training_convergence_data.png"}
    data_case_plots = {"comparison_convergence_objective.png", "comparison_convergence_data.png"}
    missing: list[str] = []
    expected: list[str] = []

    def require(path: Path) -> None:
        relative = str(path.relative_to(root))
        expected.append(relative)
        if not path.is_file():
            missing.append(relative)

    for mode in modes:
        mode_root = root / MODE_DIR[mode]
        require(mode_root / "global_test_rel_l2_heatmap.png")
        for pde in dict.fromkeys(spec.pde for spec in cases):
            require(mode_root / pde / "frequency_component_error_comparison.png")
        for spec in cases:
            case_dir = mode_root / spec.pde / spec.case
            case_plots = set(common_case_plots)
            if mode in ("hybrid", "data"):
                case_plots.update(data_case_plots)
            for filename in sorted(case_plots):
                require(case_dir / filename)
            for variant in variants:
                output_dir = case_dir / variant
                model_plots = set(common_model_plots)
                if mode in ("hybrid", "data"):
                    model_plots.update(data_model_plots)
                for filename in sorted(model_plots):
                    require(output_dir / filename)
    report = {
        "reference_roots": [
            "LegacyHF_PDE_Benchmark_PhysicsOnly_20260705",
            "LegacyHF_PDE_Benchmark_DataOnly_20260705",
            "LegacyHF_PDE_Benchmark_Hybrid_20260705",
        ],
        "definition": "Required names mirror the figure inventory in each corresponding reference training-mode root.",
        "expected_plot_files": len(expected),
        "present_plot_files": len(expected) - len(missing),
        "missing_plot_files": missing,
        "all_required_plots_present": not missing,
    }
    (root / "plot_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if missing:
        raise RuntimeError(f"Plot inventory is incomplete: {len(missing)} required files are missing")
    return report


def validate_native_1d(
    cases: list[Steady1DCase],
    variants: tuple[str, ...],
    device: torch.device,
    local_activation: str,
) -> None:
    for spec in cases:
        widths = match_widths(spec, MODEL_ORDER)
        dataset, scaling = split_and_scale_data(spec, argparse.Namespace(train_samples=4, val_samples=2, test_samples=2), 1234)
        for variant in variants:
            configuration = widths[variant]
            model = make_model(
                variant,
                int(configuration["width"]),
                int(configuration["head_hidden"]),
                spec,
                scaling,
                local_activation,
            ).to(device)
            inputs = dataset["x_train"][:2].to(device)
            output = model(inputs)
            if output.shape != (2, 1, spec.nx):
                raise AssertionError((variant, output.shape))
            if count_parameters(model) != int(configuration["params"]):
                raise AssertionError(f"parameter formula mismatch for {spec.slug}/{variant}")
            if not torch.allclose(output[..., 0], inputs[:, 1:2, 0], atol=2e-6, rtol=0.0):
                raise AssertionError(f"left boundary failed for {spec.slug}/{variant}")
            if spec.pde == "Poisson" and not torch.allclose(output[..., -1], inputs[:, 2:3, 0], atol=2e-6, rtol=0.0):
                raise AssertionError(f"right boundary failed for {spec.slug}/{variant}")
            output.square().mean().backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            if not gradients or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
                raise AssertionError(f"non-finite or missing GPU gradient for {spec.slug}/{variant}")
            model.zero_grad(set_to_none=True)
            del model, inputs, output, gradients
        xi = torch.zeros((), device=device, requires_grad=True)
        mask = xi.pow(3) if spec.pde == "KdV" else xi.square()
        first = torch.autograd.grad(mask, xi, create_graph=spec.pde == "KdV")[0]
        if not torch.equal(mask.detach(), torch.zeros_like(mask)) or not torch.equal(first.detach(), torch.zeros_like(first)):
            raise AssertionError(f"hard correction does not preserve the left value/slope for {spec.slug}")
        if spec.pde == "KdV":
            second = torch.autograd.grad(first, xi)[0]
            if not torch.equal(second.detach(), torch.zeros_like(second)):
                raise AssertionError(f"hard correction does not preserve left curvature for {spec.slug}")
        if "FNO" in widths and "HF_FNO" in widths and int(widths["FNO"]["params"]) < int(widths["HF_FNO"]["params"]):
            raise AssertionError(f"FNO is smaller than HF_FNO for {spec.slug}")
        if "CFNO" in widths and "HF_CFNO" in widths and int(widths["CFNO"]["params"]) < int(widths["HF_CFNO"]["params"]):
            raise AssertionError(f"CFNO is smaller than HF_CFNO for {spec.slug}")
        torch.cuda.empty_cache()
        print(f"validated {spec.slug}: widths={widths}", flush=True)

    length = 64
    signal = torch.randn(2, 3, length, device=device)
    transform = RealFourierTransform1d(length).to(device)
    one_sided = project_self_conjugate_bins(transform.analyze(signal), length)
    reconstructed = transform.synthesize(one_sided)
    if not torch.allclose(reconstructed, signal, atol=2e-5, rtol=2e-5):
        raise AssertionError("one-sided Fourier analysis/synthesis is not invertible")
    negative = one_sided[..., 1:-1].flip(-1).conj()
    full_spectrum = torch.cat([one_sided, negative], dim=-1)
    for frequency in range(1, length // 2):
        if not torch.allclose(full_spectrum[..., -frequency], full_spectrum[..., frequency].conj(), atol=1e-5, rtol=1e-5):
            raise AssertionError(f"Hermitian branch mismatch at frequency {frequency}")
    if not torch.allclose(one_sided[..., 0].imag, torch.zeros_like(one_sided[..., 0].imag), atol=1e-6, rtol=0.0):
        raise AssertionError("DC bin must be real")
    if not torch.allclose(one_sided[..., -1].imag, torch.zeros_like(one_sided[..., -1].imag), atol=1e-6, rtol=0.0):
        raise AssertionError("Nyquist bin must be real")
    print("validated direct real-signal Fourier inversion and Hermitian frequency pairing on CUDA", flush=True)


def render_existing_results(
    root: Path,
    cases: list[Steady1DCase],
    modes: tuple[str, ...],
    variants: tuple[str, ...],
) -> Path:
    rows: list[dict] = []
    for mode in modes:
        for spec in cases:
            case_dir = root / MODE_DIR[mode] / spec.pde / spec.case
            case_records = []
            for variant in variants:
                output_dir = case_dir / variant
                metrics_path = output_dir / "metrics.json"
                history_path = output_dir / "history.csv"
                fields_path = output_dir / "sample_fields.npz"
                if not (metrics_path.exists() and history_path.exists() and fields_path.exists()):
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                with np.load(fields_path) as field_archive:
                    prediction = field_archive["prediction"].copy()
                    exact = field_archive["exact"].copy()
                    source = field_archive["source"].copy()
                plot_fields(
                    spec,
                    variant,
                    prediction,
                    exact,
                    source,
                    output_dir,
                )
                history = history_from_csv(history_path)
                plot_training_history(history, output_dir, f"{spec.pde} / {spec.case} / {variant} / {MODE_DIR[mode]}")
                marker = output_dir / "PLOTTING_DEFERRED.json"
                if marker.exists():
                    marker.unlink()
                record = summary_row(spec, mode, metrics)
                case_records.append(record)
                rows.append(record)
            if case_records:
                plot_case_comparison(case_dir, case_records)
        mode_rows = [row for row in rows if row["training_mode"] == mode]
        if mode_rows:
            mode_root = root / MODE_DIR[mode]
            write_rows(mode_root / "global_comparison", mode_rows)
            plot_pde_frequency_comparisons(mode_root, mode_rows)
            plot_global_heatmap(mode_root, mode_rows)
    if rows:
        write_rows(root / "global_comparison", rows)
        plot_global_heatmap(root, rows)
        plot_cross_mode_comparison(root, rows)
        validate_plot_inventory(root, cases, modes, variants)
    (root / "plotting_completion.json").write_text(
        json.dumps({"models_rendered": len(rows), "expected_models": len(cases) * len(modes) * len(variants)}, indent=2),
        encoding="utf-8",
    )
    completion_path = root / "completion_report.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["plots_deferred"] = False
        completion["plots_rendered"] = len(rows)
        completion["plot_inventory_verified"] = True
        completion_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    return root


def run(args: argparse.Namespace) -> Path:
    if args.target_params is None:
        args.target_params = int(round(math.sqrt(args.reference_2d_params)))
    if args.epochs < 1000 and not args.validate_only:
        raise ValueError("Full benchmark runs require at least 1000 epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark is GPU-only; CUDA is not available")
    device = torch.device("cuda")
    seed_everything(args.seed)
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    modes = tuple(value.strip() for value in args.training_modes.split(",") if value.strip())
    if any(mode not in MODE_ORDER for mode in modes):
        raise ValueError(f"training modes must be drawn from {MODE_ORDER}")
    cases = build_cases(args)
    root = Path(args.output_root).resolve()
    if args.plots_only:
        if not root.exists():
            raise FileNotFoundError(root)
        return render_existing_results(root, cases, modes, variants)
    if args.validate_only:
        validate_native_1d(cases, variants, device, args.local_activation)
        return root

    root.mkdir(parents=True, exist_ok=True)
    config = {
        "program": Path(__file__).name,
        "purpose": "Native [B,C,N] steady 1D operator benchmark; no replicated auxiliary dimension.",
        "args": vars(args),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cases": [asdict(spec) for spec in cases],
        "training_modes": {"pde": "pure physics", "hybrid": "data plus physics", "data": "data only"},
        "normalization": {
            "source": "Training-source global mean/std; allowed in all modes.",
            "coordinate": "x/domain_x in [0,1].",
            "solution": "Fixed nondimensional shift=0 and scale=1, then explicit physical-unit denormalization before the hard projection.",
            "purity": "No theoretical-solution statistics are used by pure-physics training.",
        },
        "parameter_budget": {
            "reference_2d_real_parameters": args.reference_2d_params,
            "native_1d_target_real_parameters": args.target_params,
            "rule": "round(sqrt(reference_2d_real_parameters)) unless --target-params is explicit",
        },
        "frequency_branch_policy": "Use the one-sided real DFT equivalent of rFFT, learn only independent non-negative frequencies, force DC/Nyquist real, and recover negative frequencies by Hermitian conjugacy during synthesis.",
        "hard_constraints_and_well_posedness": {spec.slug: well_posedness(spec) for spec in cases},
        "plotting": {
            "font": "Times New Roman",
            "axis_label_size": 12,
            "legend_and_colorbar_size": 10.5,
            "primary_palette": list(PRIMARY),
            "loss_normalization": "first-five all-loss maximum; case comparison uses largest model scale",
            "absolute_loss_curves": True,
        },
        "references": REFERENCES,
    }
    (root / "benchmark_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "references.json").write_text(json.dumps(REFERENCES, indent=2), encoding="utf-8")
    rows: list[dict] = []
    for mode in modes:
        mode_root = root / MODE_DIR[mode]
        mode_root.mkdir(parents=True, exist_ok=True)
        for case_index, spec in enumerate(cases, start=1):
            case_dir = mode_root / spec.pde / spec.case
            case_dir.mkdir(parents=True, exist_ok=True)
            data_seed = args.seed + case_index * 1009
            data, scaling = split_and_scale_data(spec, args, data_seed)
            widths = match_widths(spec, MODEL_ORDER)
            case_summary = {
                "spec": asdict(spec),
                "tensor_contract": "[B,C,N]",
                "data_shapes": {name: list(value.shape) for name, value in data.items()},
                "normalization": asdict(scaling),
                "width_matching": widths,
                "parameter_matching": "Within 2% of the sqrt-scaled target, maximize usable trunk width while enforcing FNO >= HF_FNO and CFNO >= HF_CFNO.",
                "well_posedness": well_posedness(spec),
                "logical_batch_size": spec.batch_size,
                "micro_batch_size": args.micro_batch_size,
                "gradient_accumulation": "Micro-batches are weighted and accumulated before one logical-batch optimizer step.",
                "training_mode": mode,
                "pure_physics_exact_solution_in_objective": False if mode == "pde" else None,
                "hard_constraint": well_posedness(spec)["hard_constraint"],
                "bc_loss": "disabled/not used",
            }
            (case_dir / "case_summary.json").write_text(json.dumps(case_summary, indent=2), encoding="utf-8")
            case_records = []
            print(f"\n=== {MODE_DIR[mode]} [{case_index}/{len(cases)}] {spec.slug} on {device} ===", flush=True)
            for model_index, variant in enumerate(variants, start=1):
                output_dir = case_dir / variant
                metrics_path = output_dir / "metrics.json"
                if args.resume and metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    print(f"Skipping completed {MODE_DIR[mode]}/{spec.slug}/{variant}", flush=True)
                else:
                    metrics = train_one(
                        spec,
                        variant,
                        mode,
                        int(widths[variant]["width"]),
                        int(widths[variant]["head_hidden"]),
                        data,
                        scaling,
                        output_dir,
                        args,
                        device,
                        args.seed + case_index * 1009 + model_index * 97,
                    )
                record = summary_row(spec, mode, metrics)
                case_records.append(record)
                rows.append(record)
                write_rows(root / "global_comparison", rows)
            write_rows(case_dir / "model_comparison_metrics", case_records)
            if not args.skip_plots:
                plot_case_comparison(case_dir, case_records)
        mode_rows = [row for row in rows if row["training_mode"] == mode]
        write_rows(mode_root / "global_comparison", mode_rows)
    write_rows(root / "global_comparison", rows)
    if not args.skip_plots:
        plot_global_heatmap(root, rows)
        plot_cross_mode_comparison(root, rows)
    report = {
        "root": str(root),
        "models_completed": len(rows),
        "expected_models": len(modes) * len(cases) * len(variants),
        "cases_per_mode": len(cases),
        "training_modes": list(modes),
        "variants": list(variants),
        "native_tensor_contract": "[B,C,N]",
        "legacy_roots_modified": False,
        "plots_deferred": bool(args.skip_plots),
    }
    (root / "completion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native steady 1D FNO/CFNO/HF benchmark suite.")
    parser.add_argument("--output-root", default="LegacyHF_PDE_Benchmark_UniqueNativeSteady1D_SqrtParams_Final_20260820")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--high-grid", type=int, default=96)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--high-modes", type=int, default=16)
    parser.add_argument("--high-profile-modes", type=int, default=12)
    parser.add_argument("--high-profile-high-modes", type=int, default=24)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--reference-2d-params", type=int, default=500000)
    parser.add_argument("--target-params", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--variants", default=",".join(MODEL_ORDER))
    parser.add_argument("--training-modes", default=",".join(MODE_ORDER))
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--local-activation", default="gelu")
    parser.add_argument("--log-interval", type=int, default=300)
    parser.add_argument("--cases", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result_root = run(parse_args())
    print(f"\nNative steady 1D benchmark complete: {result_root}", flush=True)
