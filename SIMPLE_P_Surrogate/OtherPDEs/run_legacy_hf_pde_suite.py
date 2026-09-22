from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuralOperators import (
    CFNO2d_small,
    FNO2d_small,
    HF_CFNO2d_small,
    HF_FNO2d_small,
    SpectralAttentionOperator2d,
)


PAPER_REFERENCES = [
    {
        "key": "fno",
        "title": "Fourier Neural Operator for Parametric Partial Differential Equations",
        "url": "https://arxiv.org/abs/2010.08895",
        "used_for": "Operator-learning framing and source-to-solution Poisson/Burgers-style benchmarks.",
    },
    {
        "key": "pdebench",
        "title": "PDEBench: An Extensive Benchmark for Scientific Machine Learning",
        "url": "https://arxiv.org/abs/2210.07182",
        "used_for": "PDE family selection: Burgers, wave, diffusion-reaction, Allen-Cahn-style transient PDEs.",
    },
    {
        "key": "deeponet_benchmarks",
        "title": "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators",
        "url": "https://www.nature.com/articles/s42256-021-00302-5",
        "used_for": "Manufactured/analytic operator-learning examples with supervised exact fields.",
    },
    {
        "key": "pinn",
        "title": "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations",
        "url": "https://www.sciencedirect.com/science/article/pii/S0021999118307125",
        "used_for": "Canonical Burgers, Allen-Cahn and KdV PDE forms; this script uses hard constraints and no BC loss.",
    },
    {
        "key": "zk",
        "title": "The Zakharov-Kuznetsov equation and multidimensional KdV-type waves",
        "url": "https://doi.org/10.1016/0167-2789(74)90026-5",
        "used_for": "The steady 2D KdV case is treated as a forced ZK/KdV-type equation.",
    },
]


@dataclass(frozen=True)
class CaseSpec:
    pde: str
    case: str
    kind: str
    dimension: int
    domain_x: float
    domain_y: float
    final_time: float | None
    nx: int
    ny: int
    nt: int | None
    input_channels: int
    modes: int
    high_modes: int
    depth: int
    target_params: int
    batch_size: int
    lr: float
    attention_rank: int
    attention_gate_init: float
    params: dict
    reference_keys: tuple[str, ...]

    @property
    def slug(self) -> str:
        return f"{self.pde}_{self.case}"

    @property
    def is_transient(self) -> bool:
        return self.kind == "transient"

    @property
    def grid_shape(self) -> tuple[int, int]:
        if self.is_transient:
            assert self.nt is not None
            return self.nx, self.nt
        return self.nx, self.ny


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() * (2 if torch.is_complex(p) else 1) for p in model.parameters())


def model_factory(
    variant: str,
    *,
    width: int,
    spec: CaseSpec,
    output_features: int = 1,
    local_activation: str = "gelu",
) -> nn.Module:
    common = {
        "width": int(width),
        "depth": spec.depth,
        "input_features": spec.input_channels,
        "output_features": output_features,
    }
    if variant == "FNO":
        return FNO2d_small(modes=spec.modes, **common)
    if variant == "CFNO":
        return CFNO2d_small(
            modes=spec.modes,
            cheb_modes=(spec.modes, spec.modes),
            alpha_init=0.5,
            **common,
        )
    attention_kinds = {
        "F_ATTN": "fourier_low",
        "C_ATTN": "chebyshev_low",
        "HF_ATTN": "fourier_high",
    }
    if variant in attention_kinds:
        return SpectralAttentionOperator2d(
            modes=spec.modes,
            cheb_modes=(spec.modes, spec.modes),
            attention_kind=attention_kinds[variant],
            attention_rank=spec.attention_rank,
            attention_gate_init=spec.attention_gate_init,
            **common,
        )
    hf_common = {
        **common,
        "modes": spec.modes,
        "high_modes": spec.high_modes,
        "fourier_feature_bands": (1, 2, 4, 8),
        "high_gate_init": -0.5,
        "use_local_highpass": True,
        "grid_spacing": grid_spacing(spec),
        "boundary_mode_h": "replicate",
        "boundary_mode_w": "replicate",
        "gate_mode": "legacy",
        "use_vorticity_gate": False,
        "block_activation": "gelu",
        "head_activation": "gelu",
        "local_activation": local_activation,
    }
    if variant == "HF_FNO":
        return HF_FNO2d_small(**hf_common)
    if variant == "HF_CFNO":
        return HF_CFNO2d_small(
            cheb_modes=(spec.modes, spec.modes),
            alpha_init=0.5,
            **hf_common,
        )
    raise ValueError(f"Unknown model variant: {variant}")


def match_widths(spec: CaseSpec, variants: tuple[str, ...]) -> dict[str, dict]:
    def closest_width(variant: str) -> tuple[int, int]:
        best = None
        passed_target = False
        for width in range(4, 65):
            try:
                model = model_factory(variant, width=width, spec=spec)
                params = count_parameters(model)
            except Exception:
                continue
            score = abs(params - spec.target_params)
            if best is None or score < best[0]:
                best = (score, width, params)
            if params >= spec.target_params:
                if passed_target and best is not None and score > best[0]:
                    break
                passed_target = True
        if best is None:
            raise RuntimeError(f"Could not match width for {variant}.")
        return int(best[1]), int(best[2])

    def smallest_width_at_least(variant: str, min_params: int) -> tuple[int, int]:
        above = None
        largest = None
        for width in range(4, 129):
            try:
                model = model_factory(variant, width=width, spec=spec)
                params = count_parameters(model)
            except Exception:
                continue
            largest = (width, params)
            if params >= min_params:
                if above is None or params < above[1]:
                    above = (width, params)
                break
        if above is not None:
            return int(above[0]), int(above[1])
        if largest is not None:
            return int(largest[0]), int(largest[1])
        raise RuntimeError(f"Could not match width for {variant}.")

    matched: dict[str, dict] = {}
    for variant in variants:
        width, params = closest_width(variant)
        matched[variant] = {"width": width, "params": params}

    family_pairs = (("FNO", "HF_FNO"), ("CFNO", "HF_CFNO"))
    for base_variant, hf_variant in family_pairs:
        if base_variant in matched and hf_variant in matched:
            min_params = int(matched[hf_variant]["params"])
            if int(matched[base_variant]["params"]) < min_params:
                width, params = smallest_width_at_least(base_variant, min_params)
                matched[base_variant] = {
                    "width": width,
                    "params": params,
                    "matched_to": hf_variant,
                    "min_params": min_params,
                }
    return matched


def make_mesh(spec: CaseSpec) -> dict[str, np.ndarray]:
    nx, nw = spec.grid_shape
    x = np.linspace(0.0, spec.domain_x, nx, dtype=np.float32)
    if spec.is_transient:
        assert spec.final_time is not None
        w = np.linspace(0.0, spec.final_time, nw, dtype=np.float32)
        X, T = np.meshgrid(x, w, indexing="ij")
        return {"x": x, "w": w, "X": X.astype(np.float32), "W": T.astype(np.float32)}
    y = np.linspace(0.0, spec.domain_y, nw, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return {"x": x, "w": y, "X": X.astype(np.float32), "W": Y.astype(np.float32)}


def grid_spacing(spec: CaseSpec) -> tuple[float, float]:
    nx, nw = spec.grid_shape
    dx = spec.domain_x / max(nx - 1, 1)
    if spec.is_transient:
        assert spec.final_time is not None
        dw = spec.final_time / max(nw - 1, 1)
    else:
        dw = spec.domain_y / max(nw - 1, 1)
    return float(dx), float(dw)


def x_mask(mesh: dict[str, np.ndarray], spec: CaseSpec) -> np.ndarray:
    x = mesh["X"] / max(spec.domain_x, 1e-12)
    return (4.0 * x * (1.0 - x)).astype(np.float32)


def xy_mask(mesh: dict[str, np.ndarray], spec: CaseSpec) -> np.ndarray:
    x = mesh["X"] / max(spec.domain_x, 1e-12)
    y = mesh["W"] / max(spec.domain_y, 1e-12)
    return (16.0 * x * (1.0 - x) * y * (1.0 - y)).astype(np.float32)


def is_high_nonlinear(spec: CaseSpec) -> bool:
    return spec.params.get("profile") == "high_nonlinear"


def boundary_lift_1d(
    rng: np.random.Generator,
    mesh: dict[str, np.ndarray],
    spec: CaseSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = mesh["X"].astype(np.float64)
    amp = float(spec.params.get("boundary_amplitude", 0.0))
    left, right = rng.uniform(-amp, amp, size=2)
    L = max(float(spec.domain_x), 1e-12)
    xhat = X / L
    lift = left * (1.0 - xhat) + right * xhat
    lift_x = np.full_like(X, (right - left) / L)
    zero = np.zeros_like(X)
    return lift, lift_x, zero, zero


def boundary_lift_2d(
    rng: np.random.Generator,
    mesh: dict[str, np.ndarray],
    spec: CaseSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = mesh["X"].astype(np.float64)
    Y = mesh["W"].astype(np.float64)
    amp = float(spec.params.get("boundary_amplitude", 0.0))
    a0, ax, ay, axy = rng.uniform(-amp, amp, size=4)
    xhat = X / max(float(spec.domain_x), 1e-12)
    yhat = Y / max(float(spec.domain_y), 1e-12)
    lift = a0 + ax * xhat + ay * yhat + axy * xhat * yhat
    lift_x = (ax + axy * yhat) / max(float(spec.domain_x), 1e-12)
    lift_y = (ay + axy * xhat) / max(float(spec.domain_y), 1e-12)
    return lift, lift_x, lift_y


def high_frequency_modes(rng: np.random.Generator, spec: CaseSpec, *, dimension: int) -> np.ndarray:
    mode_min = int(spec.params.get("mode_min_2d" if dimension == 2 else "mode_min", 4))
    mode_max = int(spec.params.get("mode_max_2d" if dimension == 2 else "mode_max", 14))
    count = int(spec.params.get("mode_count_2d" if dimension == 2 else "mode_count", 5))
    values = np.arange(max(1, mode_min), max(mode_min, mode_max) + 1)
    count = min(max(1, count), len(values))
    return rng.choice(values, size=count, replace=False)


def sample_steady_1d(rng: np.random.Generator, mesh: dict[str, np.ndarray], spec: CaseSpec) -> dict[str, np.ndarray]:
    X = mesh["X"].astype(np.float64)
    L = spec.domain_x
    if is_high_nonlinear(spec):
        modes = high_frequency_modes(rng, spec, dimension=1)
        amp_scale = float(spec.params.get("amp_scale", 0.85))
        amps = rng.uniform(-amp_scale, amp_scale, size=len(modes))
    else:
        modes = rng.choice(np.arange(1, 7), size=3, replace=False)
        amps = rng.uniform(-0.8, 0.8, size=3)
    u = np.zeros_like(X)
    ux = np.zeros_like(X)
    uxx = np.zeros_like(X)
    uxxx = np.zeros_like(X)
    lift = None
    if is_high_nonlinear(spec):
        lift, lift_x, _, _ = boundary_lift_1d(rng, mesh, spec)
        u += lift
        ux += lift_x
    for a, k in zip(amps, modes):
        kk = k * math.pi / L
        s = np.sin(kk * X)
        c = np.cos(kk * X)
        u += a * s
        ux += a * kk * c
        uxx += -a * kk**2 * s
        uxxx += -a * kk**3 * c
    f = steady_source(spec, u, ux, uxx, uxxx, None, None, None)
    sample = {"u": u.astype(np.float32), "source": f.astype(np.float32)}
    if lift is not None:
        sample["lift"] = lift.astype(np.float32)
    return sample


def sample_steady_2d(rng: np.random.Generator, mesh: dict[str, np.ndarray], spec: CaseSpec) -> dict[str, np.ndarray]:
    X = mesh["X"].astype(np.float64)
    Y = mesh["W"].astype(np.float64)
    Lx, Ly = spec.domain_x, spec.domain_y
    u = np.zeros_like(X)
    ux = np.zeros_like(X)
    uy = np.zeros_like(X)
    uxx = np.zeros_like(X)
    uyy = np.zeros_like(X)
    uxxx = np.zeros_like(X)
    uxyy = np.zeros_like(X)
    lift = None
    if is_high_nonlinear(spec):
        lift, lift_x, lift_y = boundary_lift_2d(rng, mesh, spec)
        u += lift
        ux += lift_x
        uy += lift_y
        modes_x = high_frequency_modes(rng, spec, dimension=2)
        modes_y = high_frequency_modes(rng, spec, dimension=2)
        pairs = [(int(k), int(l)) for k in modes_x for l in modes_y]
        rng.shuffle(pairs)
        pairs = pairs[: int(spec.params.get("mode_pair_count", 5))]
        amp_scale = float(spec.params.get("amp_scale_2d", spec.params.get("amp_scale", 0.75)))
    else:
        pairs = [(1, 1), (2, 1), (1, 2), (3, 2)]
        rng.shuffle(pairs)
        pairs = pairs[:3]
        amp_scale = 0.7
    for k, l in pairs:
        a = rng.uniform(-amp_scale, amp_scale)
        kx = k * math.pi / Lx
        ky = l * math.pi / Ly
        sx = np.sin(kx * X)
        cx = np.cos(kx * X)
        sy = np.sin(ky * Y)
        cy = np.cos(ky * Y)
        term = a * sx * sy
        u += term
        ux += a * kx * cx * sy
        uy += a * ky * sx * cy
        uxx += -kx**2 * term
        uyy += -ky**2 * term
        uxxx += -a * kx**3 * cx * sy
        uxyy += -a * kx * ky**2 * cx * sy
    f = steady_source(spec, u, ux, uxx, uxxx, uy, uyy, uxyy)
    sample = {"u": u.astype(np.float32), "source": f.astype(np.float32)}
    if lift is not None:
        sample["lift"] = lift.astype(np.float32)
    return sample


def sample_transient_1d(rng: np.random.Generator, mesh: dict[str, np.ndarray], spec: CaseSpec) -> dict[str, np.ndarray]:
    X = mesh["X"].astype(np.float64)
    T = mesh["W"].astype(np.float64)
    L = spec.domain_x
    final_time = max(float(spec.final_time or 1.0), 1e-12)
    if is_high_nonlinear(spec):
        modes = high_frequency_modes(rng, spec, dimension=1)
        amp_scale = float(spec.params.get("transient_amp_scale", spec.params.get("amp_scale", 0.70)))
        amps = rng.uniform(-amp_scale, amp_scale, size=len(modes))
    else:
        modes = rng.choice(np.arange(1, 6), size=3, replace=False)
        amps = rng.uniform(-0.55, 0.55, size=3)
    u = np.zeros_like(X)
    ut = np.zeros_like(X)
    utt = np.zeros_like(X)
    ux = np.zeros_like(X)
    uxx = np.zeros_like(X)
    uxxx = np.zeros_like(X)
    lift = None
    if is_high_nonlinear(spec):
        lift, lift_x, _, _ = boundary_lift_1d(rng, mesh, spec)
        u += lift
        ux += lift_x
    for a, k in zip(amps, modes):
        kx = k * math.pi / L
        if spec.pde == "Wave":
            omega = float(spec.params.get("c", 1.0)) * kx
        elif spec.pde == "KdV":
            lo = float(spec.params.get("omega_min", 0.6))
            hi = float(spec.params.get("omega_max", 1.2))
            omega = 2.0 * math.pi / final_time * rng.uniform(lo, hi)
        elif is_high_nonlinear(spec):
            omega = rng.uniform(
                float(spec.params.get("omega_min", 1.4)),
                float(spec.params.get("omega_max", 3.6)),
            ) * math.pi
        else:
            omega = rng.uniform(0.6, 1.6) * math.pi
        sx = np.sin(kx * X)
        cx = np.cos(kx * X)
        ct = np.cos(omega * T)
        st = np.sin(omega * T)
        u += a * sx * ct
        ut += -a * omega * sx * st
        utt += -a * omega**2 * sx * ct
        ux += a * kx * cx * ct
        uxx += -a * kx**2 * sx * ct
        uxxx += -a * kx**3 * cx * ct
    source = transient_source(spec, u, ut, utt, ux, uxx, uxxx)
    u0 = u[:, :1].repeat(u.shape[1], axis=1)
    sample = {
        "u": u.astype(np.float32),
        "source": source.astype(np.float32),
        "u0": u0.astype(np.float32),
    }
    if spec.pde == "Wave":
        sample["v0"] = ut[:, :1].repeat(u.shape[1], axis=1).astype(np.float32)
    if lift is not None:
        sample["lift"] = lift.astype(np.float32)
    return sample


def steady_source(
    spec: CaseSpec,
    u: np.ndarray,
    ux: np.ndarray,
    uxx: np.ndarray,
    uxxx: np.ndarray,
    uy: np.ndarray | None,
    uyy: np.ndarray | None,
    uxyy: np.ndarray | None,
) -> np.ndarray:
    if spec.pde == "Poisson":
        return -uxx if spec.dimension == 1 else -(uxx + np.asarray(uyy))
    if spec.pde == "KdV":
        nonlinear_coeff = float(spec.params.get("nonlinear_coeff", 6.0))
        if spec.dimension == 1:
            return uxxx + nonlinear_coeff * u * ux
        return uxxx + np.asarray(uxyy) + nonlinear_coeff * u * ux
    if spec.pde == "AllenCahn":
        eps = float(spec.params["epsilon"])
        reaction = float(spec.params.get("reaction", 1.0))
        lap = uxx if spec.dimension == 1 else uxx + np.asarray(uyy)
        return -eps**2 * lap - reaction * u + reaction * u**3
    if spec.pde == "Burgers":
        nu = float(spec.params["nu"])
        convective_coeff = float(spec.params.get("convective_coeff", 1.0))
        if spec.dimension == 1:
            return convective_coeff * u * ux - nu * uxx
        return convective_coeff * u * (ux + np.asarray(uy)) - nu * (uxx + np.asarray(uyy))
    if spec.pde == "ReactionDiffusion":
        diff = float(spec.params["diffusion"])
        rate = float(spec.params["rate"])
        lap = uxx if spec.dimension == 1 else uxx + np.asarray(uyy)
        return -diff * lap - rate * u * (1.0 - u)
    raise ValueError(f"Unsupported steady PDE: {spec.pde}")


def transient_source(
    spec: CaseSpec,
    u: np.ndarray,
    ut: np.ndarray,
    utt: np.ndarray,
    ux: np.ndarray,
    uxx: np.ndarray,
    uxxx: np.ndarray,
) -> np.ndarray:
    if spec.pde == "Wave":
        c = float(spec.params["c"])
        return utt - c**2 * uxx
    if spec.pde == "KdV":
        nonlinear_coeff = float(spec.params.get("nonlinear_coeff", 6.0))
        return ut + uxxx + nonlinear_coeff * u * ux
    if spec.pde == "AllenCahn":
        eps = float(spec.params["epsilon"])
        reaction = float(spec.params.get("reaction", 1.0))
        return ut - eps**2 * uxx - reaction * u + reaction * u**3
    if spec.pde == "Burgers":
        nu = float(spec.params["nu"])
        convective_coeff = float(spec.params.get("convective_coeff", 1.0))
        return ut + convective_coeff * u * ux - nu * uxx
    if spec.pde == "ReactionDiffusion":
        diff = float(spec.params["diffusion"])
        rate = float(spec.params["rate"])
        return ut - diff * uxx - rate * u * (1.0 - u)
    raise ValueError(f"Unsupported transient PDE: {spec.pde}")


def make_dataset(spec: CaseSpec, n_samples: int, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    mesh = make_mesh(spec)
    inputs = []
    targets = []
    sources = []
    for _ in range(n_samples):
        if spec.is_transient:
            sample = sample_transient_1d(rng, mesh, spec)
            channels = [sample["source"], sample["u0"]]
            if "v0" in sample:
                channels.append(sample["v0"])
            if "lift" in sample:
                channels.append(sample["lift"])
            channels.extend([mesh["X"] / max(spec.domain_x, 1e-12), mesh["W"] / max(spec.final_time or 1.0, 1e-12)])
        elif spec.dimension == 1:
            sample = sample_steady_1d(rng, mesh, spec)
            channels = [sample["source"]]
            if "lift" in sample:
                channels.append(sample["lift"])
            channels.extend([mesh["X"] / max(spec.domain_x, 1e-12), mesh["W"]])
        else:
            sample = sample_steady_2d(rng, mesh, spec)
            channels = [sample["source"]]
            if "lift" in sample:
                channels.append(sample["lift"])
            channels.extend([mesh["X"] / max(spec.domain_x, 1e-12), mesh["W"] / max(spec.domain_y, 1e-12)])
        if len(channels) != spec.input_channels:
            raise ValueError(f"{spec.slug} expected {spec.input_channels} input channels, got {len(channels)}")
        inputs.append(np.stack(channels, axis=0).astype(np.float32))
        targets.append(sample["u"][None, ...].astype(np.float32))
        sources.append(sample["source"][None, ...].astype(np.float32))
    return {
        "x": torch.from_numpy(np.stack(inputs, axis=0)),
        "y": torch.from_numpy(np.stack(targets, axis=0)),
        "source": torch.from_numpy(np.stack(sources, axis=0)),
    }


class HardConstraintWrapper(nn.Module):
    def __init__(self, base: nn.Module, spec: CaseSpec):
        super().__init__()
        self.base = base
        mesh = make_mesh(spec)
        self.is_wave = spec.is_transient and spec.pde == "Wave"
        self.lift_channel = 2 if spec.is_transient and is_high_nonlinear(spec) else (1 if is_high_nonlinear(spec) else None)
        if spec.is_transient:
            mask = x_mask(mesh, spec)
            tau = mesh["W"] / max(float(spec.final_time or 1.0), 1e-12)
            self.register_buffer("mask", torch.from_numpy(mask[None, None, ...]))
            self.register_buffer("time_grid", torch.from_numpy(mesh["W"][None, None, ...].astype(np.float32)))
            self.register_buffer("tau", torch.from_numpy(tau[None, None, ...].astype(np.float32)))
            self.mode = "transient"
        elif spec.dimension == 1:
            self.register_buffer("mask", torch.from_numpy(x_mask(mesh, spec)[None, None, ...]))
            self.mode = "steady"
        else:
            self.register_buffer("mask", torch.from_numpy(xy_mask(mesh, spec)[None, None, ...]))
            self.mode = "steady"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.base(x)
        if self.mode == "transient":
            u0 = x[:, 1:2]
            if self.is_wave:
                v0 = x[:, 2:3]
                return u0 + self.time_grid * v0 + (self.tau**2) * self.mask * raw
            if self.lift_channel is not None:
                lift = x[:, self.lift_channel:self.lift_channel + 1]
                return (1.0 - self.tau) * u0 + self.tau * (lift + self.mask * raw)
            return (1.0 - self.tau) * u0 + self.tau * self.mask * raw
        if self.lift_channel is not None:
            lift = x[:, self.lift_channel:self.lift_channel + 1]
            return lift + self.mask * raw
        return self.mask * raw


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.vector_norm((pred - truth).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(truth.reshape(truth.shape[0], -1), dim=1).clamp_min(1e-12)
    return (num / den).mean()


def torch_first_derivative(a: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    n = a.shape[dim]
    out = torch.zeros_like(a)
    if n < 2:
        return out
    sl = [slice(None)] * a.ndim
    if n > 2:
        mid = sl.copy()
        plus = sl.copy()
        minus = sl.copy()
        mid[dim] = slice(1, -1)
        plus[dim] = slice(2, None)
        minus[dim] = slice(None, -2)
        out[tuple(mid)] = (a[tuple(plus)] - a[tuple(minus)]) / (2.0 * spacing)
    first = sl.copy()
    second = sl.copy()
    first[dim] = 0
    second[dim] = 1
    out[tuple(first)] = (a[tuple(second)] - a[tuple(first)]) / spacing
    last = sl.copy()
    prev = sl.copy()
    last[dim] = -1
    prev[dim] = -2
    out[tuple(last)] = (a[tuple(last)] - a[tuple(prev)]) / spacing
    return out


def torch_second_derivative(a: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    n = a.shape[dim]
    out = torch.zeros_like(a)
    if n < 3:
        return out
    sl = [slice(None)] * a.ndim
    mid = sl.copy()
    plus = sl.copy()
    center = sl.copy()
    minus = sl.copy()
    mid[dim] = slice(1, -1)
    plus[dim] = slice(2, None)
    center[dim] = slice(1, -1)
    minus[dim] = slice(None, -2)
    out[tuple(mid)] = (a[tuple(plus)] - 2.0 * a[tuple(center)] + a[tuple(minus)]) / (spacing**2)
    first = sl.copy()
    second = sl.copy()
    last = sl.copy()
    prev = sl.copy()
    first[dim] = 0
    second[dim] = 1
    last[dim] = -1
    prev[dim] = -2
    out[tuple(first)] = out[tuple(second)]
    out[tuple(last)] = out[tuple(prev)]
    return out


def torch_third_derivative(a: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    return torch_first_derivative(torch_second_derivative(a, spacing, dim), spacing, dim)


def torch_pde_lhs(spec: CaseSpec, u: torch.Tensor) -> torch.Tensor:
    dx, dw = grid_spacing(spec)
    if spec.is_transient:
        ut = torch_first_derivative(u, dw, dim=-1)
        utt = torch_second_derivative(u, dw, dim=-1)
        ux = torch_first_derivative(u, dx, dim=-2)
        uxx = torch_second_derivative(u, dx, dim=-2)
        uxxx = torch_third_derivative(u, dx, dim=-2)
        if spec.pde == "Wave":
            c = float(spec.params["c"])
            return utt - c**2 * uxx
        if spec.pde == "KdV":
            nonlinear_coeff = float(spec.params.get("nonlinear_coeff", 6.0))
            return ut + uxxx + nonlinear_coeff * u * ux
        if spec.pde == "AllenCahn":
            eps = float(spec.params["epsilon"])
            reaction = float(spec.params.get("reaction", 1.0))
            return ut - eps**2 * uxx - reaction * u + reaction * u**3
        if spec.pde == "Burgers":
            nu = float(spec.params["nu"])
            convective_coeff = float(spec.params.get("convective_coeff", 1.0))
            return ut + convective_coeff * u * ux - nu * uxx
        if spec.pde == "ReactionDiffusion":
            diff = float(spec.params["diffusion"])
            rate = float(spec.params["rate"])
            return ut - diff * uxx - rate * u * (1.0 - u)
    else:
        ux = torch_first_derivative(u, dx, dim=-2)
        uxx = torch_second_derivative(u, dx, dim=-2)
        uxxx = torch_third_derivative(u, dx, dim=-2)
        if spec.dimension == 2:
            uy = torch_first_derivative(u, dw, dim=-1)
            uyy = torch_second_derivative(u, dw, dim=-1)
            uxyy = torch_first_derivative(uyy, dx, dim=-2)
        else:
            uy = uyy = uxyy = None
        if spec.pde == "Poisson":
            return -uxx if spec.dimension == 1 else -(uxx + uyy)
        if spec.pde == "KdV":
            nonlinear_coeff = float(spec.params.get("nonlinear_coeff", 6.0))
            return uxxx + nonlinear_coeff * u * ux if spec.dimension == 1 else uxxx + uxyy + nonlinear_coeff * u * ux
        if spec.pde == "AllenCahn":
            eps = float(spec.params["epsilon"])
            reaction = float(spec.params.get("reaction", 1.0))
            lap = uxx if spec.dimension == 1 else uxx + uyy
            return -eps**2 * lap - reaction * u + reaction * u**3
        if spec.pde == "Burgers":
            nu = float(spec.params["nu"])
            convective_coeff = float(spec.params.get("convective_coeff", 1.0))
            return convective_coeff * u * ux - nu * uxx if spec.dimension == 1 else convective_coeff * u * (ux + uy) - nu * (uxx + uyy)
        if spec.pde == "ReactionDiffusion":
            diff = float(spec.params["diffusion"])
            rate = float(spec.params["rate"])
            lap = uxx if spec.dimension == 1 else uxx + uyy
            return -diff * lap - rate * u * (1.0 - u)
    raise ValueError(f"Unsupported PDE residual for {spec.pde}/{spec.case}")


def crop_pde_residual(spec: CaseSpec, residual: torch.Tensor) -> torch.Tensor:
    x_pad = 3 if spec.pde == "KdV" else 2
    if spec.is_transient:
        w_pad = 2 if spec.pde == "Wave" else 1
    elif spec.dimension == 2:
        w_pad = 2
    else:
        w_pad = 0
    x_stop = -x_pad if x_pad > 0 else None
    w_stop = -w_pad if w_pad > 0 else None
    if w_pad > 0:
        return residual[..., x_pad:x_stop, w_pad:w_stop]
    return residual[..., x_pad:x_stop, :]


def pde_residual_mses(
    spec: CaseSpec,
    pred: torch.Tensor,
    source: torch.Tensor,
    source_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = crop_pde_residual(spec, torch_pde_lhs(spec, pred) - source)
    absolute = torch.mean(residual ** 2)
    normalized = torch.mean((residual / source_scale.clamp_min(1.0)) ** 2)
    return normalized, absolute


def pde_residual_loss(spec: CaseSpec, pred: torch.Tensor, source: torch.Tensor, source_scale: torch.Tensor) -> torch.Tensor:
    normalized, _ = pde_residual_mses(spec, pred, source, source_scale)
    return normalized


def compose_training_loss(
    mode: str,
    data_mse: torch.Tensor,
    pde_mse: torch.Tensor,
    physics_weight: float,
) -> torch.Tensor:
    if mode == "data":
        return data_mse
    if mode == "pde":
        return pde_mse
    if mode == "hybrid":
        return data_mse + float(physics_weight) * pde_mse
    raise ValueError(f"Unsupported training mode: {mode}")


def selection_metric_for_mode(mode: str) -> str:
    if mode == "data":
        return "val_mse"
    if mode == "pde":
        return "val_pde_mse"
    if mode == "hybrid":
        return "val_loss"
    raise ValueError(f"Unsupported training mode: {mode}")


def train_one_model(
    spec: CaseSpec,
    variant: str,
    width: int,
    data: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    epochs: int,
    device: torch.device,
    seed: int,
    training_mode: str,
    physics_weight: float,
    local_activation: str = "gelu",
) -> dict:
    seed_everything(seed)
    base = model_factory(variant, width=width, spec=spec, local_activation=local_activation)
    model = HardConstraintWrapper(base, spec).to(device)
    params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=spec.lr * 0.05)
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
    n_train = x_train.shape[0]
    history = []
    best_metric = selection_metric_for_mode(training_mode)
    best = {"epoch": -1, best_metric: float("inf"), "selection_metric": best_metric}
    best_state = None
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_pde_sum = 0.0
        train_pde_abs_sum = 0.0
        train_rel_sum = 0.0
        seen = 0
        for start in range(0, n_train, spec.batch_size):
            idx = perm[start:start + spec.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            sb = source_train[idx]
            pred = model(xb)
            data_mse = F.mse_loss(pred, yb)
            pde_mse, pde_mse_abs = pde_residual_mses(spec, pred, sb, source_scale)
            loss = compose_training_loss(training_mode, data_mse, pde_mse, physics_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bsz = xb.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * bsz
            train_mse_sum += float(data_mse.detach().cpu()) * bsz
            train_pde_sum += float(pde_mse.detach().cpu()) * bsz
            train_pde_abs_sum += float(pde_mse_abs.detach().cpu()) * bsz
            train_rel_sum += float(relative_l2(pred.detach(), yb).cpu()) * bsz
            seen += bsz
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_pde, val_pde_abs = pde_residual_mses(spec, val_pred, source_val, source_scale)
            val_pde_mse = float(val_pde.detach().cpu())
            val_pde_mse_absolute = float(val_pde_abs.detach().cpu())
            val_data_mse_tensor = F.mse_loss(val_pred, y_val)
            val_state_mse = float(val_data_mse_tensor.detach().cpu())
            val_rel = float(relative_l2(val_pred, y_val).detach().cpu())
            val_loss = float(
                compose_training_loss(training_mode, val_data_mse_tensor, val_pde, physics_weight)
                .detach()
                .cpu()
            )
            test_pred_epoch = model(x_test)
            test_pde_epoch, test_pde_abs_epoch = pde_residual_mses(spec, test_pred_epoch, source_test, source_scale)
            test_data_mse_tensor = F.mse_loss(test_pred_epoch, y_test)
            test_loss = float(
                compose_training_loss(training_mode, test_data_mse_tensor, test_pde_epoch, physics_weight)
                .detach()
                .cpu()
            )
            test_state_mse = float(test_data_mse_tensor.detach().cpu())
            test_pde_mse = float(test_pde_epoch.detach().cpu())
            test_pde_mse_absolute = float(test_pde_abs_epoch.detach().cpu())
            test_rel = float(relative_l2(test_pred_epoch, y_test).detach().cpu())
        train_loss = train_loss_sum / max(seen, 1)
        train_mse = train_mse_sum / max(seen, 1)
        train_pde_mse = train_pde_sum / max(seen, 1)
        train_pde_mse_absolute = train_pde_abs_sum / max(seen, 1)
        train_rel = train_rel_sum / max(seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_pde_mse": train_pde_mse,
            "train_pde_mse_absolute": train_pde_mse_absolute,
            "train_state_rel_l2": train_rel,
            "val_loss": val_loss,
            "val_pde_mse": val_pde_mse,
            "val_pde_mse_absolute": val_pde_mse_absolute,
            "val_state_mse": val_state_mse,
            "val_mse": val_state_mse,
            "val_state_rel_l2": val_rel,
            "test_loss": test_loss,
            "test_mse": test_state_mse,
            "test_pde_mse": test_pde_mse,
            "test_pde_mse_absolute": test_pde_mse_absolute,
            "test_state_rel_l2": test_rel,
            "lr": scheduler.get_last_lr()[0],
        }
        if hasattr(base, "attention_summary"):
            attention = base.attention_summary()
            row.update(
                {
                    "attention_gate": attention["gate"],
                    "attention_source_rms": attention["source_rms"],
                    "attention_source_to_state_rms": attention["source_to_state_rms"],
                    "attention_output_rms": attention["attention_output_rms"],
                }
            )
        history.append(row)
        if row[best_metric] < best.get(best_metric, float("inf")):
            best = dict(row)
            best["selection_metric"] = best_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(
                f"{spec.slug}/{variant} epoch {epoch:04d}/{epochs} "
                f"mode={training_mode} loss={train_loss:.3e}/{val_loss:.3e} "
                f"test_loss={test_loss:.3e} mse_val={val_state_mse:.3e} "
                f"pde_val={val_pde_mse:.3e} state_rel={val_rel:.3e}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {
            "variant": variant,
            "spec": asdict(spec),
            "width": width,
            "params": params,
            "best": best,
            "training_mode": training_mode,
            "physics_weight": float(physics_weight),
            "local_activation": local_activation,
            "state_dict": model.state_dict(),
        },
        out_dir / "best_checkpoint.pt",
    )
    metrics = evaluate_and_plot(spec, variant, model, data, out_dir, device)
    metrics.update({
        "variant": variant,
        "width": int(width),
        "params": int(params),
        "elapsed_s": elapsed,
        "training_mode": training_mode,
        "physics_weight": float(physics_weight),
        "local_activation": local_activation,
        "best": best,
        "final": history[-1],
    })
    if hasattr(base, "attention_summary"):
        metrics["attention"] = base.attention_summary()
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_model_convergence(history, out_dir / "training_convergence.png", variant)
    plot_attention_diagnostics(history, out_dir, variant)
    return metrics


def finite_diff_first(a: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return np.gradient(a, dx, axis=axis, edge_order=2)


def finite_diff_second(a: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return np.gradient(np.gradient(a, dx, axis=axis, edge_order=2), dx, axis=axis, edge_order=2)


def finite_diff_third(a: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return np.gradient(finite_diff_second(a, dx, axis), dx, axis=axis, edge_order=2)


def residual_field(spec: CaseSpec, u: np.ndarray, source: np.ndarray) -> np.ndarray:
    dx, dw = grid_spacing(spec)
    if spec.is_transient:
        ut = finite_diff_first(u, dw, axis=1)
        utt = finite_diff_second(u, dw, axis=1)
        ux = finite_diff_first(u, dx, axis=0)
        uxx = finite_diff_second(u, dx, axis=0)
        uxxx = finite_diff_third(u, dx, axis=0)
        lhs = transient_source(spec, u, ut, utt, ux, uxx, uxxx)
        return lhs - source
    if spec.dimension == 1:
        ux = finite_diff_first(u, dx, axis=0)
        uxx = finite_diff_second(u, dx, axis=0)
        uxxx = finite_diff_third(u, dx, axis=0)
        lhs = steady_source(spec, u, ux, uxx, uxxx, None, None, None)
        return lhs - source
    ux = finite_diff_first(u, dx, axis=0)
    uy = finite_diff_first(u, dw, axis=1)
    uxx = finite_diff_second(u, dx, axis=0)
    uyy = finite_diff_second(u, dw, axis=1)
    uxxx = finite_diff_third(u, dx, axis=0)
    uxyy = finite_diff_first(uyy, dx, axis=0)
    lhs = steady_source(spec, u, ux, uxx, uxxx, uy, uyy, uxyy)
    return lhs - source


def metric_summary(arr: np.ndarray) -> dict:
    flat = np.asarray(arr).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "mean_abs": float(np.mean(np.abs(flat))),
        "rms": float(np.sqrt(np.mean(flat**2))),
        "p95_abs": float(np.percentile(np.abs(flat), 95)),
        "max_abs": float(np.max(np.abs(flat))),
    }


def evaluate_and_plot(
    spec: CaseSpec,
    variant: str,
    model: nn.Module,
    data: dict[str, torch.Tensor],
    out_dir: Path,
    device: torch.device,
) -> dict:
    model.eval()
    x_test = data["x_test"].to(device)
    y_test = data["y_test"].to(device)
    source_test = data["source_test"].to(device)
    source_scale = data["source_train"].to(device).std().clamp_min(1.0)
    with torch.no_grad():
        pred = model(x_test)
        test_pde, test_pde_abs = pde_residual_mses(spec, pred, source_test, source_scale)
        test_pde_mse = float(test_pde.detach().cpu())
        test_pde_mse_absolute = float(test_pde_abs.detach().cpu())
        test_mse = float(F.mse_loss(pred, y_test).detach().cpu())
        test_rel = float(relative_l2(pred, y_test).detach().cpu())
    pred_np = pred.detach().cpu().numpy()
    truth_np = y_test.detach().cpu().numpy()
    source_np = source_test.detach().cpu().numpy()
    idx = 0
    u_pred = pred_np[idx, 0]
    u_true = truth_np[idx, 0]
    src = source_np[idx, 0]
    err = np.abs(u_pred - u_true)
    res_pred = residual_field(spec, u_pred, src)
    res_true = residual_field(spec, u_true, src)
    np.savez_compressed(
        out_dir / "sample_fields.npz",
        prediction=u_pred,
        exact=u_true,
        abs_error=err,
        residual_prediction=res_pred,
        residual_exact=res_true,
        source=src,
    )
    plot_prediction_triplet(spec, u_pred, u_true, err, out_dir / "prediction_exact_error.png", variant)
    plot_residual_pair(spec, res_pred, res_true, out_dir / "residual_distribution.png", variant)
    plot_spectrum(spec, u_pred, u_true, err, out_dir / "spectrum_analysis.png", variant)
    gate_summary = None
    if hasattr(model.base, "high_pass_gate_summary"):
        gate_summary = model.base.high_pass_gate_summary()
    return {
        "test_pde_mse": test_pde_mse,
        "test_pde_mse_absolute": test_pde_mse_absolute,
        "test_mse": test_mse,
        "test_rel_l2": test_rel,
        "abs_error": metric_summary(err),
        "residual_prediction": metric_summary(res_pred),
        "residual_exact": metric_summary(res_true),
        "high_pass_gate": gate_summary,
    }


def image_extent(spec: CaseSpec) -> tuple[float, float, float, float]:
    if spec.is_transient:
        return 0.0, float(spec.final_time or 1.0), 0.0, spec.domain_x
    if spec.dimension == 1:
        return 0.0, 1.0, 0.0, spec.domain_x
    return 0.0, spec.domain_y, 0.0, spec.domain_x


def plot_prediction_triplet(spec: CaseSpec, pred: np.ndarray, truth: np.ndarray, err: np.ndarray, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    fields = [(pred, "Prediction"), (truth, "Theory"), (err, "Abs error")]
    vmin = min(float(pred.min()), float(truth.min()))
    vmax = max(float(pred.max()), float(truth.max()))
    for ax, (arr, label) in zip(axes, fields):
        if label == "Abs error":
            im = ax.imshow(arr, origin="lower", aspect="auto", extent=image_extent(spec), cmap="magma")
        else:
            im = ax.imshow(arr, origin="lower", aspect="auto", extent=image_extent(spec), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("t" if spec.is_transient else ("dummy" if spec.dimension == 1 else "y"))
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, shrink=0.86)
    fig.suptitle(f"{spec.slug} / {title}")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    if spec.dimension == 1 and not spec.is_transient:
        center = pred.shape[1] // 2
        x = np.linspace(0.0, spec.domain_x, pred.shape[0])
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(x, truth[:, center], label="Theory", linewidth=2)
        ax.plot(x, pred[:, center], label="Prediction", linestyle="--")
        ax.plot(x, err[:, center], label="Abs error", linestyle=":")
        ax.set_title(f"{spec.slug} / {title} center-line")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.savefig(path.with_name("prediction_exact_error_line.png"), dpi=180)
        plt.close(fig)


def plot_residual_pair(spec: CaseSpec, pred_res: np.ndarray, true_res: np.ndarray, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    vmax = max(float(np.percentile(np.abs(pred_res), 99)), float(np.percentile(np.abs(true_res), 99)), 1e-12)
    for ax, arr, label in [(axes[0], pred_res, "Prediction residual"), (axes[1], true_res, "Theory residual")]:
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=image_extent(spec), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("t" if spec.is_transient else ("dummy" if spec.dimension == 1 else "y"))
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, shrink=0.86)
    fig.suptitle(f"{spec.slug} / {title}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def spectral_energy(arr: np.ndarray) -> np.ndarray:
    ft = np.fft.rfft2(arr)
    energy = np.abs(ft) ** 2
    if energy.shape[0] == 0:
        return energy.reshape(-1)
    yy, xx = np.indices(energy.shape)
    radius = np.sqrt(yy**2 + xx**2).astype(int)
    max_r = int(radius.max()) + 1
    out = np.zeros(max_r)
    counts = np.zeros(max_r)
    np.add.at(out, radius.reshape(-1), energy.reshape(-1))
    np.add.at(counts, radius.reshape(-1), 1.0)
    return out / np.maximum(counts, 1.0)


def plot_spectrum(spec: CaseSpec, pred: np.ndarray, truth: np.ndarray, err: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for arr, label in [(truth, "Theory"), (pred, "Prediction"), (err, "Abs error")]:
        e = spectral_energy(arr)
        ax.semilogy(np.arange(len(e)), e + 1e-30, label=label)
    ax.set_title(f"{spec.slug} / {title} spectral energy")
    ax.set_xlabel("radial spectral bin")
    ax.set_ylabel("mean energy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def has_numeric_history(history: list[dict], train_key: str, val_key: str) -> bool:
    return bool(history) and all(train_key in h and val_key in h for h in history)


def plot_model_convergence(history: list[dict], path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    panels: list[tuple[str, list[tuple[str, list[float]]], str, str]] = []

    def make_series(train_key: str, val_key: str, test_key: str | None = None) -> list[tuple[str, list[float]]]:
        series = [
            ("train", [h[train_key] for h in history]),
            ("validation", [h[val_key] for h in history]),
        ]
        if test_key is not None and all(test_key in h for h in history):
            series.append(("test", [h[test_key] for h in history]))
        return series

    if has_numeric_history(history, "train_loss", "val_loss"):
        panels.append((
            "objective",
            make_series("train_loss", "val_loss", "test_loss"),
            "training objective / total loss",
            "objective",
        ))
    elif has_numeric_history(history, "train_mse", "val_mse"):
        panels.append((
            "data",
            make_series("train_mse", "val_mse", "test_mse"),
            "data MSE",
            "data",
        ))
    if has_numeric_history(history, "train_pde_mse", "val_pde_mse"):
        panels.append((
            "physics",
            make_series("train_pde_mse", "val_pde_mse", "test_pde_mse"),
            "physics loss / normalized PDE residual MSE",
            "physics",
        ))
    if has_numeric_history(history, "train_pde_mse_absolute", "val_pde_mse_absolute"):
        panels.append((
            "absolute",
            make_series("train_pde_mse_absolute", "val_pde_mse_absolute", "test_pde_mse_absolute"),
            "absolute PDE residual MSE",
            "absolute",
        ))
    if not panels:
        return

    def draw_single(ax: plt.Axes, series: list[tuple[str, list[float]]], ylabel: str) -> None:
        for label, vals in series:
            ax.semilogy(epochs, vals, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), constrained_layout=True)
    axes_list = np.atleast_1d(axes).tolist()
    for ax, (panel_title, series, ylabel, _) in zip(axes_list, panels):
        ax.set_title(panel_title)
        draw_single(ax, series, ylabel)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)

    for _, series, ylabel, suffix in panels:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.set_title(f"{title} / {suffix}")
        draw_single(ax, series, ylabel)
        fig.savefig(path.with_name(f"{path.stem}_{suffix}{path.suffix}"), dpi=180)
        plt.close(fig)


def plot_attention_diagnostics(history: list[dict], out_dir: Path, title: str) -> None:
    if "attention_gate" not in history[0]:
        return
    epochs = [row["epoch"] for row in history]
    diagnostics = [
        ("attention_gate", "attention gate", "linear"),
        ("attention_source_to_state_rms", "spectral source / state RMS", "log"),
        ("attention_output_rms", "attention output RMS", "log"),
    ]
    for key, ylabel, scale in diagnostics:
        fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        ax.plot(epochs, [max(float(row[key]), 1e-14) for row in history])
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} / {ylabel}")
        ax.set_yscale(scale)
        ax.grid(True, which="both", alpha=0.25)
        fig.savefig(out_dir / f"{key}.png", dpi=180)
        plt.close(fig)

def plot_case_comparison(case_dir: Path, metrics: list[dict]) -> None:
    curve_specs = {
        "objective": ("val_loss", "validation training objective"),
        "data": ("val_mse", "validation data MSE"),
        "normalized": ("val_pde_mse", "validation normalized PDE residual MSE"),
        "absolute": ("val_pde_mse_absolute", "validation absolute PDE residual MSE"),
    }
    curves: dict[str, list[tuple[list[int], list[float], str]]] = {k: [] for k in curve_specs}
    for item in metrics:
        hpath = case_dir / item["variant"] / "history.csv"
        history_rows = []
        with hpath.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                history_rows.append(row)
        label = f"{item['variant']} ({item['params']:,})"
        for key, (column, _) in curve_specs.items():
            if history_rows and column in history_rows[0]:
                epochs, vals = [], []
                for row in history_rows:
                    value = row.get(column, "")
                    if value == "":
                        continue
                    epochs.append(int(row["epoch"]))
                    vals.append(float(value))
                if vals:
                    curves[key].append((epochs, vals, label))

    def draw_case_panel(ax: plt.Axes, curve_set: list[tuple[list[int], list[float], str]], ylabel: str) -> None:
        for epochs, vals, label in curve_set:
            ax.semilogy(epochs, vals, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    available = [(key, ylabel) for key, (_, ylabel) in curve_specs.items() if curves[key]]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5.8 * len(available), 4.6), constrained_layout=True)
        axes_list = np.atleast_1d(axes).tolist()
        for ax, (key, ylabel) in zip(axes_list, available):
            ax.set_title(key)
            draw_case_panel(ax, curves[key], ylabel)
        fig.suptitle(f"{case_dir.name} convergence")
        fig.savefig(case_dir / "comparison_convergence.png", dpi=180)
        plt.close(fig)

    for suffix, ylabel in available:
        fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        ax.set_title(f"{case_dir.name} convergence / {suffix}")
        draw_case_panel(ax, curves[suffix], ylabel)
        fig.savefig(case_dir / f"comparison_convergence_{suffix}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = [m["variant"] for m in metrics]
    rel = [m["test_rel_l2"] for m in metrics]
    ax.bar(labels, rel)
    ax.set_yscale("log")
    ax.set_ylabel("test relative L2")
    ax.set_title(f"{case_dir.name} model comparison")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(case_dir / "comparison_test_rel_l2.png", dpi=180)
    plt.close(fig)


def metric_value(mapping: dict, key: str, default: float = float("nan")) -> float:
    try:
        value = mapping.get(key, default)
    except AttributeError:
        return default
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def best_metric_value(metrics: dict, key: str, default: float = float("nan")) -> float:
    return metric_value(metrics.get("best", {}), key, default)


def write_case_tables(case_dir: Path, metrics: list[dict]) -> None:
    rows = []
    for m in metrics:
        attention = m.get("attention", {})
        rows.append({
            "variant": m["variant"],
            "training_mode": m.get("training_mode", ""),
            "params": m["params"],
            "width": m["width"],
            "best_epoch": m["best"]["epoch"],
            "selection_metric": m.get("best", {}).get("selection_metric", ""),
            "best_val_loss": best_metric_value(m, "val_loss"),
            "best_val_mse": best_metric_value(m, "val_mse"),
            "best_val_pde_mse": best_metric_value(m, "val_pde_mse"),
            "best_val_pde_mse_absolute": best_metric_value(m, "val_pde_mse_absolute"),
            "best_val_state_rel_l2": best_metric_value(m, "val_state_rel_l2", best_metric_value(m, "val_rel_l2")),
            "test_pde_mse": metric_value(m, "test_pde_mse"),
            "test_pde_mse_absolute": metric_value(m, "test_pde_mse_absolute"),
            "test_mse": m["test_mse"],
            "test_rel_l2": m["test_rel_l2"],
            "abs_error_mean": m["abs_error"]["mean_abs"],
            "abs_error_p95": m["abs_error"]["p95_abs"],
            "residual_rms": m["residual_prediction"]["rms"],
            "attention_kind": attention.get("spectral_kind", ""),
            "attention_gate": attention.get("gate", float("nan")),
            "attention_source_to_state_rms": attention.get("source_to_state_rms", float("nan")),
            "attention_output_rms": attention.get("attention_output_rms", float("nan")),
            "elapsed_s": m["elapsed_s"],
        })
    preferred = [
        "variant",
        "training_mode",
        "params",
        "width",
        "best_epoch",
        "selection_metric",
        "best_val_loss",
        "best_val_mse",
        "best_val_pde_mse",
        "best_val_pde_mse_absolute",
        "best_val_state_rel_l2",
        "test_pde_mse",
        "test_pde_mse_absolute",
        "test_mse",
        "test_rel_l2",
        "abs_error_mean",
        "abs_error_p95",
        "residual_rms",
        "attention_kind",
        "attention_gate",
        "attention_source_to_state_rms",
        "attention_output_rms",
        "elapsed_s",
    ]
    fieldnames = [k for k in preferred if any(k in row for row in rows)]
    fieldnames.extend(k for row in rows for k in row.keys() if k not in fieldnames)
    with (case_dir / "model_comparison_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (case_dir / "model_comparison_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_existing_global_rows(run_root: Path, current_keys: set[tuple[str, str, str]]) -> list[dict]:
    path = run_root / "global_comparison.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("pde", ""), row.get("case", ""), row.get("variant", ""))
            if key not in current_keys:
                rows.append(dict(row))
    return rows


def stability_summary(spec: CaseSpec) -> dict:
    dx, dt_or_dy = grid_spacing(spec)
    if not spec.is_transient:
        return {}
    dt = dt_or_dy
    if spec.pde == "Wave":
        c = float(spec.params["c"])
        return {"criterion": "c*dt/dx <= 1", "value": c * dt / dx, "satisfied": c * dt / dx <= 1.0}
    if spec.pde == "KdV":
        value = dt / max(dx**3, 1e-30)
        return {"criterion": "dt/dx^3 <= 0.25", "value": value, "satisfied": value <= 0.25}
    if spec.pde in {"Burgers", "AllenCahn", "ReactionDiffusion"}:
        diff = float(spec.params.get("nu", spec.params.get("epsilon", spec.params.get("diffusion", 0.01))))
        if spec.pde == "AllenCahn":
            diff = diff**2
        value = diff * dt / max(dx**2, 1e-30)
        return {"criterion": "diffusion*dt/dx^2 <= 0.5", "value": value, "satisfied": value <= 0.5}
    return {}


def apply_spec_filters(specs: list[CaseSpec], args: argparse.Namespace) -> list[CaseSpec]:
    if args.pdes:
        wanted = {p.strip().lower() for p in args.pdes.split(",") if p.strip()}
        specs = [s for s in specs if s.pde.lower() in wanted or s.slug.lower() in wanted]
    if args.cases:
        wanted_cases = {c.strip().lower() for c in args.cases.split(",") if c.strip()}
        specs = [
            s
            for s in specs
            if s.case.lower() in wanted_cases or s.slug.lower() in wanted_cases or f"{s.pde}/{s.case}".lower() in wanted_cases
        ]
    return specs


def build_case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    steady_1d_ny = max(16, args.modes * 2)
    common = {
        "modes": args.modes,
        "high_modes": args.high_modes,
        "depth": args.depth,
        "target_params": args.target_params,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "attention_rank": args.attention_rank,
        "attention_gate_init": args.attention_gate_init,
    }
    specs: list[CaseSpec] = []
    if args.profile == "high_nonlinear":
        stress_base = {
            "profile": "high_nonlinear",
            "mode_min": 4,
            "mode_max": 15,
            "mode_count": 5,
            "mode_min_2d": 3,
            "mode_max_2d": 10,
            "mode_count_2d": 5,
            "mode_pair_count": 5,
            "amp_scale": 0.80,
            "amp_scale_2d": 0.70,
            "transient_amp_scale": 0.68,
            "omega_min": 1.0,
            "omega_max": 2.0,
        }

        def stress_params(**extra: float | str) -> dict:
            out = dict(stress_base)
            out.update(extra)
            return out

        specs.extend([
            CaseSpec(
                "KdV",
                "HighNonlinear_Steady1D",
                "steady",
                1,
                2.0 * math.pi,
                1.0,
                None,
                args.grid,
                steady_1d_ny,
                None,
                4,
                **common,
                params=stress_params(nonlinear_coeff=12.0, boundary_amplitude=0.35, omega_min=0.8, omega_max=1.8),
                reference_keys=("pinn",),
            ),
            CaseSpec(
                "KdV",
                "HighNonlinear_Steady2D",
                "steady",
                2,
                2.0 * math.pi,
                2.0 * math.pi,
                None,
                args.grid,
                args.grid,
                None,
                4,
                **common,
                params=stress_params(nonlinear_coeff=12.0, boundary_amplitude=0.30, omega_min=0.8, omega_max=1.8),
                reference_keys=("zk", "pinn"),
            ),
            CaseSpec(
                "KdV",
                "HighNonlinear_Transient1D",
                "transient",
                1,
                2.0 * math.pi,
                1.0,
                0.006,
                args.grid,
                args.grid,
                args.grid,
                5,
                **common,
                params=stress_params(nonlinear_coeff=12.0, boundary_amplitude=0.35, omega_min=0.8, omega_max=1.8),
                reference_keys=("pinn",),
            ),
            CaseSpec(
                "AllenCahn",
                "HighNonlinear_Steady1D",
                "steady",
                1,
                1.0,
                1.0,
                None,
                args.grid,
                steady_1d_ny,
                None,
                4,
                **common,
                params=stress_params(epsilon=0.015, reaction=2.0, boundary_amplitude=0.25),
                reference_keys=("pdebench", "pinn"),
            ),
            CaseSpec(
                "AllenCahn",
                "HighNonlinear_Steady2D",
                "steady",
                2,
                1.0,
                1.0,
                None,
                args.grid,
                args.grid,
                None,
                4,
                **common,
                params=stress_params(epsilon=0.015, reaction=2.0, boundary_amplitude=0.22),
                reference_keys=("pdebench", "pinn"),
            ),
            CaseSpec(
                "AllenCahn",
                "HighNonlinear_Transient1D",
                "transient",
                1,
                1.0,
                1.0,
                0.50,
                args.grid,
                args.grid,
                args.grid,
                5,
                **common,
                params=stress_params(epsilon=0.015, reaction=2.0, boundary_amplitude=0.25),
                reference_keys=("pdebench", "pinn"),
            ),
            CaseSpec(
                "Burgers",
                "HighNonlinear_Steady1D",
                "steady",
                1,
                1.0,
                1.0,
                None,
                args.grid,
                steady_1d_ny,
                None,
                4,
                **common,
                params=stress_params(nu=0.0025, convective_coeff=2.0, boundary_amplitude=0.25),
                reference_keys=("fno", "pdebench", "pinn"),
            ),
            CaseSpec(
                "Burgers",
                "HighNonlinear_Steady2D",
                "steady",
                2,
                1.0,
                1.0,
                None,
                args.grid,
                args.grid,
                None,
                4,
                **common,
                params=stress_params(nu=0.0025, convective_coeff=2.0, boundary_amplitude=0.22),
                reference_keys=("fno", "pdebench", "pinn"),
            ),
            CaseSpec(
                "Burgers",
                "HighNonlinear_Transient1D",
                "transient",
                1,
                1.0,
                1.0,
                0.30,
                args.grid,
                args.grid,
                args.grid,
                5,
                **common,
                params=stress_params(nu=0.0025, convective_coeff=2.0, boundary_amplitude=0.25),
                reference_keys=("fno", "pdebench", "pinn"),
            ),
            CaseSpec(
                "ReactionDiffusion",
                "HighNonlinear_Steady1D",
                "steady",
                1,
                1.0,
                1.0,
                None,
                args.grid,
                steady_1d_ny,
                None,
                4,
                **common,
                params=stress_params(diffusion=0.0025, rate=4.0, boundary_amplitude=0.25),
                reference_keys=("pdebench",),
            ),
            CaseSpec(
                "ReactionDiffusion",
                "HighNonlinear_Steady2D",
                "steady",
                2,
                1.0,
                1.0,
                None,
                args.grid,
                args.grid,
                None,
                4,
                **common,
                params=stress_params(diffusion=0.0025, rate=4.0, boundary_amplitude=0.22),
                reference_keys=("pdebench",),
            ),
            CaseSpec(
                "ReactionDiffusion",
                "HighNonlinear_Transient1D",
                "transient",
                1,
                1.0,
                1.0,
                0.30,
                args.grid,
                args.grid,
                args.grid,
                5,
                **common,
                params=stress_params(diffusion=0.0025, rate=4.0, boundary_amplitude=0.25),
                reference_keys=("pdebench",),
            ),
        ])
        return apply_spec_filters(specs, args)

    # todo load cases here to calc

    specs.extend([
        CaseSpec("Poisson", "Steady1D", "steady", 1, 1.0, 1.0, None, args.grid, steady_1d_ny, None, 3, **common, params={}, reference_keys=("fno",)),
        CaseSpec("Poisson", "Steady2D", "steady", 2, 1.0, 1.0, None, args.grid, args.grid, None, 3, **common, params={}, reference_keys=("fno",)),
        CaseSpec("Wave", "Transient1D", "transient", 1, 1.0, 1.0, 0.80, args.grid, args.grid, args.grid, 5, **common, params={"c": 1.0}, reference_keys=("pdebench",)),
        CaseSpec("KdV", "Steady1D", "steady", 1, 2.0 * math.pi, 1.0, None, args.grid, steady_1d_ny, None, 3, **common, params={}, reference_keys=("pinn",)),
        CaseSpec("KdV", "Steady2D", "steady", 2, 2.0 * math.pi, 2.0 * math.pi, None, args.grid, args.grid, None, 3, **common, params={}, reference_keys=("zk", "pinn")),
        CaseSpec("KdV", "Transient1D", "transient", 1, 2.0 * math.pi, 1.0, 0.01, args.grid, args.grid, args.grid, 4, **common, params={}, reference_keys=("pinn",)),
        CaseSpec("AllenCahn", "Steady1D", "steady", 1, 1.0, 1.0, None, args.grid, steady_1d_ny, None, 3, **common, params={"epsilon": 0.03}, reference_keys=("pdebench", "pinn")),
        CaseSpec("AllenCahn", "Steady2D", "steady", 2, 1.0, 1.0, None, args.grid, args.grid, None, 3, **common, params={"epsilon": 0.03}, reference_keys=("pdebench", "pinn")),
        CaseSpec("AllenCahn", "Transient1D", "transient", 1, 1.0, 1.0, 0.50, args.grid, args.grid, args.grid, 4, **common, params={"epsilon": 0.03}, reference_keys=("pdebench", "pinn")),
        CaseSpec("Burgers", "Steady1D", "steady", 1, 1.0, 1.0, None, args.grid, steady_1d_ny, None, 3, **common, params={"nu": 0.01}, reference_keys=("fno", "pdebench", "pinn")),
        CaseSpec("Burgers", "Steady2D", "steady", 2, 1.0, 1.0, None, args.grid, args.grid, None, 3, **common, params={"nu": 0.01}, reference_keys=("fno", "pdebench", "pinn")),
        CaseSpec("Burgers", "Transient1D", "transient", 1, 1.0, 1.0, 0.30, args.grid, args.grid, args.grid, 4, **common, params={"nu": 0.01}, reference_keys=("fno", "pdebench", "pinn")),
        CaseSpec("ReactionDiffusion", "Steady1D", "steady", 1, 1.0, 1.0, None, args.grid, steady_1d_ny, None, 3, **common, params={"diffusion": 0.01, "rate": 1.0}, reference_keys=("pdebench",)),
        CaseSpec("ReactionDiffusion", "Steady2D", "steady", 2, 1.0, 1.0, None, args.grid, args.grid, None, 3, **common, params={"diffusion": 0.01, "rate": 1.0}, reference_keys=("pdebench",)),
        CaseSpec("ReactionDiffusion", "Transient1D", "transient", 1, 1.0, 1.0, 0.30, args.grid, args.grid, args.grid, 4, **common, params={"diffusion": 0.01, "rate": 1.0}, reference_keys=("pdebench",)),
    ])
    return apply_spec_filters(specs, args)


def split_data(spec: CaseSpec, args: argparse.Namespace, seed: int) -> dict[str, torch.Tensor]:
    total = args.train_samples + args.val_samples + args.test_samples
    raw = make_dataset(spec, total, seed)
    n_train = args.train_samples
    n_val = args.val_samples
    x = raw["x"].clone()
    source_mean = x[:n_train, 0:1].mean()
    source_std = x[:n_train, 0:1].std().clamp_min(1e-6)
    x[:, 0:1] = (x[:, 0:1] - source_mean) / source_std
    return {
        "x_train": x[:n_train],
        "y_train": raw["y"][:n_train],
        "source_train": raw["source"][:n_train],
        "x_val": x[n_train:n_train + n_val],
        "y_val": raw["y"][n_train:n_train + n_val],
        "source_val": raw["source"][n_train:n_train + n_val],
        "x_test": x[n_train + n_val:],
        "y_test": raw["y"][n_train + n_val:],
        "source_test": raw["source"][n_train + n_val:],
        "source_normalization_mean": source_mean.reshape(1),
        "source_normalization_std": source_std.reshape(1),
    }


def write_references(root: Path) -> None:
    with (root / "references.json").open("w", encoding="utf-8") as f:
        json.dump(PAPER_REFERENCES, f, indent=2)
    lines = [
        "# References",
        "",
        "This benchmark uses manufactured exact solutions while keeping PDE forms close to common operator-learning and PINN benchmarks.",
        "Boundary and initial conditions are enforced by hard output projection. No boundary-condition loss term is used.",
        "",
    ]
    for ref in PAPER_REFERENCES:
        lines.append(f"- [{ref['title']}]({ref['url']})")
        lines.append(f"  Used for: {ref['used_for']}")
    (root / "README_REFERENCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_root = Path(args.output_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    write_references(run_root)
    specs = build_case_specs(args)
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    current_keys = {(spec.pde, spec.case, variant) for spec in specs for variant in variants}
    global_rows = load_existing_global_rows(run_root, current_keys) if args.append_existing_global else []
    config = {
        "args": vars(args),
        "device": str(device),
        "torch": torch.__version__,
        "references": PAPER_REFERENCES,
        "notes": [
            "HF branches use gate_mode='legacy'.",
            "F_ATTN, C_ATTN, and HF_ATTN share one CFNO backbone and differ only in the single attention layer's spectral source.",
            "The kernel is input-conditioned and evaluated with normalized linear attention, avoiding an O(N^2) matrix.",
            "All local HF padding uses replicate, not circular.",
            "Width matching raises FNO width to at least HF_FNO parameters and CFNO width to at least HF_CFNO parameters when both family members are present.",
            "Dirichlet boundaries and transient initial conditions are hard projected in the model output.",
            f"Training mode is {args.training_mode}; hybrid loss is data MSE + physics_weight * normalized PDE residual MSE.",
            "Absolute PDE residual loss is logged and plotted in the original equation scale.",
            "Exact fields are used for data/hybrid supervision and for evaluation; pure PDE mode does not use exact fields as supervision.",
            "No boundary-condition loss is present.",
        ],
    }
    config_name = "benchmark_config.json" if args.profile == "baseline" else f"benchmark_config_{args.profile}.json"
    with (run_root / config_name).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    for case_index, spec in enumerate(specs, start=1):
        case_dir = run_root / spec.pde / spec.case
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [{case_index}/{len(specs)}] {spec.slug} on {device} ===")
        widths = match_widths(spec, variants)
        data = split_data(spec, args, args.seed + case_index * 1009)
        case_summary = {
            "spec": asdict(spec),
            "stability": stability_summary(spec),
            "width_matching": widths,
            "data_shapes": {k: list(v.shape) for k, v in data.items()},
            "hard_constraints": (
                "nonzero Dirichlet boundary lift plus masked interior correction; exact t=0 projection for transient cases"
                if is_high_nonlinear(spec)
                else "homogeneous Dirichlet boundary, plus exact t=0 projection for transient cases"
            ),
            "training_objective": args.training_mode,
            "physics_weight": args.physics_weight,
            "parameter_matching": "FNO >= HF_FNO and CFNO >= HF_CFNO when both family members are present",
            "bc_loss": "disabled/not used",
            "profile": args.profile,
        }
        with (case_dir / "case_summary.json").open("w", encoding="utf-8") as f:
            json.dump(case_summary, f, indent=2)
        case_metrics = []
        for model_index, variant in enumerate(variants, start=1):
            out_dir = case_dir / variant
            if args.resume and (out_dir / "metrics.json").exists():
                print(f"Skipping completed {spec.slug}/{variant}")
                with (out_dir / "metrics.json").open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                metrics = train_one_model(
                    spec,
                    variant,
                    widths[variant]["width"],
                    data,
                    out_dir,
                    epochs=args.epochs,
                    device=device,
                    seed=args.seed + case_index * 1009 + model_index * 97,
                    training_mode=args.training_mode,
                    physics_weight=args.physics_weight,
                    local_activation=args.local_activation,
                )
            metrics.setdefault("training_mode", args.training_mode)
            metrics.setdefault("physics_weight", args.physics_weight)
            metrics["pde"] = spec.pde
            metrics["case"] = spec.case
            case_metrics.append(metrics)
            attention = metrics.get("attention", {})
            global_rows.append({
                "pde": spec.pde,
                "case": spec.case,
                "variant": variant,
                "training_mode": metrics.get("training_mode", args.training_mode),
                "params": metrics["params"],
                "width": metrics["width"],
                "best_epoch": metrics["best"]["epoch"],
                "selection_metric": metrics.get("best", {}).get("selection_metric", ""),
                "best_val_loss": best_metric_value(metrics, "val_loss"),
                "best_val_mse": best_metric_value(metrics, "val_mse"),
                "best_val_pde_mse": best_metric_value(metrics, "val_pde_mse"),
                "best_val_pde_mse_absolute": best_metric_value(metrics, "val_pde_mse_absolute"),
                "test_pde_mse": metric_value(metrics, "test_pde_mse"),
                "test_pde_mse_absolute": metric_value(metrics, "test_pde_mse_absolute"),
                "test_mse": metrics["test_mse"],
                "test_rel_l2": metrics["test_rel_l2"],
                "residual_rms": metrics["residual_prediction"]["rms"],
                "attention_kind": attention.get("spectral_kind", ""),
                "attention_gate": attention.get("gate", float("nan")),
                "attention_source_to_state_rms": attention.get("source_to_state_rms", float("nan")),
                "attention_output_rms": attention.get("attention_output_rms", float("nan")),
                "elapsed_s": metrics["elapsed_s"],
            })
        write_case_tables(case_dir, case_metrics)
        plot_case_comparison(case_dir, case_metrics)
        write_global_tables(run_root, global_rows)
    write_global_tables(run_root, global_rows)
    plot_global_summary(run_root, global_rows)
    return run_root


def write_global_tables(run_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    preferred = [
        "pde",
        "case",
        "variant",
        "training_mode",
        "params",
        "width",
        "best_epoch",
        "selection_metric",
        "best_val_loss",
        "best_val_mse",
        "best_val_pde_mse",
        "best_val_pde_mse_absolute",
        "test_pde_mse",
        "test_pde_mse_absolute",
        "test_mse",
        "test_rel_l2",
        "residual_rms",
        "attention_kind",
        "attention_gate",
        "attention_source_to_state_rms",
        "attention_output_rms",
        "elapsed_s",
    ]
    fieldnames = [k for k in preferred if any(k in row for row in rows)]
    fieldnames.extend(k for row in rows for k in row.keys() if k not in fieldnames)
    with (run_root / "global_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (run_root / "global_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_winner_summary(run_root, rows)


def write_winner_summary(run_root: Path, rows: list[dict]) -> None:
    best_by_case: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["pde"], row["case"])
        value = float(row["test_rel_l2"])
        if key not in best_by_case or value < float(best_by_case[key]["test_rel_l2"]):
            best_by_case[key] = row
    winners = []
    for (pde, case), row in sorted(best_by_case.items()):
        winners.append({
            "pde": pde,
            "case": case,
            "winner": row["variant"],
            "test_rel_l2": float(row["test_rel_l2"]),
            "params": int(float(row["params"])),
            "width": int(float(row["width"])),
            "best_epoch": int(float(row["best_epoch"])),
            "residual_rms": float(row["residual_rms"]),
        })
    with (run_root / "global_winner_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(winners[0].keys()))
        writer.writeheader()
        writer.writerows(winners)
    with (run_root / "global_winner_summary.json").open("w", encoding="utf-8") as f:
        json.dump(winners, f, indent=2)


def plot_global_summary(run_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    variants = sorted({r["variant"] for r in rows})
    cases = []
    for r in rows:
        label = f"{r['pde']}/{r['case']}"
        if label not in cases:
            cases.append(label)
    matrix = np.full((len(cases), len(variants)), np.nan)
    for r in rows:
        matrix[cases.index(f"{r['pde']}/{r['case']}"), variants.index(r["variant"])] = float(r["test_rel_l2"])
    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(cases))), constrained_layout=True)
    im = ax.imshow(np.log10(matrix + 1e-16), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(variants)), variants, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(cases)), cases)
    ax.set_title("Global comparison: log10(test relative L2)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(run_root / "global_test_rel_l2_heatmap.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy HF-FNO/HF-CFNO PDE benchmark suite.")
    parser.add_argument("--output-root", default="LegacyHF_PDE_Benchmark_Normalized_20260705")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--high-modes", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--target-params", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--variants", default="FNO,CFNO,HF_FNO,HF_CFNO")
    parser.add_argument("--attention-rank", type=int, default=8)
    parser.add_argument("--attention-gate-init", type=float, default=-2.0)
    parser.add_argument("--pdes", default="", help="Comma-separated PDE or case slug filter. Empty means all.")
    parser.add_argument("--cases", default="", help="Comma-separated case names, PDE/case labels, or full slugs to run.")
    parser.add_argument("--profile", choices=("baseline", "high_nonlinear"), default="baseline")
    parser.add_argument("--training-mode", choices=("data", "pde", "hybrid"), default="pde")
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument(
        "--local-activation",
        default="gelu",
        help="Inner LocalHighPassBlock2d activation; use 'identity' to disable only the nonlinearity.",
    )
    parser.add_argument("--append-existing-global", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1000:
        raise ValueError("--epochs must be at least 1000 for this benchmark.")
    return args


if __name__ == "__main__":
    root = run_suite(parse_args())
    print(f"\nBenchmark complete: {root}")
