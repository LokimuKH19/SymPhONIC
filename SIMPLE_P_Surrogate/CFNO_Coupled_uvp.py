import time
import argparse
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
from NeuroOperators import seed_everything, FNO2d_small, CFNO2d_small, HF_CFNO2d_small, HF_FNO2d_small


class NeuralCavityPressure:
    def __init__(
            self,
            N=64,
            L=1.0,
            rho=1.0,
            mu=0.01,
            lid_velocity=1.0,
            max_iter=5000,
            tol=1e-6,
            device="cuda",
            operator_variant="hf_cfno",
            output_mode="streamfunction",
            lr=1e-3,
            interior_margin=1,
            modes=16,
            high_modes=32,
            width=24,
            depth=5,
            boundary_mode="replicate",
            high_gate_init=-2.0,
            gate_threshold=1.0,
            gate_slope=2.0,
            gate_subgrid_weight=1.0,
            gate_alignment_weight=1e-2,
            gate_mode="subgrid",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.N = N
        self.L = L
        self.rho = rho
        self.mu = mu
        self.lid_velocity = lid_velocity
        self.reynolds = self.rho * self.lid_velocity * self.L / self.mu
        self.max_iter = max_iter
        self.tol = tol
        self.operator_variant = str(operator_variant).lower()
        self.output_mode = str(output_mode).lower()
        if self.operator_variant not in {"fno", "cfno", "hf_cfno", "hf_fno"}:
            raise ValueError("operator_variant must be one of: fno, cfno, hf_cfno, hf_fno.")
        if self.output_mode not in {"streamfunction", "uvp"}:
            raise ValueError("output_mode must be 'streamfunction' or 'uvp'.")
        self.interior_margin = int(max(interior_margin, 0))
        self.modes = int(modes)
        self.high_modes = int(high_modes)
        self.width = int(width)
        self.depth = int(depth)
        self.boundary_mode = str(boundary_mode).lower()
        if self.boundary_mode not in {"replicate", "reflect", "circular", "legacy_mixed"}:
            raise ValueError("boundary_mode must be one of: replicate, reflect, circular, legacy_mixed.")
        self.high_gate_init = float(high_gate_init)
        self.gate_threshold = float(gate_threshold)
        self.gate_slope = float(gate_slope)
        self.gate_subgrid_weight = float(gate_subgrid_weight)
        self.gate_alignment_weight = float(gate_alignment_weight)
        self.gate_mode = str(gate_mode).lower()
        if self.gate_mode not in {"subgrid", "legacy", "subgrid_gated_fuse"}:
            raise ValueError("gate_mode must be 'subgrid', 'legacy', or 'subgrid_gated_fuse'.")

        # ---------- 算子网络 FNO ----------
        # 输入：单通道常数场 (lid_velocity) -> 输出：修正前的 (u_hat, v_hat, p_hat)
        output_features = 2 if self.output_mode == "streamfunction" else 3
        boundary_mode_h = "circular" if self.boundary_mode == "legacy_mixed" else self.boundary_mode
        boundary_mode_w = "replicate" if self.boundary_mode == "legacy_mixed" else self.boundary_mode
        if self.operator_variant == "fno":
            net_cls = FNO2d_small
            net_kwargs = dict(
                modes=self.modes,
                width=self.width,
                depth=self.depth,
                input_features=1,
                output_features=output_features,
                fourier_feature_bands=(1, 2, 4, 8),
            )
        elif self.operator_variant == "cfno":
            net_cls = CFNO2d_small
            net_kwargs = dict(
                modes=self.modes,
                cheb_modes=(self.modes, self.modes),
                width=self.width,
                depth=self.depth,
                input_features=1,
                output_features=output_features,
                fourier_feature_bands=(1, 2, 4, 8),
            )
        else:
            net_cls = HF_CFNO2d_small if self.operator_variant == "hf_cfno" else HF_FNO2d_small
            net_kwargs = dict(
                modes=self.modes,
                high_modes=self.high_modes,
                width=self.width,
                depth=self.depth,
                input_features=1,
                output_features=output_features,
                fourier_feature_bands=(1, 2, 4, 8),
                high_gate_init=self.high_gate_init,
                use_local_highpass=True,
                grid_spacing=(1.0 / (N - 1), 1.0 / (N - 1)),
                boundary_mode_h=boundary_mode_h,
                boundary_mode_w=boundary_mode_w,
                gate_threshold=self.gate_threshold,
                gate_slope=self.gate_slope,
                gate_subgrid_weight=self.gate_subgrid_weight,
                use_vorticity_gate=True,
                gate_mode=self.gate_mode,
            )
            if self.operator_variant == "hf_cfno":
                net_kwargs["cheb_modes"] = (self.modes, self.modes)
        self.net = net_cls(**net_kwargs).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        param_count = sum(p.numel() * (2 if p.is_complex() else 1) for p in self.net.parameters())
        print(
            f"Network: {net_cls.__name__}(modes={self.modes}, high_modes={self.high_modes}, "
            f"width={self.width}, depth={self.depth}, output_mode={self.output_mode}, "
            f"Re={self.reynolds:.6g}, mu={self.mu:.6e})"
        )
        if self.operator_variant.startswith("hf_"):
            print(
                "High-pass gate: "
                f"boundary_mode={self.boundary_mode}, high_gate_init={self.high_gate_init}, "
                f"gate_mode={self.gate_mode}, "
                f"threshold={self.gate_threshold}, slope={self.gate_slope}, "
                f"subgrid_weight={self.gate_subgrid_weight}, "
                f"alignment_weight={self.gate_alignment_weight}"
            )
        else:
            print("High-pass gate: disabled for plain operator.")
        print(f"Real-equivalent trainable parameters: {param_count:,}")

        # 构建网格坐标（仅用于边界掩码的位置信息，不作为网络输入）
        x = torch.linspace(0, 1, N, device=self.device)
        y = torch.linspace(0, 1, N, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.xy_grid = torch.stack([X, Y], dim=0).unsqueeze(0).requires_grad_(False)  # [1,2,H,W]

        # 构造网络输入：常数场，值为 lid_velocity，形状 [1,1,H,W]
        self.input_field = torch.ones(1, 1, N, N, device=self.device) * lid_velocity

        # 网格步长
        self.dx = 1.0 / (N - 1)
        self.dy = 1.0 / (N - 1)

        # ---------- 预计算边界掩码 ----------
        mask = torch.zeros(1, 1, N, N, device=self.device, dtype=torch.bool)
        mask[:, :, 0, :] = True   # 下边界
        mask[:, :, -1, :] = True  # 上边界
        mask[:, :, :, 0] = True   # 左边界
        mask[:, :, :, -1] = True  # 右边界
        self.boundary_mask = mask.float()

        top_mask = torch.zeros(1, 1, N, N, device=self.device, dtype=torch.float)
        top_mask[:, :, -1, :] = 1.0
        self.top_mask = top_mask

        # 压力参考点掩码：p(0,0) = 0
        ref_mask = torch.zeros(1, 1, N, N, device=self.device, dtype=torch.float)
        ref_mask[:, :, 0, 0] = 1.0
        self.ref_mask = ref_mask

    # ---------- 基础差分算子 (中心差分，用于扩散项和压力梯度) ----------
    def _gradient_x_central(self, f):
        if f.dim() == 4:
            f = f.squeeze(1)
        B, H, W = f.shape
        fx = torch.zeros_like(f)
        fx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * self.dx)
        fx[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / self.dx
        fx[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / self.dx
        return fx

    def _gradient_y_central(self, f):
        if f.dim() == 4:
            f = f.squeeze(1)
        B, H, W = f.shape
        fy = torch.zeros_like(f)
        fy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * self.dy)
        fy[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / self.dy
        fy[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / self.dy
        return fy

    def _laplacian(self, f):
        if f.dim() == 4:
            f = f.squeeze(1)
        B, H, W = f.shape
        fxx = torch.zeros_like(f)
        fxx[:, :, 1:-1] = (f[:, :, 2:] - 2*f[:, :, 1:-1] + f[:, :, :-2]) / (self.dx**2)
        fxx[:, :, 0] = (f[:, :, 2] - 2*f[:, :, 1] + f[:, :, 0]) / (self.dx**2)
        fxx[:, :, -1] = (f[:, :, -1] - 2*f[:, :, -2] + f[:, :, -3]) / (self.dx**2)

        fyy = torch.zeros_like(f)
        fyy[:, 1:-1, :] = (f[:, 2:, :] - 2*f[:, 1:-1, :] + f[:, :-2, :]) / (self.dy**2)
        fyy[:, 0, :] = (f[:, 2, :] - 2*f[:, 1, :] + f[:, 0, :]) / (self.dy**2)
        fyy[:, -1, :] = (f[:, -1, :] - 2*f[:, -2, :] + f[:, -3, :]) / (self.dy**2)

        return fxx + fyy

    # ---------- 迎风格式 (二阶迎风用于对流项) ----------
    def _upwind_x(self, phi, u):
        B, H, W = phi.shape
        dphi_dx = self._gradient_x_central(phi)
        if W >= 5:
            start = 2
            end = W - 2
            term_pos = (
                3 * phi[:, :, start:end]
                - 4 * phi[:, :, start - 1:end - 1]
                + phi[:, :, start - 2:end - 2]
            ) / (2 * self.dx)
            term_neg = (
                -3 * phi[:, :, start:end]
                + 4 * phi[:, :, start + 1:end + 1]
                - phi[:, :, start + 2:end + 2]
            ) / (2 * self.dx)
            dphi_dx[:, :, start:end] = torch.where(u[:, :, start:end] > 0, term_pos, term_neg)
        return dphi_dx

    def _upwind_y(self, phi, v):
        B, H, W = phi.shape
        dphi_dy = self._gradient_y_central(phi)
        if H >= 5:
            start = 2
            end = H - 2
            term_pos = (
                3 * phi[:, start:end, :]
                - 4 * phi[:, start - 1:end - 1, :]
                + phi[:, start - 2:end - 2, :]
            ) / (2 * self.dy)
            term_neg = (
                -3 * phi[:, start:end, :]
                + 4 * phi[:, start + 1:end + 1, :]
                - phi[:, start + 2:end + 2, :]
            ) / (2 * self.dy)
            dphi_dy[:, start:end, :] = torch.where(v[:, start:end, :] > 0, term_pos, term_neg)
        return dphi_dy

    # ---------- 边界条件施加 (硬约束掩码) ----------
    def apply_bc(self, u_hat, v_hat, p_hat):
        """
        输入: u_hat, v_hat, p_hat: [1,1,H,W]
        输出: u, v, p: [1,1,H,W] 满足边界条件
        """
        u_boundary = torch.zeros_like(u_hat)
        u_boundary += self.lid_velocity * self.top_mask   # 顶盖赋值
        v_boundary = torch.zeros_like(v_hat)
        p_boundary = torch.zeros_like(p_hat)              # 压力参考点 0

        u = u_hat * (1 - self.boundary_mask) + u_boundary * self.boundary_mask
        v = v_hat * (1 - self.boundary_mask) + v_boundary * self.boundary_mask
        p = p_hat * (1 - self.ref_mask) + p_boundary * self.ref_mask

        return u, v, p

    def apply_pressure_reference(self, p_hat):
        # Pure physics mode should not constrain pressure on the walls; only
        # one reference point removes the arbitrary pressure constant.
        return p_hat * (1 - self.ref_mask)

    def fields_from_network(self):
        out = self.net(self.input_field)
        if self.output_mode == "streamfunction":
            psi = out[:, 0:1, :, :]
            p_hat = out[:, 1:2, :, :]
            psi_s = psi.squeeze(1)
            u_hat = self._gradient_y_central(psi_s).unsqueeze(1)
            v_hat = -self._gradient_x_central(psi_s).unsqueeze(1)
        else:
            u_hat = out[:, 0:1, :, :]
            v_hat = out[:, 1:2, :, :]
            p_hat = out[:, 2:3, :, :]
        u, v, p = self.apply_bc(u_hat, v_hat, p_hat)
        return u, v, self.apply_pressure_reference(p)

    def high_pass_gate_summary(self):
        if hasattr(self.net, "high_pass_gate_summary"):
            return self.net.high_pass_gate_summary()
        return None

    def high_pass_spatial_gate_summary(self):
        if hasattr(self.net, "high_pass_spatial_gate_summary"):
            return self.net.high_pass_spatial_gate_summary()
        return None

    def physical_vorticity_gate(self, omega):
        delta = 3.0 * (self.dx * self.dy) ** 0.5
        activity = delta * torch.sqrt(omega.pow(2) + 1e-12)
        eta = activity / (activity.mean(dim=(-2, -1), keepdim=True) + 1e-12)
        return torch.sigmoid(self.gate_slope * (eta - self.gate_threshold))

    def interior(self, f):
        margin = self.interior_margin
        if margin <= 0 or f.shape[-1] <= 2 * margin or f.shape[-2] <= 2 * margin:
            return f
        return f[:, margin:-margin, margin:-margin]

    # ---------- 求解主循环 ----------
    def solve(self):
        self.history = []
        best_loss = float("inf")
        best_state = None
        for it in range(self.max_iter):
            # 前向传播：网络输入为常数场
            u, v, p = self.fields_from_network()
            

            # 施加边界条件
            

            # 转换为 [B,H,W] 形式
            u_s = u.squeeze(1)
            v_s = v.squeeze(1)
            p_s = p.squeeze(1)

            # 对流项导数（迎风）
            u_x = self._upwind_x(u_s, u_s)
            u_y = self._upwind_y(u_s, v_s)
            v_x = self._upwind_x(v_s, u_s)
            v_y = self._upwind_y(v_s, v_s)
            div_x = self._gradient_x_central(u_s)
            div_y = self._gradient_y_central(v_s)

            # 压力梯度（中心差分）
            p_x = self._gradient_x_central(p_s)
            p_y = self._gradient_y_central(p_s)

            # 扩散项（拉普拉斯）
            lap_u = self._laplacian(u_s)
            lap_v = self._laplacian(v_s)

            # 连续性
            div = div_x + div_y
            L_cont = torch.mean(self.interior(div) ** 2)

            # 动量方程
            conv_u = u_s * u_x + v_s * u_y
            conv_v = u_s * v_x + v_s * v_y
            mom_x = self.rho * conv_u + p_x - self.mu * lap_u
            mom_y = self.rho * conv_v + p_y - self.mu * lap_v
            L_mom = torch.mean(self.interior(mom_x) ** 2 + self.interior(mom_y) ** 2)

            L_gate = torch.zeros((), device=self.device, dtype=L_cont.dtype)
            if (
                self.gate_mode in {"subgrid", "subgrid_gated_fuse"}
                and self.gate_alignment_weight > 0.0
                and hasattr(self.net, "high_pass_spatial_gate_map")
            ):
                spatial_gate_map = self.net.high_pass_spatial_gate_map()
                if spatial_gate_map is not None:
                    omega = v_x - u_y
                    target_gate = self.physical_vorticity_gate(omega)
                    L_gate = torch.mean((spatial_gate_map - target_gate.detach()) ** 2)

            loss = L_cont + L_mom + self.gate_alignment_weight * L_gate
            gate_summary = self.high_pass_gate_summary() or {}
            gate_mean = float(gate_summary.get("mean", float("nan")))
            gate_min = float(gate_summary.get("min", float("nan")))
            gate_max = float(gate_summary.get("max", float("nan")))
            spatial_summary = self.high_pass_spatial_gate_summary() or {}
            spatial_gate_mean = float(spatial_summary.get("mean", float("nan")))
            spatial_gate_min = float(spatial_summary.get("min", float("nan")))
            spatial_gate_max = float(spatial_summary.get("max", float("nan")))

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.opt.step()
            self.history.append((
                it,
                float(L_cont.detach().cpu()),
                float(L_mom.detach().cpu()),
                float(loss.detach().cpu()),
                float(L_gate.detach().cpu()),
                gate_mean,
                gate_min,
                gate_max,
                spatial_gate_mean,
                spatial_gate_min,
                spatial_gate_max,
            ))
            current_loss = float(loss.detach().cpu())
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.net.state_dict().items()})

            if it % 100 == 0:
                gate_text = ""
                if np.isfinite(gate_mean):
                    gate_text = (
                        f", gate_mean: {gate_mean:.3e}, "
                        f"gate_range: [{gate_min:.3e}, {gate_max:.3e}]"
                    )
                if np.isfinite(spatial_gate_mean):
                    gate_text += (
                        f", spatial_gate_mean: {spatial_gate_mean:.3e}, "
                        f"spatial_gate_range: [{spatial_gate_min:.3e}, {spatial_gate_max:.3e}]"
                    )
                print(
                    f"iter {it:5d}, L_cont: {L_cont.item():.3e}, "
                    f"L_mom: {L_mom.item():.3e}, L_gate: {L_gate.item():.3e}, "
                    f"loss: {loss.item():.3e}{gate_text}"
                )

            if not torch.isfinite(loss):
                print(f"Non-finite loss at iter {it}; stopping.")
                break

            if loss.item() < self.tol:
                break

        # 保存结果
        if best_state is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            print(f"Restored best network state with loss={best_loss:.6e}")

        with torch.no_grad():
            u, v, p = self.fields_from_network()

        self.U = u.squeeze().cpu().numpy()
        self.V = v.squeeze().cpu().numpy()
        self.P = p.squeeze().cpu().numpy()

    # ---------- 绘图 ----------
    def plot(self, output_dir="HF_CFNO_uvp_results", show=False):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.operator_variant}_{self.output_mode}"
        N = self.N
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)

        U = self.U
        V = self.V
        P = self.P
        speed = np.sqrt(U ** 2 + V ** 2)
        omega = np.gradient(V, self.dx, axis=1) - np.gradient(U, self.dy, axis=0)
        np.save(output_dir / f"{prefix}_U.npy", U)
        np.save(output_dir / f"{prefix}_V.npy", V)
        np.save(output_dir / f"{prefix}_P.npy", P)
        np.save(output_dir / f"{prefix}_Speed.npy", speed)
        np.save(output_dir / f"{prefix}_Vorticity.npy", omega)

        mid = N // 2
        centerline = np.column_stack([y, U[:, mid], x, V[mid, :]])
        np.savetxt(
            output_dir / f"{prefix}_centerline.csv",
            centerline,
            delimiter=",",
            header="y,u_at_x_mid,x,v_at_y_mid",
            comments="",
        )

        plt.figure(figsize=(6, 6))
        plt.streamplot(X, Y, U, V, density=2)
        plt.title(f"Streamlines, Re={self.reynolds:.3g}")
        plt.savefig(output_dir / f"{prefix}_Streamlines.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, speed, 20, cmap="jet")
        plt.colorbar()
        plt.title(f"Speed magnitude, Re={self.reynolds:.3g}")
        plt.savefig(output_dir / f"{prefix}_Speed.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, omega, 30, cmap="coolwarm")
        plt.colorbar()
        plt.title(f"Vorticity, Re={self.reynolds:.3g}")
        plt.savefig(output_dir / f"{prefix}_Vorticity.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        u_mid = U[:, mid]
        v_mid = V[mid, :]
        if np.isclose(self.reynolds, 100.0, rtol=1e-6, atol=1e-8):
            ghia_u = np.array([
                [1.0000, 1.00000],
                [0.9766, 0.84123],
                [0.9688, 0.78871],
                [0.9609, 0.73722],
                [0.9531, 0.68717],
                [0.8516, 0.23151],
                [0.7344, 0.00332],
                [0.6172, -0.13641],
                [0.5000, -0.20581],
                [0.4531, -0.21090],
                [0.2813, -0.15662],
                [0.1719, -0.10150],
                [0.1016, -0.06434],
                [0.0703, -0.04775],
                [0.0625, -0.04192],
                [0.0547, -0.03717],
                [0.0000, 0.00000],
            ], dtype=np.float64)
            ghia_v = np.array([
                [1.0000, 0.00000],
                [0.9688, -0.05906],
                [0.9609, -0.07391],
                [0.9531, -0.08864],
                [0.9453, -0.10313],
                [0.9063, -0.16914],
                [0.8594, -0.22445],
                [0.8047, -0.24533],
                [0.5000, 0.05454],
                [0.2344, 0.17527],
                [0.2266, 0.17507],
                [0.1563, 0.16077],
                [0.0938, 0.12317],
                [0.0781, 0.10890],
                [0.0703, 0.10091],
                [0.0625, 0.09233],
                [0.0000, 0.00000],
            ], dtype=np.float64)
            ghia_u_sorted = ghia_u[np.argsort(ghia_u[:, 0])]
            ghia_v_sorted = ghia_v[np.argsort(ghia_v[:, 0])]
            u_interp = np.interp(ghia_u_sorted[:, 0], y, u_mid)
            v_interp = np.interp(ghia_v_sorted[:, 0], x, v_mid)
            benchmark = np.concatenate([
                np.column_stack([
                    np.zeros(len(ghia_u_sorted), dtype=np.int32),
                    ghia_u_sorted[:, 0],
                    ghia_u_sorted[:, 1],
                    u_interp,
                    u_interp - ghia_u_sorted[:, 1],
                ]),
                np.column_stack([
                    np.ones(len(ghia_v_sorted), dtype=np.int32),
                    ghia_v_sorted[:, 0],
                    ghia_v_sorted[:, 1],
                    v_interp,
                    v_interp - ghia_v_sorted[:, 1],
                ]),
            ])
            np.savetxt(
                output_dir / f"{prefix}_ghia_re100_centerline_error.csv",
                benchmark,
                delimiter=",",
                header="component(0=u_xmid,1=v_ymid),coord,ghia,pred,error",
                comments="",
            )
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(u_mid, y, label="model")
            plt.scatter(ghia_u[:, 1], ghia_u[:, 0], s=18, label="Ghia Re=100")
            plt.xlabel("u at x=0.5")
            plt.ylabel("y")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(x, v_mid, label="model")
            plt.scatter(ghia_v[:, 0], ghia_v[:, 1], s=18, label="Ghia Re=100")
            plt.xlabel("x")
            plt.ylabel("v at y=0.5")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{prefix}_GhiaCenterline.png", dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
        else:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(u_mid, y, label="model")
            plt.xlabel("u at x=0.5")
            plt.ylabel("y")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(x, v_mid, label="model")
            plt.xlabel("x")
            plt.ylabel("v at y=0.5")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{prefix}_Centerline.png", dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()

        if hasattr(self, "history") and self.history:
            history = np.array(self.history, dtype=np.float64)
            np.savetxt(
                output_dir / f"{prefix}_history.csv",
                history,
                delimiter=",",
                header=(
                    "iter,L_cont,L_mom,loss,L_gate,"
                    "gate_mean,gate_min,gate_max,"
                    "spatial_gate_mean,spatial_gate_min,spatial_gate_max"
                ),
                comments="",
            )
            plt.figure(figsize=(7, 4))
            plt.semilogy(history[:, 0], history[:, 1], label="continuity")
            plt.semilogy(history[:, 0], history[:, 2], label="momentum")
            plt.semilogy(history[:, 0], history[:, 3], label="total")
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{prefix}_Loss.png", dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()

            if history.shape[1] >= 8 and np.isfinite(history[:, 5]).any():
                plt.figure(figsize=(7, 4))
                plt.plot(history[:, 0], history[:, 5], label="mean")
                plt.plot(history[:, 0], history[:, 6], label="min", alpha=0.75)
                plt.plot(history[:, 0], history[:, 7], label="max", alpha=0.75)
                plt.xlabel("iteration")
                plt.ylabel("high-pass gate")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f"{prefix}_Gate.png", dpi=200, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close()

            if history.shape[1] >= 11 and np.isfinite(history[:, 8]).any():
                plt.figure(figsize=(7, 4))
                plt.plot(history[:, 0], history[:, 8], label="mean")
                plt.plot(history[:, 0], history[:, 9], label="min", alpha=0.75)
                plt.plot(history[:, 0], history[:, 10], label="max", alpha=0.75)
                plt.xlabel("iteration")
                plt.ylabel("spatial high-pass gate")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f"{prefix}_SpatialGate.png", dpi=200, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close()
            elif history.shape[1] >= 7 and np.isfinite(history[:, 4]).any():
                plt.figure(figsize=(7, 4))
                plt.plot(history[:, 0], history[:, 4], label="mean")
                plt.plot(history[:, 0], history[:, 5], label="min", alpha=0.75)
                plt.plot(history[:, 0], history[:, 6], label="max", alpha=0.75)
                plt.xlabel("iteration")
                plt.ylabel("high-pass gate")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f"{prefix}_Gate.png", dpi=200, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close()

        if hasattr(self.net, "high_pass_spatial_gate_map"):
            spatial_gate_map = self.net.high_pass_spatial_gate_map()
            if spatial_gate_map is not None:
                spatial_gate_np = spatial_gate_map[0].detach().cpu().numpy()
                np.save(output_dir / f"{prefix}_SpatialGateMap.npy", spatial_gate_np)
                plt.figure(figsize=(6, 6))
                plt.contourf(X, Y, spatial_gate_np, 20, cmap="viridis")
                plt.colorbar()
                plt.title("Spatial high-pass gate")
                plt.savefig(output_dir / f"{prefix}_SpatialGateMap.png", dpi=200, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close()

        if hasattr(self.net, "high_pass_gate_map"):
            gate_map = self.net.high_pass_gate_map()
            if gate_map is not None:
                gate_np = gate_map[0].detach().cpu().numpy()
                np.save(output_dir / f"{prefix}_GateMap.npy", gate_np)
                plt.figure(figsize=(6, 6))
                plt.contourf(X, Y, gate_np, 20, cmap="viridis")
                plt.colorbar()
                plt.title("High-pass gate")
                plt.savefig(output_dir / f"{prefix}_GateMap.png", dpi=200, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close()

        v_x_np = np.gradient(V, self.dx, axis=1)
        u_y_np = np.gradient(U, self.dy, axis=0)
        omega_np = v_x_np - u_y_np
        delta = 3.0 * (self.dx * self.dy) ** 0.5
        activity_np = delta * np.sqrt(omega_np ** 2 + 1e-12)
        eta_np = activity_np / (np.mean(activity_np) + 1e-12)
        physical_gate_np = 1.0 / (1.0 + np.exp(-self.gate_slope * (eta_np - self.gate_threshold)))
        np.save(output_dir / f"{prefix}_PhysicalGateTarget.npy", physical_gate_np)
        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, physical_gate_np, 20, cmap="viridis")
        plt.colorbar()
        plt.title("Physical vorticity gate target")
        plt.savefig(output_dir / f"{prefix}_PhysicalGateTarget.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=["fno", "cfno", "hf_cfno", "hf_fno"], default="hf_cfno")
    parser.add_argument("--output-mode", choices=["streamfunction", "uvp"], default="streamfunction")
    parser.add_argument("--n", type=int, default=129)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--lid-velocity", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--re", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--high-modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--interior-margin", type=int, default=1)
    parser.add_argument("--boundary-mode", choices=["replicate", "reflect", "circular", "legacy_mixed"], default="replicate")
    parser.add_argument("--gate-mode", choices=["subgrid", "legacy", "subgrid_gated_fuse"], default="subgrid")
    parser.add_argument("--high-gate-init", type=float, default=-2.0)
    parser.add_argument("--gate-threshold", type=float, default=1.0)
    parser.add_argument("--gate-slope", type=float, default=2.0)
    parser.add_argument("--gate-subgrid-weight", type=float, default=1.0)
    parser.add_argument("--gate-alignment-weight", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=10492)
    args = parser.parse_args()

    mu = args.mu
    if args.re is not None:
        if args.re <= 0:
            raise ValueError("--re must be positive.")
        mu = args.rho * args.lid_velocity * args.length / args.re

    seed_everything(args.seed)
    solver = NeuralCavityPressure(
        N=args.n,
        L=args.length,
        rho=args.rho,
        mu=mu,
        lid_velocity=args.lid_velocity,
        tol=args.tol,
        max_iter=args.max_iter,
        device=args.device,
        operator_variant=args.operator,
        output_mode=args.output_mode,
        lr=args.lr,
        interior_margin=args.interior_margin,
        modes=args.modes,
        high_modes=args.high_modes,
        width=args.width,
        depth=args.depth,
        boundary_mode=args.boundary_mode,
        high_gate_init=args.high_gate_init,
        gate_threshold=args.gate_threshold,
        gate_slope=args.gate_slope,
        gate_subgrid_weight=args.gate_subgrid_weight,
        gate_alignment_weight=args.gate_alignment_weight,
        gate_mode=args.gate_mode,
    )
    t1 = time.time()
    solver.solve()
    t2 = time.time()
    print(f"Time Consumed: {t2 - t1}s")
    output_dir = args.output_dir or f"{args.operator}_{args.output_mode}_results"
    solver.plot(output_dir=output_dir, show=False)
