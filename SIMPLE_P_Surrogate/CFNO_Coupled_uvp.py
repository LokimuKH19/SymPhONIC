import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from NeuroOperators import seed_everything, FNO2d_small


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
            device="cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.N = N
        self.L = L
        self.rho = rho
        self.mu = mu
        self.lid_velocity = lid_velocity
        self.max_iter = max_iter
        self.tol = tol

        # ---------- 算子网络 FNO ----------
        # 输入：单通道常数场 (lid_velocity) -> 输出：修正前的 (u_hat, v_hat, p_hat)
        self.net = FNO2d_small(
            modes=64, width=32, depth=5,
            input_features=1, output_features=3
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

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
        if H >= 5:
            start = 2
            end = H - 2
            mask_pos = u[:, start:end, :] > 0
            mask_neg = ~mask_pos
            if mask_pos.any():
                term_pos = (3*phi[:, start:end, :][mask_pos] -
                            4*phi[:, start-1:end-1, :][mask_pos] +
                            phi[:, start-2:end-2, :][mask_pos]) / (2 * self.dx)
                dphi_dx[:, start:end, :][mask_pos] = term_pos
            if mask_neg.any():
                term_neg = (-3*phi[:, start:end, :][mask_neg] +
                            4*phi[:, start+1:end+1, :][mask_neg] -
                            phi[:, start+2:end+2, :][mask_neg]) / (2 * self.dx)
                dphi_dx[:, start:end, :][mask_neg] = term_neg
        return dphi_dx

    def _upwind_y(self, phi, v):
        B, H, W = phi.shape
        dphi_dy = self._gradient_y_central(phi)
        if H >= 5:
            start = 2
            end = H - 2
            mask_pos = v[:, start:end, :] > 0
            mask_neg = ~mask_pos
            if mask_pos.any():
                term_pos = (3*phi[:, start:end, :][mask_pos] -
                            4*phi[:, start-1:end-1, :][mask_pos] +
                            phi[:, start-2:end-2, :][mask_pos]) / (2 * self.dy)
                dphi_dy[:, start:end, :][mask_pos] = term_pos
            if mask_neg.any():
                term_neg = (-3*phi[:, start:end, :][mask_neg] +
                            4*phi[:, start+1:end+1, :][mask_neg] -
                            phi[:, start+2:end+2, :][mask_neg]) / (2 * self.dy)
                dphi_dy[:, start:end, :][mask_neg] = term_neg
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

    # ---------- 求解主循环 ----------
    def solve(self):
        for it in range(self.max_iter):
            # 前向传播：网络输入为常数场
            out = self.net(self.input_field)              # [1, 3, H, W]
            u_hat = out[:, 0:1, :, :]                     # [1,1,H,W]
            v_hat = out[:, 1:2, :, :]
            p_hat = out[:, 2:3, :, :]

            # 施加边界条件
            u, v, p = self.apply_bc(u_hat, v_hat, p_hat)  # [1,1,H,W]

            # 转换为 [B,H,W] 形式
            u_s = u.squeeze(1)
            v_s = v.squeeze(1)
            p_s = p.squeeze(1)

            # 对流项导数（迎风）
            u_x = self._upwind_x(u_s, u_s)
            u_y = self._upwind_y(u_s, v_s)
            v_x = self._upwind_x(v_s, u_s)
            v_y = self._upwind_y(v_s, v_s)

            # 压力梯度（中心差分）
            p_x = self._gradient_x_central(p_s)
            p_y = self._gradient_y_central(p_s)

            # 扩散项（拉普拉斯）
            lap_u = self._laplacian(u_s)
            lap_v = self._laplacian(v_s)

            # 连续性
            div = u_x + v_y
            L_cont = torch.mean(div ** 2)

            # 动量方程
            conv_u = u_s * u_x + v_s * u_y
            conv_v = u_s * v_x + v_s * v_y
            mom_x = self.rho * conv_u + p_x - self.mu * lap_u
            mom_y = self.rho * conv_v + p_y - self.mu * lap_v
            L_mom = torch.mean(mom_x ** 2 + mom_y ** 2)

            loss = L_cont + L_mom

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if it % 100 == 0:
                print(f"iter {it:5d}, L_cont: {L_cont.item():.3e}, L_mom: {L_mom.item():.3e}")

            if loss.item() < self.tol:
                break

        # 保存结果
        with torch.no_grad():
            out = self.net(self.input_field)
            u_hat = out[:, 0:1, :, :]
            v_hat = out[:, 1:2, :, :]
            p_hat = out[:, 2:3, :, :]
            u, v, p = self.apply_bc(u_hat, v_hat, p_hat)

        self.U = u.squeeze().cpu().numpy()
        self.V = v.squeeze().cpu().numpy()
        self.P = p.squeeze().cpu().numpy()

    # ---------- 绘图 ----------
    def plot(self):
        N = self.N
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)

        U = self.U
        V = self.V
        speed = np.sqrt(U ** 2 + V ** 2)

        plt.figure(figsize=(6, 6))
        plt.streamplot(X, Y, U, V, density=2)
        plt.title("Streamlines")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, speed, 20, cmap="jet")
        plt.colorbar()
        plt.title("Speed magnitude")
        plt.show()


if __name__ == "__main__":
    seed_everything(10492)
    solver = NeuralCavityPressure(
        N=129,
        rho=1,
        mu=0.01,
        lid_velocity=1,
        tol=1e-8,
        max_iter=5000,
        device="cuda"
    )
    t1 = time.time()
    solver.solve()
    t2 = time.time()
    print(f"Time Consumed: {t2 - t1}s")
    solver.plot()