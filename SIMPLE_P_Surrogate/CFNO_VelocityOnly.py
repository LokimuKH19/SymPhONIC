import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from NeuroOperators import seed_everything, CFNO2d_small


class NeuralCavity:
    def __init__(self,
                 N=64,
                 rho=1.0,
                 mu=0.01,
                 lid_velocity=1.0,
                 weak_iter=5000,
                 strong_iter=1000,
                 tol=1e-6,
                 device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.N = N
        self.rho = rho
        self.mu = mu
        self.U = lid_velocity
        self.weak_iter = weak_iter
        self.strong_iter = strong_iter
        self.tol = tol

        # 构建均匀网格坐标（用于边界条件掩码的坐标信息）
        x = torch.linspace(0, 1, N, device=self.device)
        y = torch.linspace(0, 1, N, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.xy_grid = torch.stack([X, Y], dim=0).unsqueeze(0).requires_grad_(False)

        # 构造网络输入：常数场，值为 lid_velocity，形状 [1,1,H,W]
        self.input_field = torch.ones(1, 1, N, N, device=self.device) * lid_velocity

        # 算子网络: CFNO2d_small，输入为单通道场，输出为修正前的速度场 (2通道)
        self.net = CFNO2d_small(modes=8, cheb_modes=(8,8), width=16, depth=3,
                                input_features=1, output_features=2).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        # 网格步长
        self.dx = 1.0 / (N - 1)
        self.dy = 1.0 / (N - 1)

        # 预计算边界掩码（用于边界条件硬约束）
        mask = torch.zeros(1, 1, N, N, device=self.device, dtype=torch.bool)
        mask[:, :, 0, :] = True   # 下边界
        mask[:, :, -1, :] = True  # 上边界
        mask[:, :, :, 0] = True   # 左边界
        mask[:, :, :, -1] = True  # 右边界
        self.boundary_mask = mask.float()

        top_mask = torch.zeros(1, 1, N, N, device=self.device, dtype=torch.float)
        top_mask[:, :, -1, :] = 1.0
        self.top_mask = top_mask

    # ---------- 有限差分工具 (二阶中心差分，边界一阶) ----------
    def _gradient_x(self, f):
        """计算 ∂f/∂x, f: [B,1,H,W] 或 [B,H,W] 返回 [B,H,W]"""
        if f.dim() == 4:
            f = f.squeeze(1)
        fx = torch.zeros_like(f)
        fx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * self.dx)
        fx[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / self.dx
        fx[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / self.dx
        return fx

    def _gradient_y(self, f):
        """计算 ∂f/∂y, f: [B,1,H,W] 或 [B,H,W] 返回 [B,H,W]"""
        if f.dim() == 4:
            f = f.squeeze(1)
        fy = torch.zeros_like(f)
        fy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * self.dy)
        fy[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / self.dy
        fy[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / self.dy
        return fy

    def _laplacian(self, f):
        """计算 ∇²f = ∂²f/∂x² + ∂²f/∂y², f: [B,1,H,W] 或 [B,H,W] 返回 [B,H,W]"""
        if f.dim() == 4:
            f = f.squeeze(1)
        fxx = torch.zeros_like(f)
        fxx[:, :, 1:-1] = (f[:, :, 2:] - 2*f[:, :, 1:-1] + f[:, :, :-2]) / (self.dx**2)
        fxx[:, :, 0] = (f[:, :, 2] - 2*f[:, :, 1] + f[:, :, 0]) / (self.dx**2)
        fxx[:, :, -1] = (f[:, :, -1] - 2*f[:, :, -2] + f[:, :, -3]) / (self.dx**2)

        fyy = torch.zeros_like(f)
        fyy[:, 1:-1, :] = (f[:, 2:, :] - 2*f[:, 1:-1, :] + f[:, :-2, :]) / (self.dy**2)
        fyy[:, 0, :] = (f[:, 2, :] - 2*f[:, 1, :] + f[:, 0, :]) / (self.dy**2)
        fyy[:, -1, :] = (f[:, -1, :] - 2*f[:, -2, :] + f[:, -3, :]) / (self.dy**2)

        return fxx + fyy

    # ---------- 边界条件硬约束 (掩码法) ----------
    def apply_bc(self, u_hat, v_hat):
        """
        输入: u_hat, v_hat: [1,1,H,W]
        输出: u, v: [1,1,H,W] 满足边界条件
        """
        u_boundary = torch.zeros_like(u_hat)
        u_boundary += self.U * self.top_mask   # 顶盖处赋值 lid_velocity
        v_boundary = torch.zeros_like(v_hat)

        u = u_hat * (1 - self.boundary_mask) + u_boundary * self.boundary_mask
        v = v_hat * (1 - self.boundary_mask) + v_boundary * self.boundary_mask
        return u, v

    # ---------- 弱形式损失 (基于有限差分) ----------
    def _weak_loss(self, u, v):
        """
        输入: u, v: [B,H,W] (已去除通道维)
        """
        u_x = self._gradient_x(u)
        u_y = self._gradient_y(u)
        v_x = self._gradient_x(v)
        v_y = self._gradient_y(v)

        div = u_x + v_y
        conv_u = u * u_x + v * u_y
        conv_v = u * v_x + v * v_y
        grad_sq = u_x**2 + u_y**2 + v_x**2 + v_y**2

        L_conv = torch.mean(conv_u**2 + conv_v**2)
        L_visc = self.mu * torch.mean(grad_sq)
        L_div = torch.mean(div**2)
        loss = L_conv + L_visc + 10 * L_div
        return loss, (L_conv, L_visc, L_div)

    # ---------- 强形式损失 (保留但默认不使用) ----------
    def _strong_loss(self, u, v):
        """
        输入: u, v: [B,H,W]
        """
        u_x = self._gradient_x(u)
        u_y = self._gradient_y(u)
        v_x = self._gradient_x(v)
        v_y = self._gradient_y(v)

        div = u_x + v_y
        L_cont = torch.mean(div**2)

        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        dpdx = -self.rho * (u * u_x + v * u_y) + self.mu * lap_u
        dpdy = -self.rho * (u * v_x + v * v_y) + self.mu * lap_v

        dpdx_x = self._gradient_x(dpdx.unsqueeze(1)).squeeze(1)
        dpdy_y = self._gradient_y(dpdy.unsqueeze(1)).squeeze(1)
        div_gradp = dpdx_x + dpdy_y
        L_divp = torch.mean(div_gradp**2)

        dpdy_x = self._gradient_x(dpdy.unsqueeze(1)).squeeze(1)
        dpdx_y = self._gradient_y(dpdx.unsqueeze(1)).squeeze(1)
        curl = dpdy_x - dpdx_y
        L_curl = torch.mean(curl**2)

        loss = L_cont + L_divp + L_curl
        return loss, (L_cont, L_divp, L_curl)

    # ---------- 训练循环 ----------
    def solve_weak(self, iters):
        for it in range(iters):
            out = self.net(self.input_field)          # [1,2,H,W]
            u_hat, v_hat = out.split(1, dim=1)        # each [1,1,H,W]
            u, v = self.apply_bc(u_hat, v_hat)        # [1,1,H,W]

            # 去除通道维以计算损失
            u_s = u.squeeze(1)                         # [1,H,W]
            v_s = v.squeeze(1)

            loss, (L_conv, L_visc, L_div) = self._weak_loss(u_s, v_s)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if it % 100 == 0:
                print(f"weak iter {it:5d}, L_conv: {L_conv.item():.3e}, "
                      f"L_visc: {L_visc.item():.3e}, L_div: {L_div.item():.3e}")

    def solve_strong(self, iters):
        for it in range(iters):
            out = self.net(self.input_field)
            u_hat, v_hat = out.split(1, dim=1)
            u, v = self.apply_bc(u_hat, v_hat)
            u_s = u.squeeze(1)
            v_s = v.squeeze(1)

            loss, (L_cont, L_divp, L_curl) = self._strong_loss(u_s, v_s)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if it % 100 == 0:
                print(f"strong iter {it:5d}, L_cont: {L_cont.item():.3e}, "
                      f"L_divp: {L_divp.item():.3e}, L_curl: {L_curl.item():.3e}")
            if loss.item() < self.tol:
                break

    def solve(self):
        self.solve_weak(self.weak_iter)
        # 可选的强形式微调（默认关闭，可自行开启）
        self.solve_strong(self.strong_iter)

        # 保存最终结果
        with torch.no_grad():
            out = self.net(self.input_field)
            u_hat, v_hat = out.split(1, dim=1)
            u, v = self.apply_bc(u_hat, v_hat)
        self.Uf = u.squeeze().cpu()
        self.Vf = v.squeeze().cpu()

    def plot(self):
        N = self.N
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)

        U = self.Uf.numpy()
        V = self.Vf.numpy()
        speed = np.sqrt(U**2 + V**2)

        plt.figure(figsize=(6,6))
        plt.streamplot(X, Y, U, V, density=2)
        plt.title("Streamlines")
        plt.show()

        plt.figure(figsize=(6,6))
        plt.contourf(X, Y, speed, 20, cmap="jet")
        plt.colorbar()
        plt.title("Speed magnitude")
        plt.show()


if __name__ == "__main__":
    seed_everything(10492)
    solver = NeuralCavity(
        N=64,
        rho=1,
        mu=0.01,
        lid_velocity=1,
        weak_iter=1000,
        strong_iter=5000,
        tol=1e-6,
    )

    t1 = time.time()
    solver.solve()
    t2 = time.time()
    print("time:", t2 - t1)

    solver.plot()