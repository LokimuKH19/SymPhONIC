import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.autograd as autograd
from NeuroOperators import seed_everything


class VelocityMLP(nn.Module):

    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward_old(self, xy, U):
        out = self.net(xy)

        u_hat = out[:, 0]
        v_hat = out[:, 1]

        x = xy[:, 0]
        y = xy[:, 1]
        B = x * (1 - x) * y * (1 - y)
        u = y * (1 - x) * x * u_hat      # 三面约束
        v = B * v_hat

        return u, v

    def forward(self, xy, U):
        out = self.net(xy)

        u_hat = out[:, 0]
        v_hat = out[:, 1]

        x = xy[:, 0]
        y = xy[:, 1]

        # ---- ψ(x) ----
        eps = 0.05
        sharp = 50.0    # 允许角点不连续，范围为0.01(1格)，这样实现了四面约束
        psi = torch.sigmoid(sharp * (x - eps)) * torch.sigmoid(sharp * ((1 - eps) - x))
        # ---- boundary lifting ----
        B = x * (1 - x) * y * (1 - y)
        u = U * y * psi + B * u_hat
        v = B * v_hat
        return u, v


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

        self.Vf = None
        self.Uf = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.N = N
        self.rho = rho
        self.mu = mu
        self.U = lid_velocity
        self.weak_iter = weak_iter
        self.strong_iter = strong_iter
        self.net = VelocityMLP().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        x = torch.linspace(0, 1, N, device=self.device)
        y = torch.linspace(0, 1, N, device=self.device)

        X, Y = torch.meshgrid(x, y, indexing="ij")

        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)

        self.tol = tol
        self.xy = coords.requires_grad_(True)

        # 掩码：内部点
        mask_inner = torch.ones(N * N, dtype=torch.bool, device=self.device).reshape(N, N)
        mask_inner[0, :] = False
        mask_inner[-1, :] = False
        mask_inner[:, 0] = False
        mask_inner[:, -1] = False
        self.mask_inner = mask_inner.flatten()

        # 掩码：边界点（排除角点）用于边界速度损失
        mask_bc = torch.zeros(N * N, dtype=torch.bool, device=self.device).reshape(N, N)
        mask_bc[1:-1, 0] = True   # 底部（不含角）
        mask_bc[1:-1, -1] = True  # 顶部（不含角）
        mask_bc[0, 1:-1] = True   # 左侧（不含角）
        mask_bc[-1, 1:-1] = True  # 右侧（不含角）
        self.mask_bc = mask_bc.flatten()

    # 计算一阶导数
    def derivatives_1(self, u, v, create_graph=True):
        xy = self.xy
        grad_u = autograd.grad(u, xy, torch.ones_like(u),
                               create_graph=create_graph)[0]
        grad_v = autograd.grad(v, xy, torch.ones_like(v),
                               create_graph=create_graph)[0]
        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        v_x = grad_v[:, 0]
        v_y = grad_v[:, 1]
        return u_x, u_y, v_x, v_y

    # 计算二阶导数
    def derivatives_2(self, u_x, u_y, v_x, v_y, create_graph=True):
        xy = self.xy
        # u 的二阶
        u_xx = autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x),
                             create_graph=create_graph, retain_graph=True)[0][:, 0]
        u_yy = autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y),
                             create_graph=create_graph, retain_graph=True)[0][:, 1]
        # v 的二阶
        v_xx = autograd.grad(v_x, xy, grad_outputs=torch.ones_like(v_x),
                             create_graph=create_graph, retain_graph=True)[0][:, 0]
        v_yy = autograd.grad(v_y, xy, grad_outputs=torch.ones_like(v_y),
                             create_graph=create_graph, retain_graph=True)[0][:, 1]
        return u_xx, u_yy, v_xx, v_yy

    # 边界软约束
    def soft_boundary(self, u, v):
        xy = self.xy
        X, Y = xy[:, 0], xy[:, 1]
        u_target = torch.zeros_like(u)
        v_target = torch.zeros_like(v)
        # 顶部边界 (y=1, 0<x<1)
        top_mask = (Y == 1) & (X > 0) & (X < 1)
        u_target[top_mask] = self.U
        v_target[top_mask] = 0
        # 底部边界 (y=0, 0<x<1)
        bottom_mask = (Y == 0) & (X > 0) & (X < 1)
        u_target[bottom_mask] = 0
        v_target[bottom_mask] = 0
        # 左侧边界 (x=0, 0<y<1)
        left_mask = (X == 0) & (Y > 0) & (Y < 1)
        u_target[left_mask] = 0
        v_target[left_mask] = 0
        # 右侧边界 (x=1, 0<y<1)
        right_mask = (X == 1) & (Y > 0) & (Y < 1)
        u_target[right_mask] = 0
        v_target[right_mask] = 0
        # 边界损失（仅对 mask_bc 点，即已排除角点）
        u_pred_bc = u[self.mask_bc]
        v_pred_bc = v[self.mask_bc]
        u_tar_bc = u_target[self.mask_bc]
        v_tar_bc = v_target[self.mask_bc]
        return torch.mean((u_pred_bc - u_tar_bc) ** 2) + torch.mean((v_pred_bc - v_tar_bc) ** 2)

    def solve_weak(self, iters, boundary=False):
        xy = self.xy
        for it in range(iters):
            u, v = self.net(xy, self.U)
            u_x, u_y, v_x, v_y = self.derivatives_1(u, v)
            # divergence
            div = u_x + v_y
            # convection
            conv_u = u * u_x + v * u_y
            conv_v = u * v_x + v * v_y
            # viscous dissipation
            grad_sq = u_x ** 2 + u_y ** 2 + v_x ** 2 + v_y ** 2
            # energy functional
            L_conv = torch.mean(conv_u ** 2 + conv_v ** 2)
            L_visc = self.mu * torch.mean(grad_sq)
            L_div = torch.mean(div ** 2)
            loss = L_conv + L_visc + 10 * L_div
            # boundary
            L_bc = self.soft_boundary(u, v)
            if boundary:
                loss += L_bc
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if it % 100 == 0:
                print(
                    f"weak iter {it:5d}, L_conv: {L_conv.item():.3e}, "
                    f"L_visc: {L_visc.item():.3e}, L_div: {L_div.item():.3e}" +
                    (f', L_bc: {L_bc.item():.3e}' if boundary else '')
                )

    # strong formation，收敛太慢，实战价值不高
    def solve_strong(self, iters, boundary=False):
        xy = self.xy

        for it in range(iters):
            u, v = self.net.forward(xy, self.U)  # 直接得到速度场
            # 计算 u, v 的导数
            u_x, u_y, v_x, v_y,  = self.derivatives_1(u, v, create_graph=True)
            u_xx, u_yy, v_xx, v_yy = self.derivatives_2(u_x, v_y, v_x, v_y, create_graph=True)
            # 速度的拉普拉斯
            lap_u = u_xx + u_yy
            lap_v = v_xx + v_yy
            # ---------- 物理约束损失 ----------
            # 1. 连续性方程 (div u = 0)
            div_u = u_x + v_y
            L_cont = torch.mean(div_u ** 2)
            # 2. 压力梯度分量（来自稳态 Navier-Stokes）
            dpdx = -self.rho * (u * u_x + v * u_y) + self.mu * lap_u
            dpdy = -self.rho * (u * v_x + v * v_y) + self.mu * lap_v
            # 3. 压力梯度的旋度（确保压力场存在）
            dpdy_x = autograd.grad(dpdy, xy, grad_outputs=torch.ones_like(dpdy),
                                   create_graph=True, retain_graph=True)[0][:, 0]
            dpdx_y = autograd.grad(dpdx, xy, grad_outputs=torch.ones_like(dpdx),
                                   create_graph=True, retain_graph=True)[0][:, 1]
            curl = dpdy_x - dpdx_y    # 核心技术
            L_curl = torch.mean(curl ** 2)
            # 4. 压力泊松方程残差（∇·(∇p)）
            div_gradp = (autograd.grad(dpdx, xy, grad_outputs=torch.ones_like(dpdx),
                                       create_graph=True, retain_graph=True)[0][:, 0] +
                         autograd.grad(dpdy, xy, grad_outputs=torch.ones_like(dpdy),
                                       create_graph=True, retain_graph=True)[0][:, 1])
            L_divp = torch.mean(div_gradp ** 2)
            # 5.边界
            L_bc = self.soft_boundary(u, v)

            # ---------- 总损失（可调整权重）----------
            loss = L_cont + L_curl + L_divp
            if boundary:
                loss += L_bc

            # 反向传播
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if it % 100 == 0:
                print(f"strong iter {it:5d}, L_cont: {L_cont.item():.3e}, "
                      f"L_curl: {L_curl.item():.3e}, L_divp: {L_divp.item():.3e}" +
                      (f', L_bc: {L_bc.item():.3e}' if boundary else ''))
            if loss.item() < self.tol:
                break

    # 求解流程：先用弱迭代，然后用强迭代，最后再输出结果
    def solve(self, boundary=False):
        self.solve_weak(self.weak_iter, boundary)
        self.solve_strong(self.strong_iter, boundary)
        with torch.no_grad():
            u, v = self.net(self.xy, self.U)
        N = self.N
        self.Uf = u.reshape(N, N).cpu().T
        self.Vf = v.reshape(N, N).cpu().T

    def plot(self):

        N = self.N

        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)

        X, Y = np.meshgrid(x, y)

        U = self.Uf.numpy()
        V = self.Vf.numpy()

        speed = np.sqrt(U ** 2 + V ** 2)

        plt.figure(figsize=(6, 6))
        plt.streamplot(X, Y, U, V, density=2)
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, speed, 20, cmap="jet")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    seed_everything(10492)
    solver = NeuralCavity(
        N=64,
        rho=1,
        mu=0.01,
        lid_velocity=1,
        weak_iter=4500,
        strong_iter=0,
        tol=1e-6,
    )

    t1 = time.time()
    solver.solve()
    t2 = time.time()

    print("time:", t2 - t1)

    solver.plot()
