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

        layers.append(nn.Linear(hidden_dim, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, xy, lid_velocity):
        out = self.net(xy)
        u_hat = out[:,0]
        v_hat = out[:,1]
        p_hat = out[:,2]
        x = xy[:,0]
        y = xy[:,1]
        # ---- ψ(x) ----
        eps = 0.05
        sharp = 50.0    # 允许角点不连续，范围为0.05(不到1格)，这样实现了四面约束
        psi = torch.sigmoid(sharp * (x - eps)) * torch.sigmoid(sharp * ((1 - eps) - x))
        u = lid_velocity*y*psi+x*(1-x)*(1-y)*y*u_hat    # 换用四面约束了
        v = x*(1-x)*y*(1-y)*v_hat
        p = p_hat - self.net(torch.stack([torch.tensor([0.]), torch.tensor([0.])], dim=1).to(device='cuda').requires_grad_(True))[:,2]
        return u, v, p


class NeuralSimpleCavity:

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
        self.dx = L / (N - 1)
        self.dy = self.dx

        self.rho = rho
        self.mu = mu
        self.lid_velocity = lid_velocity

        self.max_iter = max_iter
        self.tol = tol

        self.net = VelocityMLP(hidden_dim=128, num_layers=4).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        # 生成网格点（所有节点）
        self.x = torch.linspace(0, 1, N, device=self.device)
        self.y = torch.linspace(0, 1, N, device=self.device)
        X, Y = torch.meshgrid(self.x, self.y, indexing='ij')
        self.coords_flat = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)

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

    def compute_uvp_derivatives(self, u, v, p, xy, create_graph=True):
        """计算 u, v, p 的一阶和二阶导数"""
        # u 的一阶
        u_x = autograd.grad(u, xy, grad_outputs=torch.ones_like(u),
                            create_graph=create_graph, retain_graph=True)[0][:, 0]
        u_y = autograd.grad(u, xy, grad_outputs=torch.ones_like(u),
                            create_graph=create_graph, retain_graph=True)[0][:, 1]
        # v 的一阶
        v_x = autograd.grad(v, xy, grad_outputs=torch.ones_like(v),
                            create_graph=create_graph, retain_graph=True)[0][:, 0]
        v_y = autograd.grad(v, xy, grad_outputs=torch.ones_like(v),
                            create_graph=create_graph, retain_graph=True)[0][:, 1]
        # p 的一阶
        p_x = autograd.grad(p, xy, grad_outputs=torch.ones_like(p),
                            create_graph=create_graph, retain_graph=True)[0][:, 0]
        p_y = autograd.grad(p, xy, grad_outputs=torch.ones_like(p),
                            create_graph=create_graph, retain_graph=True)[0][:, 1]

        if create_graph:
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
        else:
            u_xx = u_yy = v_xx = v_yy = None

        return u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy

    # 边界软约束
    def soft_boundary(self, u, v):
        xy = self.coords_flat
        X, Y = xy[:, 0], xy[:, 1]
        u_target = torch.zeros_like(u)
        v_target = torch.zeros_like(v)
        # 顶部边界 (y=1, 0<x<1)
        top_mask = (Y == 1) & (X > 0) & (X < 1)
        u_target[top_mask] = self.lid_velocity
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

    def solve(self, boundary=False):
        xy = self.coords_flat
        N = self.N

        for it in range(self.max_iter):
            u, v, p = self.net.forward(xy, self.lid_velocity)  # 直接得到速度场

            # 计算 u, v, p 的导数
            u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy = self.compute_uvp_derivatives(u, v, p, xy, create_graph=True)

            # 速度的拉普拉斯
            lap_u = u_xx + u_yy
            lap_v = v_xx + v_yy

            # ---------- 物理约束损失 ----------
            # 1. 连续性方程 (div u = 0)
            div_u = u_x + v_y
            L_cont = torch.mean(div_u ** 2)

            # 2. 动量方程（ρu▽u=-▽p+μ▲u）
            mom_x = self.rho * (u * u_x + v * u_y) + p_x - self.mu * lap_u
            mom_y = self.rho * (u * v_x + v * v_y) + p_y - self.mu * lap_v
            L_mom = torch.mean(mom_x**2 + mom_y**2)

            # ---------- 总损失（可调整权重）----------
            loss = L_cont + L_mom

            # 3. 边界损失
            L_bc = self.soft_boundary(u, v)
            if boundary:
                loss += L_bc
            # 反向传播
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if it % 100 == 0:
                print(f"iter {it:5d}, L_cont: {L_cont.item():.3e}, L_mom: {L_mom.item():.3e}" +
                      (f', L_bc: {L_bc.item():.3e}' if boundary else ''))

            if loss.item() < self.tol:
                break

        # ---------- 预测阶段：获取最终速度场用于绘图 ----------
        with torch.no_grad():
            u_vis, v_vis, p_vis = self.net.forward(self.coords_flat, self.lid_velocity)

        self.U = u_vis.detach().reshape(N, N).cpu().T
        self.V = v_vis.detach().reshape(N, N).cpu().T
        self.P = p_vis.detach().reshape(N, N).cpu().T

    def plot(self):

        N = self.N

        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)

        X, Y = np.meshgrid(x, y)

        U = self.U.numpy()
        V = self.V.numpy()

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
    solver = NeuralSimpleCavity(
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
