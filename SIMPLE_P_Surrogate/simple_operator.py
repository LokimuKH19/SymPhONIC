# 代码：使用SIMPLE求解顶盖方腔驱动流
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import NeuroOperators
from torch import nn


class PressureCorrectionCFNO(nn.Module):
    def __init__(self, modes=8, cheb_modes=(8,8), width=16, depth=3):
        super().__init__()
        self.model = NeuroOperators.CFNO2d_small(
            modes=modes,
            cheb_modes=cheb_modes,
            width=width,
            depth=depth,
            input_features=6,      # aE, aW, aN, aS, aP, b
            output_features=1      # Pp
        )

    def forward(self, coeffs):
        # coeffs shape: [B, 6, H, W]，最好再多个什么P‘的neumann边界条件，就是边界零压力梯度
        return self.model(coeffs)


class SimpleCavity:
    # 定义问题
    def __init__(
            self,
            N=128,
            L=1.0,
            rho=1.0,
            mu=0.01,
            lid_velocity=1.0,
            max_iter=5000,
            tol=1e-6,
            inner_iter=1145,
            iter_with_pressure=20,
            device="cuda"
    ):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.N = N
        self.L = L
        self.dx = L / (N - 1)
        self.dy = self.dx

        self.rho = rho
        self.mu = mu
        self.lid_velocity = lid_velocity

        self.max_iter = max_iter
        self.tol = tol
        self.inner_iter = inner_iter
        self.iter_with_pressure = iter_with_pressure

        self.init_fields()


    # 初始化，生成交错网格
    def init_fields(self):

        N = self.N
        device = self.device

        self.P = torch.zeros((N, N), device=device)

        self.U = torch.zeros((N, N + 1), device=device)
        self.V = torch.zeros((N + 1, N), device=device)

        self.U_star = torch.zeros_like(self.U)
        self.V_star = torch.zeros_like(self.V)

        self.dU = torch.zeros_like(self.U)
        self.dV = torch.zeros_like(self.V)
        self.Pp = torch.zeros_like(self.P)

        self.apply_boundary()

    # 边界约束，注意P在中心节点，u,v需进行交错布置，需要插值
    def apply_boundary(self):

        N = self.N

        # lid
        self.U[N - 1, 1:N] = self.lid_velocity

        # walls
        self.U[:, 0] = 0
        self.U[:, N] = 0

        self.V[0, :] = 0
        self.V[N, :] = 0

        self.V[:, 0] = 0
        self.V[:, N - 1] = 0

    # 更新动量方程
    def solve_momentum(self):

        rho = self.rho
        mu = self.mu
        dx = self.dx
        dy = self.dy
        N = self.N

        U = self.U
        V = self.V
        P = self.P

        # U cell
        Uc = U[1:N - 1, 1:N]
        Ue = U[1:N - 1, 2:N + 1]
        Uw = U[1:N - 1, 0:N - 1]
        Un = U[2:N, 1:N]
        Us = U[0:N - 2, 1:N]

        Vn = 0.5 * (V[2:N, 1:N] + V[2:N, 0:N - 1])
        Vs = 0.5 * (V[1:N - 1, 1:N] + V[1:N - 1, 0:N - 1])

        rho_u_e = rho * 0.5 * (Uc + Ue)
        rho_u_w = rho * 0.5 * (Uc + Uw)
        rho_v_n = rho * Vn
        rho_v_s = rho * Vs

        AE = torch.maximum(-rho_u_e, torch.zeros_like(rho_u_e)) * dy + mu * dy / dx
        AW = torch.maximum(rho_u_w, torch.zeros_like(rho_u_w)) * dy + mu * dy / dx
        AN = torch.maximum(-rho_v_n, torch.zeros_like(rho_v_n)) * dx + mu * dx / dy
        AS = torch.maximum(rho_v_s, torch.zeros_like(rho_v_s)) * dx + mu * dx / dy

        AP_u = AE + AW + AN + AS + 1e-12
        self.U_star[1:N - 1, 1:N] = (AE * Ue + AW * Uw + AN * Un + AS * Us -
                                     (P[1:N - 1, 1:N] - P[1:N - 1, 0:N - 1]) * dy) / AP_u

        # 保存 U 动量方程主系数（填充到全场尺寸，边界置为 1 防止除零）
        self.AP_u = torch.ones_like(self.U)
        self.AP_u[1:N - 1, 1:N] = AP_u

        # V cell
        Vc = V[1:N - 1, 1:N - 1]
        Ve = V[1:N - 1, 2:N]
        Vw = V[1:N - 1, 0:N - 2]
        Vn = V[2:N, 1:N - 1]
        Vs = V[0:N - 2, 1:N - 1]

        Ue = 0.5 * (U[1:N - 1, 1:N - 1] + U[2:N, 1:N - 1])
        Uw = 0.5 * (U[1:N - 1, 0:N - 2] + U[2:N, 0:N - 2])

        rho_v_n = rho * 0.5 * (Vc + Vn)
        rho_v_s = rho * 0.5 * (Vc + Vs)

        rho_u_e = rho * Ue
        rho_u_w = rho * Uw

        AE = torch.maximum(-rho_u_e, torch.zeros_like(rho_u_e)) * dy + mu * dy / dx
        AW = torch.maximum(rho_u_w, torch.zeros_like(rho_u_w)) * dy + mu * dy / dx
        AN = torch.maximum(-rho_v_n, torch.zeros_like(rho_v_n)) * dx + mu * dx / dy
        AS = torch.maximum(rho_v_s, torch.zeros_like(rho_v_s)) * dx + mu * dx / dy

        AP_v = AE + AW + AN + AS + 1e-12
        self.V_star[1:N - 1, 1:N - 1] = (AE * Ve + AW * Vw + AN * Vn + AS * Vs -
                                         (P[1:N - 1, 1:N - 1] - P[0:N - 2, 1:N - 1]) * dx) / AP_v
        # 保存 V 动量方程主系数
        self.AP_v = torch.ones_like(self.V)
        self.AP_v[1:N - 1, 1:N - 1] = AP_v

    def pressure_correction(self, iteration):

        rho = self.rho
        dx = self.dx
        dy = self.dy
        N = self.N
        U = self.U_star
        V = self.V_star

        # 使用动量方程的主系数计算速度修正系数
        self.dU[1:N - 1, 1:N] = self.dy / self.AP_u[1:N - 1, 1:N]
        self.dV[1:N - 1, 1:N - 1] = self.dx / self.AP_v[1:N - 1, 1:N - 1]

        # 计算系数场（内部网格区域）
        b = rho * ((U[1:N - 1, 2:N] - U[1:N - 1, 1:N - 1]) * dy +
                   (V[2:N, 1:N - 1] - V[1:N - 1, 1:N - 1]) * dx)  # [N-2, N-2]
        # b=0间接导致P’=0平凡解，阴成啥了，必须想法增加边界条件

        aE = rho * self.dU[1:N - 1, 2:N] * dy
        aW = rho * self.dU[1:N - 1, 1:N - 1] * dy
        aN = rho * self.dV[2:N, 1:N - 1] * dx
        aS = rho * self.dV[1:N - 1, 1:N - 1] * dx
        aP = aE + aW + aN + aS + 1e-12

        if iteration > self.iter_with_pressure:
            return self.Pp

        # 堆叠为 6 通道输入，添加 batch 维度
        inputs = torch.stack([aE, aW, aN, aS, aP, b], dim=0).unsqueeze(0)  # [1, 6, H_in, W_in]

        # 定义 CFNO 模型（使用本地定义的 CFNO2d_small）
        model = PressureCorrectionCFNO(
            modes=16, cheb_modes=(8, 8), width=16, depth=3,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # 辅助函数：将内部压力修正场扩展为全场（Neumann 边界）
        def expand_to_full(p_internal):
            # p_internal: [H_in, W_in], H_in = W_in = N-2
            p_full = torch.zeros((N, N), device=self.device, dtype=p_internal.dtype)
            p_full[1:N-1, 1:N-1] = p_internal
            # Neumann 边界：外推等于相邻内部值
            p_full[0, :] = p_full[1, :]          # 上边界
            p_full[N-1, :] = p_full[N-2, :]      # 下边界
            p_full[:, 0] = p_full[:, 1]          # 左边界
            p_full[:, N-1] = p_full[:, N-2]      # 右边界
            return p_full

        max_pp_old = 0
        # 训练循环
        for it in range(self.inner_iter):
            optimizer.zero_grad()
            p_internal = model(inputs)                # [1, 1, H_in, W_in]
            p_internal = p_internal[0, 0]             # [H_in, W_in]
            p_full = expand_to_full(p_internal)       # [N, N]

            # 计算残差 r = A p' - b （在内部网格点上）
            # 取 p_full 的五个 stencil 切片
            p_center = p_full[1:N-1, 1:N-1]           # 内部点中心
            p_east = p_full[1:N-1, 2:N]             # 东邻点
            p_west = p_full[1:N-1, 0:N-2]           # 西邻点
            p_north = p_full[2:N, 1:N-1]             # 北邻点
            p_south = p_full[0:N-2, 1:N-1]           # 南邻点

            residual = (aE * p_east + aW * p_west + aN * p_north + aS * p_south - b)/aP - p_center          # [N-2, N-2]

            loss = loss_fn(residual, torch.zeros_like(residual))
            loss.backward()
            optimizer.step()
            max_pp = torch.max(torch.abs(p_center)).detach().cpu().numpy()

            # 可选：打印损失信息
            if (it+1) % 20 == 0:
                print(f"  Pressure correction inner iter {it+1}, loss = {loss.item():.6e}, max p' = {max_pp:.6e}")
            # 设置收敛早停
            if abs(max_pp_old-max_pp) < 1e-8:
                print(f" Inner iteration converged at iter {it+1}, loss = {loss.item():.6e}, max p' = {max_pp:.6e}")
                break

            max_pp_old = max_pp

        # 训练完成后，最终计算 p'
        with torch.no_grad():
            p_internal_final = model(inputs)[0, 0]
            p_full_final = expand_to_full(p_internal_final)

        # 释放模型和优化器，防止内存堆积
        del model, optimizer

        self.Pp = p_full_final

        return self.Pp

    # 更新速度
    def update_velocity(self, Pp):
        N = self.N
        self.U[1:N - 1, 1:N] = (
                self.U_star[1:N - 1, 1:N] +
                self.dU[1:N - 1, 1:N] *
                (Pp[1:N - 1, 0:N - 1] - Pp[1:N - 1, 1:N])
        )

        self.V[1:N - 1, 1:N - 1] = (
                self.V_star[1:N - 1, 1:N - 1] +
                self.dV[1:N - 1, 1:N - 1] *
                (Pp[0:N - 2, 1:N - 1] - Pp[1:N - 1, 1:N - 1])
        )
        self.P += 0.3 * Pp      # 压力更新松弛因子取0.3
        self.apply_boundary()

    # 求解流程
    def solve(self):
        for it in range(self.max_iter):
            U_old = self.U.clone()
            V_old = self.V.clone()
            self.solve_momentum()
            Pp = self.pressure_correction(it)
            self.update_velocity(Pp)
            err = max(torch.max(torch.abs(self.U - U_old)).detach().cpu().numpy(),
                      torch.max(torch.abs(self.V - V_old)).detach().cpu().numpy())
            print("iter", it, "max_err", err.item())
            if err < self.tol:
                break

    def plot(self):
        N = self.N
        dx = self.dx

        x = np.linspace(dx / 2, 1 - dx / 2, N)
        y = np.linspace(dx / 2, 1 - dx / 2, N)

        X, Y = np.meshgrid(x, y)

        Uc = 0.5 * (self.U[:, 1:] + self.U[:, :-1]).cpu().numpy()
        Vc = 0.5 * (self.V[1:, :] + self.V[:-1, :]).cpu().numpy()

        speed = np.sqrt(Uc ** 2 + Vc ** 2)

        plt.figure(figsize=(6, 6))
        plt.streamplot(X, Y, Uc, Vc, density=2)
        plt.title(f"Streamlines, Re={self.lid_velocity*self.L*self.rho/self.mu}")
        plt.savefig(f"./Simple_CFNO_Streamlines_{self.max_iter}.png")
        plt.cla()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, speed, 20, cmap="jet")
        plt.colorbar()
        plt.title(f"Velocity magnitude, Re={self.lid_velocity*self.L*self.rho/self.mu}")
        plt.savefig(f"./Simple_CFNO_Velocity Magnitude_{self.max_iter}.png")

        plt.cla()


if __name__ == "__main__":
    NeuroOperators.seed_everything(32)
    solver = SimpleCavity(
        N=128,
        rho=1,
        mu=0.01,
        lid_velocity=1.0,
        tol=1e-12,
        max_iter=200,
        inner_iter=200,
        iter_with_pressure=200,
        device="cuda"
    )

    t1 = time.time()
    solver.solve()
    t2 = time.time()
    print(f"Time Consumed: {t2 - t1}s")

    solver.plot()
