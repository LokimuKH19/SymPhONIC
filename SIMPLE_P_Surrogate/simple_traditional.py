# 代码：使用SIMPLE求解顶盖方腔驱动流
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


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

        AP = AE + AW + AN + AS + 1e-12

        self.U_star[1:N - 1, 1:N] = (AE * Ue + AW * Uw + AN * Un + AS * Us - (P[1:N - 1, 1:N] - P[1:N - 1, 0:N - 1]) * dy) / AP

        self.dU[1:N - 1, 1:N] = dy / AP

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

        AP = AE + AW + AN + AS + 1e-12

        self.V_star[1:N - 1, 1:N - 1] = (AE * Ve + AW * Vw + AN * Vn + AS * Vs - (P[1:N - 1, 1:N - 1] - P[0:N - 2, 1:N - 1]) * dx) / AP
        self.dV[1:N - 1, 1:N - 1] = dx / AP

    def pressure_correction(self):
        rho = self.rho
        dx = self.dx
        dy = self.dy
        N = self.N
        U = self.U_star
        V = self.V_star
        b = rho * ((U[1:N - 1, 2:N] - U[1:N - 1, 1:N - 1]) * dy +
                   (V[2:N, 1:N - 1] - V[1:N - 1, 1:N - 1]) * dx)
        aE = rho * self.dU[1:N - 1, 2:N] * dy
        aW = rho * self.dU[1:N - 1, 1:N - 1] * dy
        aN = rho * self.dV[2:N, 1:N - 1] * dx
        aS = rho * self.dV[1:N - 1, 1:N - 1] * dx
        aP = aE + aW + aN + aS + 1e-12
        Pp = torch.zeros_like(self.P)
        # 4000是empirical值，最好是残差驱动的，不过对于128*128网格来说128*128/2=8192iters肯定够，4000satisfied，如严格按照标准simple正常解线性代数方程的话需要O(n2)的时间复杂度
        # 它是整段代码中运行最慢的来源
        for _ in range(4000):
            Pp[1:N - 1, 1:N - 1] = (aE*Pp[1:N-1, 2:N] + aW*Pp[1:N-1, 0:N-2] +
                                    aN*Pp[2:N, 1:N-1] + aS*Pp[0:N-2, 1:N-1] - b)/aP
        return Pp

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
            Pp = self.pressure_correction()
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
        plt.savefig(f"./Simple_Traditional_Streamlines_{self.max_iter}.png")
        plt.cla()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, speed, 20, cmap="jet")
        plt.colorbar()
        plt.title(f"Velocity magnitude, Re={self.lid_velocity*self.L*self.rho/self.mu}")
        plt.savefig(f"./Simple_Traditional_Velocity Magnitude_{self.max_iter}.png")

        plt.cla()


if __name__ == "__main__":
    solver = SimpleCavity(
        N=128,
        rho=1,
        mu=0.01,
        lid_velocity=1.0,
        tol=1e-6,
        max_iter=5000,
        device="cuda"
    )

    t1 = time.time()
    solver.solve()
    t2 = time.time()
    print(f"Time Consumed: {t2 - t1}s")

    solver.plot()
