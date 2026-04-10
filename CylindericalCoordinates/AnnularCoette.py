import torch
import numpy as np
import matplotlib.pyplot as plt


class AnnularCouette:
    """
    二维环形库埃特流动求解器
    使用有限体积法和SIMPLE算法
    坐标为归一化柱坐标 R 和 Theta
    注意周期性边界条件的Theta控制体j的特殊性
    i = 1~N-2
    j = 0~N-1
    之后扩展出z轴也是
    k = 1~N-2
    """

    def __init__(self, n=64, rh=2.0, rs=4.0,
                 mu=1.0, rho=1.0, omega_out=1.0,
                 max_iter=5000, tol=1e-6,
                 u_relax=0.5, p_relax=0.3,
                 device="cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.n = n
        self.rh = rh
        self.rs = rs
        self.mu = mu
        self.rho = rho
        self.nu = mu / rho
        self.omega_out = omega_out

        self.max_iter = max_iter
        self.tol = tol
        self.u_relax = u_relax
        self.p_relax = p_relax

        self.delta_r = rs - rh
        self.u_omega = rs * self.omega_out
        self.P0 = 1/2 * self.rho * self.u_omega**2
        self.Re_omega = self.u_omega * self.delta_r / self.nu
        self.Eu_omega = self.P0 / (self.rho * self.u_omega ** 2)
        self.delta = self.delta_r / self.rs
        self.sgn_omega = 1.0 if self.omega_out >= 0 else -1.0

        # 初始化各场变量
        self.P = torch.zeros((n, n), device=self.device)
        self.UR = torch.zeros((n, n), device=self.device)
        self.UT = torch.zeros((n, n), device=self.device)

        self.AP_R = torch.ones((n, n), device=self.device)
        self.AP_T = torch.ones((n, n), device=self.device)

        self._build_grid()
        self._apply_bc()

    # 第一步：生成n*n网格
    def _build_grid(self):
        N = self.n

        self.dR = 1.0 / (N - 1)
        self.dTheta = 1.0 / (N - 1)

        R = torch.linspace(0, 1, N, device=self.device)
        Theta = torch.linspace(0, 1, N, device=self.device)

        RR, TT = torch.meshgrid(R, Theta, indexing='ij')

        r = self.rh + RR * self.delta_r

        self.r = r
        self.r_hat = r / self.delta_r

        theta0 = 2 * np.pi
        self.K_theta = 1.0 / (r * theta0)

    # 辅助整定边界条件
    def _apply_bc(self):
        self.UR[0, :] = 0.0
        self.UR[-1, :] = 0.0
        self.UT[0, :] = 0.0
        self.UT[-1, :] = 1.0
        self.P[0, 0] = 0.0

    # 第二步：Rhie-Chow插值
    def rhie_chow(self, U, P, AP):
        N = self.n
        dR = self.dR
        dTheta = self.dTheta

        # ===== Θ方向周期 =====
        P_jp = torch.roll(P, -1, dims=1)
        P_jm = torch.roll(P, 1, dims=1)

        U_jp = torch.roll(U, -1, dims=1)
        U_jm = torch.roll(U, 1, dims=1)

        AP_jp = torch.roll(AP, -1, dims=1)
        AP_jm = torch.roll(AP, 1, dims=1)

        # N face
        U_n = 0.5 * (U + U_jp) - 0.5 * (1 / AP + 1 / AP_jp) * (
                self.K_theta * (P_jp - P) / dTheta - self.K_theta * (P_jp - P_jm) / (2 * dTheta)
        )

        # S face
        U_s = 0.5 * (U + U_jm) - 0.5 * (1 / AP + 1 / AP_jm) * (
                self.K_theta * (P - P_jm) / dTheta - self.K_theta * (P_jp - P_jm) / (2 * dTheta)
        )

        # ===== R方向 interior =====
        P_ip = P[2:, :]
        P_im = P[:-2, :]

        U_ip = U[2:, :]
        U_im = U[:-2, :]

        AP_ip = AP[2:, :]
        AP_im = AP[:-2, :]

        U_c = U[1:-1, :]
        P_c = P[1:-1, :]
        AP_c = AP[1:-1, :]

        # E face
        U_e = 0.5 * (U_c + U_ip) - 0.5 * (1 / AP_c + 1 / AP_ip) * (
                (P_ip - P_c) / dR - (P_ip - P_im) / (2 * dR)
        )

        # W face
        U_w = 0.5 * (U_c + U_im) - 0.5 * (1 / AP_c + 1 / AP_im) * (
                (P_c - P_im) / dR - (P_ip - P_im) / (2 * dR)
        )

        # ===== R向的需要截断到控制体 =====
        return U_e, U_w, U_n[1:N - 1, :], U_s[1:N - 1, :]

    def momentum(self):
        N = self.n
        # 内部点切片对象
        i_inner = slice(1, N - 1)  # 径向内部索引 1..N-2
        j_all = slice(0, N)  # 周向全部（周期性）

        UR = self.UR
        UT = self.UT
        P = self.P
        AP_R = self.AP_R
        AP_T = self.AP_T

        # 周期性邻居（在完整网格上操作）
        UR_jp = torch.roll(UR, -1, dims=1)
        UR_jm = torch.roll(UR, 1, dims=1)
        UT_jp = torch.roll(UT, -1, dims=1)
        UT_jm = torch.roll(UT, 1, dims=1)
        P_jp = torch.roll(P, -1, dims=1)
        P_jm = torch.roll(P, 1, dims=1)

        # 径向邻居（使用切片，尺寸均为 (N-2, N)）
        UR_ip = UR[2:N, :]
        UR_im = UR[0:N - 2, :]
        UT_ip = UT[2:N, :]
        UT_im = UT[0:N - 2, :]
        P_ip = P[2:N, :]
        P_im = P[0:N - 2, :]

        # 几何量（内部点，尺寸 (N-2, N)）
        rE = self.r[2:N, :]
        rW = self.r[0:N - 2, :]
        rC = self.r[i_inner, :]  # 等价于 [1:N-1, :]
        r_hatC = rC / self.delta_r
        K_theta_C = self.K_theta[i_inner, :]

        # 面积（内部点对应的界面）
        AE = rE * self.dTheta
        AW = rW * self.dTheta
        AN = self.dR
        AS = self.dR

        # 无量纲数
        Re = self.Re_omega
        Eu = self.Eu_omega

        # Rhie‑Chow 界面速度（均为 (N-2, N)）
        uRe, uRw, uRn, uRs = self.rhie_chow(UR, P, AP_R)
        uTe, uTw, uTn, uTs = self.rhie_chow(UT, P, AP_T)

        # 通量（已带方向符号）
        Fe = uRe * AE
        Fw = -uRw * AW
        Fn = K_theta_C * uTn * AN * r_hatC
        Fs = -K_theta_C * uTs * AS * r_hatC

        # 扩散系数
        De = (1.0 / Re) * AE / self.dR
        Dw = (1.0 / Re) * AW / self.dR
        Dn = (1.0 / Re) * AN * r_hatC * K_theta_C**2 / self.dTheta
        Ds = (1.0 / Re) * AS * r_hatC * K_theta_C**2 / self.dTheta

        # 总的系数aF（迎风部分），周向通量需乘 r̂_C（连续性方程要求）
        aE = De - torch.clamp(Fe, max=0.0)
        aW = Dw - torch.clamp(Fw, max=0.0)
        aN = Dn - torch.clamp(Fn, max=0.0)
        aS = Ds - torch.clamp(Fs, max=0.0)

        # 对流对中心的贡献
        aP_conv = (torch.clamp(Fe, min=0.0) + torch.clamp(Fw, min=0.0) +
                   torch.clamp(Fn, min=0.0) + torch.clamp(Fs, min=0.0))

        # 基础中心系数
        aP_base = aP_conv + De + Dw + Dn + Ds + 1e-12

        # 控制体无量纲体积
        dV = r_hatC * self.dR * self.dTheta

        # 扩散附加项（这一部分是由于扩散项的第二项以及线性化非线性项引发的，和曲率源项一并处理，不过仍希望对角占优）
        aP_R = aP_base + (1.0 / Re) * dV / (r_hatC ** 2)
        aP_T = aP_base + (1.0 / Re) * dV / (r_hatC ** 2) + UR[i_inner, :]/r_hatC

        # 显式源项准备：交叉导数 ∂_Θ U_Θ 和 ∂_Θ U_R
        UT_jp_inner = UT_jp[i_inner, :]  # 尺寸 (N-2, N)
        UT_jm_inner = UT_jm[i_inner, :]
        dUT_dTheta = (UT_jp_inner - UT_jm_inner) / (2.0 * self.dTheta)

        UR_jp_inner = UR_jp[i_inner, :]
        UR_jm_inner = UR_jm[i_inner, :]
        dUR_dTheta = (UR_jp_inner - UR_jm_inner) / (2.0 * self.dTheta)

        # 内部点速度值
        UR_inner = UR[i_inner, :]
        UT_inner = UT[i_inner, :]

        # 曲率源项
        S_R_curve = - (UT_inner ** 2) / r_hatC
        S_T_couple = 0

        # 扩散交叉导数项
        S_R_diff_cross = (1.0 / Re) * (2.0 * K_theta_C / r_hatC) * dUT_dTheta
        S_T_diff_cross = - (1.0 / Re) * (2.0 * K_theta_C / r_hatC) * dUR_dTheta

        # 体积力（离心力、科氏力，做了坐标转换才会遇到，这个算例只是为了把咱们这个无敌算法调通）
        S_R_body = 0  # delta ** 2 * r_hatC - 2.0 * sgn * delta * UT_inner
        S_T_body = 0  # 2.0 * sgn * delta * UR_inner

        # 总源项（乘体积）
        S_R_total = (-S_R_curve + S_R_diff_cross + S_R_body) * dV
        S_T_total = (-S_T_couple + S_T_diff_cross + S_T_body) * dV

        # 压力梯度（FVM 中心差分，乘体积）
        # 径向：Eu * ( (r̂ P)_E - (r̂ P)_W ) / (2 ΔR) * dV / r̂_C ？直接用节点梯度乘体积亦可
        gradP_R = - Eu * (P_ip - P_im) / (2.0 * self.dR) * dV
        gradP_T = - Eu * K_theta_C * (P_jp[i_inner, :] - P_jm[i_inner, :]) / (2.0 * self.dTheta) * dV

        # 右端项（邻居值均使用上一迭代步的值）
        b_R = (aE * UR_ip + aW * UR_im +
               aN * UR_jp_inner + aS * UR_jm_inner +
               S_R_total + gradP_R)
        b_T = (aE * UT_ip + aW * UT_im +
               aN * UT_jp_inner + aS * UT_jm_inner +
               S_T_total + gradP_T)

        # 求解新值
        UR_new = b_R / aP_R
        UT_new = b_T / aP_T

        # 亚松弛并更新内部点
        self.UR[i_inner, :] = (1 - self.u_relax) * UR_inner + self.u_relax * UR_new
        self.UT[i_inner, :] = (1 - self.u_relax) * UT_inner + self.u_relax * UT_new

        # 保存 AP 供压力修正使用
        self.AP_R[i_inner, :] = aP_R
        self.AP_T[i_inner, :] = aP_T

    def pressure(self):
        N = self.n

        UR = self.UR
        UT = self.UT
        P = self.P

        UR_ip = UR[2:N, :]
        UR_im = UR[0:N - 2, :]

        UT_jp = torch.roll(UT, -1, dims=1)[1:N - 1, :]
        UT_jm = torch.roll(UT, 1, dims=1)[1:N - 1, :]

        rC = self.r[1:N - 1, :]

        div = ((rC * UR_ip - rC * UR_im) / self.dR +
               (UT_jp - UT_jm) / self.dTheta)
        P_new = P.clone()
        source = 0    # 占位符，还得写梯度
        P_new[1:N - 1, :] = 0.25 * (
                P[2:N, :] + P[0:N - 2, :] +
                torch.roll(P, -1, dims=1)[1:N - 1, :] +
                torch.roll(P, 1, dims=1)[1:N - 1, :] -
                div + source
        )

        self.P = (1 - self.p_relax) * P + self.p_relax * P_new

    def solve(self):
        for it in range(self.max_iter):

            UR_old = self.UR.clone()
            UT_old = self.UT.clone()

            self.momentum()
            self.pressure()
            self._apply_bc()

            res = torch.mean(torch.abs(self.UR - UR_old)) + \
                  torch.mean(torch.abs(self.UT - UT_old))

            print(it, res.item())

            if res < self.tol:
                print("converged", it)
                break

    def post(self):
        N = self.n

        # ===== 构造计算空间网格 =====
        R = np.linspace(0, 1, N)
        Theta = np.linspace(0, 1, N)

        RR, TT = np.meshgrid(R, Theta, indexing='ij')

        # ===== 映射到物理空间 =====
        r = self.rh + RR * (self.rs - self.rh)
        theta = TT * 2 * np.pi

        X = r * np.cos(theta)
        Y = r * np.sin(theta)

        # ===== 数据 =====
        UT = self.UT.cpu().numpy()
        UR = self.UR.cpu().numpy()
        P = self.P.cpu().numpy()

        # 映射回标准空间
        ut = UT * self.u_omega
        ur = UR * self.u_omega
        p = P*self.P0

        # ===== 1️⃣ 周向速度 =====
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, ut, shading='auto', cmap='cividis')
        plt.colorbar(label="ut/m s-1")
        plt.title("Tangential Velocity (Physical Space)")
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # ===== 2️⃣ 径向速度 =====
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, ur, shading='auto', cmap='cividis')
        plt.colorbar(label="ur/m s-1")
        plt.title("Radial Velocity (Physical Space)")
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # ===== 3️⃣ 压力 =====
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, p, shading='auto', cmap='cividis')
        plt.colorbar(label="p/Pa")
        plt.title("Pressure (Physical Space)")
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # ===== 4️⃣ 与理论解对比 =====
        # 取某一周向位置（如 theta=0 对应的 j=0）的径向剖面
        j_slice = 0
        r_1d = r[:, j_slice]  # 物理半径
        UT_num = UT[:, j_slice]  # 数值解

        # 理论解（物理空间）
        A = self.omega_out * self.rs ** 2 / (self.rs ** 2 - self.rh ** 2)
        B = -A * self.rh ** 2
        u_theta_theory = A * r_1d + B / r_1d
        UT_theory = u_theta_theory / (self.omega_out * self.rs)  # 无量纲化理论值

        plt.figure(figsize=(8, 5))
        plt.plot(r_1d, UT_num, 'bo-', label='Numerical', markersize=4)
        plt.plot(r_1d, UT_theory, 'r-', label='Analytical')
        plt.xlabel('Radius r')
        plt.ylabel('Dimensionless Tangential Velocity $U_\Theta$')
        plt.title('Comparison with Analytical Solution (Radial Profile)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 计算误差
        error = np.abs(UT_num - UT_theory)
        max_err = np.max(error)
        rmse = np.sqrt(np.mean(error ** 2))
        print(f"Max absolute error: {max_err:.3e}")
        print(f"RMSE: {rmse:.3e}")


if __name__ == "__main__":
    solver = AnnularCouette(n=64,
                            rh=2.0, rs=4.0, mu=1.0, rho=1.0,
                            omega_out=1.0,
                            max_iter=5000, tol=1e-6,
                            u_relax=0.5, p_relax=0.3,
                            device="cuda")
    solver.solve()
    solver.post()
