# PressureUpdaters.py
from AnnularCoette import AnnularCouette
import torch
from NeuralOperators import CFNO2d_small
# 压力求解器合集
# 用于求解位移线性系统 acPc = sum afPf + sum affPff + beta


# 设法兼容一下3D和2D
class Jacobi:
    def __init__(self, coef, beta, device, max_inner=40000, tol=1e-8, report_interval=500):
        """
            coef: dict, including ac, af, aff
            beta: RHS
        """
        self.coef = coef
        self.beta = beta
        self.device = device
        self.max_inner = max_inner
        self.tol = tol
        self.report_interval = report_interval

    # 测试用的雅各比迭代(2D用于验证程序正确性), 参数为P_stencil模板, 2d形态，每轮迭代作为预条件器使用
    def solve2d(self, P_prime, P_prime_old):
        alpha_C = self.coef["C"]
        alpha_E = self.coef["E"]
        alpha_W = self.coef["W"]
        alpha_N = self.coef["N"]
        alpha_S = self.coef["S"]
        alpha_NE = self.coef["NE"]
        alpha_NW = self.coef["NW"]
        alpha_SE = self.coef["SE"]
        alpha_SW = self.coef["SW"]
        beta = self.beta
        # 约束边界
        alpha_W[0, :], alpha_NW[0, :], alpha_SW[0, :] = 0, 0, 0
        alpha_E[-1, :], alpha_NE[-1, :], alpha_SE[-1, :] = 0, 0, 0

        for inner_iter in range(self.max_inner):
            # 获取邻居（周期性已在 neighbor 中处理）
            P_E = AnnularCouette.neighbor(P_prime, "E")
            P_W = AnnularCouette.neighbor(P_prime, "W")
            P_N = AnnularCouette.neighbor(P_prime, "N")
            P_S = AnnularCouette.neighbor(P_prime, "S")
            P_NE = AnnularCouette.neighbor(AnnularCouette.neighbor(P_prime_old, "N"), "E")
            P_NW = AnnularCouette.neighbor(AnnularCouette.neighbor(P_prime_old, "N"), "W")
            P_SE = AnnularCouette.neighbor(AnnularCouette.neighbor(P_prime_old, "S"), "E")
            P_SW = AnnularCouette.neighbor(AnnularCouette.neighbor(P_prime_old, "S"), "W")

            # 径向边界镜像修正（与 momentum 一致）
            P_E[-1, :] = P_prime[-1, :]  # 外边界 E 面用外壁值
            P_W[0, :] = P_prime[0, :]  # 内边界 W 面用内壁值
            # 角点也需相应修正（因涉及 NE, SE 等）
            P_NE[-1, :] = P_N[-1, :]
            P_SE[-1, :] = P_S[-1, :]
            P_NW[0, :] = P_N[0, :]
            P_SW[0, :] = P_S[0, :]

            rhs = (alpha_E * P_E + alpha_W * P_W + alpha_N * P_N + alpha_S * P_S +
                   alpha_NE * P_NE + alpha_NW * P_NW + alpha_SE * P_SE + alpha_SW * P_SW +
                   beta)
            P_prime_new = rhs / (alpha_C + 1e-12)
            omega = 0.3
            P_prime = (1 - omega) * P_prime + omega * P_prime_new

            # ===== 内迭代收敛监控 =====
            residual = rhs-P_prime*alpha_C
            res_inner = torch.max(torch.abs(residual)).item()
            res = torch.mean(torch.abs(P_prime - P_prime_new))

            # 验证压力方程
            if (inner_iter+1) % self.report_interval == 0:
                print(f"  Jacobi inner_iter={inner_iter+1}, res={res:.3e}")

            # 内迭代提前停止说明
            if res < self.tol:
                print(f"  inner converged at {inner_iter+1}, res={res_inner:.3e}")
                break
            # 固定一个压力点
            P_prime[0, 0] = 0.0

        print("Maximum P_prime: ", torch.max(torch.abs(P_prime)).cpu().numpy(),
              ", Avg P_prime: ", torch.mean(torch.abs(P_prime)).cpu().numpy())
        return P_prime


# 工程中用的BiCGStab
class BiCGStab:
    def __init__(self, coef, beta, device="cuda", max_inner=40000, tol=1e-8, report_interval=500):
        self.coef = coef
        self.beta = beta
        self.device = device
        self.max_inner = max_inner
        self.tol = tol
        self.report_interval = report_interval

    def apply_A_2D(self, P):
        """
        计算 A @ P
        """
        alpha_C = self.coef["C"]
        alpha_E = self.coef["E"]
        alpha_W = self.coef["W"]
        alpha_N = self.coef["N"]
        alpha_S = self.coef["S"]
        alpha_NE = self.coef["NE"]
        alpha_NW = self.coef["NW"]
        alpha_SE = self.coef["SE"]
        alpha_SW = self.coef["SW"]

        # neighbor
        def nb(x, d):
            return torch.roll(x, shifts={"E": -1, "W": 1, "N": -1, "S": 1}[d],
                              dims={"E": 0, "W": 0, "N": 1, "S": 1}[d])

        P_E = nb(P, "E")
        P_W = nb(P, "W")
        P_N = nb(P, "N")
        P_S = nb(P, "S")

        P_NE = nb(nb(P, "N"), "E")
        P_NW = nb(nb(P, "N"), "W")
        P_SE = nb(nb(P, "S"), "E")
        P_SW = nb(nb(P, "S"), "W")

        # 径向边界镜像 (零梯度边界条件)
        P_E[-1, :] = P[-1, :]
        P_W[0, :] = P[0, :]

        P_NE[-1, :] = P_N[-1, :]
        P_SE[-1, :] = P_S[-1, :]
        P_NW[0, :] = P_N[0, :]
        P_SW[0, :] = P_S[0, :]

        AP = (
                alpha_C * P
                - (alpha_E * P_E + alpha_W * P_W +
                   alpha_N * P_N + alpha_S * P_S +
                   alpha_NE * P_NE + alpha_NW * P_NW +
                   alpha_SE * P_SE + alpha_SW * P_SW)
        )

        # 【核心修复1】：在矩阵映射层面锁定 [0,0] 点，打破奇异性
        AP[0, 0] = P[0, 0]

        return AP

    def solve2d(self, x0):
        """
        BiCGStab 主求解器
        """
        # 在SIMPLE算法中，压力修正量 P' 的初值永远应该给 0，不要继承上一步的 P'
        x = torch.zeros_like(x0)

        b = self.beta.clone()
        # 对应 AP[0,0] = P[0,0]，将源项的这个点设为 0
        b[0, 0] = 0.0

        r = b - self.apply_A_2D(x)
        r_hat = r.clone()

        rho_old = torch.tensor(1.0, device=self.device)
        alpha = torch.tensor(1.0, device=self.device)
        omega = torch.tensor(1.0, device=self.device)

        v = torch.zeros_like(x)
        p = torch.zeros_like(x)
        res = torch.norm(r)
        for k in range(self.max_inner):
            rho_new = torch.sum(r_hat * r)
            if (k + 1) % self.report_interval == 0:
                print(f"  BiCG iter={k + 1}, res={res.item():.3e}")
            # 安全检查：允许负数，但如果是真正的 0 则中断避免除零错误
            if torch.abs(rho_new) < 1e-10 or torch.abs(rho_old) < 1e-10 or torch.abs(omega) < 1e-10:
                break
            beta_k = (rho_new / rho_old) * (alpha / omega)
            p = r + beta_k * (p - omega * v)
            v = self.apply_A_2D(p)
            den_alpha = torch.sum(r_hat * v)
            if torch.abs(den_alpha) < 1e-10:
                break
            alpha = rho_new / den_alpha
            s = r - alpha * v
            if torch.norm(s) < self.tol:
                x = x + alpha * p
                break
            t = self.apply_A_2D(s)
            tt = torch.sum(t * t)
            if torch.abs(tt) < 1e-10:
                break
            omega = torch.sum(t * s) / tt

            x = x + alpha * p + omega * s
            r = s - omega * t

            rho_old = rho_new

            res = torch.norm(r)

            if res < self.tol:
                print(f"BiCG converged at iter={k + 1}, res={res.item():.3e}")
                break

        print(
            f"Maximum P_prime: {torch.max(torch.abs(x)).item():.6f}, Avg P_prime: {torch.mean(torch.abs(x)).item():.6f}")
        return x

