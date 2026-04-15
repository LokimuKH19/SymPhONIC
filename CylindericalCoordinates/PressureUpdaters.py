# PressureUpdaters.py
from AnnularCoette import AnnularCouette
import torch
# 压力求解器合集
# 用于求解位移线性系统 acPc = sum afPf + sum affPff + beta


# 设法兼容一下3D和2D
class PressureUpdater:
    def __init__(self, coef, beta):
        """
            coef: dict, including ac, af, aff
            beta: RHS
        """
        self.coef = coef
        self.beta = beta

    # 测试用的雅各比迭代(2D用于验证程序正确性), 参数为P_stencil模板, 2d形态
    def jacobi_iter2d(self, max_inner, P_prime):
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

        for inner_iter in range(max_inner):
            # 获取邻居（周期性已在 neighbor 中处理）
            P_E = AnnularCouette.neighbor(P_prime, "E")
            P_W = AnnularCouette.neighbor(P_prime, "W")
            P_N = AnnularCouette.neighbor(P_prime, "N")
            P_S = AnnularCouette.neighbor(P_prime, "S")
            P_NE = AnnularCouette.neighbor(P_N, "E")
            P_NW = AnnularCouette.neighbor(P_N, "W")
            P_SE = AnnularCouette.neighbor(P_S, "E")
            P_SW = AnnularCouette.neighbor(P_S, "W")

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

            # 强制壁面 Neumann 条件：内外壁面法向梯度为零，直接令边界值等于相邻内点
            P_prime_new[0, :] = P_prime_new[1, :]
            P_prime_new[-1, :] = P_prime_new[-2, :]

            omega = 0.1
            P_prime = (1 - omega) * P_prime + omega * P_prime_new

            # ===== 内迭代收敛监控 =====
            res_inner = torch.max(torch.abs(P_prime_new - P_prime)).item()

            if (inner_iter+1) % 50 == 0:
                print(f"  inner_iter={inner_iter+1}, res={res_inner:.3e}")

            # 内迭代提前停止说明
            if res_inner < 1e-4:
                print(f"  inner converged at {inner_iter+1}, res={res_inner:.3e}")
                break

        # 约束一下零点
        return P_prime - torch.mean(P_prime)

    # 好一点的求解器，例如FNO
    def GMRES(self, P_prime):
        ...

    # 算子学习风格的，例如基于Krylov子空间的GMRES
    def OperatorLearning(self, P_prime):
        ...

