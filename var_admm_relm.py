# -*- coding:utf-8 -*-
# author: Lu Huihuang
# 2023/7/2 10:32
import numpy as np
from numpy.linalg import norm, inv
from .base import ELM


def var_admm(H, T, rho, lamb, mu, alpha, max_loops, err_tol=1e-12):
    """
        变步长迭代admm算法
    """
    L = H.shape[1]
    m = T.shape[1]

    xk = np.random.randn(L, m)
    zk = np.random.randn(L, m)
    uk = np.random.randn(L, m)

    # 计算一些中间变量
    Ht = np.transpose(H)
    HtH = np.dot(Ht, H)
    I = np.identity(HtH.shape[0], dtype='float64')
    HtT = np.dot(Ht, T)

    for k in range(max_loops):
        xk_old = xk
        TEMP = inv(HtH + (mu + rho) * I)
        xk = np.dot(TEMP, (HtT + rho * (zk - uk)))
        thresold = lamb / rho
        for i in range(L):
            for j in range(m):
                TEMP2 = xk[i][j] + uk[i][j]
                if TEMP2 > thresold:
                    zk = TEMP2 - thresold
                elif TEMP2 < -thresold:
                    zk = TEMP2 + thresold
                else:
                    zk = 0
        uk = uk + xk - zk
        rho = rho / (1 + alpha * k)
        rho = max(1e-20, rho)
        err = norm(xk - xk_old)
        if err < err_tol:
            # print("var-admm提前退出迭代，次数为{}".format(k))
            break
    return xk


class VarAdmmRELM(ELM):
    """
        var-admm-relm
    """
    def __init__(self,
                 n_hidden: int,
                 n_input: int,
                 n_output: int,
                 rho: float = 1e2,
                 lamb: float = 1e-2,
                 mu: float = 1e-5,
                 alpha: float = 1e3,
                 max_loops: int = 300,
                 act_func: str = 'sigmoid',
                 type_: str = "CLASSIFIER",
                 seed: int = None) -> None:
        super().__init__(n_hidden, n_input, n_output, act_func, type_, seed)
        self.rho = rho
        self.lamb = lamb
        self.mu = mu
        self.alpha = alpha
        self.max_loops = max_loops

    def calculate_beta(self, H, T):
        return var_admm(H, T, self.rho, self.lamb, self.mu, self.alpha, self.max_loops)
