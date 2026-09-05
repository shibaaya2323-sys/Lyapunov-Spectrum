
# ============================================================
# セル3
# Lyapunov spectrum の描画
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Lyapunov指数の収束
#
# 4本ずつ分けて描画
# lambda_1 ～ lambda_4
# lambda_5 ～ lambda_8
# ...
# lambda_41 ～ lambda_44
# ============================================================

num_exponents = len(lambdas)

group_size = 4

for start in range(0, num_exponents, group_size):

    end = min(start + group_size,num_exponents)

    plt.figure(figsize=(7, 5.5))

    # --------------------------------------------------------
    # 各Lyapunov指数を描画
    # --------------------------------------------------------

    for i in range(start, end):

        plt.plot(times,lambda_history[:, i],label=rf"$\lambda_{{{i+1}}}$")

    # --------------------------------------------------------
    # lambda = 0 の基準線
    # --------------------------------------------------------

    plt.axhline(0.0,linewidth=1.0)

    # --------------------------------------------------------
    # 軸ラベル
    # --------------------------------------------------------

    plt.xlabel(r"$t$")

    plt.ylabel(r"$\lambda_i(t)$")

    # --------------------------------------------------------
    # タイトル
    # --------------------------------------------------------

    plt.title(
        rf"Convergence of "
        rf"$\lambda_{{{start+1}}}$ -- "
        rf"$\lambda_{{{end}}}$"
    )

    plt.legend()

    plt.grid()

    plt.show()
