
import numpy as np
import matplotlib.pyplot as plt

from numba import njit


# ============================================================
# 1. パラメータ
# ============================================================

sigma = 16.0
b = 4.0
gamma = 40.0

# ------------------------------------------------------------
# 数値積分
# ------------------------------------------------------------

dt = 0.01

# QR 分解を行う時間間隔
tau = 1.0

# 総積分時間
t_max = 4096.0

# tau の間に何ステップ積分するか
steps_per_tau = int(round(tau / dt))

# QR 分解を行う回数
num_tau = int(round(t_max / tau))

# ============================================================
# 2. Lorenz 方程式
#
# dX/dt = -sigma X + sigma Y
#
# dY/dt = (gamma - Z)X - Y
#
# dZ/dt = XY - bZ
# ============================================================

@njit
def lorenz(n_x):

    X = n_x[0]
    Y = n_x[1]
    Z = n_x[2]

    n_dx = np.empty(3, dtype=np.float64)

    n_dx[0] = (-sigma * X + sigma * Y)
    n_dx[1] = ((gamma - Z) * X - Y)
    n_dx[2] = (X * Y - b * Z)

    return n_dx

# ============================================================
# 3. Jacobian
#
# 第一変分方程式
#
# d(delta x)/dt
# =
# J(x) delta x
#
#
# J(x)
# =
#
# [ -sigma       sigma       0 ]
# [ gamma-Z      -1         -X ]
# [ Y             X         -b ]
# ============================================================

@njit
def jacobian(n_x):

    X = n_x[0]
    Y = n_x[1]
    Z = n_x[2]

    n_J = np.empty((3, 3),dtype=np.float64)

    n_J[0, 0] = -sigma
    n_J[0, 1] = sigma
    n_J[0, 2] = 0.0

    n_J[1, 0] = gamma - Z
    n_J[1, 1] = -1.0
    n_J[1, 2] = -X

    n_J[2, 0] = Y
    n_J[2, 1] = X
    n_J[2, 2] = -b

    return n_J

# ============================================================
# 4. 軌道方程式 + 第一変分方程式
#
# E の各列が
#
# e1, e2, e3
#
# に対応する。
#
# dE/dt = J(x) E
# ============================================================

@njit
def rhs(n_x, n_E):

    n_dx = lorenz(n_x)

    n_J = jacobian(n_x)

    n_dE = n_J @ n_E

    return n_dx, n_dE

# ============================================================
# 5. Runge-Kutta-Gill 法
# ============================================================

sqrt2 = np.sqrt(2.0)

a31 = (sqrt2 - 1.0) / 2.0
a32 = (2.0 - sqrt2) / 2.0

a42 = -sqrt2 / 2.0
a43 = (2.0 + sqrt2) / 2.0

b1 = 1.0 / 6.0
b2 = (2.0 - sqrt2) / 6.0
b3 = (2.0 + sqrt2) / 6.0
b4 = 1.0 / 6.0


@njit
def rkg_step(n_x, n_E, h):

    # --------------------------------------------------------
    # 第1段
    # --------------------------------------------------------

    n_k1x, n_k1E = rhs(n_x,n_E)

    # --------------------------------------------------------
    # 第2段
    # --------------------------------------------------------

    n_x2 = (n_x + 0.5 * h * n_k1x)
    n_E2 = (n_E + 0.5 * h * n_k1E)
    n_k2x, n_k2E = rhs(n_x2,n_E2)

    # --------------------------------------------------------
    # 第3段
    # --------------------------------------------------------

    n_x3 = (n_x + h * (a31 * n_k1x + a32 * n_k2x))
    n_E3 = (n_E + h * (a31 * n_k1E + a32 * n_k2E))
    n_k3x, n_k3E = rhs(n_x3,n_E3)

    # --------------------------------------------------------
    # 第4段
    # --------------------------------------------------------

    n_x4 = (n_x + h * (a42 * n_k2x + a43 * n_k3x))
    n_E4 = (n_E + h * (a42 * n_k2E + a43 * n_k3E))
    n_k4x, n_k4E = rhs(n_x4,n_E4)

    # --------------------------------------------------------
    # 更新
    # --------------------------------------------------------

    n_x_new = (n_x + h * (b1 * n_k1x + b2 * n_k2x + b3 * n_k3x + b4 * n_k4x))
    n_E_new = (n_E + h * (b1 * n_k1E + b2 * n_k2E + b3 * n_k3E + b4 * n_k4E))

    return n_x_new, n_E_new

# ============================================================
# 6. Gram-Schmidt による QR 分解
#
#
# 入力
#
# A
# =
# D Phi_tau(x_n) E_n
#
#
# 出力
#
# A
# =
# E_{n+1} R_n
#
#
# R_n
# =
#
# [ r11  r12  r13 ]
# [  0   r22  r23 ]
# [  0    0   r33 ]
#
#
# Lyapunov 指数に必要なのは
#
# r11, r22, r33
#
# ============================================================

@njit
def qr_gram_schmidt(n_A):

    # ========================================================
    # 1本目
    # ========================================================

    n_v1 = (n_A[:, 0].copy())

    r11 = np.sqrt(np.dot(n_v1,n_v1))

    n_e1 = (n_v1 / r11)

    # ========================================================
    # 2本目
    #
    # v2 から
    #
    # e1 方向
    #
    # を取り除く
    # ========================================================

    n_v2 = (n_A[:, 1].copy())

    r12 = np.dot(n_e1,n_v2)

    n_w2 = (n_v2 - r12 * n_e1)

    r22 = np.sqrt(np.dot(n_w2,n_w2))

    n_e2 = (n_w2 / r22)


    # ========================================================
    # 3本目
    #
    # v3 から
    #
    # e1 方向
    # e2 方向
    #
    # を取り除く
    # ========================================================

    n_v3 = (n_A[:, 2].copy())

    r13 = np.dot(n_e1,n_v3)

    r23 = np.dot(n_e2,n_v3)

    n_w3 = (n_v3 - r13 * n_e1 - r23 * n_e2)

    r33 = np.sqrt(np.dot(n_w3,n_w3))

    n_e3 = (n_w3 / r33)

    # ========================================================
    # E_{n+1}
    # ========================================================

    n_E_new = np.empty((3, 3),dtype=np.float64)

    n_E_new[:, 0] = n_e1
    n_E_new[:, 1] = n_e2
    n_E_new[:, 2] = n_e3

    # ========================================================
    # R_n
    #
    # 確認や理解のために作る。
    #
    # Lyapunov 指数の計算自体には
    # 対角成分だけあればよい。
    # ========================================================

    n_R = np.zeros((3, 3),dtype=np.float64)

    n_R[0, 0] = r11
    n_R[0, 1] = r12
    n_R[0, 2] = r13

    n_R[1, 1] = r22
    n_R[1, 2] = r23

    n_R[2, 2] = r33


    return (
        n_E_new,
        n_R,
        r11,
        r22,
        r33
    )

# ============================================================
# 7. 初期基底を正規直交化
#
#
# QR 法では計算開始時に
#
# E_0^T E_0 = I
#
# としておく。
#
#
# Case 3 のように
# 初期基底が正規直交していなくても、
# ここで正規直交化する。
# ============================================================

@njit
def orthonormalize_initial_basis(n_basis0):

    # n_basis0 は
    #
    # 各行が基底ベクトル
    #
    # なので転置して
    #
    # 各列が基底ベクトル
    #
    # になるようにする。

    n_A = (n_basis0.T.copy())

    (n_E0,_,_,_,_) = qr_gram_schmidt(n_A)

    return n_E0

# ============================================================
# 8. QR 法による Lyapunov spectrum の計算
#
#
# D Phi_tau(x_n) E_n
# =
# E_{n+1} R_n
#
#
# lambda_i
# =
# lim
#
# 1/(N tau)
#
# sum log |r_ii^(n)|
#
# ============================================================

@njit
def calculate_lyapunov_qr_core(n_x0,n_basis0):

    # ========================================================
    # 初期状態
    # ========================================================

    n_x = (n_x0.copy())

    # ========================================================
    # 初期正規直交基底
    # ========================================================

    n_E = (orthonormalize_initial_basis(n_basis0))

    # ========================================================
    # 累積対数
    #
    # sum log r11
    #
    # sum log r22
    #
    # sum log r33
    # ========================================================

    sum_log_r11 = 0.0

    sum_log_r22 = 0.0

    sum_log_r33 = 0.0


    # ========================================================
    # 保存配列
    # ========================================================

    n_times = np.empty(num_tau,dtype=np.float64)

    n_lambda1 = np.empty(num_tau,dtype=np.float64)

    n_lambda2 = np.empty(num_tau,dtype=np.float64)

    n_lambda3 = np.empty(num_tau,dtype=np.float64)

    # ========================================================
    # 時間発展
    # ========================================================

    for j in range(num_tau):

        # ====================================================
        # tau 時間だけ
        #
        # dx/dt
        # =
        # F(x)
        #
        #
        # dE/dt
        # =
        # J(x) E
        #
        # を同時積分
        # ====================================================

        for _ in range(steps_per_tau):

            (
                n_x,
                n_E
            ) = rkg_step(
                n_x,
                n_E,
                dt
            )


        # ====================================================
        # 現在時刻
        # ====================================================

        t = ((j + 1) * tau)

        # ====================================================
        # QR 分解
        #
        #
        # 現在 n_E に入っているのは
        #
        # D Phi_tau(x_n) E_n
        #
        #
        # これを
        #
        # E_{n+1} R_n
        #
        # に分解する。
        # ====================================================

        (
            n_E,
            n_R,
            r11,
            r22,
            r33
        ) = qr_gram_schmidt(
            n_E
        )


        # ====================================================
        # R の対角成分の対数を累積
        #
        #
        # r11
        # ↓
        # lambda1
        #
        #
        # r22
        # ↓
        # lambda2
        #
        #
        # r33
        # ↓
        # lambda3
        # ====================================================

        sum_log_r11 += (np.log(abs(r11)))

        sum_log_r22 += (np.log(abs(r22)))

        sum_log_r33 += (np.log(abs(r33)))

        # ====================================================
        # 有限時間 Lyapunov 指数
        # ====================================================

        lambda1 = (sum_log_r11 / t)

        lambda2 = (sum_log_r22 / t)

        lambda3 = (sum_log_r33 / t)

        # ====================================================
        # 保存
        # ====================================================

        n_times[j] = t

        n_lambda1[j] = lambda1

        n_lambda2[j] = lambda2

        n_lambda3[j] = lambda3


    return (
        n_times,
        n_lambda1,
        n_lambda2,
        n_lambda3
    )


# ============================================================
# 9. Python 側ラッパー
# ============================================================

def calculate_lyapunov_qr(n_x0,n_basis0):

    (
        n_times,
        n_lambda1,
        n_lambda2,
        n_lambda3
    ) = calculate_lyapunov_qr_core(
        n_x0,
        n_basis0
    )

    return {

        "t":
            n_times,

        "lambda1":
            n_lambda1,

        "lambda2":
            n_lambda2,

        "lambda3":
            n_lambda3
    }


# ============================================================
# 10. 初期条件
#
# 元の論文再現コードと同じ3 Case
# ============================================================


# ------------------------------------------------------------
# Case 1
# ------------------------------------------------------------

n_x0_1 = np.array(
    [
        10.0,
        0.0,
        30.0
    ],
    dtype=np.float64
)

n_basis_1 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    dtype=np.float64
)


# ------------------------------------------------------------
# Case 2
# ------------------------------------------------------------

n_x0_2 = np.array(
    [
        10.0,
        10.0,
        30.0
    ],
    dtype=np.float64
)

n_basis_2 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    dtype=np.float64
)


# ------------------------------------------------------------
# Case 3
#
# 最初は正規直交していない。
#
# QR 法では計算開始前に
# orthonormalize_initial_basis()
# により正規直交化する。
# ------------------------------------------------------------

n_x0_3 = np.array(
    [
        10.0,
        10.0,
        30.0
    ],
    dtype=np.float64
)

n_basis_3 = np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ],
    dtype=np.float64
)

# ============================================================
# 11. 計算
# ============================================================

print("Case 1 を計算中...")

result1 = calculate_lyapunov_qr(n_x0_1,n_basis_1)

print("Case 2 を計算中...")

result2 = calculate_lyapunov_qr(n_x0_2,n_basis_2)

print("Case 3 を計算中...")

result3 = calculate_lyapunov_qr(n_x0_3,n_basis_3)

print("計算終了")

# ============================================================
# 12. 指定時刻のデータを抽出
# ============================================================

n_table_times = np.array(
    [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096
    ],
    dtype=np.float64
)


def sample_result(
    result,
    n_sample_times
):

    n_indices = np.empty(
        len(n_sample_times),
        dtype=np.int64
    )

    for i in range(
        len(n_sample_times)
    ):

        target_t = (
            n_sample_times[i]
        )

        n_indices[i] = np.argmin(
            np.abs(
                result["t"]
                - target_t
            )
        )

    return {

        "t":
            result["t"][n_indices],

        "lambda1":
            result["lambda1"][n_indices],

        "lambda2":
            result["lambda2"][n_indices],

        "lambda3":
            result["lambda3"][n_indices]
    }

# ============================================================
# 13. Case 1 の収束表
# ============================================================

s1_table = sample_result(
    result1,
    n_table_times
)


print()

print(
    "QR method : Lyapunov spectrum"
)

print(
    "=" * 70
)

print(
    f"{'t':>8}"
    f"{'lambda1':>20}"
    f"{'lambda2':>20}"
    f"{'lambda3':>20}"
)

print(
    "-" * 70
)


for (
    t,
    lambda1,
    lambda2,
    lambda3
) in zip(

    s1_table["t"],

    s1_table["lambda1"],

    s1_table["lambda2"],

    s1_table["lambda3"]
):

    print(
        f"{t:8.0f}"
        f"{lambda1:20.10f}"
        f"{lambda2:20.10f}"
        f"{lambda3:20.10f}"
    )

# ============================================================
# 14. Lyapunov 指数の収束を描画
#
# Case 1
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(
    result1["t"],
    result1["lambda1"],
    label=r"$\lambda_1$"
)

plt.plot(
    result1["t"],
    result1["lambda2"],
    label=r"$\lambda_2$"
)

plt.plot(
    result1["t"],
    result1["lambda3"],
    label=r"$\lambda_3$"
)


plt.xscale(
    "log",
    base=2
)

plt.xlabel(
    r"$t$",
    fontsize=13
)

plt.ylabel(
    r"$\lambda_i(t)$",
    fontsize=13
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()

# ============================================================
# 15. lambda1 と lambda2 を拡大して描画
#
# lambda3 は約 -22 なので、
# 同じ図では lambda2 ≈ 0 が見づらい。
# ============================================================

# plt.figure(figsize=(7.5, 5.5))
# plt.plot(result1["t"],result1["lambda1"],label=r"$\lambda_1$")
# plt.plot(result1["t"],result1["lambda2"],label=r"$\lambda_2$")
# plt.xscale("log",base=2)
# plt.xlabel(r"$t$",fontsize=13)
# plt.ylabel(r"$\lambda_i(t)$",fontsize=13)
# plt.legend()
# plt.grid(True,alpha=0.3)
# plt.tight_layout()
# plt.show()

# ============================================================
# 16. Kaplan-Yorke dimension
# ============================================================

def kaplan_yorke_dimension(n_lyapunov_exponents):

    # --------------------------------------------------------
    # 大きい順
    # --------------------------------------------------------

    n_lambdas = np.sort(np.asarray(n_lyapunov_exponents,dtype=np.float64))[::-1]

    # --------------------------------------------------------
    # 部分和
    #
    # S_j
    # =
    # lambda1 + ... + lambda_j
    # --------------------------------------------------------

    n_cumulative_sum = np.cumsum(n_lambdas)

    # --------------------------------------------------------
    # S_j >= 0
    #
    # となる最大の j
    # --------------------------------------------------------

    n_nonnegative_indices = np.where(n_cumulative_sum >= 0.0)[0]

    if len(n_nonnegative_indices) == 0:

        return 0.0

    j_index = (n_nonnegative_indices[-1])

    j = (j_index + 1)

    # --------------------------------------------------------
    # 全部足しても非負の場合
    # --------------------------------------------------------

    if j == len(n_lambdas):

        return float(j)

    # --------------------------------------------------------
    # Kaplan-Yorke
    #
    # D_KY
    # =
    #
    # j
    # +
    # S_j / |lambda_{j+1}|
    # --------------------------------------------------------

    S_j = (n_cumulative_sum[j_index])

    lambda_next = (n_lambdas[j_index + 1])

    D_KY = (j + S_j / abs(lambda_next))

    return D_KY

# ============================================================
# 17. 最終 Lyapunov spectrum
#     +
#     Kaplan-Yorke dimension
#     +
#     divergence check
#
#
# Lorenz 系
#
# div F
# =
# -(sigma + 1 + b)
#
# =
# -21
#
#
# したがって
#
# lambda1 + lambda2 + lambda3
# =
# -21
#
# のはず。
# ============================================================

divergence_exact = (-(sigma + b + 1.0))

for i, result in enumerate([result1,result2,result3],start=1):

    # --------------------------------------------------------
    # 最終値
    # --------------------------------------------------------

    lambda1 = (result["lambda1"][-1])

    lambda2 = (result["lambda2"][-1])

    lambda3 = (result["lambda3"][-1])

    # --------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------

    n_lyapunov_spectrum = np.array([lambda1,lambda2,lambda3],dtype=np.float64)

    # --------------------------------------------------------
    # Kaplan-Yorke dimension
    # --------------------------------------------------------

    D_KY = (kaplan_yorke_dimension(n_lyapunov_spectrum))

    # --------------------------------------------------------
    # Lyapunov 指数の和
    # --------------------------------------------------------

    spectrum_sum = np.sum(n_lyapunov_spectrum)

    # --------------------------------------------------------
    # div F = -21 との誤差
    # --------------------------------------------------------

    absolute_error = abs(spectrum_sum - divergence_exact)

    relative_error = (absolute_error / abs(divergence_exact))

    # ========================================================
    # 表示
    # ========================================================

    print()

    print("=" * 60)

    print(f"Case {i}")

    print("=" * 60)

    print()

    print("Lyapunov spectrum by QR method")

    print(f"lambda1 = {lambda1:.10f}")

    print(f"lambda2 = {lambda2:.10f}")

    print(f"lambda3 = {lambda3:.10f}")

    print()

    print("Kaplan-Yorke dimension")

    print(f"D_KY = {D_KY:.10f}")

    print()

    print("Divergence check")

    print("lambda1 + lambda2 + lambda3"f" = {spectrum_sum:.10f}")

    print(f"div F = {divergence_exact:.10f}")

    print(f"absolute error = "f"{absolute_error:.10e}")

    print(f"relative error = "f"{relative_error:.10e}")
