
# ============================================================
# シェルモデル Lyapunov spectrum 計算・検証コード
#
# 構成
#   1. 固定パラメータ
#   2. シェルパラメータ
#   3. 基準方程式・第一変分方程式
#   4. 初期条件・初期摂動基底
#   5. 時間積分
#   6. Modified Gram-Schmidt
#   7. 物理量・補助量
#   8. Lyapunov spectrum 本体
#   9. 検証用関数
#  10. 描画用関数
#
# 残している検証
#   ・中心差分による第一変分方程式検証
#   ・1ステップ写像による第一変分方程式検証
#   ・Modified Gram-Schmidt の直交性検証
#   ・基準軌道の統計的定常性確認
#   ・基準軌道の続き計算
# ============================================================

import numpy as np
from numba import njit
import matplotlib.pyplot as plt


# ============================================================
# 1. 固定パラメータ
# ============================================================

q = np.float64(2.0)
k0 = np.float64(2.0 ** (-4))
beta = np.float64(0.5)

# 第4シェルへの複素外力
f = np.complex128(5.0e-3 * (1.0 + 1.0j))

# ============================================================
# 2. シェルパラメータ
# ============================================================

def make_shell_parameters(N):

    N = int(N)

    n_arr = np.arange(1, N + 1, dtype=np.float64)

    n_k = k0 * q**n_arr
    n_k_sq = n_k**2

    n_c1 = np.zeros(N, dtype=np.float64)
    n_c2 = np.zeros(N, dtype=np.float64)
    n_c3 = np.zeros(N, dtype=np.float64)

    for i in range(N - 2):
        n_c1[i] = n_k[i]

    for i in range(1, N - 1):
        n_c2[i] = -beta * n_k[i - 1]

    for i in range(2, N):
        n_c3[i] = (beta - 1.0) * n_k[i - 2]

    return (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
    )

# ============================================================
# 3. 基準方程式・第一変分方程式
# ============================================================

@njit
def nonlinear_into_numba(
    n_u,
    n_c1,
    n_c2,
    n_c3,
    f,
    n_nl,
):

    N_local = n_u.size

    # 第1シェル
    n_nl[0] = 1j * (
        n_c1[0]
        * np.conj(n_u[1])
        * np.conj(n_u[2])
    )

    # 第2シェル
    n_nl[1] = 1j * (
        n_c1[1] * np.conj(n_u[2]) * np.conj(n_u[3])
        + n_c2[1] * np.conj(n_u[0]) * np.conj(n_u[2])
    )

    # 内部のシェル
    for i in range(2, N_local - 2):

        n_nl[i] = 1j * (
            n_c1[i] * np.conj(n_u[i + 1]) * np.conj(n_u[i + 2])
            + n_c2[i] * np.conj(n_u[i - 1]) * np.conj(n_u[i + 1])
            + n_c3[i] * np.conj(n_u[i - 2]) * np.conj(n_u[i - 1])
        )

    # 最後から2番目
    n_nl[N_local - 2] = 1j * (
        n_c2[N_local - 2]
        * np.conj(n_u[N_local - 3])
        * np.conj(n_u[N_local - 1])
        + n_c3[N_local - 2]
        * np.conj(n_u[N_local - 4])
        * np.conj(n_u[N_local - 3])
    )

    # 最後のシェル
    n_nl[N_local - 1] = 1j * (
        n_c3[N_local - 1]
        * np.conj(n_u[N_local - 3])
        * np.conj(n_u[N_local - 2])
    )

    # 第4シェルへの外力
    n_nl[3] += f

@njit
def variational_nonlinear_into_numba(
    n_u,
    n_E,
    n_c1,
    n_c2,
    n_c3,
    n_dE,
):

    N_local = n_u.size
    dim = n_E.shape[1]

    # 第1シェル
    for j in range(dim):

        n_dE[0, j] = (
            1j
            * n_c1[0]
            * (
                np.conj(n_E[1, j]) * np.conj(n_u[2])
                + np.conj(n_u[1]) * np.conj(n_E[2, j])
            )
        )

    # 第2シェル
    for j in range(dim):

        n_dE[1, j] = 1j * (
            n_c1[1]
            * (
                np.conj(n_E[2, j]) * np.conj(n_u[3])
                + np.conj(n_u[2]) * np.conj(n_E[3, j])
            )
            + n_c2[1]
            * (
                np.conj(n_E[0, j]) * np.conj(n_u[2])
                + np.conj(n_u[0]) * np.conj(n_E[2, j])
            )
        )

    # 内部のシェル
    for i in range(2, N_local - 2):

        for j in range(dim):

            n_dE[i, j] = 1j * (
                n_c1[i]
                * (
                    np.conj(n_E[i + 1, j]) * np.conj(n_u[i + 2])
                    + np.conj(n_u[i + 1]) * np.conj(n_E[i + 2, j])
                )
                + n_c2[i]
                * (
                    np.conj(n_E[i - 1, j]) * np.conj(n_u[i + 1])
                    + np.conj(n_u[i - 1]) * np.conj(n_E[i + 1, j])
                )
                + n_c3[i]
                * (
                    np.conj(n_E[i - 2, j]) * np.conj(n_u[i - 1])
                    + np.conj(n_u[i - 2]) * np.conj(n_E[i - 1, j])
                )
            )

    # 最後から2番目
    i = N_local - 2

    for j in range(dim):

        n_dE[i, j] = 1j * (
            n_c2[i]
            * (
                np.conj(n_E[i - 1, j]) * np.conj(n_u[i + 1])
                + np.conj(n_u[i - 1]) * np.conj(n_E[i + 1, j])
            )
            + n_c3[i]
            * (
                np.conj(n_E[i - 2, j]) * np.conj(n_u[i - 1])
                + np.conj(n_u[i - 2]) * np.conj(n_E[i - 1, j])
            )
        )

    # 最後のシェル
    i = N_local - 1

    for j in range(dim):

        n_dE[i, j] = (
            1j
            * n_c3[i]
            * (
                np.conj(n_E[i - 2, j]) * np.conj(n_u[i - 1])
                + np.conj(n_u[i - 2]) * np.conj(n_E[i - 1, j])
            )
        )

# ============================================================
# 4. 初期条件・初期摂動基底
# ============================================================

@njit
def make_initial_condition_numba(n_k, n_k_sq, seed=42):

    N_local = n_k.size

    np.random.seed(seed)

    n_u = np.zeros(N_local, dtype=np.complex128)

    for i in range(N_local):

        # E(k_n) = k_n^2 exp(-k_n^2)
        initial_energy = n_k_sq[i] * np.exp(-n_k_sq[i])

        phase = np.random.uniform(0.0, 2.0 * np.pi)

        # E(k_n) = |u_n|^2 / (2k_n)
        n_u[i] = (np.sqrt(2.0 * n_k[i] * initial_energy) * np.exp(1j * phase))

    return n_u

@njit
def make_initial_tangent_basis_numba(N):

    dim = 2 * N

    n_E = np.zeros((N, dim), dtype=np.complex128)

    for i in range(N):

        # 実部方向
        n_E[i, 2 * i] = 1.0 + 0j

        # 虚部方向
        n_E[i, 2 * i + 1] = 0.0 + 1j

    return n_E

# ============================================================
# 5. 時間積分
# ============================================================

@njit
def rk4_step_u_inplace_numba(
    n_u,
    dt,
    n_E_visc,
    n_E_visc_half,
    n_c1,
    n_c2,
    n_c3,
    f,
    work_u,
):

    N_local = n_u.size

    # --------------------------------------------------------
    # RK4 第1段
    # --------------------------------------------------------

    nonlinear_into_numba(n_u, n_c1, n_c2, n_c3, f, work_u[0])

    for i in range(N_local):
        work_u[4, i] = (n_u[i] + 0.5 * dt * work_u[0, i]) * n_E_visc_half[i]

    # --------------------------------------------------------
    # RK4 第2段
    # --------------------------------------------------------

    nonlinear_into_numba(work_u[4], n_c1, n_c2, n_c3, f, work_u[1])

    for i in range(N_local):
        work_u[4, i] = (n_u[i] * n_E_visc_half[i] + 0.5 * dt * work_u[1, i])

    # --------------------------------------------------------
    # RK4 第3段
    # --------------------------------------------------------

    nonlinear_into_numba(work_u[4], n_c1, n_c2, n_c3, f, work_u[2])

    for i in range(N_local):
        work_u[4, i] = (n_E_visc[i] * n_u[i] + dt * n_E_visc_half[i] * work_u[2, i])

    # --------------------------------------------------------
    # RK4 第4段
    # --------------------------------------------------------

    nonlinear_into_numba(work_u[4], n_c1, n_c2, n_c3, f, work_u[3])

    # --------------------------------------------------------
    # 基準軌道を更新
    # --------------------------------------------------------

    for i in range(N_local):
        n_u[i] = n_u[i] * n_E_visc[i] + dt / 6.0 * (
            work_u[0, i] * n_E_visc[i]
            + 2.0 * work_u[1, i] * n_E_visc_half[i]
            + 2.0 * work_u[2, i] * n_E_visc_half[i]
            + work_u[3, i]
        )

@njit
def rk4_step_u_E_inplace_numba(
    n_u,
    n_E,
    dt,
    n_E_visc,
    n_E_visc_half,
    n_c1,
    n_c2,
    n_c3,
    f,
    work_u,
    work_E,
):

    N_local = n_u.size
    dim = n_E.shape[1]

    # --------------------------------------------------------
    # RK4 第1段
    # --------------------------------------------------------

    nonlinear_into_numba(n_u, n_c1, n_c2, n_c3, f, work_u[0])

    variational_nonlinear_into_numba(n_u, n_E, n_c1, n_c2, n_c3, work_E[0])

    for i in range(N_local):

        work_u[4, i] = (n_u[i] + 0.5 * dt * work_u[0, i]) * n_E_visc_half[i]

        for j in range(dim):
            work_E[4, i, j] = (n_E[i, j] + 0.5 * dt * work_E[0, i, j]) * n_E_visc_half[i]

    # --------------------------------------------------------
    # RK4 第2段
    # --------------------------------------------------------

    nonlinear_into_numba(work_u[4], n_c1, n_c2, n_c3, f, work_u[1])

    variational_nonlinear_into_numba(work_u[4], work_E[4],n_c1, n_c2, n_c3, work_E[1])

    for i in range(N_local):

        work_u[4, i] = (n_u[i] * n_E_visc_half[i] + 0.5 * dt * work_u[1, i])

        for j in range(dim):
            work_E[4, i, j] = (n_E[i, j] * n_E_visc_half[i] + 0.5 * dt * work_E[1, i, j])

    # --------------------------------------------------------
    # RK4 第3段
    # --------------------------------------------------------

    nonlinear_into_numba(work_u[4], n_c1, n_c2, n_c3, f, work_u[2])

    variational_nonlinear_into_numba(work_u[4], work_E[4],n_c1, n_c2, n_c3, work_E[2])

    for i in range(N_local):

        work_u[4, i] = (n_E_visc[i] * n_u[i] + dt * n_E_visc_half[i] * work_u[2, i])

        for j in range(dim):
            work_E[4, i, j] = (n_E[i, j] * n_E_visc[i] + dt * n_E_visc_half[i] * work_E[2, i, j])

    # --------------------------------------------------------
    # RK4 第4段
    # --------------------------------------------------------

    nonlinear_into_numba(work_u[4], n_c1, n_c2, n_c3, f, work_u[3])

    variational_nonlinear_into_numba(work_u[4], work_E[4],n_c1, n_c2, n_c3, work_E[3])

    # --------------------------------------------------------
    # 基準軌道と摂動基底を更新
    # --------------------------------------------------------

    for i in range(N_local):

        n_u[i] = n_u[i] * n_E_visc[i] + dt / 6.0 * (
            work_u[0, i] * n_E_visc[i]
            + 2.0 * work_u[1, i] * n_E_visc_half[i]
            + 2.0 * work_u[2, i] * n_E_visc_half[i]
            + work_u[3, i]
        )

        for j in range(dim):
            n_E[i, j] = n_E[i, j] * n_E_visc[i] + dt / 6.0 * (
                work_E[0, i, j] * n_E_visc[i]
                + 2.0 * work_E[1, i, j] * n_E_visc_half[i]
                + 2.0 * work_E[2, i, j] * n_E_visc_half[i]
                + work_E[3, i, j]
            )

@njit
def rk4_step_E_inplace_numba(
    n_E,
    dt,
    n_E_visc,
    n_E_visc_half,
    n_u_start,
    n_u_half,
    n_u_end,
    n_c1,
    n_c2,
    n_c3,
    work_E,
):

    N_local = n_E.shape[0]
    dim = n_E.shape[1]

    # --------------------------------------------------------
    # RK4 第1段
    #
    # 時刻 t
    #
    # u(t), E(t)
    # --------------------------------------------------------

    variational_nonlinear_into_numba(
        n_u_start,
        n_E,
        n_c1,
        n_c2,
        n_c3,
        work_E[0],
    )

    # --------------------------------------------------------
    # RK4 第2段用の E
    #
    # 時刻 t + dt/2
    # --------------------------------------------------------

    for i in range(N_local):

        for j in range(dim):

            work_E[4, i, j] = (
                n_E[i, j]
                + 0.5 * dt * work_E[0, i, j]
            ) * n_E_visc_half[i]

    # --------------------------------------------------------
    # RK4 第2段
    #
    # u(t + dt/2), E_2
    # --------------------------------------------------------

    variational_nonlinear_into_numba(
        n_u_half,
        work_E[4],
        n_c1,
        n_c2,
        n_c3,
        work_E[1],
    )

    # --------------------------------------------------------
    # RK4 第3段用の E
    #
    # 時刻 t + dt/2
    # --------------------------------------------------------

    for i in range(N_local):

        for j in range(dim):

            work_E[4, i, j] = (
                n_E[i, j] * n_E_visc_half[i]
                + 0.5 * dt * work_E[1, i, j]
            )

    # --------------------------------------------------------
    # RK4 第3段
    #
    # u(t + dt/2), E_3
    # --------------------------------------------------------

    variational_nonlinear_into_numba(
        n_u_half,
        work_E[4],
        n_c1,
        n_c2,
        n_c3,
        work_E[2],
    )

    # --------------------------------------------------------
    # RK4 第4段用の E
    #
    # 時刻 t + dt
    # --------------------------------------------------------

    for i in range(N_local):

        for j in range(dim):

            work_E[4, i, j] = (
                n_E[i, j] * n_E_visc[i]
                + dt
                * n_E_visc_half[i]
                * work_E[2, i, j]
            )

    # --------------------------------------------------------
    # RK4 第4段
    #
    # u(t + dt), E_4
    # --------------------------------------------------------

    variational_nonlinear_into_numba(
        n_u_end,
        work_E[4],
        n_c1,
        n_c2,
        n_c3,
        work_E[3],
    )

    # --------------------------------------------------------
    # 第一変分方程式を更新
    #
    # E(t) -> E(t + dt)
    # --------------------------------------------------------

    for i in range(N_local):

        for j in range(dim):

            n_E[i, j] = (
                n_E[i, j] * n_E_visc[i]
                + dt / 6.0
                * (
                    work_E[0, i, j] * n_E_visc[i]
                    + 2.0
                    * work_E[1, i, j]
                    * n_E_visc_half[i]
                    + 2.0
                    * work_E[2, i, j]
                    * n_E_visc_half[i]
                    + work_E[3, i, j]
                )
            )

@njit
def calculate_transient_numba(
    n_u,
    dt,
    transient_time,
    n_E_visc_half,
    n_E_visc_quarter,
    n_c1,
    n_c2,
    n_c3,
    f,
):

    n_u = n_u.copy()

    work_u = np.empty((5, n_u.size),dtype=np.complex128,)

    transient_dt = 0.5 * dt
    transient_steps = int(round(transient_time / transient_dt))

    for step in range(transient_steps):

        rk4_step_u_inplace_numba(
            n_u,
            transient_dt,
            n_E_visc_half,
            n_E_visc_quarter,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u,
        )

    return n_u

# ============================================================
# 6. Modified Gram-Schmidt
# ============================================================

@njit
def real_inner_product_numba(n_v, n_w):

    value = 0.0
    N_local = n_v.size

    for i in range(N_local):
        value += np.real(np.conj(n_v[i]) * n_w[i])

    return value

@njit
def gram_schmidt_real_into_numba(
    n_A,
    n_Q,
    n_r_diag,
    n_w,
):

    N_local, dim = n_A.shape

    for j in range(dim):

        # j本目のベクトルを作業配列へコピー
        for k in range(N_local):
            n_w[k] = n_A[k, j]

        # すでに得た基底方向を順に引く
        for i in range(j):

            r_ij = real_inner_product_numba(n_Q[:, i], n_w)

            for k in range(N_local):
                n_w[k] -= r_ij * n_Q[k, i]

        # 長さ
        norm_sq = real_inner_product_numba(n_w, n_w)

        if not np.isfinite(norm_sq) or norm_sq <= 0.0:
            raise ValueError("Gram-Schmidt法でゼロまたは非有限のノルムが発生しました。")

        r_jj = np.sqrt(norm_sq)

        # 正規化
        for k in range(N_local):
            n_Q[k, j] = n_w[k] / r_jj

        n_r_diag[j] = r_jj

@njit
def calculate_max_orthogonality_error_numba(n_Q):

    dim = n_Q.shape[1]
    max_error = 0.0

    for i in range(dim):
        for j in range(dim):

            inner = real_inner_product_numba(
                n_Q[:, i],
                n_Q[:, j],
            )

            if i == j:
                error = abs(inner - 1.0)
            else:
                error = abs(inner)

            if error > max_error:
                max_error = error

    return max_error

# ============================================================
# 7. 物理量・補助量
# ============================================================

@njit
def calculate_energy_dissipation_numba(n_u,n_k_sq,nu):

    epsilon = 0.0

    for i in range(n_u.size):
        epsilon += (nu * n_k_sq[i] * (n_u[i].real**2 + n_u[i].imag**2))

    return epsilon

@njit
def calculate_energy_injection_numba(
    n_u,
    f,
):

    injection = np.real(
        np.conj(n_u[3]) * f
    )

    return injection

def calculate_kaplan_yorke_dimension(lambdas):

    lambdas_sorted = np.sort(np.asarray(lambdas, dtype=np.float64))[::-1]

    dim = lambdas_sorted.size

    cumulative_sum = 0.0
    j_max = 0
    S_j = 0.0

    for i in range(dim):

        cumulative_sum += lambdas_sorted[i]

        if cumulative_sum >= 0.0:
            j_max = i + 1
            S_j = cumulative_sum
        else:
            break

    if j_max == 0:
        return 0.0

    if j_max == dim:
        return float(dim)

    D_KY = (j_max + S_j / abs(lambdas_sorted[j_max]))

    return D_KY

def calculate_kolmogorov_entropy(lambdas):

    H = np.sum(lambdas[lambdas > 0.0])

    return H

def calculate_dissipation_wavenumber(epsilon_mean, nu):

    k_d = epsilon_mean**0.25 * nu**(-0.75)

    return k_d

# ============================================================
# 8. Lyapunov spectrum 本体
# ============================================================

@njit
def calculate_lyapunov_numba(
    n_u,
    dt,
    t_max,
    tau,
    save_interval,
    n_E_visc,
    n_E_visc_half,
    n_E_visc_quarter,
    n_c1,
    n_c2,
    n_c3,
    f,
    n_k_sq,
    nu,
):

    n_u = n_u.copy()

    N_local = n_u.size
    dim = 2 * N_local

    steps_per_tau = int(round(tau / dt))
    num_tau = int(round(t_max / tau))
    save_every_tau = int(round(save_interval / tau))
    num_save = int(round(t_max / save_interval))

    # --------------------------------------------------------
    # 初期摂動基底
    # --------------------------------------------------------

    n_E = make_initial_tangent_basis_numba(N_local)

    # --------------------------------------------------------
    # 正規直交化の作業配列
    # --------------------------------------------------------

    n_Q = np.empty((N_local, dim),dtype=np.complex128)

    n_r_diag = np.empty(dim, dtype=np.float64)
    n_w = np.empty(N_local, dtype=np.complex128)

    # --------------------------------------------------------
    # RK4の作業配列
    # --------------------------------------------------------

    work_u = np.empty((5, N_local),dtype=np.complex128)

    work_E = np.empty((5, N_local, dim),dtype=np.complex128)

    # --------------------------------------------------------
    # 基準軌道の保存用配列
    #
    # n_u_start = u(t)
    # n_u_half  = u(t + dt/2)
    # --------------------------------------------------------

    n_u_start = np.empty(N_local,dtype=np.complex128)

    n_u_half = np.empty(N_local,dtype=np.complex128)

    # --------------------------------------------------------
    # 累積・保存用配列
    # --------------------------------------------------------

    n_sum_log = np.zeros(dim, dtype=np.float64)
    n_times = np.zeros(num_save, dtype=np.float64)

    n_lambda_history = np.zeros((num_save, dim),dtype=np.float64)

    epsilon_sum = 0.0
    epsilon_count = 0
    save_index = 0

    # MGS直交性の最大誤差
    max_orthogonality_error = 0.0

    # --------------------------------------------------------
    # 測定
    # --------------------------------------------------------

    for m in range(num_tau):

        # tauだけ時間発展
        for step in range(steps_per_tau):

            # --------------------------------------------------------
            # 1. 現在の基準軌道を保存
            #
            # n_u_start = u(t)
            # --------------------------------------------------------

            for i in range(N_local):
                n_u_start[i] = n_u[i]

            # --------------------------------------------------------
            # 2. 基準軌道を dt/2 進める
            #
            # u(t)
            # ->
            # u(t + dt/2)
            # --------------------------------------------------------

            rk4_step_u_inplace_numba(
                n_u,
                0.5 * dt,
                n_E_visc_half,
                n_E_visc_quarter,
                n_c1,
                n_c2,
                n_c3,
                f,
                work_u,
            )

            # --------------------------------------------------------
            # 3. 中間時刻の基準軌道を保存
            #
            # n_u_half = u(t + dt/2)
            # --------------------------------------------------------

            for i in range(N_local):
                n_u_half[i] = n_u[i]

            # --------------------------------------------------------
            # 4. 基準軌道をさらに dt/2 進める
            #
            # u(t + dt/2)
            # ->
            # u(t + dt)
            #
            # この時点で
            #
            # n_u = u(t + dt)
            #
            # となる
            # --------------------------------------------------------

            rk4_step_u_inplace_numba(
                n_u,
                0.5 * dt,
                n_E_visc_half,
                n_E_visc_quarter,
                n_c1,
                n_c2,
                n_c3,
                f,
                work_u,
            )

            # --------------------------------------------------------
            # 5. 第一変分方程式を dt 進める
            #
            # 使用する基準軌道
            #
            # n_u_start = u(t)
            # n_u_half  = u(t + dt/2)
            # n_u       = u(t + dt)
            # --------------------------------------------------------

            rk4_step_E_inplace_numba(
                n_E,
                dt,
                n_E_visc,
                n_E_visc_half,
                n_u_start,
                n_u_half,
                n_u,
                n_c1,
                n_c2,
                n_c3,
                work_E,
            )

            # --------------------------------------------------------
            # 6. エネルギー散逸率
            #
            # n_u はすでに u(t + dt)
            # --------------------------------------------------------

            epsilon = calculate_energy_dissipation_numba(
                n_u,
                n_k_sq,
                nu
            )

            epsilon_sum += epsilon
            epsilon_count += 1

        # 正規直交化
        gram_schmidt_real_into_numba(n_E,n_Q,n_r_diag,n_w)

        # Qを次の摂動基底として使用。
        # 古いEは、次回のQの書き込み先として再利用。
        n_E, n_Q = n_Q, n_E

        # ----------------------------------------------------
        # MGS直交性の検証
        #
        # n_E には直交化後の基底 Q が入っている。
        # max |<q_i,q_j> - delta_ij| を測る。
        # ----------------------------------------------------

        orthogonality_error = (
            calculate_max_orthogonality_error_numba(
                n_E
            )
        )

        if orthogonality_error > max_orthogonality_error:
            max_orthogonality_error = orthogonality_error

        # log(r_jj)の累積
        for j in range(dim):
            n_sum_log[j] += np.log(abs(n_r_diag[j]))

        t = (m + 1) * tau

        # 指定間隔で履歴を保存
        if ((m + 1) % save_every_tau) == 0:

            n_times[save_index] = t

            for j in range(dim):
                n_lambda_history[save_index, j] = (n_sum_log[j] / t)

            save_index += 1

    epsilon_mean = epsilon_sum / epsilon_count

    return (
        n_u,
        n_times,
        n_lambda_history,
        epsilon_mean,
        max_orthogonality_error,
    )

def run_lyapunov(
    N,
    nu,
    dt,
    transient_time,
    t_max,
    tau,
    save_interval,
    seed=42,
):

    # --------------------------------------------------------
    # 入力値の確認と変換
    # --------------------------------------------------------

    if not np.isfinite(N) or N != int(N):
        raise ValueError("Nは4以上の整数にしてください。")

    if (
        not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= 4294967295
    ):
        raise ValueError("seedは0～4294967295の整数にしてください。")

    if not np.all(
        np.isfinite(
            np.asarray(
                [
                    nu,
                    dt,
                    transient_time,
                    t_max,
                    tau,
                    save_interval,
                ],
                dtype=np.float64,
            )
        )
    ):
        raise ValueError("計算条件には有限の数値を指定してください。")

    N = int(N)
    nu = np.float64(nu)
    dt = np.float64(dt)
    transient_time = np.float64(transient_time)
    t_max = np.float64(t_max)
    tau = np.float64(tau)
    save_interval = np.float64(save_interval)

    if N < 4:
        raise ValueError("Nは4以上の整数にしてください。")

    if nu <= 0.0:
        raise ValueError("nuは正の値にしてください。")

    if dt <= 0.0:
        raise ValueError("dtは正の値にしてください。")

    if transient_time < 0.0:
        raise ValueError("transient_timeは0以上にしてください。")

    if t_max <= 0.0:
        raise ValueError("t_maxは正の値にしてください。")

    if tau <= 0.0:
        raise ValueError("tauは正の値にしてください。")

    if tau < dt:
        raise ValueError("tauはdt以上にしてください。")

    if save_interval < tau:
        raise ValueError("save_intervalはtau以上にしてください。")

    # --------------------------------------------------------
    # tauがdtの整数倍か確認
    # --------------------------------------------------------

    steps_per_tau = int(round(tau / dt))
    actual_tau = steps_per_tau * dt

    if not np.isclose(
        actual_tau, tau,
        rtol=1.0e-12, atol=1.0e-14,
    ):
        raise ValueError("tauはdtの整数倍にしてください。")

    # --------------------------------------------------------
    # t_maxがtauの整数倍か確認
    # --------------------------------------------------------

    num_tau = int(round(t_max / tau))
    actual_t_max = num_tau * tau

    if not np.isclose(
        actual_t_max, t_max,
        rtol=1.0e-12, atol=1.0e-14,
    ):
        raise ValueError("t_maxはtauの整数倍にしてください。")

    # --------------------------------------------------------
    # save_intervalがtauの整数倍か確認
    # --------------------------------------------------------

    save_every_tau = int(round(save_interval / tau))
    actual_save_interval = save_every_tau * tau

    if not np.isclose(
        actual_save_interval, save_interval,
        rtol=1.0e-12, atol=1.0e-14,
    ):
        raise ValueError("save_intervalはtauの整数倍にしてください。")

    # --------------------------------------------------------
    # t_maxがsave_intervalの整数倍か確認
    # --------------------------------------------------------

    num_save = int(round(t_max / save_interval))
    actual_t_max_from_save = num_save * save_interval

    if not np.isclose(
        actual_t_max_from_save, t_max,
        rtol=1.0e-12, atol=1.0e-14,
    ):
        raise ValueError("t_maxはsave_intervalの整数倍にしてください。")

    # --------------------------------------------------------
    # シェルパラメータと初期条件
    # --------------------------------------------------------

    (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
    ) = make_shell_parameters(N)

    n_u = make_initial_condition_numba(n_k, n_k_sq, seed)

    # 積分因子
    n_E_visc = np.exp(-nu * n_k_sq * dt)
    n_E_visc_half = np.exp(-nu * n_k_sq * dt * 0.5)
    n_E_visc_quarter = np.exp(-nu * n_k_sq * dt * 0.25)

    # --------------------------------------------------------
    # 計算条件の表示
    # --------------------------------------------------------

    orbit_dt = 0.5 * dt

    transient_steps_orbit = int(round(transient_time / orbit_dt))

    print("--- 計算条件 ---")
    print(f"シェル数 N: {N}")
    print(f"実次元 2N: {2 * N}")
    print(f"初期位相の乱数 seed: {seed}")
    print(f"動粘性係数 nu: {nu:.10e}")

    print(f"変分方程式の時間刻み dt: {dt}")
    print(f"基準軌道の時間刻み dt/2: {orbit_dt}")

    print(f"過渡時間 transient_time: {transient_time}")
    print(f"過渡期間の基準軌道ステップ数: "f"{transient_steps_orbit:,}")

    print(f"Lyapunov指数の測定時間 t_max: {t_max}")
    print(f"Gram-Schmidt間隔 tau: {tau}")
    print(f"Lyapunov指数の保存間隔 save_interval: {save_interval}")

    print(f"1回のGram-Schmidtまでの変分方程式ステップ数: "f"{steps_per_tau:,}")

    print(f"1回のGram-Schmidtまでの基準軌道ステップ数: "f"{2 * steps_per_tau:,}")

    print(f"Gram-Schmidt回数: {num_tau:,}")
    print()

    # --------------------------------------------------------
    # 過渡状態
    # --------------------------------------------------------

    print("過渡状態を計算中...")

    n_u = calculate_transient_numba(
        n_u,
        dt,
        transient_time,
        n_E_visc_half,
        n_E_visc_quarter,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    print("過渡状態の計算完了")
    print()

    # --------------------------------------------------------
    # Lyapunov spectrum
    # --------------------------------------------------------

    print("Lyapunov spectrum を計算中...")

    (
        n_u_final,
        n_times,
        n_lambda_history,
        epsilon_mean,
        max_orthogonality_error,
    ) = calculate_lyapunov_numba(
        n_u,
        dt,
        t_max,
        tau,
        save_interval,
        n_E_visc,
        n_E_visc_half,
        n_E_visc_quarter,
        n_c1,
        n_c2,
        n_c3,
        f,
        n_k_sq,
        nu,
    )

    # --------------------------------------------------------
    # 結果の計算
    # --------------------------------------------------------

    lambdas = n_lambda_history[-1].copy()

    D_KY = calculate_kaplan_yorke_dimension(lambdas)

    k_d = calculate_dissipation_wavenumber(epsilon_mean, nu)

    divergence = -2.0 * nu * np.sum(n_k_sq)
    lambda_sum = np.sum(lambdas)

    relative_error = (abs(lambda_sum - divergence) / abs(divergence))

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print("計算完了！")
    print()

    print("--- Lyapunov spectrum ---")

    for j in range(lambdas.size):
        print(f"lambda_{j + 1:2d} = {lambdas[j]: .10e}")

    print()
    print(f"Kaplan-Yorke dimension D_KY = {D_KY:.10f}")

    print()
    print("--- 散逸に関する値 ---")
    print(f"平均エネルギー散逸率 <epsilon> = {epsilon_mean:.10e}")
    print(f"散逸波数 k_d = {k_d:.10e}")

    print()
    print("--- 発散との比較 ---")
    print(f"sum(lambda_i) = {lambda_sum:.10e}")
    print(f"div F = {divergence:.10e}")
    print(f"relative error = {relative_error:.10e}")
    print(f"relative error (%) = {100.0 * relative_error:.6f} %")

    print()
    print("--- Modified Gram-Schmidt の直交性 ---")
    print(
        "max |<q_i,q_j> - delta_ij| = "
        f"{max_orthogonality_error:.10e}"
    )

    # --------------------------------------------------------
    # 辞書形式で返す
    # --------------------------------------------------------

    return {
        "N": N,
        "nu": nu,
        "dt": dt,
        "transient_time": transient_time,
        "t_max": t_max,
        "tau": tau,
        "save_interval": save_interval,
        "seed": seed,

        "lambda_1": np.max(lambdas),
        "H": calculate_kolmogorov_entropy(lambdas),

        "k": n_k.copy(),
        "u_final": n_u_final.copy(),

        "t": n_times.copy(),
        "lambdas": lambdas.copy(),
        "lambda_history": n_lambda_history.copy(),

        "D_KY": D_KY,
        "epsilon_mean": epsilon_mean,
        "k_d": k_d,

        "lambda_sum": lambda_sum,
        "divergence": divergence,
    }

# ============================================================
# 9. 検証用関数
# ============================================================

@njit
def full_rhs_numba(
    n_u,
    n_k_sq,
    nu,
    n_c1,
    n_c2,
    n_c3,
    f,
    n_rhs,
    work_nl,
):

    N_local = n_u.size

    # 非線形項 + 外力
    nonlinear_into_numba(
        n_u,
        n_c1,
        n_c2,
        n_c3,
        f,
        work_nl,
    )

    # 粘性項も加えて完全な右辺 F(u) を作る
    for i in range(N_local):

        n_rhs[i] = (
            -nu * n_k_sq[i] * n_u[i]
            + work_nl[i]
        )

@njit
def central_difference_numba(
    n_u,
    n_delta_u,
    epsilon,
    n_k_sq,
    nu,
    n_c1,
    n_c2,
    n_c3,
    f,
    n_center_diff,
    n_u_plus,
    n_u_minus,
    n_rhs_plus,
    n_rhs_minus,
    work_nl_plus,
    work_nl_minus,
):

    N_local = n_u.size

    # --------------------------------------------------------
    # u + epsilon delta_u
    # u - epsilon delta_u
    # を作る
    # --------------------------------------------------------

    for i in range(N_local):

        n_u_plus[i] = (
            n_u[i]
            + epsilon * n_delta_u[i]
        )

        n_u_minus[i] = (
            n_u[i]
            - epsilon * n_delta_u[i]
        )

    # --------------------------------------------------------
    # F(u + epsilon delta_u)
    # --------------------------------------------------------

    full_rhs_numba(
        n_u_plus,
        n_k_sq,
        nu,
        n_c1,
        n_c2,
        n_c3,
        f,
        n_rhs_plus,
        work_nl_plus,
    )

    # --------------------------------------------------------
    # F(u - epsilon delta_u)
    # --------------------------------------------------------

    full_rhs_numba(
        n_u_minus,
        n_k_sq,
        nu,
        n_c1,
        n_c2,
        n_c3,
        f,
        n_rhs_minus,
        work_nl_minus,
    )

    # --------------------------------------------------------
    # 中心差分
    # --------------------------------------------------------

    for i in range(N_local):

        n_center_diff[i] = (
            n_rhs_plus[i]
            - n_rhs_minus[i]
        ) / (2.0 * epsilon)

@njit
def analytical_variation_numba(
    n_u,
    n_delta_u,
    n_k_sq,
    nu,
    n_c1,
    n_c2,
    n_c3,
    n_analytic,
    n_E_temp,
    n_dE_temp,
):

    N_local = n_u.size

    # --------------------------------------------------------
    # 1本の摂動ベクトル delta_u を
    # N x 1 の行列として入れる
    # --------------------------------------------------------

    for i in range(N_local):
        n_E_temp[i, 0] = n_delta_u[i]

    # --------------------------------------------------------
    # 非線形項の変分 DN(u) delta_u
    # --------------------------------------------------------

    variational_nonlinear_into_numba(
        n_u,
        n_E_temp,
        n_c1,
        n_c2,
        n_c3,
        n_dE_temp,
    )

    # --------------------------------------------------------
    # 粘性項も加えて
    #
    # DF(u) delta_u
    #
    # を作る
    # --------------------------------------------------------

    for i in range(N_local):

        n_analytic[i] = (
            -nu * n_k_sq[i] * n_delta_u[i]
            + n_dE_temp[i, 0]
        )

@njit
def calculate_variation_error_numba(
    n_analytic,
    n_center_diff,
):

    N_local = n_analytic.size

    diff_norm_sq = 0.0
    analytic_norm_sq = 0.0

    for i in range(N_local):

        diff = (
            n_analytic[i]
            - n_center_diff[i]
        )

        diff_norm_sq += (
            diff.real**2
            + diff.imag**2
        )

        analytic_norm_sq += (
            n_analytic[i].real**2
            + n_analytic[i].imag**2
        )

    absolute_error = np.sqrt(
        diff_norm_sq
    )

    analytic_norm = np.sqrt(
        analytic_norm_sq
    )

    if analytic_norm == 0.0:
        relative_error = np.nan
    else:
        relative_error = (
            absolute_error
            / analytic_norm
        )

    return (
        absolute_error,
        relative_error,
    )

def check_variational_equation(
    N,
    nu,
    dt,
    transient_time,
    epsilon_values,
    basis_index=0,
    seed=42,
):

    # --------------------------------------------------------
    # シェルパラメータ
    # --------------------------------------------------------

    (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
    ) = make_shell_parameters(N)

    # --------------------------------------------------------
    # 初期条件
    # --------------------------------------------------------

    n_u = make_initial_condition_numba(
        n_k,
        n_k_sq,
        seed,
    )

    # --------------------------------------------------------
    # 積分因子
    # --------------------------------------------------------

    n_E_visc_half = np.exp(
        -nu * n_k_sq * dt * 0.5
    )

    n_E_visc_quarter = np.exp(
        -nu * n_k_sq * dt * 0.25
    )

    # --------------------------------------------------------
    # 過渡状態
    #
    # 基準状態 u をアトラクタ上へ移す
    # --------------------------------------------------------

    n_u = calculate_transient_numba(
        n_u,
        dt,
        transient_time,
        n_E_visc_half,
        n_E_visc_quarter,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    # --------------------------------------------------------
    # 初期摂動基底
    # --------------------------------------------------------

    n_E = make_initial_tangent_basis_numba(N)

    dim = 2 * N

    if not (0 <= basis_index < dim):
        raise ValueError(
            f"basis_index は 0 から {dim - 1} の範囲にしてください。"
        )

    # --------------------------------------------------------
    # 検証に使う摂動ベクトル delta_u
    #
    # n_E の basis_index 列を使う
    # --------------------------------------------------------

    n_delta_u = n_E[:, basis_index].copy()

    # --------------------------------------------------------
    # 作業配列
    # --------------------------------------------------------

    n_analytic = np.empty(
        N,
        dtype=np.complex128,
    )

    n_center_diff = np.empty(
        N,
        dtype=np.complex128,
    )

    n_u_plus = np.empty(
        N,
        dtype=np.complex128,
    )

    n_u_minus = np.empty(
        N,
        dtype=np.complex128,
    )

    n_rhs_plus = np.empty(
        N,
        dtype=np.complex128,
    )

    n_rhs_minus = np.empty(
        N,
        dtype=np.complex128,
    )

    work_nl_plus = np.empty(
        N,
        dtype=np.complex128,
    )

    work_nl_minus = np.empty(
        N,
        dtype=np.complex128,
    )

    n_E_temp = np.empty(
        (N, 1),
        dtype=np.complex128,
    )

    n_dE_temp = np.empty(
        (N, 1),
        dtype=np.complex128,
    )

    # --------------------------------------------------------
    # 解析的な DF(u) delta_u
    #
    # epsilon には依存しないので1回だけ計算
    # --------------------------------------------------------

    analytical_variation_numba(
        n_u,
        n_delta_u,
        n_k_sq,
        nu,
        n_c1,
        n_c2,
        n_c3,
        n_analytic,
        n_E_temp,
        n_dE_temp,
    )

    # --------------------------------------------------------
    # 誤差保存用
    # --------------------------------------------------------

    epsilon_values = np.asarray(
        epsilon_values,
        dtype=np.float64,
    )

    absolute_errors = np.empty(
        epsilon_values.size,
        dtype=np.float64,
    )

    relative_errors = np.empty(
        epsilon_values.size,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # epsilon を変えながら中心差分を計算
    # --------------------------------------------------------

    for m in range(epsilon_values.size):

        epsilon = epsilon_values[m]

        central_difference_numba(
            n_u,
            n_delta_u,
            epsilon,
            n_k_sq,
            nu,
            n_c1,
            n_c2,
            n_c3,
            f,
            n_center_diff,
            n_u_plus,
            n_u_minus,
            n_rhs_plus,
            n_rhs_minus,
            work_nl_plus,
            work_nl_minus,
        )

        (
            absolute_error,
            relative_error,
        ) = calculate_variation_error_numba(
            n_analytic,
            n_center_diff,
        )

        absolute_errors[m] = absolute_error
        relative_errors[m] = relative_error

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print("--- 中心差分による第一変分方程式の検証 ---")
    print(f"N = {N}")
    print(f"nu = {nu:.10e}")
    print(f"dt = {dt}")
    print(f"transient_time = {transient_time}")
    print(f"basis_index = {basis_index}")
    print()

    for m in range(epsilon_values.size):

        print(
            f"epsilon = {epsilon_values[m]:.1e}, "
            f"absolute error = {absolute_errors[m]:.10e}, "
            f"relative error = {relative_errors[m]:.10e}"
        )

    return {
        "epsilon": epsilon_values,
        "absolute_error": absolute_errors,
        "relative_error": relative_errors,
        "u": n_u.copy(),
        "delta_u": n_delta_u.copy(),
        "analytic": n_analytic.copy(),
    }

def check_variational_one_step(
    N,
    nu,
    dt,
    transient_time,
    epsilon_values,
    basis_index=0,
    seed=42,
):

    # --------------------------------------------------------
    # 1. シェルパラメータ
    # --------------------------------------------------------

    (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
    ) = make_shell_parameters(N)

    # --------------------------------------------------------
    # 2. 初期条件
    # --------------------------------------------------------

    n_u = make_initial_condition_numba(
        n_k,
        n_k_sq,
        seed,
    )

    # --------------------------------------------------------
    # 3. 積分因子
    #
    # 変分方程式は dt で1ステップ進める。
    # 基準軌道は dt/2 ずつ2回進める。
    # --------------------------------------------------------

    n_E_visc = np.exp(
        -nu * n_k_sq * dt
    )

    n_E_visc_half = np.exp(
        -nu * n_k_sq * dt * 0.5
    )

    n_E_visc_quarter = np.exp(
        -nu * n_k_sq * dt * 0.25
    )

    # --------------------------------------------------------
    # 4. 過渡状態
    #
    # 基準状態をアトラクタ上へ移す
    # --------------------------------------------------------

    n_u = calculate_transient_numba(
        n_u,
        dt,
        transient_time,
        n_E_visc_half,
        n_E_visc_quarter,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    # --------------------------------------------------------
    # 5. 検証開始時刻の基準状態
    #
    # u_start = u(t)
    # --------------------------------------------------------

    n_u_start = n_u.copy()

    # --------------------------------------------------------
    # 6. 初期摂動基底
    # --------------------------------------------------------

    n_basis = make_initial_tangent_basis_numba(N)

    dim = 2 * N

    if not (0 <= basis_index < dim):
        raise ValueError(
            f"basis_index は 0 から {dim - 1} の範囲にしてください。"
        )

    # 検証する1本の摂動
    n_delta_u = n_basis[:, basis_index].copy()

    # --------------------------------------------------------
    # 7. 基準軌道
    #
    # u(t)
    #   ->
    # u(t + dt/2)
    #   ->
    # u(t + dt)
    #
    # を作る。
    # --------------------------------------------------------

    n_u_half = n_u_start.copy()

    work_u_base = np.empty(
        (5, N),
        dtype=np.complex128,
    )

    # u(t) -> u(t + dt/2)
    rk4_step_u_inplace_numba(
        n_u_half,
        0.5 * dt,
        n_E_visc_half,
        n_E_visc_quarter,
        n_c1,
        n_c2,
        n_c3,
        f,
        work_u_base,
    )

    # 中間状態を保存
    n_u_end = n_u_half.copy()

    # u(t + dt/2) -> u(t + dt)
    rk4_step_u_inplace_numba(
        n_u_end,
        0.5 * dt,
        n_E_visc_half,
        n_E_visc_quarter,
        n_c1,
        n_c2,
        n_c3,
        f,
        work_u_base,
    )

    # --------------------------------------------------------
    # 8. 第一変分方程式側
    #
    # delta_u(t)
    #   ->
    # delta_u(t + dt)
    #
    # を rk4_step_E_inplace_numba() で計算する。
    # --------------------------------------------------------

    n_E_test = np.empty(
        (N, 1),
        dtype=np.complex128,
    )

    for i in range(N):
        n_E_test[i, 0] = n_delta_u[i]

    work_E_test = np.empty(
        (5, N, 1),
        dtype=np.complex128,
    )

    rk4_step_E_inplace_numba(
        n_E_test,
        dt,
        n_E_visc,
        n_E_visc_half,
        n_u_start,
        n_u_half,
        n_u_end,
        n_c1,
        n_c2,
        n_c3,
        work_E_test,
    )

    # 第一変分方程式で得た1ステップ後の摂動
    n_variational = n_E_test[:, 0].copy()

    # --------------------------------------------------------
    # 9. epsilon の準備
    # --------------------------------------------------------

    epsilon_values = np.asarray(
        epsilon_values,
        dtype=np.float64,
    )

    absolute_errors = np.empty(
        epsilon_values.size,
        dtype=np.float64,
    )

    relative_errors = np.empty(
        epsilon_values.size,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 10. 作業配列
    # --------------------------------------------------------

    work_u_plus = np.empty(
        (5, N),
        dtype=np.complex128,
    )

    work_u_minus = np.empty(
        (5, N),
        dtype=np.complex128,
    )

    # --------------------------------------------------------
    # 11. epsilon を変えながら
    #
    # Phi_dt(u + eps delta_u)
    #
    # Phi_dt(u - eps delta_u)
    #
    # を計算する。
    # --------------------------------------------------------

    for m in range(epsilon_values.size):

        epsilon = epsilon_values[m]

        # ----------------------------------------------------
        # u + epsilon delta_u
        # u - epsilon delta_u
        # ----------------------------------------------------

        n_u_plus = (
            n_u_start
            + epsilon * n_delta_u
        ).copy()

        n_u_minus = (
            n_u_start
            - epsilon * n_delta_u
        ).copy()

        # ----------------------------------------------------
        # plus側
        #
        # dt/2 を2回
        # ----------------------------------------------------

        rk4_step_u_inplace_numba(
            n_u_plus,
            0.5 * dt,
            n_E_visc_half,
            n_E_visc_quarter,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u_plus,
        )

        rk4_step_u_inplace_numba(
            n_u_plus,
            0.5 * dt,
            n_E_visc_half,
            n_E_visc_quarter,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u_plus,
        )

        # ----------------------------------------------------
        # minus側
        #
        # dt/2 を2回
        # ----------------------------------------------------

        rk4_step_u_inplace_numba(
            n_u_minus,
            0.5 * dt,
            n_E_visc_half,
            n_E_visc_quarter,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u_minus,
        )

        rk4_step_u_inplace_numba(
            n_u_minus,
            0.5 * dt,
            n_E_visc_half,
            n_E_visc_quarter,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u_minus,
        )

        # ----------------------------------------------------
        # 12. 非線形軌道による中心差分
        #
        # [Phi_dt(u+eps delta_u)-Phi_dt(u-eps delta_u)]
        # ---------------------------------------------------
        #                     2 eps
        # ----------------------------------------------------

        n_center_diff = (
            n_u_plus
            - n_u_minus
        ) / (2.0 * epsilon)

        # ----------------------------------------------------
        # 13. 第一変分方程式との誤差
        # ----------------------------------------------------

        difference = (
            n_center_diff
            - n_variational
        )

        absolute_error = np.linalg.norm(
            difference
        )

        variational_norm = np.linalg.norm(
            n_variational
        )

        if variational_norm == 0.0:
            relative_error = np.nan
        else:
            relative_error = (
                absolute_error
                / variational_norm
            )

        absolute_errors[m] = absolute_error
        relative_errors[m] = relative_error

    # --------------------------------------------------------
    # 14. 結果表示
    # --------------------------------------------------------

    print(
        "--- 第一変分方程式の1ステップ検証 ---"
    )

    print(f"N = {N}")
    print(f"nu = {nu:.10e}")
    print(f"dt = {dt}")
    print(
        f"基準軌道の時間刻み = {0.5 * dt}"
    )
    print(
        f"transient_time = {transient_time}"
    )
    print(
        f"basis_index = {basis_index}"
    )
    print()

    for m in range(epsilon_values.size):

        print(
            f"epsilon = {epsilon_values[m]:.1e}, "
            f"absolute error = "
            f"{absolute_errors[m]:.10e}, "
            f"relative error = "
            f"{relative_errors[m]:.10e}"
        )

    # --------------------------------------------------------
    # 15. 結果を返す
    # --------------------------------------------------------

    return {
        "epsilon": epsilon_values,
        "absolute_error": absolute_errors,
        "relative_error": relative_errors,
        "u_start": n_u_start.copy(),
        "u_half": n_u_half.copy(),
        "u_end": n_u_end.copy(),
        "delta_u": n_delta_u.copy(),
        "variational": n_variational.copy(),
    }

def check_mgs_orthogonality(
    N,
    seed=42,
):

    dim = 2 * N

    rng = np.random.default_rng(seed)

    n_A = (
        rng.standard_normal((N, dim))
        + 1j * rng.standard_normal((N, dim))
    )

    n_Q = np.empty(
        (N, dim),
        dtype=np.complex128,
    )

    n_r_diag = np.empty(
        dim,
        dtype=np.float64,
    )

    n_w = np.empty(
        N,
        dtype=np.complex128,
    )

    gram_schmidt_real_into_numba(
        n_A,
        n_Q,
        n_r_diag,
        n_w,
    )

    max_error = (
        calculate_max_orthogonality_error_numba(
            n_Q
        )
    )

    print(
        "--- Modified Gram-Schmidt の直交性検証 ---"
    )
    print(f"N = {N}")
    print(f"実次元 2N = {dim}")
    print(
        "max |<q_i,q_j> - delta_ij| = "
        f"{max_error:.10e}"
    )

    return {
        "Q": n_Q.copy(),
        "r_diag": n_r_diag.copy(),
        "max_orthogonality_error": max_error,
    }

def check_base_orbit_stationarity(
    N,
    nu,
    dt,
    transient_time,
    measurement_time,
    save_interval=1.0,
    seed=42,
):

    # ========================================================
    # 1. シェルパラメータ
    # ========================================================

    (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
    ) = make_shell_parameters(N)

    # ========================================================
    # 2. 初期条件
    # ========================================================

    n_u = make_initial_condition_numba(
        n_k,
        n_k_sq,
        seed,
    )

    # ========================================================
    # 3. 基準軌道の時間刻み
    #
    # Lyapunov計算と同じく dt/2 を使う
    # ========================================================

    orbit_dt = 0.5 * dt

    n_E_visc_orbit = np.exp(
        -nu * n_k_sq * orbit_dt
    )

    n_E_visc_orbit_half = np.exp(
        -nu * n_k_sq * orbit_dt * 0.5
    )

    # ========================================================
    # 4. 過渡状態
    # ========================================================

    n_u = calculate_transient_numba(
        n_u,
        dt,
        transient_time,
        n_E_visc_orbit,
        n_E_visc_orbit_half,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    # ========================================================
    # 5. 保存条件
    # ========================================================

    num_steps = int(
        round(measurement_time / orbit_dt)
    )

    save_every = int(
        round(save_interval / orbit_dt)
    )

    if save_every < 1:
        raise ValueError(
            "save_interval は orbit_dt 以上にしてください。"
        )

    num_save = num_steps // save_every

    times = np.empty(
        num_save,
        dtype=np.float64,
    )

    epsilon_history = np.empty(
        num_save,
        dtype=np.float64,
    )

    injection_history = np.empty(
        num_save,
        dtype=np.float64,
    )

    energy_history = np.empty(
        num_save,
        dtype=np.float64,
    )

    # RK4 作業配列
    work_u = np.empty(
        (5, N),
        dtype=np.complex128,
    )

    save_index = 0

    # ========================================================
    # 6. 基準軌道だけを時間発展
    # ========================================================

    for step in range(num_steps):

        rk4_step_u_inplace_numba(
            n_u,
            orbit_dt,
            n_E_visc_orbit,
            n_E_visc_orbit_half,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u,
        )

        # ====================================================
        # 指定間隔で保存
        # ====================================================

        if (step + 1) % save_every == 0:

            t = (step + 1) * orbit_dt

            # ------------------------------
            # 散逸率
            # ------------------------------

            epsilon = (
                calculate_energy_dissipation_numba(
                    n_u,
                    n_k_sq,
                    nu,
                )
            )

            # ------------------------------
            # 注入率
            # ------------------------------

            injection = (
                calculate_energy_injection_numba(
                    n_u,
                    f,
                )
            )

            # ------------------------------
            # 全エネルギー
            #
            # E_total = 1/2 sum |u_n|^2
            # ------------------------------

            energy_total = (
                0.5
                * np.sum(
                    np.abs(n_u) ** 2
                )
            )

            times[save_index] = t

            epsilon_history[save_index] = epsilon
            injection_history[save_index] = injection
            energy_history[save_index] = energy_total

            save_index += 1

    # ========================================================
    # 7. 平均値
    # ========================================================

    epsilon_mean = np.mean(
        epsilon_history
    )

    injection_mean = np.mean(
        injection_history
    )

    energy_mean = np.mean(
        energy_history
    )

    print(
        "--- 基準軌道の定常性確認 ---"
    )
    print(f"N = {N}")
    print(f"nu = {nu:.10e}")
    print(f"dt = {dt}")
    print(f"orbit_dt = {orbit_dt}")
    print(
        f"transient_time = {transient_time}"
    )
    print(
        f"measurement_time = {measurement_time}"
    )
    print(
        f"<P> = {injection_mean:.10e}"
    )
    print(
        f"<epsilon> = {epsilon_mean:.10e}"
    )
    print(
        f"<P> - <epsilon> = "
        f"{injection_mean - epsilon_mean:.10e}"
    )
    print(
        f"<E_total> = {energy_mean:.10e}"
    )

    return {
        "t": times,
        "epsilon": epsilon_history,
        "injection": injection_history,
        "energy": energy_history,
        "epsilon_mean": epsilon_mean,
        "injection_mean": injection_mean,
        "energy_mean": energy_mean,
        "u_final": n_u.copy(),
    }

def continue_base_orbit(
    u_initial,
    N,
    nu,
    dt,
    additional_time,
    save_interval=1.0,
):

    # --------------------------------------------------------
    # 1. シェルパラメータ
    # --------------------------------------------------------

    (
        n_k,
        n_k_sq,
        n_c1,
        n_c2,
        n_c3,
    ) = make_shell_parameters(N)

    # --------------------------------------------------------
    # 2. 開始状態
    #
    # seed から作り直さず、
    # 既存の u_final をそのまま使う
    # --------------------------------------------------------

    n_u = np.asarray(
        u_initial,
        dtype=np.complex128,
    ).copy()

    # --------------------------------------------------------
    # 3. 基準軌道の時間刻み
    # --------------------------------------------------------

    orbit_dt = 0.5 * dt

    n_E_visc_orbit = np.exp(
        -nu * n_k_sq * orbit_dt
    )

    n_E_visc_orbit_half = np.exp(
        -nu * n_k_sq * orbit_dt * 0.5
    )

    # --------------------------------------------------------
    # 4. 計算回数
    # --------------------------------------------------------

    num_steps = int(
        round(additional_time / orbit_dt)
    )

    save_every = int(
        round(save_interval / orbit_dt)
    )

    if save_every < 1:
        raise ValueError(
            "save_interval は orbit_dt 以上にしてください。"
        )

    num_save = num_steps // save_every

    # --------------------------------------------------------
    # 5. 保存用配列
    # --------------------------------------------------------

    times = np.empty(
        num_save,
        dtype=np.float64,
    )

    epsilon_history = np.empty(
        num_save,
        dtype=np.float64,
    )

    injection_history = np.empty(
        num_save,
        dtype=np.float64,
    )

    energy_history = np.empty(
        num_save,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 6. RK4の作業配列
    # --------------------------------------------------------

    work_u = np.empty(
        (5, N),
        dtype=np.complex128,
    )

    # 保存先の番号
    save_index = 0

    # --------------------------------------------------------
    # 7. 続きの時間発展
    # --------------------------------------------------------

    for step in range(num_steps):

        rk4_step_u_inplace_numba(
            n_u,
            orbit_dt,
            n_E_visc_orbit,
            n_E_visc_orbit_half,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u,
        )

        # ----------------------------------------------------
        # save_interval ごとに保存
        # ----------------------------------------------------

        if (step + 1) % save_every == 0:

            t = (step + 1) * orbit_dt

            # 散逸率
            epsilon = (
                calculate_energy_dissipation_numba(
                    n_u,
                    n_k_sq,
                    nu,
                )
            )

            # 注入率
            injection = (
                calculate_energy_injection_numba(
                    n_u,
                    f,
                )
            )

            # 全エネルギー
            energy_total = (
                0.5
                * np.sum(
                    np.abs(n_u) ** 2
                )
            )

            # 保存
            times[save_index] = t
            epsilon_history[save_index] = epsilon
            injection_history[save_index] = injection
            energy_history[save_index] = energy_total

            save_index += 1

    # --------------------------------------------------------
    # 8. 平均値
    # --------------------------------------------------------

    epsilon_mean = np.mean(
        epsilon_history
    )

    injection_mean = np.mean(
        injection_history
    )

    energy_mean = np.mean(
        energy_history
    )

    print("--- 基準軌道の続き計算 ---")
    print(f"N = {N}")
    print(f"nu = {nu:.10e}")
    print(f"dt = {dt}")
    print(f"orbit_dt = {orbit_dt}")
    print(f"additional_time = {additional_time}")

    print(
        f"<P> = {injection_mean:.10e}"
    )

    print(
        f"<epsilon> = {epsilon_mean:.10e}"
    )

    print(
        f"<P> - <epsilon> = "
        f"{injection_mean - epsilon_mean:.10e}"
    )

    print(
        f"<E_total> = {energy_mean:.10e}"
    )

    # --------------------------------------------------------
    # 9. 結果を返す
    # --------------------------------------------------------

    return {
        "t": times,

        "epsilon": epsilon_history,
        "injection": injection_history,
        "energy": energy_history,

        "epsilon_mean": epsilon_mean,
        "injection_mean": injection_mean,
        "energy_mean": energy_mean,

        "u_final": n_u.copy(),
    }

# ============================================================
# 10. 描画用関数
# ============================================================

def plot_lyapunov_history(
    result,
    index_start,
    index_end,
):

    # --------------------------------------------------------
    # データ
    # --------------------------------------------------------

    t = result["t"]
    history = result["lambda_history"]

    dim = history.shape[1]

    # --------------------------------------------------------
    # 指数番号の確認
    # --------------------------------------------------------

    if not (1 <= index_start <= index_end <= dim):
        raise ValueError(
            f"1 <= index_start <= index_end <= {dim} "
            "となるように指定してください。"
        )

    # --------------------------------------------------------
    # 描画
    # --------------------------------------------------------

    plt.figure(figsize=(7, 5))

    for i in range(index_start - 1, index_end):

        plt.plot(
            t,
            history[:, i],
            label=rf"$\lambda_{i + 1}$",
        )

    plt.axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$\lambda_i(t)$")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lyapunov_spectrum(
    result,
    index_start=1,
    index_end=None,
):

    # --------------------------------------------------------
    # 計算結果の取得
    # --------------------------------------------------------

    lambdas = np.sort(
        np.asarray(
            result["lambdas"],
            dtype=np.float64,
        )
    )[::-1]

    num_exponents = lambdas.size

    j = np.arange(
        1,
        num_exponents + 1,
    )

    # --------------------------------------------------------
    # 拡大表示する終了番号
    # --------------------------------------------------------

    if index_end is None:
        actual_end = num_exponents
    else:
        actual_end = min(
            index_end,
            num_exponents,
        )

    # --------------------------------------------------------
    # 表示範囲の確認
    # --------------------------------------------------------

    if not (
        1
        <= index_start
        <= actual_end
    ):
        raise ValueError(
            f"表示範囲を確認してください。"
            f"この結果の指数は{num_exponents}本です。"
        )

    # ========================================================
    # 1. Lyapunov spectrum 全体
    # ========================================================

    plt.figure(
        figsize=(7, 5.5)
    )

    plt.plot(
        j,
        lambdas,
        marker="o",
        markersize=4,
    )

    plt.axhline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    plt.xlabel(r"$j$")
    plt.ylabel(r"$\lambda_j$")

    plt.title(
        "Lyapunov spectrum"
    )

    plt.grid()

    plt.tight_layout()
    plt.show()

    # ========================================================
    # 2. 指定範囲を拡大
    # ========================================================

    selected = slice(
        index_start - 1,
        actual_end,
    )

    plt.figure(
        figsize=(7, 5.5)
    )

    plt.plot(
        j[selected],
        lambdas[selected],
        marker="o",
        markersize=4,
    )

    plt.axhline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    plt.xlabel(r"$j$")
    plt.ylabel(r"$\lambda_j$")

    plt.title(
        rf"Lyapunov spectrum "
        rf"($j={index_start},\ldots,{actual_end}$)"
    )

    plt.grid()

    plt.tight_layout()
    plt.show()
