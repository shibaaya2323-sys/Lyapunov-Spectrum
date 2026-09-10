
# ============================================================
# セル1
# シェルモデルの Lyapunov spectrum / Kaplan–Yorke 次元
# 作業配列を再利用する高速化版
#
# このコード全体を1つの関数定義セルで実行してください。
# 計算・描画は、これまで通り別のセルで行います。
#
# ・積分因子入りRK4
# ・実2N次元のModified Gram–Schmidt
# ・全2N本のリアプノフ指数
# ・seedの既定値は42
# ・fastmath、並列化は使用しない
# ============================================================

import numpy as np
from numba import njit

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
# 3. 基準軌道の非線形項
#
# 新しい配列を作らず、受け取ったn_nlへ書き込む。
# すべての要素を上書きしてから、第4シェルに外力を加える。
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

# ============================================================
# 4. 第一変分方程式の非線形項
#
# 新しい配列を作らず、受け取ったn_dEへ書き込む。
# 各シェルで摂動列jを順に処理して、連続するメモリを読む。
# 各要素の計算式と加算順序は元のコードに合わせる。
# ============================================================

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
# 5. 初期条件
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

# ============================================================
# 6. 初期摂動基底
# ============================================================

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
# 7. 基準軌道だけのRK4
#
# work_u[0]～work_u[3]：RK4各段の非線形項
# work_u[4]           ：各段の途中状態
#
# n_uをその場で更新する。戻り値はない。
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

# ============================================================
# 8. 基準軌道と第一変分方程式の同時RK4
#
# work_E[0]～work_E[3]：RK4各段の摂動の非線形項
# work_E[4]           ：各段の途中の摂動基底
#
# n_uとn_Eをその場で更新する。戻り値はない。
# ============================================================

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

# ============================================================
# 9. 実2N次元の内積
# ============================================================

@njit
def real_inner_product_numba(n_v, n_w):

    value = 0.0
    N_local = n_v.size

    for i in range(N_local):
        value += np.real(np.conj(n_v[i]) * n_w[i])

    return value

# ============================================================
# 10. Modified Gram–Schmidt
#
# Q、r、wを受け取り、再利用する。
# n_Aとn_Qには必ず別の配列を渡す。
# ============================================================

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

# ============================================================
# 11. 過渡状態の時間積分
#
# 作業配列をループの外で一度だけ確保する。
# ============================================================

@njit
def calculate_transient_numba(
    n_u,
    dt,
    transient_time,
    n_E_visc,
    n_E_visc_half,
    n_c1,
    n_c2,
    n_c3,
    f,
):

    n_u = n_u.copy()

    work_u = np.empty((5, n_u.size),dtype=np.complex128,)

    transient_steps = int(round(transient_time / dt))

    for step in range(transient_steps):

        rk4_step_u_inplace_numba(
            n_u,
            dt,
            n_E_visc,
            n_E_visc_half,
            n_c1,
            n_c2,
            n_c3,
            f,
            work_u,
        )

    return n_u

# ============================================================
# 12. エネルギー散逸率
#
# epsilon = nu sum_n k_n^2 |u_n|^2
# ============================================================

@njit
def calculate_energy_dissipation_numba(n_u,n_k_sq,nu):

    epsilon = 0.0

    for i in range(n_u.size):
        epsilon += (nu * n_k_sq[i] * (n_u[i].real**2 + n_u[i].imag**2))

    return epsilon

# ============================================================
# 13. Lyapunov spectrum の計算
#
# 時間積分・正規直交化の作業配列を再利用する。
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
    # 累積・保存用配列
    # --------------------------------------------------------

    n_sum_log = np.zeros(dim, dtype=np.float64)
    n_times = np.zeros(num_save, dtype=np.float64)

    n_lambda_history = np.zeros((num_save, dim),dtype=np.float64)

    epsilon_sum = 0.0
    epsilon_count = 0
    save_index = 0

    # --------------------------------------------------------
    # 測定
    # --------------------------------------------------------

    for m in range(num_tau):

        # tauだけ時間発展
        for step in range(steps_per_tau):

            rk4_step_u_E_inplace_numba(
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
            )

            epsilon = calculate_energy_dissipation_numba(n_u, n_k_sq, nu)

            epsilon_sum += epsilon
            epsilon_count += 1

        # 正規直交化
        gram_schmidt_real_into_numba(n_E,n_Q,n_r_diag,n_w)

        # Qを次の摂動基底として使用。
        # 古いEは、次回のQの書き込み先として再利用。
        n_E, n_Q = n_Q, n_E

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
    )

# ============================================================
# 14. Kaplan–Yorke次元
# ============================================================

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

# ============================================================
# 15. 正の指数の和 H
# ============================================================

def calculate_kolmogorov_entropy(lambdas):

    H = np.sum(lambdas[lambdas > 0.0])

    return H

# ============================================================
# 16. 散逸波数
#
# k_d = <epsilon>^(1/4) nu^(-3/4)
# ============================================================

def calculate_dissipation_wavenumber(epsilon_mean, nu):

    k_d = epsilon_mean**0.25 * nu**(-0.75)

    return k_d

# ============================================================
# 17. 実行用関数
#
# 既存の計算・描画セルに対応。
# seedを指定しなければ42を使用。
# ============================================================

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

    # --------------------------------------------------------
    # 計算条件の表示
    # --------------------------------------------------------

    print("--- 計算条件 ---")
    print(f"シェル数 N: {N}")
    print(f"実次元 2N: {2 * N}")
    print(f"初期位相の乱数 seed: {seed}")
    print(f"動粘性係数 nu: {nu:.10e}")
    print(f"時間刻み dt: {dt}")
    print(f"過渡時間 transient_time: {transient_time}")
    print(f"Lyapunov指数の測定時間 t_max: {t_max}")
    print(f"Gram-Schmidt間隔 tau: {tau}")
    print(f"Lyapunov指数の保存間隔 save_interval: {save_interval}")
    print(f"1回のGram-Schmidtまでのステップ数: {steps_per_tau:,}")
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
        n_E_visc,
        n_E_visc_half,
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
    ) = calculate_lyapunov_numba(
        n_u,
        dt,
        t_max,
        tau,
        save_interval,
        n_E_visc,
        n_E_visc_half,
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
    print("--- Fig.4 用の値 ---")
    print(f"平均エネルギー散逸率 <epsilon> = {epsilon_mean:.10e}")
    print(f"散逸波数 k_d = {k_d:.10e}")

    print()
    print("--- 発散との比較 ---")
    print(f"sum(lambda_i) = {lambda_sum:.10e}")
    print(f"div F = {divergence:.10e}")
    print(f"relative error = {relative_error:.10e}")
    print(f"relative error (%) = {100.0 * relative_error:.6f} %")

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

