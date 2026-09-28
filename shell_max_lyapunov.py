# 最大リアプノフ指数のみ（添付コードの第1摂動列を追跡）
# このファイル全体を Colab の定義セルで実行してください。
# 実行セルの数値条件は元のものをそのまま使用してください。
# 基準軌道 dt/2、変分 dt、積分因子 RK4、二段階の過渡を維持。
# lambda_1 は第1列の有限時間推定値（元の全列の np.max とは異なる）。
import numpy as np
from numba import njit

q = np.float64(2.0)
k0 = np.float64(2.0 ** (-4))
beta = np.float64(0.5)
f = np.complex128(5.0e-3 * (1.0 + 1.0j))


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
        n_u[i] = (
            np.sqrt(2.0 * n_k[i] * initial_energy)
            * np.exp(1j * phase)
        )

    return n_u


@njit
def make_initial_tangent_vector_numba(N, basis_seed=12345, random_basis=True):
    # 元の初期基底の第1列を完全に再現する。
    n_E = np.zeros((N, 1), dtype=np.complex128)
    if random_basis:
        np.random.seed(basis_seed)
        dim = 2 * N
        # 乱数の配列形状と QR を変えると初期方向が変わるため、ここだけ維持。
        A = np.random.normal(0.0, 1.0, (dim, dim))
        Q, R = np.linalg.qr(A)
        for i in range(N):
            n_E[i, 0] = Q[2 * i, 0] + 1j * Q[2 * i + 1, 0]
    else:
        n_E[0, 0] = 1.0 + 0.0j
    return n_E


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

    # RK4 第1段
    nonlinear_into_numba(
        n_u,
        n_c1,
        n_c2,
        n_c3,
        f,
        work_u[0],
    )

    for i in range(N_local):
        work_u[4, i] = (
            n_u[i] + 0.5 * dt * work_u[0, i]
        ) * n_E_visc_half[i]

    # RK4 第2段
    nonlinear_into_numba(
        work_u[4],
        n_c1,
        n_c2,
        n_c3,
        f,
        work_u[1],
    )

    for i in range(N_local):
        work_u[4, i] = (
            n_u[i] * n_E_visc_half[i]
            + 0.5 * dt * work_u[1, i]
        )

    # RK4 第3段
    nonlinear_into_numba(
        work_u[4],
        n_c1,
        n_c2,
        n_c3,
        f,
        work_u[2],
    )

    for i in range(N_local):
        work_u[4, i] = (
            n_E_visc[i] * n_u[i]
            + dt * n_E_visc_half[i] * work_u[2, i]
        )

    # RK4 第4段
    nonlinear_into_numba(
        work_u[4],
        n_c1,
        n_c2,
        n_c3,
        f,
        work_u[3],
    )

    # 基準軌道を更新
    for i in range(N_local):
        n_u[i] = n_u[i] * n_E_visc[i] + dt / 6.0 * (
            work_u[0, i] * n_E_visc[i]
            + 2.0 * work_u[1, i] * n_E_visc_half[i]
            + 2.0 * work_u[2, i] * n_E_visc_half[i]
            + work_u[3, i]
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

    # RK4 第1段: 時刻 t, u(t), E(t)
    variational_nonlinear_into_numba(
        n_u_start,
        n_E,
        n_c1,
        n_c2,
        n_c3,
        work_E[0],
    )

    # RK4 第2段用の E: 時刻 t + dt/2
    for i in range(N_local):
        for j in range(dim):
            work_E[4, i, j] = (
                n_E[i, j]
                + 0.5 * dt * work_E[0, i, j]
            ) * n_E_visc_half[i]

    # RK4 第2段: u(t + dt/2), E_2
    variational_nonlinear_into_numba(
        n_u_half,
        work_E[4],
        n_c1,
        n_c2,
        n_c3,
        work_E[1],
    )

    # RK4 第3段用の E: 時刻 t + dt/2
    for i in range(N_local):
        for j in range(dim):
            work_E[4, i, j] = (
                n_E[i, j] * n_E_visc_half[i]
                + 0.5 * dt * work_E[1, i, j]
            )

    # RK4 第3段: u(t + dt/2), E_3
    variational_nonlinear_into_numba(
        n_u_half,
        work_E[4],
        n_c1,
        n_c2,
        n_c3,
        work_E[2],
    )

    # RK4 第4段用の E: 時刻 t + dt
    for i in range(N_local):
        for j in range(dim):
            work_E[4, i, j] = (
                n_E[i, j] * n_E_visc[i]
                + dt * n_E_visc_half[i] * work_E[2, i, j]
            )

    # RK4 第4段: u(t + dt), E_4
    variational_nonlinear_into_numba(
        n_u_end,
        work_E[4],
        n_c1,
        n_c2,
        n_c3,
        work_E[3],
    )

    # 第一変分方程式を更新: E(t) -> E(t + dt)
    for i in range(N_local):
        for j in range(dim):
            n_E[i, j] = (
                n_E[i, j] * n_E_visc[i]
                + dt / 6.0
                * (
                    work_E[0, i, j] * n_E_visc[i]
                    + 2.0 * work_E[1, i, j] * n_E_visc_half[i]
                    + 2.0 * work_E[2, i, j] * n_E_visc_half[i]
                    + work_E[3, i, j]
                )
            )


@njit
def real_inner_product_numba(n_v, n_w):
    value = 0.0
    N_local = n_v.size

    for i in range(N_local):
        value += np.real(np.conj(n_v[i]) * n_w[i])

    return value


@njit
def normalize_tangent_into_numba(n_A, n_Q, n_r_diag, n_w):
    # 元の Modified Gram–Schmidt の j=0 と同じ演算順序。
    N_local = n_A.shape[0]
    for k in range(N_local):
        n_w[k] = n_A[k, 0]
    norm_sq = real_inner_product_numba(n_w, n_w)
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("正規化でゼロまたは非有限のノルムが発生しました。")
    r_jj = np.sqrt(norm_sq)
    for k in range(N_local):
        n_Q[k, 0] = n_w[k] / r_jj
    n_r_diag[0] = r_jj


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

    work_u = np.empty(
        (5, n_u.size),
        dtype=np.complex128,
    )

    transient_dt = 0.5 * dt
    transient_steps = int(round(transient_time / transient_dt))

    for _ in range(transient_steps):
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


@njit
def calculate_max_lyapunov_numba(
    n_u,
    dt,
    basis_transient_time,
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
    basis_seed,
    random_basis,
):
    n_u = n_u.copy()

    N_local = n_u.size
    dim = 1

    steps_per_tau = int(round(tau / dt))
    num_tau = int(round(t_max / tau))
    save_every_tau = int(round(save_interval / tau))
    num_save = int(round(t_max / save_interval))

    num_basis_transient_tau = int(round(basis_transient_time / tau))

    # --------------------------------------------------------
    # 初期摂動基底
    # --------------------------------------------------------

    n_E = make_initial_tangent_vector_numba(
        N_local,
        basis_seed,
        random_basis,
    )

    # --------------------------------------------------------
    # 正規化の作業配列
    # --------------------------------------------------------

    n_Q = np.empty((N_local, dim),dtype=np.complex128)

    n_r_diag = np.empty(dim,dtype=np.float64)

    n_w = np.empty(N_local,dtype=np.complex128)

    # --------------------------------------------------------
    # RK4 の作業配列
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

    n_sum_log = np.zeros(dim,dtype=np.float64)

    n_times = np.zeros(num_save,dtype=np.float64)

    n_lambda_history = np.zeros((num_save, dim),dtype=np.float64)

    save_index = 0

    # ============================================================
    # 摂動基底の過渡
    #
    # この期間は
    # ・基準軌道
    # ・第一変分方程式
    # ・正規化
    # を進める。
    #
    # ただし Lyapunov 指数 は累積しない。
    # ============================================================

    for _ in range(num_basis_transient_tau):

        # tau だけ時間発展
        for _ in range(steps_per_tau):

            # ----------------------------------------------------
            # 1. u(t) を保存
            # ----------------------------------------------------

            for i in range(N_local):
                n_u_start[i] = n_u[i]

            # ----------------------------------------------------
            # 2. u(t) -> u(t + dt/2)
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # 3. u(t + dt/2) を保存
            # ----------------------------------------------------

            for i in range(N_local):
                n_u_half[i] = n_u[i]

            # ----------------------------------------------------
            # 4. u(t + dt/2) -> u(t + dt)
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # 5. 第一変分方程式を dt 進める
            #
            # 使用する基準軌道:
            #
            # n_u_start = u(t)
            # n_u_half  = u(t + dt/2)
            # n_u       = u(t + dt)
            # ----------------------------------------------------

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
        # tau ごとに 正規化
        # --------------------------------------------------------

        normalize_tangent_into_numba(
            n_E,
            n_Q,
            n_r_diag,
            n_w,
        )

        # 正規化後の基底を次へ
        n_E, n_Q = n_Q, n_E

    # ============================================================
    # 本測定
    #
    # ここから
    # ・Lyapunov指数
    # を累積する。
    # ============================================================

    for m in range(num_tau):

        # tau だけ時間発展
        for _ in range(steps_per_tau):

            # ----------------------------------------------------
            # 1. 現在の基準軌道を保存
            #
            # n_u_start = u(t)
            # ----------------------------------------------------

            for i in range(N_local):
                n_u_start[i] = n_u[i]

            # ----------------------------------------------------
            # 2. 基準軌道を dt/2 進める
            #
            # u(t) -> u(t + dt/2)
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # 3. 中間時刻の基準軌道を保存
            #
            # n_u_half = u(t + dt/2)
            # ----------------------------------------------------

            for i in range(N_local):
                n_u_half[i] = n_u[i]

            # ----------------------------------------------------
            # 4. 基準軌道をさらに dt/2 進める
            #
            # u(t + dt/2) -> u(t + dt)
            #
            # この時点で
            # n_u = u(t + dt)
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # 5. 第一変分方程式を dt 進める
            #
            # 使用する基準軌道:
            #
            # n_u_start = u(t)
            # n_u_half  = u(t + dt/2)
            # n_u       = u(t + dt)
            # ----------------------------------------------------

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
        # tau ごとに 正規化
        # --------------------------------------------------------

        normalize_tangent_into_numba(
            n_E,
            n_Q,
            n_r_diag,
            n_w,
        )

        # Q を次の摂動基底として使用する
        # 古い E は次回の Q の書き込み先として再利用する
        n_E, n_Q = n_Q, n_E

        # --------------------------------------------------------
        # Lyapunov指数のために log(r_jj) を累積
        # --------------------------------------------------------

        for j in range(dim):
            n_sum_log[j] += np.log(n_r_diag[j])

        # 本測定開始からの時間
        t = (m + 1) * tau

        # --------------------------------------------------------
        # 指定間隔で Lyapunov 指数の履歴を保存
        # --------------------------------------------------------

        if ((m + 1) % save_every_tau) == 0:

            n_times[save_index] = t

            for j in range(dim):
                n_lambda_history[save_index, j] = (
                    n_sum_log[j] / t
                )

            save_index += 1

    # ============================================================
    # 本測定終了後
    # ============================================================

    return n_u, n_times, n_lambda_history


def run_max_lyapunov(
    N,
    nu,
    dt,
    orbit_transient_time,
    basis_transient_time,
    t_max,
    tau,
    save_interval,
    seed=42,
    basis_seed=12345,
    basis_type="standard",
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

    if (
        not isinstance(basis_seed, (int, np.integer))
        or not 0 <= basis_seed <= 4294967295
    ):
        raise ValueError("basis_seedは0～4294967295の整数にしてください。")

    if basis_type not in ("standard", "random"):
        raise ValueError('basis_typeは"standard"または"random"にしてください。')

    if not np.all(
        np.isfinite(
            np.asarray(
                [
                    nu,
                    dt,
                    orbit_transient_time,
                    basis_transient_time,
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
    orbit_transient_time = np.float64(orbit_transient_time)
    basis_transient_time = np.float64(basis_transient_time)
    t_max = np.float64(t_max)
    tau = np.float64(tau)
    save_interval = np.float64(save_interval)

    if N < 4:
        raise ValueError("Nは4以上の整数にしてください。")

    if nu <= 0.0:
        raise ValueError("nuは正の値にしてください。")

    if dt <= 0.0:
        raise ValueError("dtは正の値にしてください。")

    if orbit_transient_time < 0.0:
        raise ValueError("orbit_transient_timeは0以上にしてください。")

    if basis_transient_time < 0.0:
        raise ValueError("basis_transient_timeは0以上にしてください。")

    if t_max <= 0.0:
        raise ValueError("t_maxは正の値にしてください。")

    if tau <= 0.0:
        raise ValueError("tauは正の値にしてください。")

    if tau < dt:
        raise ValueError("tauはdt以上にしてください。")

    if save_interval < tau:
        raise ValueError("save_intervalはtau以上にしてください。")

    # --------------------------------------------------------
    # tau が dt の整数倍か確認
    # --------------------------------------------------------

    steps_per_tau = int(round(tau / dt))
    actual_tau = steps_per_tau * dt

    if not np.isclose(
        actual_tau,
        tau,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("tauはdtの整数倍にしてください。")

    # --------------------------------------------------------
    # basis_transient_time が tau の整数倍か確認
    # --------------------------------------------------------

    num_basis_transient_tau = int(round(basis_transient_time / tau))

    actual_basis_transient_time = (num_basis_transient_tau * tau)

    if not np.isclose(
        actual_basis_transient_time,
        basis_transient_time,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("basis_transient_timeは""tauの整数倍にしてください。")

    # --------------------------------------------------------
    # t_max が tau の整数倍か確認
    # --------------------------------------------------------

    num_tau = int(round(t_max / tau))
    actual_t_max = num_tau * tau

    if not np.isclose(
        actual_t_max,
        t_max,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("t_maxはtauの整数倍にしてください。")

    # --------------------------------------------------------
    # save_interval が tau の整数倍か確認
    # --------------------------------------------------------

    save_every_tau = int(round(save_interval / tau))
    actual_save_interval = save_every_tau * tau

    if not np.isclose(
        actual_save_interval,
        save_interval,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("save_intervalはtauの整数倍にしてください。")

    # --------------------------------------------------------
    # t_max が save_interval の整数倍か確認
    # --------------------------------------------------------

    num_save = int(round(t_max / save_interval))
    actual_t_max_from_save = num_save * save_interval

    if not np.isclose(
        actual_t_max_from_save,
        t_max,
        rtol=1.0e-12,
        atol=1.0e-14,
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

    n_u = make_initial_condition_numba(
        n_k,
        n_k_sq,
        seed,
    )

    # 積分因子
    n_E_visc = np.exp(-nu * n_k_sq * dt)
    n_E_visc_half = np.exp(-nu * n_k_sq * dt * 0.5)
    n_E_visc_quarter = np.exp(-nu * n_k_sq * dt * 0.25)

    # --------------------------------------------------------
    # 計算条件の表示
    # --------------------------------------------------------

    orbit_dt = 0.5 * dt
    transient_steps_orbit = int(round(orbit_transient_time / orbit_dt))
    basis_transient_gs = int(round(basis_transient_time / tau))

    print("--- 計算条件 ---")
    print(f"シェル数 N: {N}")
    print(f"実次元 2N: {2 * N}")
    print(f"初期位相の乱数 seed: {seed}")
    print(f"初期摂動基底: {basis_type}")

    if basis_type == "random":
        print(f"初期摂動基底の乱数 basis_seed: {basis_seed}")
    print(f"動粘性係数 nu: {nu:.10e}")
    print(f"変分方程式の時間刻み dt: {dt}")
    print(f"基準軌道の時間刻み dt/2: {orbit_dt}")

    print()
    print("--- 過渡計算 ---")

    print(f"基準軌道の過渡時間 orbit_transient_time: "f"{orbit_transient_time}")
    print(f"基準軌道の過渡期間のステップ数: "f"{transient_steps_orbit:,}")
    print(f"摂動基底の過渡時間 basis_transient_time: "f"{basis_transient_time}")
    print(f"摂動基底の過渡中の "f"正規化回数: "f"{basis_transient_gs:,}")

    print()
    print("--- 本測定 ---")

    print(f"Lyapunov指数の測定時間 t_max: {t_max}")
    print(f"正規化間隔 tau: {tau}")
    print("Lyapunov指数の保存間隔 "f"save_interval: {save_interval}")
    print("1回の正規化までの変分方程式ステップ数: "f"{steps_per_tau:,}")
    print("1回の正規化までの基準軌道ステップ数: "f"{2 * steps_per_tau:,}")
    print(f"正規化回数: {num_tau:,}")
    print()

    # --------------------------------------------------------
    # 過渡状態
    # --------------------------------------------------------

    print("基準軌道の過渡を計算中...")

    n_u = calculate_transient_numba(
        n_u,
        dt,
        orbit_transient_time,
        n_E_visc_half,
        n_E_visc_quarter,
        n_c1,
        n_c2,
        n_c3,
        f,
    )

    print("基準軌道の過渡計算完了")
    print()

    # --------------------------------------------------------
    # 最大 Lyapunov 指数
    # --------------------------------------------------------

    print("摂動基底の過渡と 最大 Lyapunov 指数 の本測定を開始します...")

    (
        n_u_final,
        n_times,
        n_lambda_history,
    ) = calculate_max_lyapunov_numba(
        n_u,
        dt,
        basis_transient_time,
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
        basis_seed,
        basis_type == "random",
    )

    # --------------------------------------------------------
    # 結果の計算
    # --------------------------------------------------------

    lambda_1 = float(n_lambda_history[-1, 0])
    print()
    print("計算完了！")
    print(f"lambda_1 = {lambda_1:.10e}")
    return {
        "N": N, "nu": nu, "dt": dt,
        "orbit_transient_time": orbit_transient_time,
        "basis_transient_time": basis_transient_time,
        "t_max": t_max, "tau": tau, "save_interval": save_interval,
        "seed": seed, "basis_seed": basis_seed, "basis_type": basis_type,
        "lambda_1": lambda_1,
        "u_final": n_u_final.copy(),
        "t": n_times.copy(),
        "lambda_history": n_lambda_history[:, 0].copy(),
    }


# 実行例：既存の実行セルの値をそのまま渡してください。
# result_max = run_max_lyapunov(
#     N=N, nu=nu, dt=dt,
#     orbit_transient_time=orbit_transient_time,
#     basis_transient_time=basis_transient_time,
#     t_max=t_max, tau=tau, save_interval=save_interval,
#     seed=seed, basis_seed=basis_seed, basis_type=basis_type,
# )
