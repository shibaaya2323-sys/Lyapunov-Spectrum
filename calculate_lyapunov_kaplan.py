
# ============================================================
# セル2
# Lyapunov spectrum / Kaplan-Yorke dimension
# 数値計算
# Fig.3～Fig.6 共通
# ============================================================


# ============================================================
# 1. パラメータ
# ============================================================

N = 12

nu = np.float64(1.0e-3)

# 時間刻み
dt = np.float64(0.001)


# Lyapunov指数を測定する前の過渡時間
transient_time = np.float64(1000.0)


# Lyapunov指数の測定時間
t_max = np.float64(4000.0)


# Gram-Schmidt正規直交化を行う時間間隔
tau = np.float64(0.001)


# Lyapunov指数の時間履歴を保存する間隔
save_interval = np.float64(1.0)


# ============================================================
# 2. Lyapunov spectrum の計算
# ============================================================

result = run_lyapunov(

    N=N,

    nu=nu,

    dt=dt,

    transient_time=transient_time,

    t_max=t_max,

    tau=tau,

    save_interval=save_interval,

)


# ============================================================
# 3. 結果の取り出し
# ============================================================

times = result["t"]

lambdas = result["lambdas"]

lambda_history = result["lambda_history"]

D_KY = result["D_KY"]

lambda_sum = result["lambda_sum"]

divergence = result["divergence"]


# ============================================================
# 4. Fig.3 で使用する量
# ============================================================

# 最大 Lyapunov 指数
lambda_1 = lambdas[0]

# Kolmogorov entropy
# H = sum_{lambda_i > 0} lambda_i
H = calculate_kolmogorov_entropy(lambdas)


# ============================================================
# 5. 結果表示
# ============================================================

print()

print("============================================================")

print("Fig.3～Fig.6 共通結果")

print("============================================================")

print(f"N        = "f"{N}")

print(f"nu       = "f"{nu:.10e}")

print()

print(f"lambda_1 = "f"{lambda_1:.10e}")

print(f"H        = "f"{H:.10e}")

print(f"D_KY     = "f"{D_KY:.10f}")

print()

print(f"sum(lambda_i) = "f"{lambda_sum:.10e}")

print(f"div F         = "f"{divergence:.10e}")

print(f"relative error = "f"{abs(lambda_sum - divergence) / abs(divergence):.10e}")

print(f"relative error (%) = "f"{100.0 * abs(lambda_sum - divergence) / abs(divergence):.6f} %")
