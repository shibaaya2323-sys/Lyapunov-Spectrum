
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5.5))


# ============================================================
# ν = 10^{-6} （十字）
# ============================================================

plt.semilogx(
    result_0["x_normalized"],
    result_0["flux_normalized"],
    linestyle="None",
    marker="+",
    color="black",
    markersize=8,
    markeredgewidth=1.3,
    label=r"$\nu=10^{-6}$",
)


# ============================================================
# ν = 10^{-7} （×）
# ============================================================

plt.semilogx(
    result_1["x_normalized"],
    result_1["flux_normalized"],
    linestyle="None",
    marker="x",
    color="red",
    markersize=6,
    markeredgewidth=1.3,
    label=r"$\nu=10^{-7}$",
)


# ============================================================
# ν = 10^{-8} （ひし形）
# ============================================================

plt.semilogx(
    result_2["x_normalized"],
    result_2["flux_normalized"],
    linestyle="None",
    marker="D",
    color="blue",
    markerfacecolor="none",
    markeredgewidth=1.3,
    markersize=5,
    label=r"$\nu=10^{-8}$",
)


# ============================================================
# ν = 10^{-9} （星）
# ============================================================

plt.semilogx(
    result_3["x_normalized"],
    result_3["flux_normalized"],
    linestyle="None",
    marker="*",
    color="green",
    markersize=8,
    markeredgewidth=1.3,
    label=r"$\nu=10^{-9}$",
)


# ============================================================
# 軸の設定
# ============================================================

# 横軸の範囲
plt.xlim(
    1.0e-8,
    1.0e2
)

# 縦軸の範囲
plt.ylim(
    -0.2,
    1.2
)

# 縦軸の主目盛
plt.yticks(
    np.arange(
        -0.2,
        1.21,
        0.2
    )
)

# 軸ラベル
plt.xlabel(
    r"$k/k_d$",
    fontsize=15,
)

plt.ylabel(
    r"$\Pi(k)/\varepsilon$",
    fontsize=15,
)


# ============================================================
# グラフの見た目
# ============================================================

plt.grid(False)

# 目盛をグラフの内側に向ける
plt.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True,
)

plt.tick_params(
    axis="both",
    which="major",
    length=7,
    width=1.2,
    labelsize=12,
)

plt.tick_params(
    axis="both",
    which="minor",
    length=4,
    width=1.0,
)

# 枠線を少し太くする
ax = plt.gca()

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 凡例
plt.legend(
    loc="upper right",
    frameon=False,
    fontsize=11,
)


# ============================================================
# 保存
# ============================================================

plt.tight_layout()

#plt.savefig(
#    "Fig1_energy_flux_compare.png",
#    dpi=600,
#    bbox_inches="tight"
#)

#plt.savefig(
#    "Fig1_energy_flux_compare.pdf",
#    bbox_inches="tight"
#)

plt.show()
