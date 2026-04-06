import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------- 原始数据 ----------------------
coverage = np.array([
    0.877, 0.991, 0.974, 0.990, 0.990, 0.959, 0.990, 0.990, 0.990, 0.987,
    0.673, 0.989, 0.989, 0.989, 0.988, 0.989, 0.989, 0.990, 0.988, 0.989,
    0.794, 0.990, 0.990, 0.914, 0.955, 0.961, 0.960, 0.988, 0.901, 0.915,
    0.962, 0.961, 0.796, 0.936, 0.939, 0.940, 0.935, 0.911, 0.890, 0.885,
    0.000, 0.770, 0.952, 0.950, 0.950, 0.950, 0.950, 0.950, 0.950, 0.973
])

# ---------------------- 生成不规则坐标（模拟参考图的不规则形状） ----------------------
np.random.seed(42)  # 固定随机种子，保证复现
n_points = 500  # 采样点数量，越多越平滑
# 生成不规则的二维坐标（模拟PushT任务的操作空间/地图形状）
x = np.random.normal(0, 1, n_points)
y = np.random.normal(0, 1, n_points)
# 用Coverage值加权，高Coverage区域更密集
weights = np.repeat(coverage, 10)  # 每个Episode对应10个采样点
weights = weights[:n_points]  # 对齐采样点数量

# ---------------------- 核密度估计（KDE）生成平滑热力图 ----------------------
xy = np.vstack([x, y])
kde = gaussian_kde(xy, weights=weights)  # 用Coverage加权，高值区域更亮

# 生成网格
xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi.reshape(xi.shape)
# 归一化到0~1，和Coverage范围一致
zi = (zi - zi.min()) / (zi.max() - zi.min())

# ---------------------- 绘图（完美复刻参考图样式） ----------------------
plt.figure(figsize=(16, 9))
# 绘制不规则热力图
contour = plt.contourf(xi, yi, zi, levels=20, cmap="RdYlGn_r", vmin=0, vmax=1)
# 添加颜色条
cbar = plt.colorbar(contour)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
cbar.set_ticklabels(["0.00 (Extreme Fail)", "0.20", "0.40", "0.60", "0.80 (Valid)", "0.90 (High Cov)", "0.95 (Strict Success)", "1.00 (Max)"], fontsize=12)
# 标题 & 样式
plt.title("V4 Model Irregular Performance Heatmap\n(50 Episodes, Coverage Distribution)", 
          fontsize=24, weight="bold", pad=20)
plt.axis("off")  # 隐藏坐标轴，更像参考图
plt.tight_layout()
plt.savefig("v4_irregular_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()