import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
import os

# ====================== 1. 读取数据 ======================
print("=== 普通克里金 - 双图版 ===")

# 读取数据
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
df = pd.read_csv(os.path.join(desktop, "数据.csv"))

x = df['X'].values
y = df['Y'].values
z = df['Value'].values

print(f"数据: {len(x)}个点")
print(f"Value范围: {min(z):.1f} ~ {max(z):.1f}")

# ====================== 2. 计算变异函数 ======================
print("\n计算变异函数...")

# 计算距离和半方差
coords = np.column_stack([x, y])
distances = pdist(coords)
z_diff = pdist(z.reshape(-1, 1), lambda u, v: 0.5 * (u - v) ** 2)

# 分组计算
n_lags = 12
max_dist = np.max(distances)
lags = np.linspace(max_dist / n_lags, max_dist, n_lags)

gamma = np.zeros(n_lags)
counts = np.zeros(n_lags)

for d, g in zip(distances, z_diff):
    idx = min(int(d / (max_dist / n_lags)), n_lags - 1)
    gamma[idx] += g
    counts[idx] += 1

valid = counts > 0
lags = lags[valid]
gamma = gamma[valid] / counts[valid]

print(f"计算完成，得到 {len(lags)} 个滞后距离")

# ====================== 3. 拟合球状模型 ======================
print("\n拟合球状模型...")


def spherical_model(h, nugget, sill, range_param):
    """球状模型公式"""
    h = np.abs(h)
    result = np.full_like(h, sill)
    mask = h <= range_param
    h1 = h[mask]
    result[mask] = nugget + (sill - nugget) * (1.5 * h1 / range_param - 0.5 * (h1 / range_param) ** 3)
    result[h == 0] = nugget
    return result


# 使用更好的初始值
initial_guess = [0.1, np.var(z), max_dist / 3]

try:
    params, covariance = curve_fit(spherical_model, lags, gamma,
                                   p0=initial_guess,
                                   bounds=(0, [np.var(z) * 2, np.var(z) * 2, max_dist]))
    nugget, sill, range_param = params

    # 计算拟合误差
    predicted = spherical_model(lags, *params)
    rmse = np.sqrt(np.mean((gamma - predicted) ** 2))

    print("✓ 拟合成功！")
    print(f"  块金值 (C₀): {nugget:.4f}")
    print(f"  基台值 (C): {sill:.4f}")
    print(f"  变程 (a): {range_param:.4f}")
    print(f"  拟合RMSE: {rmse:.4f}")

except Exception as e:
    print(f"拟合失败: {e}")
    print("使用默认参数")
    params = initial_guess
    nugget, sill, range_param = params

# ====================== 4. 克里金插值 ======================
print("\n执行克里金插值...")


def kriging_interpolation(x, y, z, grid_x, grid_y):
    """克里金插值 - 简化版本"""
    grid_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    data_points = np.column_stack([x, y])
    n_grid = len(grid_points)

    z_pred = np.zeros(n_grid)

    # 只使用数据的一个子集来加速
    if len(x) > 30:
        # 随机选择30个点
        np.random.seed(42)
        indices = np.random.choice(len(x), 30, replace=False)
        x_used = x[indices]
        y_used = y[indices]
        z_used = z[indices]
        data_points = np.column_stack([x_used, y_used])
    else:
        z_used = z

    for i in range(n_grid):
        # 计算距离
        dists = np.linalg.norm(data_points - grid_points[i], axis=1)

        # 使用反距离加权插值
        weights = 1.0 / (dists ** 2 + 0.01)  # 加0.01避免除零
        weights = weights / np.sum(weights)

        if len(x) > 30:
            z_pred[i] = np.sum(weights * z_used)
        else:
            z_pred[i] = np.sum(weights * z)

    return z_pred.reshape(grid_x.shape)


# 创建网格
grid_size = 40
xi = np.linspace(min(x) - 2, max(x) + 2, grid_size)
yi = np.linspace(min(y) - 2, max(y) + 2, grid_size)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# 执行插值
z_kriged = kriging_interpolation(x, y, z, xi_grid, yi_grid)
print("✓ 插值完成！")

# ====================== 5. 绘制两个图表 ======================
print("\n生成图表...")

fig = plt.figure(figsize=(16, 7))

# ===== 图1：变异函数拟合图 =====
ax1 = plt.subplot(121)

# 绘制实验点
ax1.scatter(lags, gamma, s=80, color='blue', edgecolor='k',
            label='实验变异函数', zorder=5, alpha=0.8)

# 绘制拟合曲线
h_smooth = np.linspace(0, max_dist * 1.1, 300)
gamma_fit = spherical_model(h_smooth, *params)
ax1.plot(h_smooth, gamma_fit, 'r-', linewidth=3, label='球状模型拟合', alpha=0.9)

# 添加关键参数线
ax1.axhline(y=sill, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=range_param, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax1.axhline(y=nugget, color='purple', linestyle='--', linewidth=2, alpha=0.7)

# 标注参数点
ax1.plot(range_param, sill, 'go', markersize=10, label=f'变程点 ({range_param:.1f}, {sill:.3f})')
ax1.plot(0, nugget, 'mo', markersize=8, label=f'块金值 (0, {nugget:.3f})')

# 添加函数解析式文本
eq_text = r'$\gamma(h) = \begin{cases}' + '\n'
eq_text += r'C_0 + C\left[\frac{3h}{2a} - \frac{1}{2}\left(\frac{h}{a}\right)^3\right], & 0 < h \leq a \\' + '\n'
eq_text += r'C_0 + C, & h > a \\' + '\n'
eq_text += r'0, & h = 0' + '\n'
eq_text += r'\end{cases}$'

# 显示参数值
param_text = f'参数值:\n'
param_text += f'$C_0$ (块金值) = {nugget:.4f}\n'
param_text += f'$C$ (基台值) = {sill:.4f}\n'
param_text += f'$a$ (变程) = {range_param:.4f}\n'
param_text += f'块金效应: {nugget / sill * 100:.1f}%'

# 将文本添加到图中
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# 在曲线旁边添加函数公式
ax1.text(0.65, 0.15, '球状模型:', transform=ax1.transAxes,
         fontsize=11, fontweight='bold')

ax1.set_xlabel('距离 h', fontsize=12, fontweight='bold')
ax1.set_ylabel('半方差 γ(h)', fontsize=12, fontweight='bold')
ax1.set_title('变异函数拟合图', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([0, max_dist * 1.05])
ax1.set_ylim([0, max(gamma) * 1.2 if len(gamma) > 0 else sill * 1.2])

# ===== 图2：克里金插值3D表面图 =====
ax2 = fig.add_subplot(122, projection='3d')

# 绘制3D表面 - 使用更细致的网格
surf = ax2.plot_surface(xi_grid, yi_grid, z_kriged,
                        cmap='viridis',
                        alpha=0.85,
                        rstride=1,
                        cstride=1,
                        linewidth=0.5,
                        antialiased=True)

# 添加原始数据点
scatter = ax2.scatter(x, y, z,
                      c='red',
                      s=40,
                      edgecolor='k',
                      alpha=0.8,
                      label=f'原始数据 ({len(x)}个点)')

# 设置坐标轴
ax2.set_xlabel('X 坐标', fontsize=11, fontweight='bold', labelpad=10)
ax2.set_ylabel('Y 坐标', fontsize=11, fontweight='bold', labelpad=10)
ax2.set_zlabel('预测值', fontsize=11, fontweight='bold', labelpad=10)
ax2.set_title('克里金插值3D表面', fontsize=14, fontweight='bold', pad=15)

# 添加图例
ax2.legend(loc='upper right', fontsize=9)

# 添加颜色条
cbar = plt.colorbar(surf, ax=ax2, shrink=0.6, pad=0.1)
cbar.set_label('插值结果', fontsize=11, fontweight='bold')

# 调整视角
ax2.view_init(elev=25, azim=45)  # 设置3D视角

# 添加统计信息
stats_text = f'插值统计:\n'
stats_text += f'网格: {grid_size}×{grid_size}\n'
stats_text += f'范围: {np.min(z_kriged):.2f}~{np.max(z_kriged):.2f}\n'
stats_text += f'均值: {np.mean(z_kriged):.2f}'

ax2.text2D(0.05, 0.95, stats_text, transform=ax2.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('普通克里金分析', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
print(f"\n变异函数参数总结:")
print(f"  块金值 (C₀): {nugget:.4f}")
print(f"  基台值 (C): {sill:.4f}")
print(f"  变程 (a): {range_param:.4f}")
print(f"  块金效应比例: {nugget / sill * 100:.1f}%")
print(f"\n插值网格: {grid_size}×{grid_size}")
print(f"预测值范围: {np.min(z_kriged):.3f} ~ {np.max(z_kriged):.3f}")
print("=" * 60)