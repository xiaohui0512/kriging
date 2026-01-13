import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.interpolate import griddata
import os

# ====================== 1. 读取数据 ======================
print("=== 普通克里金分析 ===")

# 读取数据
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
df = pd.read_csv(os.path.join(desktop, "数据.csv"))

x = df['X'].values
y = df['Y'].values
z = df['Value'].values

print(f"📊 数据统计:")
print(f"  数据点数: {len(x)}")
print(f"  X范围: {min(x):.1f} ~ {max(x):.1f}")
print(f"  Y范围: {min(y):.1f} ~ {max(y):.1f}")
print(f"  Value范围: {min(z):.1f} ~ {max(z):.1f}")
print(f"  Value均值: {np.mean(z):.2f}")
print(f"  Value方差: {np.var(z):.2f}")

# ====================== 2. 变异函数分析 ======================
print("\n🔬 计算实验变异函数...")

# 计算所有点对的距离和半方差
coords = np.column_stack([x, y])
distances = pdist(coords)
z_pairs = pdist(z.reshape(-1, 1))
gamma_exp = 0.5 * z_pairs ** 2

# 分组统计
n_lags = 15
max_dist = np.max(distances)
lag_size = max_dist / n_lags
lags = np.linspace(lag_size, max_dist, n_lags)

gamma_mean = np.zeros(n_lags)
counts = np.zeros(n_lags)

for d, g in zip(distances, gamma_exp):
    idx = min(int(d / lag_size), n_lags - 1)
    gamma_mean[idx] += g
    counts[idx] += 1

valid = counts > 0
lags = lags[valid]
gamma_mean = gamma_mean[valid] / counts[valid]

print(f"  最大距离: {max_dist:.1f}")
print(f"  滞后距离数: {len(lags)}")

# ====================== 3. 拟合变异函数模型 ======================
print("\n📐 拟合变异函数模型...")


# 球状模型函数
def spherical_model(h, nugget, sill, range_param):
    """球状模型 γ(h) = C₀ + C[1.5h/a - 0.5(h/a)³], h≤a; γ(h) = C₀ + C, h>a"""
    h = np.abs(h)
    result = np.full_like(h, sill)
    mask = h <= range_param
    h1 = h[mask]
    result[mask] = nugget + (sill - nugget) * (1.5 * h1 / range_param - 0.5 * (h1 / range_param) ** 3)
    result[h == 0] = nugget
    return result


# 指数模型函数
def exponential_model(h, nugget, sill, range_param):
    """指数模型 γ(h) = C₀ + C[1 - exp(-3h/a)]"""
    h = np.abs(h)
    return nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_param))


# 尝试拟合球状模型
try:
    # 初始参数猜测
    initial_guess = [0.1, np.var(z), max_dist / 3]

    # 拟合球状模型
    params, pcov = curve_fit(spherical_model, lags, gamma_mean,
                             p0=initial_guess,
                             bounds=(0, [np.var(z) * 2, np.var(z) * 2, max_dist * 1.5]))

    nugget, sill, range_param = params
    model_func = spherical_model
    model_name = "球状模型"

    # 计算拟合质量
    predicted = model_func(lags, *params)
    residuals = gamma_mean - predicted
    rmse = np.sqrt(np.mean(residuals ** 2))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((gamma_mean - np.mean(gamma_mean)) ** 2)

    print(f"  ✅ {model_name}拟合成功!")
    print(f"    块金值 C₀: {nugget:.4f}")
    print(f"    基台值 C: {sill:.4f}")
    print(f"    变程 a: {range_param:.4f}")
    print(f"    拟合RMSE: {rmse:.4f}")
    print(f"    拟合R²: {r2:.4f}")
    print(f"    块金效应: {nugget / sill * 100:.1f}%")

except Exception as e:
    print(f"  ⚠️ 球状模型拟合失败: {e}")
    print("  尝试指数模型...")

    try:
        params, pcov = curve_fit(exponential_model, lags, gamma_mean,
                                 p0=initial_guess)
        nugget, sill, range_param = params
        model_func = exponential_model
        model_name = "指数模型"
        print(f"  ✅ {model_name}拟合成功!")
    except:
        print("  ❌ 所有模型拟合失败，使用默认参数")
        params = initial_guess
        model_func = spherical_model
        model_name = "默认模型"

# ====================== 4. 克里金插值 ======================
print("\n🧩 执行克里金插值...")


def ordinary_kriging(x, y, z, grid_x, grid_y, max_neighbors=20):
    """普通克里金插值"""
    data_points = np.column_stack([x, y])
    grid_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])

    n_grid = len(grid_points)
    z_pred = np.zeros(n_grid)

    # 为了加快计算，对每个网格点只使用最近的max_neighbors个数据点
    for i in range(n_grid):
        # 计算到所有数据点的距离
        dists = np.linalg.norm(data_points - grid_points[i], axis=1)

        # 选择最近的max_neighbors个点
        idx = np.argsort(dists)[:max_neighbors]
        neighbor_points = data_points[idx]
        neighbor_z = z[idx]
        n_neighbors = len(idx)

        if n_neighbors < 3:
            # 如果邻居太少，使用反距离加权
            weights = 1.0 / (dists[idx] ** 2 + 0.01)
            weights = weights / np.sum(weights)
            z_pred[i] = np.sum(weights * neighbor_z)
            continue

        # 构建克里金矩阵
        K = np.ones((n_neighbors + 1, n_neighbors + 1))

        # 填充变异函数值
        for j in range(n_neighbors):
            for k in range(n_neighbors):
                d = np.linalg.norm(neighbor_points[j] - neighbor_points[k])
                K[j, k] = model_func(d, *params)

        K[-1, -1] = 0

        # 构建右侧向量
        k_vec = np.ones(n_neighbors + 1)
        for j in range(n_neighbors):
            d = dists[idx][j]
            k_vec[j] = model_func(d, *params)

        # 求解权重
        try:
            weights = np.linalg.solve(K, k_vec)
            z_pred[i] = np.sum(weights[:-1] * neighbor_z)
        except:
            # 如果求解失败，使用反距离加权
            weights = 1.0 / (dists[idx] ** 2 + 0.01)
            weights = weights / np.sum(weights)
            z_pred[i] = np.sum(weights * neighbor_z)

    return z_pred.reshape(grid_x.shape)


# 创建插值网格
grid_size = 50
xi = np.linspace(min(x) - 3, max(x) + 3, grid_size)
yi = np.linspace(min(y) - 3, max(y) + 3, grid_size)
xi_grid, yi_grid = np.meshgrid(xi, yi)

print(f"  创建{grid_size}×{grid_size}的插值网格...")
z_kriged = ordinary_kriging(x, y, z, xi_grid, yi_grid, max_neighbors=15)
print("  ✅ 克里金插值完成!")

# ====================== 5. 绘制两个核心图表 ======================
print("\n🎨 生成可视化图表...")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig = plt.figure(figsize=(18, 8))

# ===== 图表1：变异函数拟合图 =====
ax1 = plt.subplot(121)

# 1. 绘制实验变异函数点
scatter = ax1.scatter(lags, gamma_mean, s=80, color='blue',
                      edgecolor='black', linewidth=1.5,
                      label=f'实验变异函数 ({len(lags)}个滞后)',
                      zorder=10, alpha=0.8)

# 2. 绘制拟合曲线
h_fit = np.linspace(0, max_dist * 1.2, 300)
gamma_fit = model_func(h_fit, *params)
ax1.plot(h_fit, gamma_fit, 'r-', linewidth=3.5,
         label=f'{model_name}拟合曲线', alpha=0.9, zorder=5)

# 3. 添加关键参考线
ax1.axhline(y=sill, color='green', linestyle='--', linewidth=2,
            alpha=0.7, label=f'基台值 C = {sill:.3f}')
ax1.axvline(x=range_param, color='orange', linestyle='--', linewidth=2,
            alpha=0.7, label=f'变程 a = {range_param:.2f}')
ax1.axhline(y=nugget, color='purple', linestyle='--', linewidth=2,
            alpha=0.7, label=f'块金值 C₀ = {nugget:.3f}')

# 4. 标注关键点
ax1.plot(range_param, sill, 'go', markersize=12, markeredgecolor='black',
         linewidth=2, label='变程点', zorder=15)
ax1.plot(0, nugget, 'mo', markersize=10, markeredgecolor='black',
         linewidth=2, label='块金点', zorder=15)

# 5. 添加函数公式
if model_name == "球状模型":
    eq_text = r'$\gamma(h) = \begin{cases}' + '\n'
    eq_text += r'C_0 + C\left[\frac{3h}{2a} - \frac{1}{2}\left(\frac{h}{a}\right)^3\right], & 0 < h \leq a \\' + '\n'
    eq_text += r'C_0 + C, & h > a \\' + '\n'
    eq_text += r'0, & h = 0' + '\n'
    eq_text += r'\end{cases}$'
elif model_name == "指数模型":
    eq_text = r'$\gamma(h) = C_0 + C\left[1 - \exp\left(-\frac{3h}{a}\right)\right]$'
else:
    eq_text = r'$\gamma(h) = \text{默认模型}$'

# 在左上角显示公式
ax1.text(0.02, 0.98, eq_text, transform=ax1.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gold'))

# 6. 在右上角显示参数
param_text = f'{model_name}参数:\n'
param_text += f'$C_0$ = {nugget:.4f}\n'
param_text += f'$C$ = {sill:.4f}\n'
param_text += f'$a$ = {range_param:.4f}\n'
if 'r2' in locals():
    param_text += f'$R^2$ = {r2:.4f}\n'
    param_text += f'RMSE = {rmse:.4f}\n'
param_text += f'块金效应: {nugget / sill * 100:.1f}%'

ax1.text(0.98, 0.98, param_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

# 7. 设置图表属性
ax1.set_xlabel('距离 $h$', fontsize=13, fontweight='bold')
ax1.set_ylabel('半方差 $\gamma(h)$', fontsize=13, fontweight='bold')
ax1.set_title('变异函数拟合分析', fontsize=15, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([0, max_dist * 1.1])
ax1.set_ylim(bottom=0)

# ===== 图表2：克里金插值3D表面图 =====
ax2 = fig.add_subplot(122, projection='3d')

# 1. 绘制3D插值表面
# 使用细致的网格以获得平滑表面
surf = ax2.plot_surface(xi_grid, yi_grid, z_kriged,
                        cmap='viridis',
                        alpha=0.88,
                        rstride=2,
                        cstride=2,
                        linewidth=0.2,
                        antialiased=True,
                        shade=True)

# 2. 叠加原始数据点
scatter_3d = ax2.scatter(x, y, z,
                         c='red',
                         s=50,
                         edgecolor='black',
                         linewidth=1.0,
                         alpha=0.9,
                         depthshade=True,
                         label=f'原始数据点 (n={len(x)})')

# 3. 设置坐标轴标签
ax2.set_xlabel('X 坐标', fontsize=12, fontweight='bold', labelpad=12)
ax2.set_ylabel('Y 坐标', fontsize=12, fontweight='bold', labelpad=12)
ax2.set_zlabel('插值结果', fontsize=12, fontweight='bold', labelpad=12)
ax2.set_title('克里金插值3D表面', fontsize=15, fontweight='bold', pad=15)

# 4. 添加颜色条
cbar = plt.colorbar(surf, ax=ax2, shrink=0.6, pad=0.08)
cbar.set_label('数值大小', fontsize=11, fontweight='bold')

# 5. 添加图例
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

# 6. 设置3D视角
ax2.view_init(elev=28, azim=135)  # 更好的视角

# 7. 添加统计信息框
stats_text = f'插值统计信息:\n'
stats_text += f'网格分辨率: {grid_size}×{grid_size}\n'
stats_text += f'最小值: {np.min(z_kriged):.3f}\n'
stats_text += f'最大值: {np.max(z_kriged):.3f}\n'
stats_text += f'平均值: {np.mean(z_kriged):.3f}\n'
stats_text += f'标准差: {np.std(z_kriged):.3f}'

ax2.text2D(0.02, 0.98, stats_text, transform=ax2.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# 设置总标题
plt.suptitle('普通克里金插值分析结果', fontsize=17, fontweight='bold', y=0.98)

# 调整布局
plt.tight_layout()
plt.show()

# ====================== 6. 输出总结 ======================
print("\n" + "═" * 60)
print("分析完成！")
print("═" * 60)

print(f"\n📋 数据摘要:")
print(f"  数据点数: {len(x)}")
print(f"  空间范围: X[{min(x):.1f}, {max(x):.1f}], Y[{min(y):.1f}, {max(y):.1f}]")
print(f"  数值范围: [{min(z):.3f}, {max(z):.3f}]")

print(f"\n📐 变异函数参数 ({model_name}):")
print(f"  块金值 C₀: {nugget:.4f}")
print(f"  基台值 C: {sill:.4f}")
print(f"  变程 a: {range_param:.4f}")
print(f"  块金效应比例: {nugget / sill * 100:.1f}%")
if 'r2' in locals():
    print(f"  拟合优度 R²: {r2:.4f}")
    print(f"  拟合误差 RMSE: {rmse:.4f}")

print(f"\n🧭 克里金插值结果:")
print(f"  网格大小: {grid_size} × {grid_size}")
print(f"  预测范围: [{np.min(z_kriged):.3f}, {np.max(z_kriged):.3f}]")
print(f"  预测均值: {np.mean(z_kriged):.3f}")

# 计算交叉验证误差
z_pred_at_points = griddata((xi_grid.flatten(), yi_grid.flatten()),
                            z_kriged.flatten(), (x, y), method='linear')
cv_rmse = np.sqrt(np.mean((z - z_pred_at_points) ** 2))
print(f"  交叉验证RMSE: {cv_rmse:.4f}")

print("═" * 60)
print("✅ 所有分析完成！")