import os
import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("克里金插值核心计算")
print("=" * 60)

# 1. 获取桌面数据文件
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
file_path = os.path.join(desktop_path, '数据.csv')

if not os.path.exists(file_path):
    print(f"❌ 文件不存在: {file_path}")
    # 尝试其他可能的文件名
    desktop_files = os.listdir(desktop_path)
    csv_files = [f for f in desktop_files if f.lower().endswith('.csv')]
    if csv_files:
        print("找到的CSV文件:")
        for f in csv_files:
            print(f"  📄 {f}")
        file_path = os.path.join(desktop_path, csv_files[0])
        print(f"尝试使用: {file_path}")
    else:
        exit()

# 读取数据
for encoding in ['utf-8', 'gbk', 'gb2312', 'ansi']:
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"✅ 使用 {encoding} 编码读取成功")
        break
    except:
        continue

# 重命名列
if len(data.columns) >= 3:
    if 'X' not in data.columns or 'Y' not in data.columns or 'Value' not in data.columns:
        data.columns = ['X', 'Y', 'Value'] + list(data.columns[3:])
        print("📝 自动重命名列为: X, Y, Value")

x = data['X'].values
y = data['Y'].values
z = data['Value'].values

print(f"📊 数据点数: {len(x)}")
print(f"📍 坐标范围: X({x.min():.2f}~{x.max():.2f}), Y({y.min():.2f}~{y.max():.2f})")
print(f"📈 数值范围: {z.min():.2f}~{z.max():.2f}, 均值: {z.mean():.2f}")

# 2. 计算实验变异函数
print("\n" + "=" * 60)
print("1. 计算实验变异函数")
print("=" * 60)


def calculate_experimental_variogram(x, y, z, num_lags=15):
    """计算实验变异函数"""
    n = len(x)
    max_distance = np.sqrt((x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2) * 0.5

    distances = []
    variances = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist <= max_distance:
                var = 0.5 * (z[i] - z[j]) ** 2
                distances.append(dist)
                variances.append(var)

    lag_bins = np.linspace(0, max_distance, num_lags)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    gamma = np.zeros(len(lag_centers))
    counts = np.zeros(len(lag_centers))

    for d, v in zip(distances, variances):
        idx = np.digitize(d, lag_bins) - 1
        if 0 <= idx < len(lag_centers):
            gamma[idx] += v
            counts[idx] += 1

    valid = counts > 0
    return lag_centers[valid], gamma[valid] / counts[valid], counts[valid], max_distance


lag_centers, gamma, counts, max_lag = calculate_experimental_variogram(x, y, z)

print(f"📏 最大滞后距离: {max_lag:.2f}")
print(f"📊 有效滞后区间: {len(lag_centers)}")
print(f"📈 总点对数量: {np.sum(counts):.0f}")
print("\n实验变异函数值:")
for i, (lag, gam, cnt) in enumerate(zip(lag_centers, gamma, counts)):
    print(f"  距离 {lag:6.2f}: 半方差={gam:6.3f}, 点对={cnt:3.0f}")

# 3. 定义变异函数模型
print("\n" + "=" * 60)
print("2. 定义理论变异函数模型")
print("=" * 60)


def spherical(h, c0, c1, a):
    """球状模型: γ(h) = c0 + c1 * [1.5*(h/a) - 0.5*(h/a)^3] for h≤a, else c0+c1"""
    return np.where(h <= a, c0 + c1 * (1.5 * (h / a) - 0.5 * (h / a) ** 3), c0 + c1)


def exponential(h, c0, c1, a):
    """指数模型: γ(h) = c0 + c1 * [1 - exp(-3h/a)]"""
    return c0 + c1 * (1 - np.exp(-3 * h / a))


def gaussian(h, c0, c1, a):
    """高斯模型: γ(h) = c0 + c1 * [1 - exp(-3*(h/a)^2)]"""
    return c0 + c1 * (1 - np.exp(-3 * (h / a) ** 2))


def linear(h, c0, c1, a):
    """线性模型: γ(h) = c0 + c1 * min(h/a, 1)"""
    return c0 + c1 * np.minimum(h / a, 1)


models = {
    'spherical': spherical,
    'exponential': exponential,
    'gaussian': gaussian,
    'linear': linear
}

# 4. 拟合模型
print("\n" + "=" * 60)
print("3. 拟合变异函数模型")
print("=" * 60)

best_model = None
best_params = None
best_residual = float('inf')
all_fits = {}

for name, func in models.items():
    try:
        # 初始参数猜测
        c0_guess = max(0.1, gamma[0] if len(gamma) > 0 else 0.1)
        c1_guess = max(0.1, np.max(gamma) - c0_guess)
        a_guess = max_lag * 0.4

        params, _ = curve_fit(func, lag_centers, gamma,
                              p0=[c0_guess, c1_guess, a_guess],
                              bounds=([0, 0, 0], [np.inf, np.inf, max_lag * 2]),
                              maxfev=5000)

        predicted = func(lag_centers, *params)
        residual = np.sum((gamma - predicted) ** 2)

        all_fits[name] = {'params': params, 'residual': residual, 'func': func}

        print(f"\n{name.upper():12} 模型:")
        print(f"  块金值 (c0): {params[0]:.4f}")
        print(f"  偏基台值 (c1): {params[1]:.4f}")
        print(f"  变程 (a): {params[2]:.4f}")
        print(f"  基台值 (c0+c1): {params[0] + params[1]:.4f}")
        print(f"  拟合残差: {residual:.6f}")
        print(f"  块金效应: {params[0] / (params[0] + params[1]):.1%}")

        if residual < best_residual:
            best_residual = residual
            best_model = name
            best_params = params

    except Exception as e:
        print(f"\n{name.upper():12} 模型拟合失败: {str(e)[:50]}")

print(f"\n✅ 最佳模型: {best_model.upper()} (残差最小: {best_residual:.6f})")

# 5. 执行克里金插值
print("\n" + "=" * 60)
print("4. 克里金插值计算")
print("=" * 60)

grid_size = 50  # 减少网格大小以提高速度
grid_x = np.linspace(x.min(), x.max(), grid_size)
grid_y = np.linspace(y.min(), y.max(), grid_size)

print("🔄 构建克里金方程组...")
print("   未知点数量:", grid_size * grid_size)
print("   已知点数量:", len(x))

# 详细展示克里金权重计算过程（简化版）
print("\n克里金权重计算原理:")
print("   1. 基于变异函数模型计算点对之间的协方差")
print("   2. 构建克里金方程组: K * w = k")
print("   3. 解方程组得到各已知点的权重 w")
print("   4. 插值值 = Σ(w_i * z_i)")
print("   5. 估计方差 = C(0) - Σ(w_i * C(d_i))")

try:
    # 使用最佳模型进行克里金插值
    print(f"\n🔄 使用 {best_model} 模型进行插值...")
    OK = OrdinaryKriging(x, y, z,
                         variogram_model=best_model,
                         variogram_parameters={
                             'nugget': best_params[0],
                             'sill': best_params[0] + best_params[1],
                             'range': best_params[2]
                         },
                         verbose=True)  # 显示详细过程

    z_interp, sigma = OK.execute('grid', grid_x, grid_y)
    print("✅ 克里金插值完成！")

    # 显示部分权重信息
    print(f"\n插值结果统计:")
    print(f"   最小值: {z_interp.min():.4f}")
    print(f"   最大值: {z_interp.max():.4f}")
    print(f"   平均值: {z_interp.mean():.4f}")
    print(f"   平均标准差: {sigma.mean():.4f}")

except Exception as e:
    print(f"❌ 克里金插值失败: {e}")
    # 使用简单插值作为备选
    from scipy.interpolate import griddata

    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    z_interp = griddata((x, y), z, (X_grid, Y_grid), method='cubic')
    sigma = np.zeros_like(z_interp)
    print("⚠️ 使用立方插值作为备选")

# 6. 创建图表
print("\n" + "=" * 60)
print("5. 生成可视化图表")
print("=" * 60)

fig = plt.figure(figsize=(18, 6))

# 子图1：变异函数拟合比较
ax1 = plt.subplot(1, 3, 1)
# 绘制实验变异函数
ax1.scatter(lag_centers, gamma, s=60, c='black', alpha=0.8,
            edgecolors='white', linewidth=1, label='实验变异函数', zorder=5)

# 绘制不同模型的拟合曲线
h_fit = np.linspace(0, max_lag, 200)
colors = ['red', 'blue', 'green', 'orange']
linestyles = ['-', '--', '-.', ':']

for idx, (name, fit_info) in enumerate(all_fits.items()):
    if name in all_fits:
        params = fit_info['params']
        func = fit_info['func']
        gamma_fit = func(h_fit, *params)

        # 用粗线标记最佳模型
        if name == best_model:
            ax1.plot(h_fit, gamma_fit, color=colors[idx],
                     linewidth=3, linestyle=linestyles[idx],
                     label=f'{name} (最佳)', alpha=0.9)
        else:
            ax1.plot(h_fit, gamma_fit, color=colors[idx],
                     linewidth=1.5, linestyle=linestyles[idx],
                     label=f'{name}', alpha=0.7)

# 添加关键参数标注
ax1.axhline(y=best_params[0] + best_params[1], color='purple',
            linestyle=':', linewidth=1, alpha=0.5, label='基台值')
ax1.axvline(x=best_params[2], color='green',
            linestyle=':', linewidth=1, alpha=0.5, label='变程')

ax1.set_xlabel('距离 (h)', fontsize=11)
ax1.set_ylabel('半方差 γ(h)', fontsize=11)
ax1.set_title('变异函数模型拟合比较', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

# 添加模型参数文本框
param_text = (f'最佳模型: {best_model.upper()}\n'
              f'块金值: {best_params[0]:.3f}\n'
              f'偏基台值: {best_params[1]:.3f}\n'
              f'变程: {best_params[2]:.3f}\n'
              f'基台值: {best_params[0] + best_params[1]:.3f}\n'
              f'块金效应: {best_params[0] / (best_params[0] + best_params[1]):.1%}')
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes,
         verticalalignment='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 子图2：克里金中间过程示意
ax2 = plt.subplot(1, 3, 2)
# 选择一个插值点作为示例
example_x = (x.min() + x.max()) / 2
example_y = (y.min() + y.max()) / 2

# 绘制所有已知点
scatter = ax2.scatter(x, y, c=z, s=40, cmap='coolwarm',
                      alpha=0.7, edgecolors='black', linewidth=0.5)

# 标记插值点
ax2.scatter([example_x], [example_y], s=200, c='red',
            marker='*', edgecolors='black', linewidth=1, zorder=10, label='插值点')

# 标记最近的几个点（示意克里金权重）
distances = np.sqrt((x - example_x) ** 2 + (y - example_y) ** 2)
nearest_idx = np.argsort(distances)[:5]  # 最近的5个点

for idx in nearest_idx:
    # 用线连接插值点和已知点
    ax2.plot([example_x, x[idx]], [example_y, y[idx]],
             'gray', linestyle='--', alpha=0.5, linewidth=1)
    # 标记权重大小（用点的大小表示）
    weight = 1.0 / (distances[idx] + 0.1)  # 示意权重
    ax2.scatter([x[idx]], [y[idx]], s=weight * 200,
                c='green', alpha=0.6, edgecolors='black')

ax2.set_xlabel('X坐标', fontsize=11)
ax2.set_ylabel('Y坐标', fontsize=11)
ax2.set_title('克里金权重计算示意', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

# 添加示意图说明
info_text = ('克里金插值原理:\n'
             '1. 基于空间相关性\n'
             '2. 距离越近权重越大\n'
             '3. 使用变异函数模型\n'
             '4. 无偏最优估计')
ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
         verticalalignment='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# 子图3：3D插值结果
ax3 = plt.subplot(1, 3, 3, projection='3d')
X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

# 创建3D曲面
surf = ax3.plot_surface(X_grid, Y_grid, z_interp, cmap='terrain',
                        alpha=0.9, linewidth=0.2, antialiased=True,
                        rstride=1, cstride=1)

# 添加原始数据点
ax3.scatter(x, y, z, c='red', s=20, depthshade=True,
            alpha=0.8, edgecolors='black', linewidth=0.5, label='原始数据')

ax3.set_xlabel('X坐标', fontsize=11, labelpad=10)
ax3.set_ylabel('Y坐标', fontsize=11, labelpad=10)
ax3.set_zlabel('插值结果', fontsize=11, labelpad=10)
ax3.set_title('3D插值曲面', fontsize=12, fontweight='bold', pad=20)

# 调整视角
ax3.view_init(elev=30, azim=45)

plt.suptitle(f'克里金插值分析 - 最佳模型: {best_model.upper()}',
             fontsize=14, y=1.05, fontweight='bold')
plt.tight_layout()

print("\n" + "=" * 60)
print("关键结果总结:")
print("=" * 60)
print(f"1. 实验变异函数: 计算了{len(lag_centers)}个滞后区间")
print(f"2. 最佳拟合模型: {best_model.upper()}")
print(f"3. 模型参数: 块金值={best_params[0]:.3f}, "
      f"偏基台值={best_params[1]:.3f}, 变程={best_params[2]:.3f}")
print(f"4. 块金效应: {best_params[0] / (best_params[0] + best_params[1]):.1%}")
print(f"5. 插值网格: {grid_size}×{grid_size} ({grid_size * grid_size}个点)")

print("\n" + "=" * 60)
print("完成！关闭窗口结束程序。")
print("=" * 60)

plt.show()