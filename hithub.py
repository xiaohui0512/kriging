import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
from scipy import stats
import os

# ==================== 1. 读取数据 ====================
print("=" * 60)
print("克里金插值分析系统")
print("=" * 60)

# 查找数据文件
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', '数据.csv')
if os.path.exists(desktop_path):
    data = pd.read_csv(desktop_path)
    print("✓ 从桌面读取数据")
elif os.path.exists('数据.csv'):
    data = pd.read_csv('数据.csv')
    print("✓ 从当前目录读取数据")
else:
    print("✗ 找不到数据文件！")
    exit()

# 提取数据
x = data['X'].values
y = data['Y'].values
z = data['Value'].values

print(f"✓ 成功读取 {len(data)} 条数据")
print(f"  X范围: [{x.min():.1f}, {x.max():.1f}]")
print(f"  Y范围: [{y.min():.1f}, {y.max():.1f}]")
print(f"  Value范围: [{z.min():.1f}, {z.max():.1f}]")

# ==================== 2. 选择克里金类型 ====================
print("\n" + "=" * 60)
print("请选择克里金类型：")
print("1. 泛克里金 (Universal Kriging)")
print("2. 普通克里金 (Ordinary Kriging)")
print("3. 协同克里金 (CoKriging) - 基于外部漂移")
print("4. 回归克里金 (Regression Kriging)")
print("=" * 60)

choice = input("请输入选择 (1-4, 默认1): ").strip()
if choice not in ['1', '2', '3', '4']:
    choice = '1'

# ==================== 3. 选择变差函数模型 ====================
print("\n" + "=" * 60)
print("请选择变差函数模型：")
print("1. linear (线性)")
print("2. gaussian (高斯)")
print("3. spherical (球状)")
print("4. exponential (指数)")
print("=" * 60)

variogram_choice = input("请输入选择 (1-4, 默认1): ").strip()
variogram_dict = {'1': 'linear', '2': 'gaussian', '3': 'spherical', '4': 'exponential'}
variogram_model = variogram_dict.get(variogram_choice, 'linear')

print(f"\n使用变差函数模型: {variogram_model}")

# ==================== 4. 创建插值网格 ====================
grid_x = np.linspace(x.min(), x.max(), 50)
grid_y = np.linspace(y.min(), y.max(), 50)
X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

# ==================== 5. 执行不同的克里金方法 ====================
if choice == '2':
    # ==================== 普通克里金 ====================
    print("\n使用普通克里金 (Ordinary Kriging)")

    ok = OrdinaryKriging(
        x, y, z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    # 显示变差函数信息
    print("\n" + "-" * 40)
    print("变差函数参数:")
    print("-" * 40)
    params = ok.variogram_model_parameters

    if variogram_model == 'linear':
        print(f"块金值 (nugget): {params[0]:.4f}")
        print(f"斜率 (slope): {params[1]:.4f}")
        print("模型: γ(h) = nugget + slope × h")
    else:
        print(f"块金值 (nugget): {params[0]:.4f}")
        print(f"偏基台值 (sill): {params[1]:.4f}")
        print(f"变程 (range): {params[2]:.4f}")

        if variogram_model == 'gaussian':
            print("模型: γ(h) = nugget + sill × [1 - exp(-3h²/range²)]")
        elif variogram_model == 'spherical':
            print("模型: γ(h) = nugget + sill × [1.5h/range - 0.5(h/range)³]  (h≤range)")
            print("      = nugget + sill  (h>range)")
        elif variogram_model == 'exponential':
            print("模型: γ(h) = nugget + sill × [1 - exp(-3h/range)]")

    kriging_obj = ok
    method_name = "普通克里金"

elif choice == '3':
    # ==================== 协同克里金 (基于外部漂移) ====================
    print("\n使用协同克里金 (基于外部漂移的近似实现)")
    print("注: 这里使用一个合成的辅助变量")

    # 创建一个辅助变量（例如，可以基于X和Y的某种函数）
    # 这里我们假设辅助变量与主变量有相关性
    print("\n" + "-" * 40)
    print("辅助变量生成:")
    print("-" * 40)

    # 生成辅助变量（模拟一个与主变量相关的变量）
    # 使用X和Y的线性组合加上一些噪声
    z2 = 0.3 * x + 0.2 * y + 0.5 * z + np.random.normal(0, 0.3, len(z))

    print(f"主变量 (Z) 均值: {z.mean():.4f}, 标准差: {z.std():.4f}")
    print(f"辅助变量 (Z2) 均值: {z2.mean():.4f}, 标准差: {z2.std():.4f}")

    # 计算相关系数
    correlation = np.corrcoef(z, z2)[0, 1]
    print(f"主变量与辅助变量的相关系数: {correlation:.4f}")

    # 对辅助变量在整个网格上进行插值（使用普通克里金）
    print("\n对辅助变量进行克里金插值...")
    ok_z2 = OrdinaryKriging(
        x, y, z2,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )
    z2_interp, _ = ok_z2.execute('grid', grid_x, grid_y)

    # 使用泛克里金进行协同克里金（外部漂移方法）
    print("使用外部漂移进行协同克里金插值...")
    uk = UniversalKriging(
        x, y, z,
        variogram_model=variogram_model,
        drift_terms=['external_Z'],
        external_drift_x=grid_x,
        external_drift_y=grid_y,
        external_drift_z=z2_interp,
        verbose=False,
        enable_plotting=False
    )

    params = uk.variogram_model_parameters
    print("\n" + "-" * 40)
    print("协同克里金参数:")
    print("-" * 40)

    if variogram_model == 'linear':
        print(f"块金值 (nugget): {params[0]:.4f}")
        print(f"斜率 (slope): {params[1]:.4f}")
    else:
        print(f"块金值 (nugget): {params[0]:.4f}")
        print(f"偏基台值 (sill): {params[1]:.4f}")
        print(f"变程 (range): {params[2]:.4f}")

    kriging_obj = uk
    method_name = "协同克里金"

elif choice == '4':
    # ==================== 回归克里金 ====================
    print("\n使用回归克里金 (Regression Kriging)")

    # 第一步：建立回归模型
    print("\n" + "-" * 40)
    print("回归模型建立:")
    print("-" * 40)

    # 使用X, Y以及可能的交互项作为预测变量
    X_design = np.column_stack([np.ones(len(x)), x, y, x * y, x ** 2, y ** 2])
    beta = np.linalg.lstsq(X_design, z, rcond=None)[0]

    print(f"回归方程:")
    print(f"Z = {beta[0]:.4f} + {beta[1]:.4f}·X + {beta[2]:.4f}·Y + ")
    print(f"    {beta[3]:.4f}·XY + {beta[4]:.4f}·X² + {beta[5]:.4f}·Y²")

    # 计算回归预测值
    z_regression = (beta[0] + beta[1] * x + beta[2] * y +
                    beta[3] * x * y + beta[4] * x ** 2 + beta[5] * y ** 2)

    # 计算回归残差
    residuals = z - z_regression

    # 回归模型评估
    sst = np.sum((z - np.mean(z)) ** 2)
    ssr = np.sum(residuals ** 2)
    r2_reg = 1 - ssr / sst

    print(f"\n回归模型评估:")
    print(f"回归R²: {r2_reg:.4f}")
    print(f"残差均值: {residuals.mean():.4f}")
    print(f"残差标准差: {residuals.std():.4f}")

    # 第二步：对残差进行普通克里金插值
    print("\n" + "-" * 40)
    print("残差克里金插值:")
    print("-" * 40)

    ok_residual = OrdinaryKriging(
        x, y, residuals,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    params = ok_residual.variogram_model_parameters
    if variogram_model == 'linear':
        print(f"残差块金值: {params[0]:.4f}")
        print(f"残差斜率: {params[1]:.4f}")
    else:
        print(f"残差块金值: {params[0]:.4f}")
        print(f"残差基台值: {params[1]:.4f}")
        print(f"残差变程: {params[2]:.4f}")

    # 第三步：在整个网格上计算回归趋势
    trend_surface = (beta[0] + beta[1] * X_grid + beta[2] * Y_grid +
                     beta[3] * X_grid * Y_grid + beta[4] * X_grid ** 2 + beta[5] * Y_grid ** 2)

    # 对残差进行克里金插值
    residuals_interp, ss_residual = ok_residual.execute('grid', grid_x, grid_y)

    # 合并回归趋势和残差插值
    z_interp = trend_surface + residuals_interp
    ss = ss_residual  # 使用残差插值的方差

    kriging_obj = ok_residual
    method_name = "回归克里金"

else:
    # ==================== 泛克里金 ====================
    print("\n使用泛克里金 (Universal Kriging)")

    uk = UniversalKriging(
        x, y, z,
        variogram_model=variogram_model,
        drift_terms=['regional_linear'],  # 区域线性趋势
        verbose=False,
        enable_plotting=False
    )

    params = uk.variogram_model_parameters
    print("\n" + "-" * 40)
    print("泛克里金参数:")
    print("-" * 40)

    if variogram_model == 'linear':
        print(f"块金值 (nugget): {params[0]:.4f}")
        print(f"斜率 (slope): {params[1]:.4f}")
    else:
        print(f"块金值 (nugget): {params[0]:.4f}")
        print(f"偏基台值 (sill): {params[1]:.4f}")
        print(f"变程 (range): {params[2]:.4f}")

    print(f"\n趋势项: 区域线性趋势")
    print("m(x,y) = μ₀ + μ₁·x + μ₂·y")

    # 估计趋势项系数
    X_design = np.column_stack([np.ones(len(x)), x, y])
    beta_est = np.linalg.lstsq(X_design, z, rcond=None)[0]
    print(f"\n趋势项系数估计:")
    print(f"μ₀ ≈ {beta_est[0]:.4f}")
    print(f"μ₁ ≈ {beta_est[1]:.4f}")
    print(f"μ₂ ≈ {beta_est[2]:.4f}")

    kriging_obj = uk
    method_name = "泛克里金"

# ==================== 6. 执行插值（除了回归克里金）====================
if choice != '4':  # 回归克里金已经在上面计算过了
    print(f"\n正在执行{method_name}插值...")
    z_interp, ss = kriging_obj.execute('grid', grid_x, grid_y)

# ==================== 7. 显示克里金方法原理 ====================
print("\n" + "=" * 60)
print(f"{method_name} 原理:")
print("=" * 60)

if choice == '2':
    print("""
普通克里金 (Ordinary Kriging):
假设区域化变量Z(x)满足二阶平稳性，即：
  1. E[Z(x)] = m (常数)
  2. Cov[Z(x), Z(x+h)] = C(h)

估计值: Ẑ(x₀) = Σ λᵢ Z(xᵢ)
约束条件: Σ λᵢ = 1 (无偏性)

权重λ通过最小化估计方差得到，求解以下方程组：
  Σ λⱼ C(xᵢ-xⱼ) + μ = C(xᵢ-x₀)  ∀i
  Σ λᵢ = 1
其中μ是拉格朗日乘子。
""")

elif choice == '3':
    print("""
协同克里金 (CoKriging):
同时利用主变量Z和辅助变量Z₂的信息进行估计。

估计值: Ẑ(x₀) = Σ λᵢ Z(xᵢ) + Σ νⱼ Z₂(xⱼ)

需要估计主变量和辅助变量之间的互协方差函数。
本实现使用外部漂移方法近似：
  Z(x) = α + β·Z₂(x) + ε(x)
其中ε(x)是残差，满足E[ε(x)] = 0
""")

elif choice == '4':
    print("""
回归克里金 (Regression Kriging):
两步法：
1. 用回归模型估计趋势: Z(x) = f(x)β + ε(x)
2. 对残差ε(x)进行普通克里金插值

最终估计: Ẑ(x₀) = f(x₀)β̂ + Σ λᵢ ε̂(xᵢ)

优点：可以将专业知识（回归模型）与空间相关性结合。
""")

else:
    print("""
泛克里金 (Universal Kriging):
假设区域化变量Z(x)的期望是位置的函数：
  E[Z(x)] = m(x) = Σ βⱼ fⱼ(x)

估计值: Ẑ(x₀) = Σ λᵢ Z(xᵢ)
约束条件: Σ λᵢ fⱼ(xᵢ) = fⱼ(x₀)  ∀j

本实现使用区域线性趋势：m(x,y) = μ₀ + μ₁·x + μ₂·y
""")

# ==================== 8. 计算采样点处的插值值用于评估 ====================
z_at_samples = griddata((X_grid.flatten(), Y_grid.flatten()),
                        z_interp.flatten(), (x, y), method='linear')
residuals = z - z_at_samples

# ==================== 9. 权重计算示例 ====================
print("\n" + "-" * 40)
print("权重计算示例:")
print("-" * 40)

# 选择一个插值点（网格中心）
center_idx_x = len(grid_x) // 2
center_idx_y = len(grid_y) // 2
center_x = grid_x[center_idx_x]
center_y = grid_y[center_idx_y]

print(f"插值点位置: ({center_x:.1f}, {center_y:.1f})")

# 计算距离并排序
distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
sorted_idx = np.argsort(distances)

print(f"\n最近的5个采样点:")
print("ID   X      Y      距离     观测值     估计权重")
print("-" * 50)

total_weight = 0
for i in range(min(5, len(sorted_idx))):
    idx = sorted_idx[i]
    dist = distances[idx]

    # 简化权重计算（距离反比加权）
    weight = 1.0 / (1.0 + dist ** 2)
    total_weight += weight

    print(f"{idx + 1:3d}  {x[idx]:5.1f}  {y[idx]:5.1f}  {dist:6.2f}  {z[idx]:7.3f}  {weight:8.4f}")

print("-" * 50)
print(f"权重总和: {total_weight:.4f}")

# 计算插值点估计值（简化）
weights = 1.0 / (1.0 + distances ** 2)
weights = weights / np.sum(weights)  # 归一化
estimated_value = np.sum(weights * z)
print(f"加权估计值: {estimated_value:.4f}")

# ==================== 10. 3D可视化 ====================
print("\n" + "=" * 60)
print("生成3D可视化...")
print("=" * 60)

fig = plt.figure(figsize=(18, 5))

# 子图1：插值结果
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_grid, Y_grid, z_interp, cmap='viridis',
                         alpha=0.7, rstride=1, cstride=1, antialiased=True)
scatter1 = ax1.scatter(x, y, z, c='red', s=40, marker='o',
                       edgecolors='black', linewidth=1, label='采样点')
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.set_zlabel('Value', fontsize=10)
ax1.set_title(f'{method_name} 插值结果', fontsize=12, pad=15)
ax1.legend()
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label='值')

# 子图2：插值方差
ax2 = fig.add_subplot(132, projection='3d')
if choice == '3':
    # 协同克里金的方差可能不是直接可用的
    variance_plot = np.zeros_like(z_interp)
else:
    variance_plot = ss

surf2 = ax2.plot_surface(X_grid, Y_grid, variance_plot, cmap='Reds',
                         alpha=0.7, rstride=1, cstride=1, antialiased=True)
ax2.scatter(x, y, np.zeros_like(z), c='blue', s=20, marker='^', alpha=0.6)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.set_zlabel('方差', fontsize=10)
ax2.set_title('插值不确定性', fontsize=12, pad=15)
fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, label='方差')

# 子图3：残差分布
ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(x, y, z_at_samples, c=residuals, cmap='coolwarm',
                       s=40, marker='o', edgecolors='black', linewidth=0.5)

# 添加误差线
for i in range(min(15, len(x))):  # 只显示部分误差线
    ax3.plot([x[i], x[i]], [y[i], y[i]], [z[i], z_at_samples[i]],
             'gray', alpha=0.5, linewidth=0.5)

ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.set_zlabel('估计值', fontsize=10)
ax3.set_title('采样点残差分布', fontsize=12, pad=15)
fig.colorbar(scatter3, ax=ax3, shrink=0.6, aspect=10, label='残差')

# 调整视角
for ax in [ax1, ax2, ax3]:
    ax.view_init(elev=30, azim=45)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# ==================== 11. 统计评估 ====================
print("\n" + "=" * 60)
print("模型评估统计:")
print("=" * 60)

mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals ** 2))
r2 = 1 - np.sum(residuals ** 2) / np.sum((z - np.mean(z)) ** 2)
bias = np.mean(residuals)

print(f"模型: {method_name} ({variogram_model}变差函数)")
print(f"采样点数: {len(z)}")
print(f"插值网格: {z_interp.shape[0]} × {z_interp.shape[1]}")
print(f"偏差 (Bias): {bias:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 残差统计
print(f"\n残差统计:")
print(f"  最小值: {residuals.min():.4f}")
print(f"  最大值: {residuals.max():.4f}")
print(f"  均值: {residuals.mean():.4f}")
print(f"  标准差: {residuals.std():.4f}")

# 精度分类
good_fit = np.sum(np.abs(residuals) < 0.5) / len(residuals) * 100
print(f"\n拟合精度:")
print(f"  |残差| < 0.5: {good_fit:.1f}% 的采样点")
print(f"  |残差| < 1.0: {np.sum(np.abs(residuals) < 1.0) / len(residuals) * 100:.1f}% 的采样点")

print("\n" + "=" * 60)
print(f"✓ {method_name} 分析完成!")
print("=" * 60)

plt.show()