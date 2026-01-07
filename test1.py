import numpy as np
from pykrige.ok import OrdinaryKriging

# 输入数据
data = np.array([
    [1.0, 5.0, 100.0],
    [3.0, 4.0, 105.0],
    [1.0, 3.0, 105.0],
    [4.0, 5.0, 100.0],
    [5.0, 1.0, 115.0]
])

# 分离坐标和值
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

print("输入数据点：")
for i in range(len(x)):
    print(f"点({x[i]}, {y[i]}) = {z[i]}")

# 要预测的点
target_x = 1.0
target_y = 4.0

print(f"\n要预测的点: ({target_x}, {target_y})")

# 创建克里金对象并进行预测
ok = OrdinaryKriging(
    x, y, z,
    variogram_model='linear',  # 使用线性模型
    variogram_parameters=[1, 0],  # [slope, nugget]
    verbose=True
)

# 执行预测
z_pred, sigma_sq = ok.execute('points', [target_x], [target_y])

print(f"\n预测结果:")
print(f"预测值: {z_pred[0]:.2f}")
print(f"预测方差: {sigma_sq[0]:.2f}")