import sys

sys.path.append('C:/Users/huawei/PyCharmMiscProject/.venv/Lib/site-packages')

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# 设置随机种子保证可复现
np.random.seed(42)
tf.random.set_seed(42)

os.makedirs('outputs', exist_ok=True)

print("\n" + "=" * 60)
print("出行需求预测模型")
print("=" * 60)

# ========================
# 1. 数据加载与特征工程
# ========================
print("\n【步骤1: 数据加载与特征工程】")

# 加载清洗后的数据
data = pd.read_parquet('yellow_tripdata_2023-01_cleaned.parquet')
print(f"数据形状: {data.shape}")

# 提取时间特征
data['pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['pickup_hour'] = data['pickup_datetime'].dt.hour
data['pickup_weekday'] = data['pickup_datetime'].dt.weekday
data['pickup_day'] = data['pickup_datetime'].dt.day
data['is_weekend'] = data['pickup_weekday'].isin([5, 6]).astype(int)
data['is_rush_hour'] = data['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

# 策略理由：将小时转换为循环特征，保留0点和23点相邻的关系
data['hour_sin'] = np.sin(2 * np.pi * data['pickup_hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['pickup_hour'] / 24)
data['weekday_sin'] = np.sin(2 * np.pi * data['pickup_weekday'] / 7)
data['weekday_cos'] = np.cos(2 * np.pi * data['pickup_weekday'] / 7)

# ========================
# 2. 构建目标变量：区域×时段的需求量
# ========================
print("\n【步骤2: 构建目标变量】")

# 按区域和小时聚合，计算需求量
demand = data.groupby(['PULocationID', 'pickup_hour', 'pickup_weekday']).size().reset_index(name='demand')

# 添加循环时间特征
demand['hour_sin'] = np.sin(2 * np.pi * demand['pickup_hour'] / 24)
demand['hour_cos'] = np.cos(2 * np.pi * demand['pickup_hour'] / 24)
demand['weekday_sin'] = np.sin(2 * np.pi * demand['pickup_weekday'] / 7)
demand['weekday_cos'] = np.cos(2 * np.pi * demand['pickup_weekday'] / 7)
demand['is_weekend'] = demand['pickup_weekday'].isin([5, 6]).astype(int)
demand['is_rush_hour'] = demand['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

# 对区域ID进行标签编码
le = LabelEncoder()
demand['location_encoded'] = le.fit_transform(demand['PULocationID'])

print(f"聚合后数据形状: {demand.shape}")
print(f"区域数量: {demand['PULocationID'].nunique()}")
print(f"需求量范围: {demand['demand'].min()} - {demand['demand'].max()}")
print(f"平均需求量: {demand['demand'].mean():.2f}")

# ========================
# 3. 准备特征和目标变量
# ========================
print("\n【步骤3: 准备训练数据】")

# 特征列
feature_cols = [
    'location_encoded',
    'hour_sin', 'hour_cos',
    'weekday_sin', 'weekday_cos',
    'is_weekend', 'is_rush_hour'
]

X = demand[feature_cols].values
y = demand['demand'].values.reshape(-1, 1)

# 标准化特征（对神经网络很重要）
# 策略理由：标准化能让神经网络训练更稳定，收敛更快
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape[0]:,}")
print(f"测试集大小: {X_test.shape[0]:,}")

# ========================
# 4. 构建神经网络模型
# ========================
print("\n【步骤4: 构建神经网络模型】")


def build_nn_model(input_dim):
    """
    策略理由：
    - 使用3层全连接网络，逐渐减少神经元数量
    - ReLU激活函数避免梯度消失
    - Dropout防止过拟合
    - 学习率衰减帮助模型收敛
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,),
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),

        layers.Dense(1)  # 输出层：预测需求量
    ])

    # 使用Adam优化器，学习率可调节
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model


# 构建模型
input_dim = X_train.shape[1]
nn_model = build_nn_model(input_dim)

print("神经网络结构:")
nn_model.summary()

# ========================
# 5. 训练神经网络
# ========================
print("\n【步骤5: 训练神经网络】")

# 回调函数
# 策略理由：
# - EarlyStopping：验证损失不再下降时停止训练，防止过拟合
# - ReduceLROnPlateau：损失停滞时降低学习率，帮助找到更优解
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# 训练模型
history = nn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ========================
# 6. 绘制Loss曲线
# ========================
print("\n【步骤6: 绘制Loss曲线】")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss曲线
axes[0].plot(history.history['loss'], label='训练损失', linewidth=2)
axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('神经网络训练Loss曲线', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# MAE曲线
axes[1].plot(history.history['mae'], label='训练MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='验证MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('神经网络训练MAE曲线', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/nn_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ nn_training_curves.png 已保存")

# ========================
# 7. 神经网络评估
# ========================
print("\n【步骤7: 神经网络模型评估】")

# 预测
y_pred_nn_scaled = nn_model.predict(X_test)
y_pred_nn = scaler_y.inverse_transform(y_pred_nn_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# 计算指标
mae_nn = mean_absolute_error(y_test_original, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test_original, y_pred_nn))

print(f"神经网络测试结果:")
print(f"  MAE:  {mae_nn:.4f}")
print(f"  RMSE: {rmse_nn:.4f}")

# 预测值vs真实值散点图
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(y_test_original, y_pred_nn, alpha=0.3, s=10, c='steelblue')
ax.plot([y_test_original.min(), y_test_original.max()],
        [y_test_original.min(), y_test_original.max()],
        'r--', linewidth=2, label='完美预测')

ax.set_xlabel('真实需求量', fontsize=12)
ax.set_ylabel('预测需求量', fontsize=12)
ax.set_title(f'神经网络预测 vs 真实值\nMAE={mae_nn:.2f}, RMSE={rmse_nn:.2f}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/nn_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ nn_predictions.png 已保存")

# ========================
# 8. 随机森林模型
# ========================
print("\n【步骤8: 训练随机森林模型】")

# 策略理由：随机森林对特征尺度不敏感，使用原始数据即可
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X, y.ravel(), test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("训练随机森林中...")
rf_model.fit(X_train_rf, y_train_rf)
print("随机森林训练完成")

# ========================
# 9. 随机森林评估
# ========================
print("\n【步骤9: 随机森林模型评估】")

y_pred_rf = rf_model.predict(X_test_rf)

mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))

print(f"随机森林测试结果:")
print(f"  MAE:  {mae_rf:.4f}")
print(f"  RMSE: {rmse_rf:.4f}")

# 随机森林预测vs真实值
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(y_test_rf, y_pred_rf, alpha=0.3, s=10, c='forestgreen')
ax.plot([y_test_rf.min(), y_test_rf.max()],
        [y_test_rf.min(), y_test_rf.max()],
        'r--', linewidth=2, label='完美预测')

ax.set_xlabel('真实需求量', fontsize=12)
ax.set_ylabel('预测需求量', fontsize=12)
ax.set_title(f'随机森林预测 vs 真实值\nMAE={mae_rf:.2f}, RMSE={rmse_rf:.2f}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/rf_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ rf_predictions.png 已保存")

# ========================
# 10. 模型对比分析
# ========================
print("\n" + "=" * 60)
print("模型对比分析")
print("=" * 60)

# 10.1 指标对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ['神经网络', '随机森林']
mae_values = [mae_nn, mae_rf]
rmse_values = [rmse_nn, rmse_rf]
colors_bar = ['#3498db', '#2ecc71']

# MAE对比
bars1 = axes[0].bar(models, mae_values, color=colors_bar, edgecolor='white', width=0.5)
for bar, val in zip(bars1, mae_values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
axes[0].set_ylabel('MAE', fontsize=12)
axes[0].set_title('MAE 对比（越低越好）', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# RMSE对比
bars2 = axes[1].bar(models, rmse_values, color=colors_bar, edgecolor='white', width=0.5)
for bar, val in zip(bars2, rmse_values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('RMSE 对比（越低越好）', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('神经网络 vs 随机森林：预测性能对比', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ model_comparison.png 已保存")

# 10.2 特征重要性（随机森林）
fig, ax = plt.subplots(figsize=(10, 6))

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

feature_names_cn = {
    'location_encoded': '区域编码',
    'hour_sin': '小时(sin)',
    'hour_cos': '小时(cos)',
    'weekday_sin': '星期(sin)',
    'weekday_cos': '星期(cos)',
    'is_weekend': '是否周末',
    'is_rush_hour': '是否高峰时段'
}
feature_importance['feature_cn'] = feature_importance['feature'].map(feature_names_cn)

bars = ax.barh(feature_importance['feature_cn'], feature_importance['importance'],
               color='coral', edgecolor='white', height=0.6)

for bar, val in zip(bars, feature_importance['importance']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10)

ax.set_xlabel('重要性', fontsize=12)
ax.set_title('随机森林特征重要性', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ feature_importance.png 已保存")

# 10.3 残差分布对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

residuals_nn = (y_test_original - y_pred_nn).ravel()
residuals_rf = (y_test_rf - y_pred_rf).ravel()

# 神经网络残差
axes[0].hist(residuals_nn, bins=50, color='steelblue', edgecolor='white', alpha=0.7, density=True)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].axvline(x=np.mean(residuals_nn), color='darkred', linestyle='-', linewidth=2,
                label=f'均值: {np.mean(residuals_nn):.2f}')
axes[0].set_xlabel('残差（真实值-预测值）', fontsize=11)
axes[0].set_ylabel('密度', fontsize=11)
axes[0].set_title('神经网络残差分布', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 随机森林残差
axes[1].hist(residuals_rf, bins=50, color='forestgreen', edgecolor='white', alpha=0.7, density=True)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].axvline(x=np.mean(residuals_rf), color='darkred', linestyle='-', linewidth=2,
                label=f'均值: {np.mean(residuals_rf):.2f}')
axes[1].set_xlabel('残差（真实值-预测值）', fontsize=11)
axes[1].set_ylabel('密度', fontsize=11)
axes[1].set_title('随机森林残差分布', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('残差分布对比', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/residuals_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ residuals_comparison.png 已保存")

# ========================
# 11. 总结报告
# ========================
print("\n" + "=" * 60)
print("模型对比总结")
print("=" * 60)

print(f"\n{'指标':<15} {'神经网络':<15} {'随机森林':<15} {'较优模型':<15}")
print("-" * 60)
print(f"{'MAE':<15} {mae_nn:<15.4f} {mae_rf:<15.4f} {'神经网络' if mae_nn < mae_rf else '随机森林':<15}")
print(f"{'RMSE':<15} {rmse_nn:<15.4f} {rmse_rf:<15.4f} {'神经网络' if rmse_nn < rmse_rf else '随机森林':<15}")

improvement_mae = abs(mae_nn - mae_rf) / max(mae_nn, mae_rf) * 100
improvement_rmse = abs(rmse_nn - rmse_rf) / max(rmse_nn, rmse_rf) * 100

print(f"\n性能差异:")
print(f"  MAE差异: {improvement_mae:.2f}%")
print(f"  RMSE差异: {improvement_rmse:.2f}%")

print("\n分析方法优劣:")
print("-" * 40)
print("神经网络优势:")
print("  1. 能捕捉复杂的非线性关系")
print("  2. 通过循环特征更好地理解时间的周期性")
print("  3. 可扩展性强，可添加更多特征")
print("  4. 适合大规模数据训练")
print("\n随机森林优势:")
print("  1. 训练速度快，不需要特征标准化")
print("  2. 可解释性强，能输出特征重要性")
print("  3. 对超参数不那么敏感")
print("  4. 小数据集上表现更稳定")

print("\n" + "=" * 60)
print("预测模型构建完成！所有图表已保存至 outputs/ 目录")
print("=" * 60)