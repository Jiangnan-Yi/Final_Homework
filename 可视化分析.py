import sys
sys.path.append('C:/Users/huawei/PyCharmMiscProject/.venv/Lib/site-packages')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import os
os.makedirs('outputs', exist_ok=True)
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "="*60)
print("可视化分析")
print("="*60)

# ========================
# 加载清洗后的数据
# ========================
data = pd.read_parquet('yellow_tripdata_2023-01_cleaned.parquet')

print(f"数据形状: {data.shape}")

# ========================
# 提取时间特征
# ========================
print("正在提取时间特征...")

data['pickup_hour'] = pd.to_datetime(data['tpep_pickup_datetime']).dt.hour
data['pickup_weekday'] = pd.to_datetime(data['tpep_pickup_datetime']).dt.weekday
data['pickup_date'] = pd.to_datetime(data['tpep_pickup_datetime']).dt.date
data['is_weekend'] = data['pickup_weekday'].isin([5, 6]).map({True: '周末', False: '工作日'})

print("时间特征提取完成")

# ========================
# 1 出行需求时间规律
# ========================
print("\n【分析1: 出行需求时间规律】")

# 1.1 分小时平均订单量折线图（区分工作日/周末）
fig, ax = plt.subplots(figsize=(14, 6))

for period, color, label in [('工作日', '#1f77b4', '工作日'), ('周末', '#d62728', '周末')]:
    period_data = data[data['is_weekend'] == period].groupby('pickup_hour').size()
    period_data = period_data.reindex(range(24), fill_value=0)
    ax.plot(period_data.index, period_data.values, color=color, linewidth=2.5, marker='o', markersize=6, label=label)
    ax.fill_between(period_data.index, period_data.values, alpha=0.1, color=color)

ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('订单量', fontsize=12)
ax.set_title('分小时订单量分布（工作日 vs 周末）', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/01_hourly_demand.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_hourly_demand.png 已保存")

# 1.2 热力图：星期×小时矩阵
fig, ax = plt.subplots(figsize=(14, 6))

heatmap_data = data.pivot_table(
    index='pickup_weekday',
    columns='pickup_hour',
    values='total_amount',
    aggfunc='count'
)
heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)

weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax,
            xticklabels=range(0, 24), yticklabels=weekday_names,
            cbar_kws={'label': '订单量'}, linewidths=0.5, linecolor='white')

ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('星期', fontsize=12)
ax.set_title('星期×小时订单量热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_weekday_hour_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_weekday_hour_heatmap.png 已保存")

# ========================
# 2 区域热度分析
# ========================
print("\n【分析2: 区域热度分析】")

# 2.1 上车区域TOP10
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (col_name, title, color) in enumerate([
    ('PULocationID', '上车区域 TOP 10', 'steelblue'),
    ('DOLocationID', '下车区域 TOP 10', 'coral')
]):
    ax = axes[idx]
    top10 = data[col_name].value_counts().head(10)  # 修复：df_cleaned → data

    bars = ax.barh(range(len(top10)), top10.values, color=color, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([f'区域 {int(loc)}' for loc in top10.index])
    ax.set_xlabel('订单量', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, top10.values):
        ax.text(bar.get_width() + max(top10.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9)

plt.suptitle('上下客区域热度 TOP 10', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/02_top10_locations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_top10_locations.png 已保存")

# 2.2 上车区域TOP5的高峰时段分布
fig, ax = plt.subplots(figsize=(14, 6))

top5_locations = data['PULocationID'].value_counts().head(5).index  # 修复：df_cleaned → data

for loc, color in zip(top5_locations, plt.cm.Set2.colors):
    loc_data = data[data['PULocationID'] == loc].groupby('pickup_hour').size()  # 修复：df_cleaned → data
    loc_data = loc_data.reindex(range(24), fill_value=0)
    ax.plot(loc_data.index, loc_data.values, color=color, linewidth=2,
            marker='o', markersize=5, label=f'区域 {int(loc)}')

ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('订单量', fontsize=12)
ax.set_title('TOP5 上车区域分小时订单分布', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/02_top5_hourly.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_top5_hourly.png 已保存")

# 2.3 区域间流动（OD对TOP15）
fig, ax = plt.subplots(figsize=(12, 8))

od_pairs = data.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')  # 修复：df_cleaned → data
od_pairs = od_pairs.sort_values('count', ascending=False).head(15)

labels = []
for _, row in od_pairs.iterrows():
    labels.append(f"区域{int(row['PULocationID'])} → 区域{int(row['DOLocationID'])}")

colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(od_pairs)))
bars = ax.barh(range(len(od_pairs)), od_pairs['count'].values, color=colors, edgecolor='white')

ax.set_yticks(range(len(od_pairs)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('行程数量', fontsize=12)
ax.set_title('最热门区域间流动 TOP 15 (O→D)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, od_pairs['count'].values):
    ax.text(bar.get_width() + max(od_pairs['count']) * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/02_od_flows.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_od_flows.png 已保存")

# ========================
# 3 车费影响因素分析
# ========================
print("\n【分析3: 车费影响因素分析】")

# 3.1 行程距离-车费散点图
fig, ax = plt.subplots(figsize=(12, 7))

q99_dist = data['trip_distance'].quantile(0.99)
q99_fare = data['fare_amount'].quantile(0.99)
scatter_data = data[
    (data['trip_distance'] <= q99_dist) &
    (data['fare_amount'] <= q99_fare) &
    (data['trip_distance'] > 0)
]

if len(scatter_data) > 100000:
    scatter_data = scatter_data.sample(n=100000, random_state=42)
    print(f"  数据量较大，已采样至100,000条用于绘图")

sc = ax.scatter(
    scatter_data['trip_distance'],
    scatter_data['fare_amount'],
    c=scatter_data['passenger_count'],
    cmap='viridis',
    alpha=0.4,
    s=5,
    vmin=1,
    vmax=6
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('乘客人数', fontsize=11)

ax.set_xlabel('行程距离 (英里)', fontsize=12)
ax.set_ylabel('车费 (美元)', fontsize=12)
ax.set_title(f'行程距离 vs 车费 (n={len(scatter_data):,})', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/03_distance_fare.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_distance_fare.png 已保存")

# 3.2 分时段平均车费对比
fig, ax = plt.subplots(figsize=(14, 6))

for period, color, marker in [('工作日', '#1f77b4', 'o'), ('周末', '#d62728', 's')]:
    period_data = data[data['is_weekend'] == period].groupby('pickup_hour')['fare_amount'].mean()
    period_data = period_data.reindex(range(24))
    ax.plot(period_data.index, period_data.values, color=color, linewidth=2.5,
            marker=marker, markersize=6, label=period)

ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('平均车费 (美元)', fontsize=12)
ax.set_title('分小时平均车费（工作日 vs 周末）', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/03_hourly_fare.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_hourly_fare.png 已保存")

# 3.3 乘客人数对车费的影响
fig, ax = plt.subplots(figsize=(12, 6))

passenger_data = data[
    (data['passenger_count'] >= 1) &
    (data['passenger_count'] <= 6) &
    (data['fare_amount'] <= data['fare_amount'].quantile(0.99))
]

box_data = [passenger_data[passenger_data['passenger_count'] == p]['fare_amount'].values
            for p in range(1, 7)]

bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
colors_box = plt.cm.Set2(np.linspace(0, 1, 6))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means = [d.mean() for d in box_data]
ax.plot(range(1, 7), means, 'ro-', linewidth=2, markersize=8, label='平均值', zorder=5)

ax.set_xlabel('乘客人数', fontsize=12)
ax.set_ylabel('车费 (美元)', fontsize=12)
ax.set_title('乘客人数对车费的影响', fontsize=14, fontweight='bold')
ax.set_xticklabels(range(1, 7))
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/03_passenger_fare.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_passenger_fare.png 已保存")

# ========================
# 4 自选分析：支付方式与费用结构深度洞察
# ========================
print("\n【分析4: 支付方式与费用结构深度洞察】")

# 4.1 支付方式分布
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

payment_names = {1: '信用卡', 2: '现金', 3: '免费', 4: '争议/拒付'}
payment_counts = data['payment_type'].value_counts()  # 修复：df_cleaned → data
labels = [f"{payment_names.get(k, f'类型{k}')}\n({v / len(data) * 100:.1f}%)"  # 修复：df_cleaned → data
          for k, v in payment_counts.items()]
colors_pie = ['#2ecc71', '#3498db', '#95a5a6', '#e74c3c']

axes[0].pie(payment_counts.values, labels=labels, colors=colors_pie,
            autopct='', startangle=90, explode=(0.02, 0.02, 0.02, 0.02))
axes[0].set_title('支付方式分布', fontsize=13, fontweight='bold')

# 右图：小费率对比
payment_tip = data[data['payment_type'].isin([1, 2])].copy()  # 修复：df_cleaned → data
payment_tip['tip_rate'] = payment_tip['tip_amount'] / payment_tip['fare_amount'] * 100
payment_tip = payment_tip[payment_tip['tip_rate'] <= 100]

box_data_tip = [
    payment_tip[payment_tip['payment_type'] == 1]['tip_rate'].values,
    payment_tip[payment_tip['payment_type'] == 2]['tip_rate'].values
]

bp2 = axes[1].boxplot(box_data_tip, patch_artist=True, widths=0.5, labels=['信用卡', '现金'])
bp2['boxes'][0].set_facecolor('#2ecc71')
bp2['boxes'][1].set_facecolor('#3498db')
for box in bp2['boxes']:
    box.set_alpha(0.7)

axes[1].set_ylabel('小费占车费比例 (%)', fontsize=11)
axes[1].set_title('信用卡 vs 现金：小费率对比', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, d in enumerate(box_data_tip):
    median = np.median(d)
    mean = np.mean(d)
    axes[1].annotate(f'中位数: {median:.1f}%\n均值: {mean:.1f}%',
                     xy=(i + 1, median), fontsize=10, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.suptitle('支付方式与费用结构分析', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/04_payment_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_payment_analysis.png 已保存")

# 4.2 费用构成对比
fig, ax = plt.subplots(figsize=(12, 7))

payment_fees = data[data['payment_type'].isin([1, 2])].groupby('payment_type').agg({  # 修复：df_cleaned → data
    'fare_amount': 'mean',
    'tip_amount': 'mean',
    'tolls_amount': 'mean',
    'total_amount': 'mean'
}).round(2)

payment_fees['other_fees'] = payment_fees['total_amount'] - payment_fees['fare_amount'] - payment_fees['tip_amount'] - payment_fees['tolls_amount']

x = np.arange(len(payment_fees))
width = 0.2

categories = ['fare_amount', 'tip_amount', 'tolls_amount', 'other_fees']
colors_stacked = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
labels_stacked = ['基础车费', '小费', '过路费', '其他费用']

for i, (cat, color, label) in enumerate(zip(categories, colors_stacked, labels_stacked)):
    offset = width * (i - 1.5)
    ax.bar(x + offset, payment_fees[cat].values, width,
           color=color, label=label, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(['信用卡', '现金'], fontsize=12)
ax.set_ylabel('平均费用 (美元)', fontsize=12)
ax.set_title('信用卡 vs 现金：费用构成对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)

for i, (payment_type, total) in enumerate(zip(['信用卡', '现金'], payment_fees['total_amount'])):
    ax.text(i, ax.get_ylim()[1] * 0.95, f'总计: ${total:.2f}',
           ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

y_max = max(payment_fees[['fare_amount', 'tip_amount', 'tolls_amount', 'other_fees']].max()) * 1.3
ax.set_ylim(0, y_max)

plt.tight_layout()
plt.savefig('outputs/04_fee_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_fee_structure.png 已保存")

# 4.3 分小时支付方式偏好
fig, ax = plt.subplots(figsize=(14, 6))

for ptype, color, name in [(1, '#2ecc71', '信用卡'), (2, '#3498db', '现金')]:
    hourly_payment = data[data['payment_type'] == ptype].groupby('pickup_hour').size()  # 修复：df_cleaned → data
    hourly_payment = hourly_payment.reindex(range(24), fill_value=0)

    total_hourly = data[data['payment_type'].isin([1, 2])].groupby('pickup_hour').size()  # 修复：df_cleaned → data
    total_hourly = total_hourly.reindex(range(24), fill_value=1)
    ratio = hourly_payment / total_hourly * 100

    ax.plot(ratio.index, ratio.values, color=color, linewidth=2.5, marker='o', markersize=6, label=name)

ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('占比 (%)', fontsize=12)
ax.set_title('分小时信用卡/现金支付占比变化', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('outputs/04_hourly_payment_preference.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_hourly_payment_preference.png 已保存")

print("\n" + "=" * 60)
print("所有图表已保存至 outputs/ 目录")
print("=" * 60)