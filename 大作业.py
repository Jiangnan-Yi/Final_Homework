import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


## ======== M1 数据处理 ========
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
print("出租车区域对照表 - 数据质量报告\n")

print("\n【数据前5行预览】")
print(df.head().to_string())

# ------缺失值分析------
print("缺失值分析")
# 计算缺失值数量和比例
missing_count = df.isnull().sum()
missing_rate = (df.isnull().sum() / len(df)) * 100

# 创建缺失值统计表
missing_df = pd.DataFrame({
    '缺失数量': missing_count,
    '缺失率(%)': missing_rate.round(2)
})

# 只显示有缺失的列
missing_with_nan = missing_df[missing_df['缺失数量'] > 0]
if len(missing_with_nan) > 0:
    print("\n存在缺失值的列：")
    print(missing_with_nan)
else:
    print("\n未发现缺失值")

# 缺失值条形图
fig, ax = plt.subplots(figsize=(10, 5))
if len(missing_with_nan) > 0:
    bars = ax.bar(missing_with_nan.index, missing_with_nan['缺失率(%)'],
                   color='#A35488', edgecolor='darkred')
    ax.set_title('各列缺失率(%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('缺失率 (%)')
    max_rate = missing_with_nan['缺失率(%)'].max()
    ax.set_ylim(0, max_rate * 1.15)
    ax.set_xlabel('列名')
    ax.tick_params(axis='x', rotation=45)
    # 添加数值标签
    for bar, rate in zip(bars, missing_with_nan['缺失率(%)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
else:
    ax.text(0.5, 0.5, '无缺失值', ha='center', va='center',
            fontsize=14, transform=ax.transAxes)
    ax.set_title('各列缺失率(%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('missing_analysis.png', dpi=150)
plt.show()

# ------ 异常值分析（基于IQR方法）------
print("异常值分析")
# 识别数值型列和分类型列
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

print(f"\n数值型列 ({len(numeric_cols)}个): {numeric_cols[:10]}..." if len(numeric_cols) > 10 else f"\n数值型列 ({len(numeric_cols)}个): {numeric_cols}")
print(f"分类型列 ({len(categorical_cols)}个): {categorical_cols}")
print(f"日期时间列 ({len(datetime_cols)}个): {datetime_cols}")

# --- 数值型列的异常值分析（基于IQR方法）---
print("\n【数值型列异常值检测 - IQR方法】")
# 对于出租车行程数据，选择关键业务字段进行异常值检测
# 关键的连续数值字段，这些字段应该有合理的业务范围
key_numeric_cols = [
    'trip_distance', 'fare_amount', 'tip_amount', 'tolls_amount',
    'total_amount', 'passenger_count', 'extra', 'mta_tax',
    'improvement_surcharge', 'congestion_surcharge', 'airport_fee'
]

# 只分析存在的列
existing_key_cols = [col for col in key_numeric_cols if col in numeric_cols]

iqr_outliers_summary = {}

for col in existing_key_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # IQR方法基于四分位数，对数据分布不敏感，适合检测极端值。1.5倍IQR是常用的标准。
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_rate = (outlier_count / len(df)) * 100

    iqr_outliers_summary[col] = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        '下界': lower_bound, '上界': upper_bound,
        '异常值数量': outlier_count,
        '异常值比例(%)': round(outlier_rate, 2)
    }

    print(f"\n  {col}:")
    print(f"    Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"    界限: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"    异常值数量: {outlier_count:,} ({outlier_rate:.2f}%)")

# --- 业务逻辑异常值检查 ---
print("\n【业务逻辑异常值检查】")

# 对于出租车数据，需要根据业务常识检查不合理的数据
# 检查行程距离
if 'trip_distance' in df.columns:
    negative_distance = (df['trip_distance'] < 0).sum()
    zero_distance = (df['trip_distance'] == 0).sum()
    print(f"\n  trip_distance:")
    print(f"    负数: {negative_distance:,} ({negative_distance/len(df)*100:.2f}%)")
    print(f"    为0: {zero_distance:,} ({zero_distance/len(df)*100:.2f}%)")

# 检查费用相关字段
fee_cols = ['fare_amount', 'tip_amount', 'tolls_amount', 'total_amount']
for col in fee_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        zero_count = (df[col] == 0).sum()
        print(f"\n  {col}:")
        print(f"    负数: {negative_count:,} ({negative_count/len(df)*100:.2f}%)")
        print(f"    为0: {zero_count:,} ({zero_count/len(df)*100:.2f}%)")

# 检查乘客数量
if 'passenger_count' in df.columns:
    invalid_passengers = (df['passenger_count'] <= 0).sum()
    print(f"\n  passenger_count:")
    print(f"    无效（≤0）: {invalid_passengers:,} ({invalid_passengers/len(df)*100:.2f}%)")
    print(f"    值分布: \n{df['passenger_count'].value_counts().sort_index()}")

# 检查时间相关字段
if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
    pickup = pd.to_datetime(df['tpep_pickup_datetime'])
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'])
    invalid_duration = (dropoff <= pickup).sum()
    print(f"\n  行程时间:")
    print(f"    下车时间≤上车时间: {invalid_duration:,} ({invalid_duration/len(df)*100:.2f}%)")

# --- 分类型列的唯一值分析 ---
print("\n【分类型列分析】")

categorical_columns_info = {
    'VendorID': {
        'standard_values': [1, 2],  # 1=Creative Mobile Technologies, 2=VeriFone Inc.
        'description': '供应商ID',
        'value_meanings': {
            1: 'Creative Mobile Technologies',
            2: 'VeriFone Inc.',
            6: '未知供应商(异常)',
            7: '未知供应商(异常)'
        }
    },
    'RatecodeID': {
        'standard_values': [1, 2, 3, 4, 5, 6],  # 1-6为标准费率类型
        'description': '费率类型ID',
        'value_meanings': {
            1: '标准费率',
            2: 'JFK机场',
            3: '纽瓦克机场',
            4: 'Nassau或Westchester',
            5: '议价费率',
            6: '团体行程',
            99: '未知费率(异常)'
        }
    },
    'store_and_fwd_flag': {
        'standard_values': ['Y', 'N'],  # Y=存储转发, N=实时发送
        'description': '存储转发标志',
        'value_meanings': {
            'Y': '存储后转发',
            'N': '实时发送'
        }
    },
    'payment_type': {
        'standard_values': [1, 2, 3, 4],  # 1-4为标准化支付方式
        'description': '支付方式',
        'value_meanings': {
            0: '未知支付方式(异常)',
            1: '信用卡',
            2: '现金',
            3: '无费用',
            4: '争议/拒付',
            5: '未知(异常)',
            6: '无效行程(异常)'
        }
    }
}

# 存储异常值统计
categorical_anomalies = {}

# 对每个已知的分类列进行分析
for col, info in categorical_columns_info.items():
    if col not in df.columns:
        continue

    # 识别异常值
    standard_set = set(info['standard_values'])
    actual_values = set(df[col].dropna().unique())
    anomalous_values = actual_values - standard_set

    # 计算异常值总数
    total_anomalous = 0
    if anomalous_values:
        for val in anomalous_values:
            total_anomalous += (df[col] == val).sum()

    categorical_anomalies[col] = {
        'description': info['description'],
        'anomalous_values': anomalous_values,
        'total_anomalous': total_anomalous,
        'anomalous_rate': total_anomalous / len(df) * 100
    }

# 生成分类列异常总结
total_anomalous_rows = 0
for col, anomaly_info in categorical_anomalies.items():
    if anomaly_info['total_anomalous'] > 0:
        print(f"\n{col} ({categorical_columns_info[col]['description']}):")
        print(f"  异常值: {anomaly_info['anomalous_values']}")
        print(f"  异常记录数: {anomaly_info['total_anomalous']:,}")
        print(f"  异常占比: {anomaly_info['anomalous_rate']:.2f}%")
        total_anomalous_rows += anomaly_info['total_anomalous']
    else:
        print(f"\n{col} ({categorical_columns_info[col]['description']}): ✓ 无异常")

if total_anomalous_rows > 0:
    print(f"\n⚠ 总计发现 {total_anomalous_rows:,} 条分类列异常记录")
else:
    print(f"\n✓ 所有分类列数据正常，未发现异常")