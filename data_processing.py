import sys
sys.path.append('C:/Users/huawei/PyCharmMiscProject/.venv/Lib/site-packages')
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs('outputs', exist_ok=True)


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath='yellow_tripdata_2023-01.parquet'):
    """1. 数据加载"""
    try:
        df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
        print("=" * 60)
        print("黄色出租车行程数据 2023年1月 - 数据质量报告")
        print("=" * 60)
    except FileNotFoundError:
        print("错误：找不到文件 'yellow_tripdata_2023-01.parquet'")
        print("请确保文件存在于当前工作目录中")
        exit()
    except Exception as e:
        print(f"错误：无法读取Parquet文件 - {e}")
        exit()
    return df

def basic_info(df):
    """2. 基本信息概览：形状、列名、数据类型、内存使用"""
    print("\n【基本信息】")
    print(f"数据集形状: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n内存使用: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

def data_preview(df):
    """3. 数据预览：前5行、后5行、数值列统计描述"""
    print("\n【数据前5行预览】")
    print(df.head().to_string())

    print("\n【数据后5行预览】")
    print(df.tail().to_string())

    # 数据统计描述
    print("\n【数值列统计描述】")
    print(df.describe().to_string())

def missing_value_analysis(df):
    print("\n" + "=" * 60)
    print("缺失值分析")
    print("=" * 60)

    # 计算缺失值数量和比例
    missing_count = df.isnull().sum()
    missing_rate = (df.isnull().sum() / len(df)) * 100

    # 创建缺失值统计表
    missing_df = pd.DataFrame({
        '缺失数量': missing_count,
        '缺失率(%)': missing_rate.round(2)
    })

    # 显示所有列的缺失情况（按缺失率降序排列）
    missing_df_sorted = missing_df.sort_values('缺失率(%)', ascending=False)
    print("\n所有列缺失情况（按缺失率降序）:")
    print(missing_df_sorted[missing_df_sorted['缺失数量'] > 0].to_string())

    # 只显示有缺失的列
    missing_with_nan = missing_df_sorted[missing_df_sorted['缺失数量'] > 0]
    if len(missing_with_nan) > 0:
        print(f"\n共 {len(missing_with_nan)} 列存在缺失值")
    else:
        print("\n未发现缺失值")

    # 缺失值条形图
    try:
        plt.figure(figsize=(10, 6))

        if len(missing_with_nan) > 0:
            # 只显示缺失率前15的列，避免图表过于拥挤
            top_missing = missing_with_nan.head(15)
            bars = plt.barh(top_missing.index, top_missing['缺失率(%)'],
                            color='coral', edgecolor='darkred')
            plt.title('各列缺失率(%) - 前15列', fontsize=14, fontweight='bold')
            plt.xlabel('缺失率 (%)')
            plt.ylabel('列名')
            # 添加数值标签
            for bar, rate in zip(bars, top_missing['缺失率(%)']):
                plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                         f'{rate:.1f}%', ha='left', va='center', fontsize=9)
        else:
            plt.text(0.5, 0.5, '无缺失值', ha='center', va='center',
                     fontsize=14, transform=plt.gca().transAxes)
            plt.title('各列缺失率(%)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('missing_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n缺失值分析图已保存: missing_analysis.png")
    except Exception as e:
        print(f"警告：无法生成缺失值分析图 - {e}")

def outlier_analysis(df):
    """5. 异常值分析：IQR/Z-score检测 + 分类列唯一值 + 重复值检查"""
    print("\n" + "=" * 60)
    print("异常值分析")
    print("=" * 60)

    # 识别数值型列和分类型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    print(f"\n数值型列 ({len(numeric_cols)}个): {numeric_cols[:10]}..." if len(
        numeric_cols) > 10 else f"\n数值型列 ({len(numeric_cols)}个): {numeric_cols}")
    print(f"分类型列 ({len(categorical_cols)}个): {categorical_cols}")
    print(f"日期时间列 ({len(datetime_cols)}个): {datetime_cols}")

    # --- 5.1 数值型列的异常值分析（基于IQR方法）---
    print("\n【数值型列异常值检测 - IQR方法】")
    print("-" * 40)

    # 策略理由：对于出租车行程数据，选择关键业务字段进行异常值检测
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

        # 策略理由：IQR方法基于四分位数，对数据分布不敏感，
        # 适合检测极端值。1.5倍IQR是常用的标准。
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

    # --- 5.2 业务逻辑异常值检查 ---
    print("\n【业务逻辑异常值检查】")
    print("-" * 40)

    # 策略理由：对于出租车数据，需要根据业务常识检查不合理的数据
    # 例如：行程距离为0或负数，费用为负数，乘客数为0等

    # 检查行程距离
    if 'trip_distance' in df.columns:
        negative_distance = (df['trip_distance'] < 0).sum()
        zero_distance = (df['trip_distance'] == 0).sum()
        print(f"\n  trip_distance:")
        print(f"    负数: {negative_distance:,} ({negative_distance / len(df) * 100:.2f}%)")
        print(f"    为0: {zero_distance:,} ({zero_distance / len(df) * 100:.2f}%)")

    # 检查费用相关字段
    fee_cols = ['fare_amount', 'tip_amount', 'tolls_amount', 'total_amount']
    for col in fee_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            zero_count = (df[col] == 0).sum()
            print(f"\n  {col}:")
            print(f"    负数: {negative_count:,} ({negative_count / len(df) * 100:.2f}%)")
            print(f"    为0: {zero_count:,} ({zero_count / len(df) * 100:.2f}%)")

    # 检查乘客数量
    if 'passenger_count' in df.columns:
        invalid_passengers = (df['passenger_count'] <= 0).sum()
        print(f"\n  passenger_count:")
        print(f"    无效（≤0）: {invalid_passengers:,} ({invalid_passengers / len(df) * 100:.2f}%)")
        print(f"    值分布: \n{df['passenger_count'].value_counts().sort_index()}")

    # 检查时间相关字段
    if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
        pickup = pd.to_datetime(df['tpep_pickup_datetime'])
        dropoff = pd.to_datetime(df['tpep_dropoff_datetime'])
        invalid_duration = (dropoff <= pickup).sum()
        print(f"\n  行程时间:")
        print(f"    下车时间≤上车时间: {invalid_duration:,} ({invalid_duration / len(df) * 100:.2f}%)")

    # --- 5.3 分类型列的唯一值分析 ---
    print("\n【分类型列分析】")
    print("-" * 40)

    # 已知的分类列及其标准值
    # 策略理由：基于出租车数据字典，定义各分类列的业务合法值
    # 超出标准值的取值将被标记为异常
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
            print(f"\n  ⚠ 列 '{col}' 不存在于数据集中")
            continue

        print(f"\n{'=' * 50}")
        print(f"列名: {col} ({info['description']})")
        print(f"{'=' * 50}")

        # 计算基本统计
        total_count = len(df)
        null_count = df[col].isnull().sum()
        non_null_count = total_count - null_count

        print(f"总记录数: {total_count:,}")
        print(f"缺失值数: {null_count:,} ({null_count / total_count * 100:.2f}%)")
        print(f"有效记录: {non_null_count:,}")

        # 获取实际值分布
        value_counts = df[col].value_counts().sort_index()
        unique_count = df[col].nunique()

        print(f"唯一值数量: {unique_count}")
        print(f"标准值: {info['standard_values']}")

        # 识别异常值
        standard_set = set(info['standard_values'])
        actual_values = set(df[col].dropna().unique())
        anomalous_values = actual_values - standard_set
        normal_values = actual_values & standard_set

        # 处理可能的类型不匹配（字符串vs数值）
        if col == 'store_and_fwd_flag':
            # 对于字符串列，需要特别处理
            anomalous_values_str = {v for v in actual_values if str(v).strip() not in [str(s) for s in standard_set]}
            normal_values_str = {v for v in actual_values if str(v).strip() in [str(s) for s in standard_set]}
            anomalous_values = anomalous_values_str
            normal_values = normal_values_str

        # 输出标准值分布
        print("\n【标准值分布】")
        total_normal = 0
        for val in sorted(normal_values, key=str):
            count = value_counts.get(val, 0)
            percentage = count / total_count * 100
            meaning = info['value_meanings'].get(val, '')
            print(f"  {val}: {count:,} ({percentage:.2f}%) - {meaning}")
            total_normal += count

        # 输出异常值分布
        if anomalous_values:
            print(f"\n【异常值分布】(不在标准值 {info['standard_values']} 范围内)")
            total_anomalous = 0
            for val in sorted(anomalous_values, key=str):
                count = value_counts.get(val, 0)
                percentage = count / total_count * 100
                meaning = info['value_meanings'].get(val, '未知异常值')
                print(f"  ⚠ {val}: {count:,} ({percentage:.2f}%) - {meaning}")
                total_anomalous += count

            print(f"\n  异常值总计: {total_anomalous:,} ({total_anomalous / total_count * 100:.2f}%)")

            # 存储异常信息
            categorical_anomalies[col] = {
                'anomalous_values': anomalous_values,
                'total_anomalous': total_anomalous,
                'anomalous_rate': total_anomalous / total_count * 100
            }
        else:
            print(f"\n  ✓ 未发现异常值，所有值均在标准范围内")
            categorical_anomalies[col] = {
                'anomalous_values': set(),
                'total_anomalous': 0,
                'anomalous_rate': 0
            }

        # 检查数据质量问题
        if col == 'store_and_fwd_flag':
            # 检查空格问题
            has_leading_space = df[col].astype(str).str.startswith(' ').sum()
            has_trailing_space = df[col].astype(str).str.endswith(' ').sum()
            if has_leading_space > 0 or has_trailing_space > 0:
                print(f"\n  ⚠ 空格问题: 前导空格{has_leading_space}条, 尾随空格{has_trailing_space}条")

    # 生成分类列异常总结
    print("\n" + "=" * 60)
    print("分类列异常值总结")
    print("=" * 60)

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


    # --- 5.4 重复值检查 ---
    print("\n【重复值检查】")
    print("-" * 40)

    # 策略理由：对于出租车行程数据，完全重复可能表示数据重复记录
    duplicate_rows = df.duplicated().sum()
    print(f"完全重复行数: {duplicate_rows:,} ({duplicate_rows / len(df) * 100:.2f}%)")

def clean_data(df):
    """7-8. 数据清洗"""
    print("\n" + "="*60)
    print("数据清洗")
    print("="*60)

    # 创建清洗前的副本用于对比
    df_before = df.copy()
    print(f"\n清洗前数据形状: {df.shape[0]:,} 行 × {df.shape[1]} 列")

    # --- 7.1 去除缺失值 ---
    # 策略理由：
    # 1. 对于出租车行程数据，关键业务字段（如费用、距离）的缺失会影响分析
    # 2. 根据缺失率决定策略：高缺失率的列考虑删除列，低缺失率的行考虑删除行
    print("\n【步骤1: 去除缺失值】")

    # 检查关键列的缺失情况
    key_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                   'trip_distance', 'fare_amount', 'total_amount', 'passenger_count']
    existing_key_columns = [col for col in key_columns if col in df.columns]

    print("关键列缺失情况:")
    for col in existing_key_columns:
        if col in df.columns:
            col_missing = df[col].isnull().sum()
            col_missing_rate = col_missing / len(df) * 100
            print(f"  列 '{col}': 缺失{col_missing:,}行 ({col_missing_rate:.2f}%)")

    # 删除关键列中包含缺失值的行
    # 策略理由：对于出租车行程数据的关键业务字段，缺失的数据无法进行合理推断，选择删除
    rows_before = len(df)
    df_cleaned = df.dropna(subset=existing_key_columns)
    rows_removed_missing = rows_before - len(df_cleaned)
    print(f"\n  因关键列缺失值删除行数: {rows_removed_missing:,}")
    print(f"  删除后数据形状: {df_cleaned.shape[0]:,} 行 × {df_cleaned.shape[1]} 列")

    # 对于非关键列，如果缺失率过高（>50%），考虑删除该列
    # 策略理由：缺失率过高的列提供的信息有限，删除可以简化数据集
    threshold = 0.5  # 50%缺失率阈值
    high_missing_cols = []
    for col in df_cleaned.columns:
        if col not in existing_key_columns:
            missing_rate = df_cleaned[col].isnull().sum() / len(df_cleaned)
            if missing_rate > threshold:
                high_missing_cols.append(col)

    if high_missing_cols:
        print(f"  缺失率>50%的非关键列（已删除）: {high_missing_cols}")
        df_cleaned = df_cleaned.drop(columns=high_missing_cols)

    # --- 7.2 去除业务逻辑异常值 ---
    print("\n【步骤2: 去除业务逻辑异常值】")

    df_cleaned = df_cleaned.copy()

    # 策略理由：基于出租车业务规则，过滤明显不合理的数据
    rows_before_outlier = len(df_cleaned)

    # 2.1 行程距离必须大于0
    if 'trip_distance' in df_cleaned.columns:
        mask_invalid_distance = df_cleaned['trip_distance'] <= 0
        invalid_distance_count = mask_invalid_distance.sum()
        if invalid_distance_count > 0:
            df_cleaned = df_cleaned[~mask_invalid_distance]
            print(f"  删除行程距离≤0的数据: {invalid_distance_count:,}行")

    # 2.2 费用必须合法（fare_amount至少大于0）
    if 'fare_amount' in df_cleaned.columns:
        mask_invalid_fare = df_cleaned['fare_amount'] <= 0
        invalid_fare_count = mask_invalid_fare.sum()
        if invalid_fare_count > 0:
            df_cleaned = df_cleaned[~mask_invalid_fare]
            print(f"  删除费用≤0的数据: {invalid_fare_count:,}行")

    # 2.3 乘客数量必须大于0
    if 'passenger_count' in df_cleaned.columns:
        mask_invalid_passengers = df_cleaned['passenger_count'] <= 0
        invalid_passengers_count = mask_invalid_passengers.sum()
        if invalid_passengers_count > 0:
            df_cleaned = df_cleaned[~mask_invalid_passengers]
            print(f"  删除乘客数≤0的数据: {invalid_passengers_count:,}行")

    # 2.4 下车时间必须晚于上车时间
    if 'tpep_pickup_datetime' in df_cleaned.columns and 'tpep_dropoff_datetime' in df_cleaned.columns:
        pickup = pd.to_datetime(df_cleaned['tpep_pickup_datetime'])
        dropoff = pd.to_datetime(df_cleaned['tpep_dropoff_datetime'])
        mask_invalid_time = dropoff <= pickup
        invalid_time_count = mask_invalid_time.sum()
        if invalid_time_count > 0:
            df_cleaned = df_cleaned[~mask_invalid_time]
            print(f"  删除下车时间≤上车时间的数据: {invalid_time_count:,}行")

    # 2.5 total_amount应该合理（与fare_amount相关）
    if 'total_amount' in df_cleaned.columns and 'fare_amount' in df_cleaned.columns:
        mask_invalid_total = df_cleaned['total_amount'] < df_cleaned['fare_amount']
        invalid_total_count = mask_invalid_total.sum()
        if invalid_total_count > 0:
            # 策略理由：total_amount应≥fare_amount，因为包含税费等额外费用
            print(f"  检测到total_amount < fare_amount的数据: {invalid_total_count:,}行")
            # 注：不直接删除，因为这可能是数据记录问题，根据业务需求决定是否删除

    rows_removed_outlier = rows_before_outlier - len(df_cleaned)
    print(f"\n  因业务逻辑异常删除行数: {rows_removed_outlier:,}")
    print(f"  删除后数据形状: {df_cleaned.shape[0]:,} 行 × {df_cleaned.shape[1]} 列")

    # --- 7.3 去除异常值 ---
    print("\n【步骤3: 处理分类列异常值】")

    # 策略理由：根据数据字典，payment_type的0、5、6为无效值，
    # RatecodeID的99为未知费率，VendorID的6、7为非标准供应商
    # 这些异常值应该被移除或标记

    if 'payment_type' in df_cleaned.columns:
        # 删除payment_type异常值 (0, 5, 6)
        # 策略理由：0表示未知支付方式，5和6在新版数据字典中已不再使用
        anomalous_payment = df_cleaned['payment_type'].isin([0, 5, 6])
        anomalous_payment_count = anomalous_payment.sum()
        if anomalous_payment_count > 0:
            df_cleaned = df_cleaned[~anomalous_payment]
            print(f"  删除payment_type异常值(0,5,6): {anomalous_payment_count:,}行")

    if 'RatecodeID' in df_cleaned.columns:
        # 删除RatecodeID异常值 (99)
        # 策略理由：99表示未知费率类型，无法用于有效分析
        anomalous_rate = df_cleaned['RatecodeID'] == 99
        anomalous_rate_count = anomalous_rate.sum()
        if anomalous_rate_count > 0:
            df_cleaned = df_cleaned[~anomalous_rate]
            print(f"  删除RatecodeID异常值(99): {anomalous_rate_count:,}行")

    if 'VendorID' in df_cleaned.columns:
        # 删除VendorID异常值 (6, 7)
        # 策略理由：标准情况下只有供应商1和2，6和7为非标准供应商
        anomalous_vendor = df_cleaned['VendorID'].isin([6, 7])
        anomalous_vendor_count = anomalous_vendor.sum()
        if anomalous_vendor_count > 0:
            df_cleaned = df_cleaned[~anomalous_vendor]
            print(f"  删除VendorID异常值(6,7): {anomalous_vendor_count:,}行")

    if 'store_and_fwd_flag' in df_cleaned.columns:
        # 清理store_and_fwd_flag的空格问题
        # 策略理由：去除空格确保数据一致性
        df_cleaned['store_and_fwd_flag'] = df_cleaned['store_and_fwd_flag'].str.strip()
        # 删除非标准值
        anomalous_flag = ~df_cleaned['store_and_fwd_flag'].isin(['Y', 'N'])
        anomalous_flag_count = anomalous_flag.sum()
        if anomalous_flag_count > 0:
            df_cleaned = df_cleaned[~anomalous_flag]
            print(f"  删除store_and_fwd_flag异常值: {anomalous_flag_count:,}行")

    # 更新异常值删除计数
    rows_removed_categorical = rows_before_outlier - len(df_cleaned) - rows_removed_outlier
    print(f"\n  因分类列异常删除行数: {rows_removed_categorical:,}")

    # --- 7.4 去除重复值 ---
    print("\n【步骤4: 去除重复值】")
    rows_before_dedup = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    rows_removed_duplicate = rows_before_dedup - len(df_cleaned)
    print(f"  移除重复行数: {rows_removed_duplicate:,}")
    print(f"  去重后数据形状: {df_cleaned.shape[0]:,} 行 × {df_cleaned.shape[1]} 列")

    # --- 7.5 重置索引 ---
    # 策略理由：删除行后需要重置索引，保持索引连续性
    df_cleaned = df_cleaned.reset_index(drop=True)

    # ========================
    # 8. 清洗结果总结
    # ========================
    print("\n" + "="*60)
    print("数据清洗总结")
    print("="*60)

    total_removed = len(df_before) - len(df_cleaned)
    print(f"\n原始数据行数: {len(df_before):,}")
    print(f"清洗后数据行数: {len(df_cleaned):,}")
    print(f"总共删除行数: {total_removed:,} ({total_removed/len(df_before)*100:.2f}%)")
    print(f"  - 因缺失值删除: {rows_removed_missing:,}行")
    print(f"  - 因业务逻辑异常删除: {rows_removed_outlier:,}行")
    print(f"  - 因重复值删除: {rows_removed_duplicate:,}行")

    print("\n清洗后数据预览:")
    print(df_cleaned.head().to_string())
    stats = {
        'rows_before': len(df_before),
        'rows_after': len(df_cleaned),
        'removed_missing': rows_removed_missing,
        'removed_outlier': rows_removed_outlier,
        'removed_duplicate': rows_removed_duplicate
    }
    return df_cleaned, stats


def save_cleaned_data(df_cleaned, filepath='yellow_tripdata_2023-01_cleaned.parquet'):
    """9. 保存清洗后的数据"""
    output_filename = 'yellow_tripdata_2023-01_cleaned.parquet'
    try:
        df_cleaned.to_parquet(output_filename, index=False)
        print(f"\n清洗后数据已保存至: {output_filename}")
    except Exception as e:
        print(f"\n错误：无法保存Parquet文件 - {e}")
        # 如果Parquet保存失败，尝试保存为CSV
        try:
            csv_filename = 'yellow_tripdata_2023-01_cleaned.csv'
            df_cleaned.to_csv(csv_filename, index=False)
            print(f"已保存为CSV格式: {csv_filename}")
        except Exception as e2:
            print(f"错误：也无法保存为CSV - {e2}")

def generate_summary_report(df_before, df_after, stats):
    """10. 生成数据质量报告摘要"""
    print("\n" + "="*60)
    print("数据质量报告摘要")
    print("="*60)

    # 计算清洗后的数据统计
    cleaned_stats = {
        '指标': [
            '原始行数',
            '原始列数',
            '存在缺失值的列数',
            '缺失行总数',
            '异常值行数',
            '重复行数',
            '清洗后行数',
            '数据完整率(%)'
        ],
        '数值': [
            f"{len(df_before):,}",
            len(df_before.columns),
            sum(1 for col in df_before.columns if df_before[col].isnull().sum() > 0),
            f"{stats['removed_missing']:,}",      # 从字典取
            f"{stats['removed_outlier']:,}",      # 从字典取
            f"{stats['removed_duplicate']:,}",    # 从字典取
            f"{len(df_after):,}",
            round(len(df_after)/len(df_before)*100, 2)
        ]
    }

    summary_df = pd.DataFrame(cleaned_stats)
    print(summary_df.to_string(index=False))

def run_full_pipeline(input_file='yellow_tripdata_2023-01.parquet'):
    """一键运行全部10个步骤"""
    # 1. 加载
    df = load_data(input_file)
    # 2. 基本信息
    basic_info(df)
    # 3. 预览
    data_preview(df)
    # 4. 缺失值
    missing_value_analysis(df)
    # 5. 异常值
    outlier_analysis(df)
    # 7-8. 清洗
    df_cleaned, stats = clean_data(df)
    # 9. 保存
    save_cleaned_data(df_cleaned)
    # 10. 报告
    generate_summary_report(df, df_cleaned, stats)
    return df_cleaned


if __name__ == '__main__':
    df_cleaned = run_full_pipeline()
    print("\n" + "="*60)
    print("数据质量分析和清洗完成！")
    print("="*60)