import sys

sys.path.append('C:/Users/huawei/PyCharmMiscProject/.venv/Lib/site-packages')

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import re
import pickle
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('outputs', exist_ok=True)

print("\n" + "=" * 60)
print("纽约出租车数据问答系统")
print("=" * 60)

# ========================
# 1. 数据加载与初始化
# ========================
print("\n正在加载数据...")
data = pd.read_parquet('yellow_tripdata_2023-01_cleaned.parquet')

# 时间特征
data['pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['pickup_hour'] = data['pickup_datetime'].dt.hour
data['pickup_weekday'] = data['pickup_datetime'].dt.weekday
data['pickup_date'] = data['pickup_datetime'].dt.date
data['is_weekend'] = data['pickup_weekday'].isin([5, 6])

# 构建区域需求量数据
print("正在构建需求模型...")
demand = data.groupby(['PULocationID', 'pickup_hour', 'pickup_weekday']).size().reset_index(name='demand')
demand['hour_sin'] = np.sin(2 * np.pi * demand['pickup_hour'] / 24)
demand['hour_cos'] = np.cos(2 * np.pi * demand['pickup_hour'] / 24)
demand['weekday_sin'] = np.sin(2 * np.pi * demand['pickup_weekday'] / 7)
demand['weekday_cos'] = np.cos(2 * np.pi * demand['pickup_weekday'] / 7)
demand['is_weekend'] = demand['pickup_weekday'].isin([5, 6]).astype(int)
demand['is_rush_hour'] = demand['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

le = LabelEncoder()
demand['location_encoded'] = le.fit_transform(demand['PULocationID'])

feature_cols = ['location_encoded', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend', 'is_rush_hour']
X = demand[feature_cols].values
y = demand['demand'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 训练随机森林（快速）
print("正在训练预测模型...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)
rf_model.fit(X, y.ravel())

print("系统初始化完成！\n")


# ========================
# 2. 核心分析函数
# ========================

def query_hourly_demand(data, location_id=None, day_type='全部'):
    """M4-1: 时段需求量查询"""
    if location_id:
        query_data = data[data['PULocationID'] == int(location_id)]
        title_loc = f"区域 {location_id}"
    else:
        query_data = data
        title_loc = "全区域"

    if day_type == '工作日':
        query_data = query_data[~query_data['is_weekend']]
        title_type = '工作日'
    elif day_type == '周末':
        query_data = query_data[query_data['is_weekend']]
        title_type = '周末'
    else:
        title_type = '全部'

    hourly = query_data.groupby('pickup_hour').size()
    hourly = hourly.reindex(range(24), fill_value=0)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly.index, hourly.values, color='steelblue', edgecolor='white', alpha=0.8)
    ax.plot(hourly.index, hourly.values, 'ro-', markersize=5, linewidth=2)
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('订单量', fontsize=12)
    ax.set_title(f'{title_loc} - {title_type} 分时段需求量', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis='y', alpha=0.3)

    peak_hour = hourly.idxmax()
    peak_value = hourly.max()

    filepath = 'outputs/qa_hourly_demand.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'peak_hour': peak_hour,
        'peak_value': int(peak_value),
        'avg_demand': round(hourly.mean(), 1),
        'morning_peak': f"{hourly[6:10].idxmax()}时 ({hourly[6:10].max()}单)",
        'evening_peak': f"{hourly[16:20].idxmax()}时 ({hourly[16:20].max()}单)",
        'chart': filepath
    }


def query_location_ranking(data, rank_type='上车', top_n=10):
    """M4-2: 区域热度排名"""
    col = 'PULocationID' if rank_type == '上车' else 'DOLocationID'
    title = '上车' if rank_type == '上车' else '下车'

    top = data[col].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))
    bars = ax.barh(range(len(top)), top.values, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f'区域 {int(loc)}' for loc in top.index])
    ax.set_xlabel('订单量', fontsize=12)
    ax.set_title(f'{title}量 TOP {top_n} 区域', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + max(top.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=10)

    filepath = 'outputs/qa_location_ranking.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'top_list': [(int(loc), int(cnt)) for loc, cnt in zip(top.index, top.values)],
        'total_locations': data[col].nunique(),
        'chart': filepath
    }


def predict_demand(location_id, hour, weekday, demand, le, rf_model):
    """M4-3: 预测指定区域时段的出行需求量"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    is_weekend = 1 if weekday in [5, 6] else 0
    is_rush = 1 if hour in [7, 8, 9, 17, 18, 19] else 0

    # 处理未知区域
    if location_id in le.classes_:
        loc_encoded = le.transform([location_id])[0]
    else:
        loc_encoded = 0

    features = np.array([[loc_encoded, hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend, is_rush]])
    pred = rf_model.predict(features)[0]

    # 获取历史数据作为参考
    hist_data = demand[(demand['PULocationID'] == location_id) &
                       (demand['pickup_hour'] == hour) &
                       (demand['pickup_weekday'] == weekday)]
    hist_avg = hist_data['demand'].mean() if len(hist_data) > 0 else None

    weekdays_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

    return {
        'location': location_id,
        'hour': hour,
        'weekday': weekdays_cn[weekday],
        'predicted_demand': round(pred, 1),
        'historical_avg': round(hist_avg, 1) if hist_avg else '无历史数据',
        'is_rush_hour': '是' if is_rush else '否',
        'is_weekend': '是' if is_weekend else '否'
    }


def query_route_info(data, location_from=None, location_to=None):
    """M4-4: 路线信息查询（费用、距离、时间）"""
    query = data
    conditions = []
    desc_parts = []

    if location_from:
        query = query[query['PULocationID'] == int(location_from)]
        conditions.append(f"出发区域{location_from}")
    if location_to:
        query = query[query['DOLocationID'] == int(location_to)]
        conditions.append(f"到达区域{location_to}")

    if len(query) == 0:
        return {'error': '未找到匹配的行程数据'}

    result = {
        'trip_count': len(query),
        'avg_distance': round(query['trip_distance'].mean(), 2),
        'avg_fare': round(query['fare_amount'].mean(), 2),
        'avg_total': round(query['total_amount'].mean(), 2),
        'avg_duration_min': round((pd.to_datetime(query['tpep_dropoff_datetime']) - pd.to_datetime(
            query['tpep_pickup_datetime'])).dt.total_seconds().mean() / 60, 1),
        'avg_tip': round(query['tip_amount'].mean(), 2),
        'description': '、'.join(conditions) if conditions else '全路线'
    }

    return result


def query_payment_analysis(data):
    """M4-5: 支付方式与费用分析"""
    payment_names = {1: '信用卡', 2: '现金', 3: '免费', 4: '争议/拒付'}

    # 支付方式分布
    payment_counts = data['payment_type'].value_counts()

    # 各支付方式费用统计
    payment_stats = data.groupby('payment_type').agg({
        'fare_amount': 'mean',
        'tip_amount': 'mean',
        'total_amount': 'mean'
    }).round(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 饼图
    labels = [f"{payment_names.get(k, f'类型{k}')}\n({v / len(data) * 100:.1f}%)"
              for k, v in payment_counts.items()]
    colors_pie = ['#2ecc71', '#3498db', '#95a5a6', '#e74c3c']
    axes[0].pie(payment_counts.values, labels=labels, colors=colors_pie, startangle=90,
                explode=(0.02, 0.02, 0.02, 0.02))
    axes[0].set_title('支付方式分布', fontsize=13, fontweight='bold')

    # 柱状图：平均费用对比
    x = np.arange(len(payment_stats))
    width = 0.25
    axes[1].bar(x - width, payment_stats['fare_amount'], width, label='平均车费', color='#3498db')
    axes[1].bar(x, payment_stats['tip_amount'], width, label='平均小费', color='#2ecc71')
    axes[1].bar(x + width, payment_stats['total_amount'], width, label='平均总费用', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([payment_names.get(k, f'类型{k}') for k in payment_stats.index])
    axes[1].set_ylabel('金额 (美元)', fontsize=11)
    axes[1].set_title('各支付方式费用对比', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    filepath = 'outputs/qa_payment_analysis.png'
    plt.suptitle('支付方式综合分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'payment_distribution': {payment_names.get(k, f'类型{k}'): f"{v / len(data) * 100:.1f}%"
                                 for k, v in payment_counts.items()},
        'credit_card_tip_avg': round(data[data['payment_type'] == 1]['tip_amount'].mean(), 2),
        'cash_tip_avg': round(data[data['payment_type'] == 2]['tip_amount'].mean(), 2),
        'chart': filepath
    }


def query_peak_hours(data, location_id=None):
    """M4-6: 高峰时段分析"""
    if location_id:
        query_data = data[data['PULocationID'] == int(location_id)]
        title = f"区域 {location_id}"
    else:
        query_data = data
        title = "全区域"

    # 工作日和周末分别分析
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (day_type, day_label, color) in zip(axes, [
        (~query_data['is_weekend'], '工作日', 'steelblue'),  # ~True = False（工作日）
        (query_data['is_weekend'], '周末', 'coral')
    ]):
        hourly = query_data[day_type].groupby('pickup_hour').size()
        hourly = hourly.reindex(range(24), fill_value=0)
        ax.bar(hourly.index, hourly.values, color=color, edgecolor='white', alpha=0.8)
        ax.plot(hourly.index, hourly.values, 'ko-', markersize=4, linewidth=1.5)
        ax.set_xlabel('小时', fontsize=11)
        ax.set_ylabel('订单量', fontsize=11)
        ax.set_title(f'{title} - {day_label}', fontsize=13, fontweight='bold')
        ax.set_xticks(range(0, 24, 3))
        ax.grid(axis='y', alpha=0.3)

    filepath = 'outputs/qa_peak_hours.png'
    plt.suptitle(f'{title} 高峰时段分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return {'chart': filepath, 'location': title}


# ========================
# 3. 自然语言理解模块
# ========================
def parse_query(text):
    """解析用户自然语言查询，提取关键词和参数"""
    text = text.lower().strip()
    result = {
        'intent': None,
        'location': None,
        'time': None,
        'day_type': None,
        'hour': None,
        'weekday': None,
        'top_n': 10,
        'rank_type': '上车',
        'route_from': None,
        'route_to': None
    }

    # 提取区域编号
    loc_match = re.search(r'区域\s*(\d+)', text)
    if loc_match:
        result['location'] = int(loc_match.group(1))

    # 提取OD对
    od_from = re.search(r'从区域\s*(\d+)', text)
    od_to = re.search(r'(?:到|去|至)区域\s*(\d+)', text)
    if od_from:
        result['route_from'] = int(od_from.group(1))
    if od_to:
        result['route_to'] = int(od_to.group(1))

    # 提取小时
    hour_match = re.search(r'(\d{1,2})\s*点|(\d{1,2})\s*时|(\d{1,2})\s*:00', text)
    if hour_match:
        hour_val = next(h for h in hour_match.groups() if h is not None)
        result['hour'] = int(hour_val)

    # 提取星期
    weekdays_cn = {'周一': 0, '周二': 1, '周三': 2, '周四': 3, '周五': 4, '周六': 5, '周日': 6,
                   '星期一': 0, '星期二': 1, '星期三': 2, '星期四': 3, '星期五': 4, '星期六': 5, '星期日': 6}
    for cn, num in weekdays_cn.items():
        if cn in text:
            result['weekday'] = num
            break

    # 提取TOP N
    top_match = re.search(r'前\s*(\d+)|top\s*(\d+)', text, re.IGNORECASE)
    if top_match:
        n_val = next(n for n in top_match.groups() if n is not None)
        result['top_n'] = int(n_val)

    # 判断意图
    if any(w in text for w in ['预测', '预计', '预估', '会有多少', '需求量']):
        result['intent'] = 'predict'
    elif any(w in text for w in ['排名', '排行', 'top', '热门', '热度']):
        result['intent'] = 'ranking'
        if '下车' in text or '下客' in text or '到达' in text:
            result['rank_type'] = '下车'
    elif any(w in text for w in ['高峰', '繁忙', '最忙', '峰值']):
        result['intent'] = 'peak_hours'
    elif any(w in text for w in ['时段', '小时', '每小时', '分小时', '几点']):
        result['intent'] = 'hourly_demand'
    elif any(w in text for w in ['费用', '车费', '价格', '花费', '多少钱', '距离', '时间']):
        result['intent'] = 'route_info'
    elif any(w in text for w in ['支付', '信用卡', '现金', '小费', '付款']):
        result['intent'] = 'payment'
    elif any(w in text for w in ['区域', '哪里', '哪个']):
        result['intent'] = 'hourly_demand'

    # 判断工作日/周末
    if any(w in text for w in ['工作日', '平日', '上班日']):
        result['day_type'] = '工作日'
    elif any(w in text for w in ['周末', '休息日', '假日']):
        result['day_type'] = '周末'

    return result


# ========================
# 4. 回答生成模块
# ========================
def generate_answer(intent, params, data, demand, le, rf_model):
    """根据意图调用对应函数并生成回答"""

    if intent == 'hourly_demand':
        r = query_hourly_demand(data, params['location'], params['day_type'])
        return (f"时段需求分析结果：\n"
                f"   • 全天高峰时段：{r['peak_hour']}:00（{r['peak_value']:,}单）\n"
                f"   • 早高峰：{r['morning_peak']}\n"
                f"   • 晚高峰：{r['evening_peak']}\n"
                f"   • 平均每小时：{r['avg_demand']}单\n"
                f"  图表已保存：{r['chart']}")


    elif intent == 'ranking':

        r = query_location_ranking(data, params['rank_type'], params['top_n'])
        top_str = '\n'.join([f"   {i + 1}. 区域{loc}：{cnt:,}单" for i, (loc, cnt) in enumerate(r['top_list'])])
        return (f"🏆 区域热度排名（{params['rank_type']}量 TOP {params['top_n']}）：\n{top_str}\n"
                f"   📈 图表已保存：{r['chart']}")

    elif intent == 'predict':
        if params['location'] is None or params['hour'] is None or params['weekday'] is None:
            return "⚠️ 需求预测需要指定：区域编号、时间和星期。例如：'预测区域100 周一下午5点的需求量'"
        r = predict_demand(params['location'], params['hour'], params['weekday'], demand, le, rf_model)
        return (f"🔮 需求预测结果：\n"
                f"   • 区域{r['location']} {r['weekday']} {r['hour']}:00\n"
                f"   • 预测需求量：{r['predicted_demand']}单\n"
                f"   • 历史平均值：{r['historical_avg']}单\n"
                f"   • 高峰时段：{r['is_rush_hour']} | 周末：{r['is_weekend']}")

    elif intent == 'route_info':
        r = query_route_info(data, params['route_from'], params['route_to'])
        if 'error' in r:
            return f"⚠️ {r['error']}"
        return (f"🚕 路线信息（{r['description']}）：\n"
                f"   • 匹配行程：{r['trip_count']:,}条\n"
                f"   • 平均距离：{r['avg_distance']}英里\n"
                f"   • 平均车费：${r['avg_fare']}\n"
                f"   • 平均总费用：${r['avg_total']}\n"
                f"   • 平均时长：{r['avg_duration_min']}分钟\n"
                f"   • 平均小费：${r['avg_tip']}")

    elif intent == 'payment':
        r = query_payment_analysis(data)
        return (f"支付方式分析：\n"
                f"   • 分布：{r['payment_distribution']}\n"
                f"   • 信用卡平均小费：${r['credit_card_tip_avg']}\n"
                f"   • 现金平均小费：${r['cash_tip_avg']}\n"
                f"   图表已保存：{r['chart']}")

    elif intent == 'peak_hours':
        r = query_peak_hours(data, params['location'])
        return (f"高峰时段分析完成！\n   图表已保存：{r['chart']}")

    else:
        return ("抱歉，我无法理解您的问题。请尝试以下提问方式：\n"
                "   • '区域100 工作日 分时段需求量'\n"
                "   • '上车量 TOP 5 区域排名'\n"
                "   • '预测区域100 周二 下午3点的需求'\n"
                "   • '从区域100到区域200的费用和距离'\n"
                "   • '支付方式和小费分析'\n"
                "   • '周五 全天高峰时段'")


# ========================
# 5. 命令行交互循环
# ========================
def run_qa_system(data, demand, rf_model, le):
    """命令行交互问答循环 - 供 main.py 调用"""

    print(f"列名: {list(data.columns)}")
    print(f"is_weekend 是否存在: {'is_weekend' in data.columns}")
    if 'is_weekend' in data.columns:
        print(f"is_weekend 类型: {data['is_weekend'].dtype}")
        print(f"is_weekend 唯一值: {data['is_weekend'].unique()[:5]}")
    print("\n" + "=" * 60)
    print("欢迎使用纽约出租车数据问答系统！")
    print("=" * 60)
    print("\n支持的问题类型：")
    print("  1. 时段查询  - '区域100 工作日分时段有多少订单'")
    print("  2. 区域排名  - '上车量排名前10的区域'")
    print("  3. 需求预测  - '预测区域100 周一下午5点需求量'")
    print("  4. 路线查询  - '从区域100到区域200的费用'")
    print("  5. 支付分析  - '各种支付方式的小费对比'")
    print("  6. 高峰分析  - '周五哪些时段最繁忙'")
    print("\n输入 'quit' 或 'exit' 退出系统")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n请输入您的问题：").strip()

            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n感谢使用，再见！")
                break

            if not user_input:
                continue

            # 解析问题
            params = parse_query(user_input)

            # 如果没有明确意图，尝试推断
            if params['intent'] is None:
                params['intent'] = 'hourly_demand'  # 默认时段查询

            # 生成回答
            answer = generate_answer(params['intent'], params, data, demand, le, rf_model)
            print(f"\n{answer}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n⚠️ 处理出错：{e}")
            print("请尝试用其他方式提问，或输入 'quit' 退出")


if __name__ == '__main__':
    # 独立运行时：加载数据、初始化模型、启动问答
    data = pd.read_parquet('yellow_tripdata_2023-01_cleaned.parquet')
    # ... 初始化代码 ...
    run_qa_system(data, demand, rf_model, le)