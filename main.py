import sys
sys.path.append('C:/Users/huawei/PyCharmMiscProject/.venv/Lib/site-packages')
import pandas as pd
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import data_processing
import visualization
import prediction_model
import qa_system



def main():
    # 1. 数据处理
    df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
    df_cleaned, stats = data_processing.clean_data(df)

    # 2. 特征提取
    data = visualization.extract_time_features(df_cleaned)

    # 3. 可视化（可选）
    visualization.run_all_visualizations(data)

    # 4. 训练模型
    model, demand, le = prediction_model.train_prediction_model(data)

    # 5. 问答系统
    qa_system.run_qa_system(data, demand, model, le)


if __name__ == '__main__':
    main()