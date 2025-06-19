# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import random
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from sklearn.preprocessing import StandardScaler  
from torch.utils.data import Dataset, DataLoader  
import pytorch_lightning as pl  
import math  
import matplotlib.pyplot as plt 
import os
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
# 配置硬件加速  
torch.set_float32_matmul_precision('medium')  
torch.backends.cudnn.benchmark = True  
def set_seed(seed=42):  
    """全面固定随机种子"""  
    # Python 内置随机数  
    random.seed(seed)  
    
    # NumPy随机数  
    np.random.seed(seed)  
    
    # PyTorch随机数  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  # 多GPU  
    
    # CUDA随机性  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    
    # 环境变量  
    os.environ['PYTHONHASHSEED'] = str(seed)  


import mydata 
import mymodel 
import train_eval
import run_train
# 全局调用  
set_seed(2011)  # 或任何固定值   
df = pd.read_csv('data.csv',encoding='gbk')
feature_cols=[  
            'O浓度', 'N浓度', '水温', '总悬浮物', 'COD', 'pH', '氨氮',  
            '平均气压', '最高气压', '最低气压', '平均温度', '最高气温',  
            '最低气温', '平均相对湿度', '最小湿度', '降水量', '日平均风速',  
            '日照时数', '病例_2w_ma', '病例_周环比', 'O病毒_3w趋势','N病毒_3w趋势'  
        ]
processor = mydata.PandemicDataProcessor(  
    feature_cols=feature_cols,  
    data=df,  
    lookback=8  
)  
# 设置字体为SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 2. 训练模型  
model, results = run_train.run_training_example(processor, batch_size=128, epochs=5000)  

# 加载模型权重
# 使用相对路径

# 2. 初始化数据处理器
# processor = PandemicDataProcessor(
#     feature_cols=feature_cols,
#     data_path=df,
#     lookback=8
# )

# 3. 创建数据加载器
train_loader = mydata.DataLoader(processor.train_dataset, batch_size=32, shuffle=False)
val_loader = mydata.DataLoader(processor.val_dataset, batch_size=32, shuffle=False)
test_loader = mydata.DataLoader(processor.test_dataset, batch_size=32, shuffle=False)

# 4. 加载模型
model = mymodel.PandemicDualModel(
    input_channels=processor.feature_dim,
    seq_len=processor.lookback,
    site_count=processor.num_sites,
    site_embed_dim=4
)
model_path = r'.\runs\week1final\best_model.pt'
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置模型为评估模式
# 3. 进行预测的函数
def make_predictions(model, data_loader):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    model.eval()  # 将模型设置为评估模式
    y_true = []
    y_pred = []
    y_mean_true = []
    y_resid_true = []
    y_mean_pred = []
    y_resid_pred = []
    current_week = []
    with torch.no_grad():
        for X, y_mean, y_resid,y_trues, site_ids,weeks in data_loader:
            X = X.to(device)
            site_ids = site_ids.to(device)
            y_trues = y_trues.to(device)
            mean_pred, resid_pred = model(X, site_ids)
            total_pred = mean_pred + resid_pred
            current_week.extend(weeks.cpu().numpy())
            # 收集真实值和预测值
            y_true.extend(y_trues.cpu().numpy())
            y_pred.extend(total_pred.cpu().numpy())
            
            # 收集均值和残差的真实值和预测值
            y_mean_true.extend(y_mean.cpu().numpy())
            y_resid_true.extend(y_resid.cpu().numpy())
            y_mean_pred.extend(mean_pred.cpu().numpy())
            y_resid_pred.extend(resid_pred.cpu().numpy())
      # 转换为数组
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    # 计算指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true,y_pred)
    
    print(f"评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"总MAPE: {mape:.4f}")

    return y_true,y_pred,current_week

# 4. 分别对训练集、验证集和测试集进行预测
train_true,train_predictions,train_week = make_predictions(model, train_loader)
val_true,val_predictions,val_week = make_predictions(model, val_loader)
test_true,test_predictions,test_week = make_predictions(model, test_loader)

# 5. 输出预测结果
train_pred_df = pd.DataFrame({
    'Cases': train_true.flatten(),  # 确保是一维数组
    'Predicted': train_predictions.flatten(),  # 确保是一维数组
    'week':train_week
})
val_pred_df = pd.DataFrame({
    'Cases': val_true.flatten(),  # 确保是一维数组
    'Predicted': val_predictions.flatten(),  # 确保是一维数组
    'week':val_week
})
test_pred_df = pd.DataFrame({
    'Cases': test_true.flatten(),  # 确保是一维数组
    'Predicted': test_predictions.flatten(),  # 确保是一维数组
    'week':test_week
})

# 保存预测结果到 CSV 文件
train_pred_df.to_csv('train_predictions.csv', index=False)
val_pred_df.to_csv('val_predictions.csv', index=False)
test_pred_df.to_csv('test_predictions.csv', index=False)

# 打印部分预测结果
print("训练集预测结果：")
print(train_pred_df.head())
print("\n验证集预测结果：")
print(val_pred_df.head())
print("\n测试集预测结果：")
print(test_pred_df.head())