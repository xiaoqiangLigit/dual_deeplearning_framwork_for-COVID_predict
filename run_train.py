# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 21:13:20 2025

@author: Administrator
"""
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
# ========= 模型训练和使用示例 =========
def run_training_example(processor, batch_size=32, epochs=30):
    """使用修改后的双分支模型训练示例"""
    train_loader, val_loader, test_loader = processor.get_dataloaders(batch_size=batch_size)
    
    # 获取特征维度
    sample_batch = next(iter(train_loader))
    X, _,_, _, _,_ = sample_batch
    seq_len = X.shape[1]
    feature_dim = X.shape[2]
    
    # 创建模型
    model = PandemicDualModel(
        input_channels=feature_dim,
        seq_len=seq_len,
        site_count=processor.num_sites,
        site_embed_dim=4
    )
    
    # 训练模型
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=5e-4,
        weight_decay=1e-4,
        mean_weight=1.2,
        resid_weight=0.00005,  # 可以调整权重以更注重残差预测
    )
    
    # 评估模型
    results = evaluate_model(model, test_loader)
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练与验证损失")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["mean_loss"], label="Mean Loss")
    plt.plot(history["resid_loss"], label="Residual Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("均值与残差分支损失")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, results