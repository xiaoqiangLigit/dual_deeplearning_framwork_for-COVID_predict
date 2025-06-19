# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 21:12:21 2025

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
from sklearn.metrics import mean_absolute_percentage_error
# ========= 训练与评估函数 =========
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mean_weight=1.0,  # 均值损失权重
    resid_weight=1.0,  # 残差损失权重
    early_stopping=20,
    scheduler_patience=3,
    logdir=None,
):
    """训练双分支模型"""
    print(f"使用设备: {device}")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=scheduler_patience, verbose=True
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    best_model_path = None
    counter = 0  # 早停计数器
    
    if logdir is None:
        logdir = f"runs/pandemic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logdir, exist_ok=True)
    
    history = {"train_loss": [], "val_loss": [], "mean_loss": [], "resid_loss": []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mean_loss = 0.0
        train_resid_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X, y_mean, y_resid,y_true, site_ids,_ in progress_bar:
            X = X.to(device)
            y_mean = y_mean.to(device)
            y_resid = y_resid.to(device)
            y_true = y_true.to(device)
            site_ids = site_ids.to(device)
            
            optimizer.zero_grad()
            mean_pred, resid_pred = model(X, site_ids)
            
            # 分别计算均值和残差的损失
            mean_loss = criterion(mean_pred, y_mean)
            resid_loss = criterion(resid_pred, y_resid)
            
            # 加权组合损失
            loss = mean_weight * mean_loss + resid_weight * resid_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mean_loss += mean_loss.item()
            train_resid_loss += resid_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mean": f"{mean_loss.item():.4f}",
                "resid": f"{resid_loss.item():.4f}"
            })
        
        train_loss /= len(train_loader)
        train_mean_loss /= len(train_loader)
        train_resid_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mean_loss = 0.0
        val_resid_loss = 0.0
        
        with torch.no_grad():
            for X, y_mean, y_resid,y_true, site_ids,_ in val_loader:
                X = X.to(device)
                y_mean = y_mean.to(device)
                y_resid = y_resid.to(device)
                y_true =y_true.to(device)
                site_ids = site_ids.to(device)
                
                mean_pred, resid_pred = model(X, site_ids)
                mean_loss = criterion(mean_pred, y_mean)
                resid_loss = criterion(resid_pred, y_resid)
                loss = mean_weight * mean_loss + resid_weight * resid_loss
                
                val_loss += loss.item()
                val_mean_loss += mean_loss.item()
                val_resid_loss += resid_loss.item()
        
        val_loss /= len(val_loader)
        val_mean_loss /= len(val_loader)
        val_resid_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录损失
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mean_loss"].append(val_mean_loss)
        history["resid_loss"].append(val_resid_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} (Mean: {train_mean_loss:.4f}, Resid: {train_resid_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Mean: {val_mean_loss:.4f}, Resid: {val_resid_loss:.4f})")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(logdir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
        
        # 早停
        if counter >= early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存最终模型
    final_model_path = os.path.join(logdir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # 保存训练历史
    history_path = os.path.join(logdir, "history.pt")
    torch.save(history, history_path)
    
    # 加载最佳模型
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
    
    return model, history

def evaluate_model(
    model,
    test_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    show_plots=True,
):
    """评估双分支模型"""
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    y_mean_true = []
    y_resid_true = []
    y_mean_pred = []
    y_resid_pred = []
    
    with torch.no_grad():
        for X, y_mean, y_resid,y_trues, site_ids,_ in test_loader:
            X = X.to(device)
            site_ids = site_ids.to(device)
            y_trues = y_trues.to(device)
            mean_pred, resid_pred = model(X, site_ids)
            total_pred = mean_pred + resid_pred
            
            # 收集真实值和预测值
            y_true.extend(y_trues.cpu().numpy())
            y_pred.extend(total_pred.cpu().numpy())
            
            # 收集均值和残差的真实值和预测值
            y_mean_true.extend(y_mean.cpu().numpy())
            y_resid_true.extend(y_resid.cpu().numpy())
            y_mean_pred.extend(mean_pred.cpu().numpy())
            y_resid_pred.extend(resid_pred.cpu().numpy())
    
    # 转换为数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean_true = np.array(y_mean_true)
    y_resid_true = np.array(y_resid_true)
    y_mean_pred = np.array(y_mean_pred)
    y_resid_pred = np.array(y_resid_pred)
    
    # 计算指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 单独计算均值和残差的指标
    mean_rmse = np.sqrt(mean_squared_error(y_mean_true, y_mean_pred))
    resid_rmse = np.sqrt(mean_squared_error(y_resid_true, y_resid_pred))
    mape = mean_absolute_percentage_error(y_true,y_pred)
    
    print(f"测试集评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"均值分支 RMSE: {mean_rmse:.4f}")
    print(f"残差分支 RMSE: {resid_rmse:.4f}")
    print(f"总MAPE: {mape:.4f}")
    
    
    if show_plots:
        # 绘制实际值vs预测值散点图
        plt.figure(figsize=(16, 12))
        
        # 总体预测
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('总体预测 (均值+残差)')
        
        # 均值预测
        plt.subplot(2, 2, 2)
        plt.scatter(y_mean_true, y_mean_pred, alpha=0.5, color='green')
        plt.plot([y_mean_true.min(), y_mean_true.max()], [y_mean_true.min(), y_mean_true.max()], 'r--')
        plt.xlabel('实际均值')
        plt.ylabel('预测均值')
        plt.title('均值分支预测')
        
        # 残差预测
        plt.subplot(2, 2, 3)
        plt.scatter(y_resid_true, y_resid_pred, alpha=0.5, color='orange')
        plt.plot([y_resid_true.min(), y_resid_true.max()], [y_resid_true.min(), y_resid_true.max()], 'r--')
        plt.xlabel('实际残差')
        plt.ylabel('预测残差')
        plt.title('残差分支预测')
        
        # 预测误差直方图
        plt.subplot(2, 2, 4)
        errors = y_pred - y_true
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('预测误差')
        plt.ylabel('频次')
        plt.title(f'预测误差直方图 (RMSE={rmse:.4f})')
        
        plt.tight_layout()
        plt.show()
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_rmse": mean_rmse,
        "resid_rmse": resid_rmse,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_mean_true": y_mean_true,
        "y_mean_pred": y_mean_pred,
        "y_resid_true": y_resid_true,
        "y_resid_pred": y_resid_pred,
    }

