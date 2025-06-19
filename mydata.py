# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 21:05:26 2025

@author: Administrator
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

# 全局调用  
set_seed(2011)  # 或任何固定值   
# ========= FFT低通工具函数 =========
def lowpass_time_series(x: np.ndarray, cutoff: float = 0.8, d:float = 0.05):
    """
    对 1-D 实数序列做 FFT 低通，并返回：
    trend (低频部分, 平滑均值)、resid (高频残差 = 原序列-趋势)
    cutoff 取 0~0.5（相对Nyquist）；0.20≈保留20%低频
    """
    n = len(x)  
    sampling_rate = 1 / d  # 计算采样频率
    nyquist_frequency = sampling_rate / 2  # 奈奎斯特频率

    fft = np.fft.rfft(x)  # 计算FFT
    freqs = np.fft.rfftfreq(n, d)  # 计算频率

    # 根据相对奈奎斯特频率设置截止频率
    cutoff_frequency = cutoff * nyquist_frequency
    
    fft_mask = fft.copy()  
    fft_mask[freqs > cutoff_frequency] = 0  # 应用低通滤波器
    trend = np.fft.irfft(fft_mask, n=n)  # 计算趋势部分

#     增加高通滤波器确保残差有意义  
    resid = x-trend
    return trend, resid
# ========= 数据集 =========
class PandemicDataset(Dataset):
    """
    同时返回 y_mean(低频均值), y_resid(高频残差), site_id
    """
    def __init__(self, X: np.ndarray, y_mean: np.ndarray, y_resid: np.ndarray,
                 y_true: np.ndarray,
                 site_ids: Optional[np.ndarray] = None,
                 weeks: Optional[np.ndarray] = None,
                 ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float32).unsqueeze(-1)
        self.y_resid = torch.as_tensor(y_resid, dtype=torch.float32).unsqueeze(-1)
        self.y_true = torch.as_tensor(y_true, dtype=torch.float32).unsqueeze(-1)
        self.site_ids = torch.as_tensor(site_ids, dtype=torch.long)
        self.weeks = torch.as_tensor(weeks.astype(int), dtype=torch.long)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_mean[idx],
            self.y_resid[idx],
            self.y_true[idx],
            self.site_ids[idx],
            self.weeks[idx]
        )

# ========= 数据处理器 =========
class PandemicDataProcessor:
    """
    产生 train/val/test 三个 Dataset，并提供 DataLoader
    参数
      feature_cols: 作为模型输入的列名
      lookback:     用于窗口（T周）
      pred_days:    预测几天，等价于1周->y为标量
    """
    def __init__(
        self,
        feature_cols: List[str],
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        lookback: int = 4,
        pred_days: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 2025,
        
    ):
        if data is not None:
            self.raw_df = data.copy()
        elif data_path is not None:
            self.raw_df = pd.read_csv(data_path)
        else:
            raise ValueError("必须提供 data 或 data_path")

        self.feature_cols = feature_cols
        self.lookback = lookback
        self.pred_days = pred_days
        self.random_state = random_state
        self.need_cols = feature_cols+ ["周新冠数"]
        self._preprocess()

        (
            self.train_idx,
            self.val_idx,
            self.test_idx,
        ) = self._make_splits(test_size, val_size)

        self.train_dataset = PandemicDataset(
            self.X[self.train_idx],
            self.y_mean[self.train_idx],
            self.y_resid[self.train_idx],
            self.y_true[self.train_idx],
            self.site_ids[self.train_idx],
            self.weeks[self.train_idx],  # 添加周数
            
        )
        self.val_dataset = PandemicDataset(
            self.X[self.val_idx],
            self.y_mean[self.val_idx],
            self.y_resid[self.val_idx],
            self.y_true[self.val_idx],
            self.site_ids[self.val_idx],
            self.weeks[self.val_idx],  # 添加周数
            
        )
        self.test_dataset = PandemicDataset(
            self.X[self.test_idx],
            self.y_mean[self.test_idx],
            self.y_resid[self.test_idx],
            self.y_true[self.test_idx],
            self.site_ids[self.test_idx],
            self.weeks[self.test_idx],  # 添加周数
            
        )

    @staticmethod
    def _trend(x: pd.Series) -> float:
        """简单线性趋势斜率；数据不足返回 0"""
        x = x.dropna()
        if len(x) < 2:
            return 0.0
        try:
            p = np.polyfit(np.arange(len(x)), x.values, 1)
            return float(p[0])
        except Exception:
            return 0.0
    def pad_sequence(self,seq, target_length=32):  
        # 假设 seq 的形状为 (当前长度, 特征数)   
        current_length = seq.shape[0]  # 获取当前序列长度  
        if current_length < target_length:  
            # 创建一个零填充的数组  
#             padded_seq = np.zeros((target_length, seq.shape[1]))  # 创建 target_length x features 的零数组  
#             padded_seq[:current_length, :] = seq  # 将原始序列复制到零数组中  
            padded_seq = np.repeat(seq, 10, axis=0) 
#             return pd.DataFrame(padded_seq,columns=self.need_cols)   # 返回具有填充的序列 
            return pd.DataFrame(padded_seq,columns=['周新冠数']) 
        return seq  # 返回原序列，如果不需要填充 
    def _preprocess(self):
        df = self.raw_df.copy()
        df["采样日期"] = pd.to_datetime(df["采样日期"])
        df["周数"] = df["采样日期"].dt.isocalendar().week
        df["年度周数"] = df["采样日期"].dt.strftime("%Y-%W")
        grouped = df.groupby("监测点", group_keys=False)
        df["病例_2w_ma"] = grouped["周新冠数"].transform(
            lambda s: s.rolling(window=2, min_periods=1).mean()
        )
        df["病例_周环比"] = (
            grouped["周新冠数"]
            .apply(lambda s: s.replace(0, 1e-6).pct_change())
            .fillna(0)
        )
        for col, new_col in [("O浓度", "O病毒_3w趋势"), ("N浓度", "N病毒_3w趋势")]:
            df[new_col] = grouped[col].transform(
                lambda s: s.rolling(window=3, min_periods=2).apply(self._trend, raw=False)
            )

        # 归一化输入特征 (标准差+1e-6防止除零)
        df[self.feature_cols] = grouped[self.feature_cols].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-6)
        )

        df.dropna(axis=0, inplace=True)
        # print(display(df))  # 如需调试可打开
        X_list, y_mean_list, y_resid_list, site_list = [], [], [], []
        weeks_,y_trues = [],[]
        target_length = 32
        need_cols = self.need_cols
        for site, g in df.groupby("监测点"):
            g = g.sort_values("采样日期").reset_index(drop=True)
            for i in range(len(g) - self.lookback - self.pred_days + 1):
                seq = g.iloc[i : i + self.lookback]
#                 padded_seq = self.pad_sequence(seq[self.need_cols].values, target_length=target_length)
                target = df[['周新冠数']].groupby(df["监测点"]).get_group(32041101).iloc[i:i + self.lookback + self.pred_days]
                padded_seq = self.pad_sequence(target['周新冠数'].values, target_length=target_length)
                if seq[need_cols].isna().values.any():
                    continue
                if target["周新冠数"].isna().any():
                    continue
                week_ = g.iloc[i + self.lookback]["周数"]
                # --- FFT 提取均值与残差 ---
                seq_cases = seq["周新冠数"].values
                tgt_cases = padded_seq["周新冠数"].values
                # 序列部分不用，目标部分分解低频/高频
                μ_tgt, ε_tgt = lowpass_time_series(tgt_cases)
#                 X_list.append(padded_seq[self.need_cols].values)
                X_list.append(seq[self.need_cols].values)
                y_mean_list.append(float(μ_tgt[-1]))
                y_resid_list.append(float(ε_tgt[-1]))
                y_trues.append(float(padded_seq["周新冠数"].values[-1]))
                site_list.append(site)
                weeks_.append(week_)
                
        if not X_list:
            raise ValueError("全部窗口都被 NaN 删掉了，请检查 min_periods / lookback 设置")
        self.X = np.stack(X_list, axis=0)
        self.y_mean = np.asarray(y_mean_list, dtype=np.float32)
        self.y_resid = np.asarray(y_resid_list, dtype=np.float32)
        self.y_true = np.asarray(y_trues, dtype=np.float32)
        self.site_ids, _ = pd.factorize(site_list)
        self.num_sites = len(np.unique(self.site_ids))
        self.feature_dim = self.X.shape[-1]
        self.weeks = np.asarray(weeks_)
        
    def _make_splits(self, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#        """
#        先按周数随机采样，然后根据采样到的周数提取数据
#        1. 首先随机选择周数
#        2. 基于选择的周数，提取对应的数据索引
#        3. 保证分层采样（按监测点）
#        """
    # 获取唯一的周数
        unique_weeks = np.unique(self.weeks)
        np.random.seed(self.random_state)
            # 随机选择测试集的周数
#         trval,test_weeks = train_test_split(unique_weeks,test_size=test_size,random_state=self.random_state)
#     # 剩余周数作为训练和验证集
#         train_weeks,val_weeks = train_test_split(trval,test_size=val_size,random_state=self.random_state)

    # 随机选择测试集的周数
        test_weeks = np.random.choice(
            unique_weeks, 
            size=int(len(unique_weeks) * test_size), 
            replace=False
        )
    
    # 剩余周数作为训练和验证集
        remaining_weeks = np.setdiff1d(unique_weeks, test_weeks)
    
    # 从剩余周数中选择验证集周数
        val_weeks = np.random.choice(
            remaining_weeks, 
            size=int(len(remaining_weeks) * val_size), 
            replace=False
        )
    
    # 训练集周数
        train_weeks = np.setdiff1d(remaining_weeks, val_weeks)
    
    # 根据周数获取对应的索引
        test_idx = np.where(np.isin(self.weeks, test_weeks))[0]
        val_idx = np.where(np.isin(self.weeks, val_weeks))[0]
        train_idx = np.where(np.isin(self.weeks, train_weeks))[0]
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError("无法进行训练/验证/测试数据集划分，检查周数和数据源。")

    # train_idx, test_idx = train_test_split(
            # train_idx,
            # test_size=len(test_idx) / len(train_idx),
            # random_state=self.random_state,
            # stratify=self.site_ids[train_idx],
        # )
    
    # train_idx, val_idx = train_test_split(
            # train_idx,
            # test_size=len(val_idx) / len(train_idx),
            # random_state=self.random_state,
            # stratify=self.site_ids[train_idx],
        # )
    
    # 保存采样的周数信息（可选）
        self.sampled_weeks = {
            'train_weeks': train_weeks,
            'val_weeks': val_weeks,
            'test_weeks': test_weeks
        }
    
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    def print_weeks(self):
#        """打印训练、验证和测试集中的采样周数"""
        print("训练集采样周数:", self.sampled_weeks['train_weeks'])
        print("验证集采样周数:", self.sampled_weeks['val_weeks'])
        print("测试集采样周数:", self.sampled_weeks['test_weeks'])

    def get_dataloaders(self, batch_size: int = 32, num_workers: Optional[int] = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """返回 train/val/test 三个 DataLoader"""
        if num_workers is None:
            num_workers = min(os.cpu_count(), 4)
        dl_train = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        dl_val = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        dl_test = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        return dl_train, dl_val, dl_test

    def sample_series(self, split: str = "test", idx: int = 0):
        """
        用法示例：
            x, y_mean, y_resid, site_id = processor.sample_series('test', 5)
        """
        assert split in {"train", "val", "test"}
        ds = getattr(self, f"{split}_dataset")
        return ds[idx]