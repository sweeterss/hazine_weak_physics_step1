#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUV物理一致性约束神经网络故障诊断示例代码

功能：
1. 默认自动生成可运行的AUV模拟数据，不依赖外部文件；
2. 支持5类故障：正常、DVL偏置、IMU漂移、深度计漂移、推进器效率下降；
3. 将AUV物理一致性残差作为额外输入特征；
4. 在网络训练中加入“正常样本物理一致性损失”；
5. 输出测试集准确率、Precision/Recall/F1、混淆矩阵；
6. 保存模型和混淆矩阵图片。

运行：
    python auv_physics_consistency_fault_diagnosis.py

快速测试：
    python auv_physics_consistency_fault_diagnosis.py --epochs 5 --n_per_class 120

使用自己的数据：
    如果你有 X.npy 和 y.npy：
        X.npy shape = [样本数, 时间步, 13]
        y.npy shape = [样本数]
    其中13通道默认顺序为：
        0 thruster_cmd
        1 motor_current
        2 voltage
        3 power
        4 vx
        5 vy
        6 vz
        7 roll
        8 pitch
        9 yaw
        10 p
        11 q
        12 depth

    运行：
        python auv_physics_consistency_fault_diagnosis.py --data_dir ./your_data

如果你的envpower和motion是分开的：
    envpower.npy shape = [N, T, 4]
    motion.npy   shape = [N, T, 9]
    labels.npy   shape = [N]
    运行同上。
"""

import os
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# =========================
# 1. 基础配置
# =========================

CHANNEL_NAMES = [
    "thruster_cmd", "motor_current", "voltage", "power",
    "vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "depth"
]

CLASS_NAMES = [
    "Normal",
    "DVL_bias",
    "IMU_drift",
    "Depth_drift",
    "Thruster_loss"
]


@dataclass
class TrainConfig:
    seq_len: int = 80
    n_per_class: int = 250
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dt: float = 0.1
    test_ratio: float = 0.25
    seed: int = 42
    noise_std: float = 0.02
    use_physical_features: bool = True
    lambda_recon: float = 0.15
    lambda_phys: float = 0.05
    out_dir: str = "./auv_phys_outputs"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 2. 数据生成与物理残差
# =========================

def smooth_random_signal(T: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """生成平滑随机控制信号。"""
    x = rng.normal(0, 1, size=T)
    kernel = np.ones(9) / 9.0
    y = np.convolve(x, kernel, mode="same")
    y = y / (np.max(np.abs(y)) + 1e-8)
    return scale * y


def generate_one_sequence(
    label: int,
    T: int,
    dt: float,
    rng: np.random.Generator,
    noise_std: float = 0.02,
) -> np.ndarray:
    """
    生成单个AUV序列。
    输出 shape = [T, 13]
    """
    # 控制输入：推进器命令，范围大致在[0.2, 1.0]
    base = 0.65 + 0.15 * np.sin(np.linspace(0, 2 * np.pi, T) + rng.uniform(-1, 1))
    cmd = base + 0.12 * smooth_random_signal(T, rng, scale=1.0)
    cmd = np.clip(cmd, 0.05, 1.0)

    # 物理参数
    k_thruster = 0.75      # 推进器命令到纵向加速度的系数
    damping_vx = 0.55      # 速度阻尼
    damping_vz = 0.35
    depth_stiffness = 0.04

    # 故障起始时刻
    fault_start = int(T * 0.45)

    # 真实状态
    vx = np.zeros(T)
    vy = np.zeros(T)
    vz = np.zeros(T)
    roll = np.zeros(T)
    pitch = np.zeros(T)
    yaw = np.zeros(T)
    p = np.zeros(T)
    q = np.zeros(T)
    depth = np.zeros(T)

    # 初值
    vx[0] = 0.2 + rng.normal(0, 0.01)
    depth[0] = 10.0 + rng.normal(0, 0.05)
    yaw[0] = rng.normal(0, 0.02)

    # 推进器效率，正常为1；推进器故障时降低
    eta = np.ones(T)
    if label == 4:  # Thruster_loss
        eta[fault_start:] = rng.uniform(0.45, 0.70)

    # 海流/扰动：这里作为普通环境扰动，不作为类别
    current_disturbance = 0.05 * smooth_random_signal(T, rng, scale=1.0)

    for t in range(T - 1):
        # 纵向速度动力学
        ax = k_thruster * eta[t] * cmd[t] - damping_vx * vx[t] + current_disturbance[t]
        vx[t + 1] = vx[t] + dt * ax

        # 垂向速度/深度简化动力学
        az = -depth_stiffness * (depth[t] - 10.0) - damping_vz * vz[t] + 0.01 * rng.normal()
        vz[t + 1] = vz[t] + dt * az
        depth[t + 1] = depth[t] + dt * vz[t]

        # 姿态简化动力学
        p[t + 1] = 0.85 * p[t] + 0.03 * rng.normal()
        q[t + 1] = 0.85 * q[t] + 0.03 * rng.normal()
        roll[t + 1] = roll[t] + dt * p[t]
        pitch[t + 1] = pitch[t] + dt * q[t]

        # yaw与推进器命令微弱相关
        yaw_rate = 0.08 * (cmd[t] - np.mean(cmd)) + 0.02 * rng.normal()
        yaw[t + 1] = yaw[t] + dt * yaw_rate

        # 横向速度小幅波动
        vy[t + 1] = 0.9 * vy[t] + 0.02 * rng.normal()

    # 电气量
    voltage = 24.0 + 0.2 * rng.normal(size=T)
    # 推进器故障时，命令不变，但电流/功率和运动响应关系异常
    current = 2.0 + 5.0 * cmd + 0.15 * rng.normal(size=T)
    if label == 4:
        current[fault_start:] += rng.uniform(0.3, 0.8)  # 效率下降时可能出现电流偏高
    power = voltage * current + 0.3 * rng.normal(size=T)

    # 传感器测量值 = 真实状态 + 噪声 + 故障
    vx_m = vx + noise_std * rng.normal(size=T)
    vy_m = vy + noise_std * rng.normal(size=T)
    vz_m = vz + noise_std * rng.normal(size=T)
    roll_m = roll + noise_std * rng.normal(size=T)
    pitch_m = pitch + noise_std * rng.normal(size=T)
    yaw_m = yaw + noise_std * rng.normal(size=T)
    p_m = p + noise_std * rng.normal(size=T)
    q_m = q + noise_std * rng.normal(size=T)
    depth_m = depth + noise_std * rng.normal(size=T)

    if label == 1:  # DVL_bias: 速度测量偏置
        bias = rng.uniform(0.12, 0.25)
        vx_m[fault_start:] += bias
        vy_m[fault_start:] += 0.4 * bias

    elif label == 2:  # IMU_drift: 姿态/角速度漂移
        drift = np.linspace(0, rng.uniform(0.15, 0.30), T - fault_start)
        roll_m[fault_start:] += drift
        pitch_m[fault_start:] += 0.6 * drift
        p_m[fault_start:] += 0.03 + 0.3 * drift

    elif label == 3:  # Depth_drift: 深度计漂移
        drift = np.linspace(0, rng.uniform(0.6, 1.2), T - fault_start)
        depth_m[fault_start:] += drift

    x = np.stack([
        cmd, current, voltage, power,
        vx_m, vy_m, vz_m, roll_m, pitch_m, yaw_m, p_m, q_m, depth_m
    ], axis=-1).astype(np.float32)

    return x


def generate_synthetic_dataset(
    n_per_class: int,
    T: int,
    dt: float,
    seed: int,
    noise_std: float
) -> Tuple[np.ndarray, np.ndarray]:
    """生成模拟数据集。"""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for label in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            X_list.append(generate_one_sequence(label, T, dt, rng, noise_std))
            y_list.append(label)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    # 打乱
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def load_real_or_synthetic_data(args: argparse.Namespace, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    """读取真实数据；如果没有提供data_dir，则生成模拟数据。"""
    if args.data_dir is None:
        print("[INFO] 未提供 --data_dir，使用模拟AUV故障数据直接运行。")
        return generate_synthetic_dataset(
            n_per_class=cfg.n_per_class,
            T=cfg.seq_len,
            dt=cfg.dt,
            seed=cfg.seed,
            noise_std=cfg.noise_std
        )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir 不存在: {data_dir}")

    # 情况1：X.npy + y.npy
    x_file = data_dir / "X.npy"
    y_file = data_dir / "y.npy"
    if x_file.exists() and y_file.exists():
        X = np.load(x_file).astype(np.float32)
        y = np.load(y_file).astype(np.int64)
        print(f"[INFO] 加载 {x_file} 和 {y_file}")
        return X, y

    # 情况2：envpower.npy + motion.npy + labels.npy
    env_file = data_dir / "envpower.npy"
    mot_file = data_dir / "motion.npy"
    lab_file = data_dir / "labels.npy"
    if env_file.exists() and mot_file.exists() and lab_file.exists():
        env = np.load(env_file).astype(np.float32)
        mot = np.load(mot_file).astype(np.float32)
        y = np.load(lab_file).astype(np.int64)

        if env.ndim != 3 or mot.ndim != 3:
            raise ValueError("envpower.npy 和 motion.npy 必须是 [N, T, C] 三维数组。")
        if env.shape[0] != mot.shape[0] or env.shape[1] != mot.shape[1]:
            raise ValueError("envpower 和 motion 的样本数/时间步不一致。")
        if env.shape[-1] != 4 or mot.shape[-1] != 9:
            raise ValueError("默认要求 envpower=4通道，motion=9通道。")

        X = np.concatenate([env, mot], axis=-1).astype(np.float32)
        print(f"[INFO] 加载 envpower/motion/labels，并拼接为 X shape={X.shape}")
        return X, y

    raise FileNotFoundError(
        "未找到数据文件。请提供以下任一组合：\n"
        "1) X.npy + y.npy\n"
        "2) envpower.npy + motion.npy + labels.npy"
    )


def compute_physical_residuals(X_raw: np.ndarray, dt: float) -> np.ndarray:
    """
    根据简化AUV物理关系计算物理一致性残差，作为额外输入特征。
    输入 X_raw shape = [N, T, 13]
    输出 residuals shape = [N, T, 4]
    """
    cmd = X_raw[:, :, 0]
    current = X_raw[:, :, 1]
    voltage = X_raw[:, :, 2]
    power = X_raw[:, :, 3]

    vx = X_raw[:, :, 4]
    vz = X_raw[:, :, 6]
    yaw = X_raw[:, :, 9]
    q = X_raw[:, :, 11]
    depth = X_raw[:, :, 12]

    # 这些参数不需要非常精确，目的是构造物理一致性特征
    k_thruster = 0.75
    damping_vx = 0.55

    N, T = cmd.shape
    rvx = np.zeros((N, T), dtype=np.float32)
    ryaw = np.zeros((N, T), dtype=np.float32)
    rdepth = np.zeros((N, T), dtype=np.float32)
    rpower = np.zeros((N, T), dtype=np.float32)

    # 纵向速度一致性：vx[t+1] 是否符合推进器输入导致的响应
    vx_next_phy = vx[:, :-1] + dt * (k_thruster * cmd[:, :-1] - damping_vx * vx[:, :-1])
    rvx[:, :-1] = vx[:, 1:] - vx_next_phy
    rvx[:, -1] = rvx[:, -2]

    # 航向一致性：yaw[t+1] 是否和角速度/简化角运动一致
    yaw_next_phy = yaw[:, :-1] + dt * q[:, :-1]
    ryaw[:, :-1] = yaw[:, 1:] - yaw_next_phy
    ryaw[:, -1] = ryaw[:, -2]

    # 深度一致性：depth[t+1] 是否和垂向速度一致
    depth_next_phy = depth[:, :-1] + dt * vz[:, :-1]
    rdepth[:, :-1] = depth[:, 1:] - depth_next_phy
    rdepth[:, -1] = rdepth[:, -2]

    # 电气一致性：power 是否约等于 voltage * current
    rpower = power - voltage * current

    residuals = np.stack([rvx, ryaw, rdepth, rpower], axis=-1).astype(np.float32)
    return residuals


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X_train - mean) / std, (X_test - mean) / std, mean.astype(np.float32), std.astype(np.float32)


# =========================
# 3. Dataset
# =========================

class AUVDataset(Dataset):
    def __init__(
        self,
        X_feat_norm: np.ndarray,
        X_raw_norm: np.ndarray,
        X_raw_real: np.ndarray,
        y: np.ndarray
    ):
        self.X_feat_norm = torch.tensor(X_feat_norm, dtype=torch.float32)
        self.X_raw_norm = torch.tensor(X_raw_norm, dtype=torch.float32)
        self.X_raw_real = torch.tensor(X_raw_real, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x_feat": self.X_feat_norm[idx],
            "x_raw_norm": self.X_raw_norm[idx],
            "x_raw_real": self.X_raw_real[idx],
            "label": self.y[idx],
        }


# =========================
# 4. 物理一致性神经网络
# =========================

class AUVPhysicsConsistentNet(nn.Module):
    """
    输入：
        x_feat: [B, T, C]
        C可以是13个原始通道，也可以是13+4个物理残差通道。
    输出：
        logits: 故障类别
        motion_rec_norm: 重构的motion通道，shape=[B, T, 9]
    """

    def __init__(self, input_dim: int, num_classes: int = 5, hidden_dim: int = 96):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, num_classes)
        )

        # 辅助任务：重构9个motion通道，便于加入物理一致性约束
        self.motion_rec_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)
        )

    def forward(self, x_feat):
        # x_feat: [B, T, C]
        z = x_feat.transpose(1, 2)      # [B, C, T]
        z = self.conv(z)                # [B, H, T]
        z = z.transpose(1, 2)           # [B, T, H]
        z, _ = self.gru(z)              # [B, T, 2H]

        # 时序平均池化分类
        pooled = z.mean(dim=1)
        logits = self.cls_head(pooled)

        # 每个时间步重构motion
        motion_rec_norm = self.motion_rec_head(z)
        return logits, motion_rec_norm


# =========================
# 5. 损失函数与评价指标
# =========================

def physics_consistency_loss(
    motion_rec_norm: torch.Tensor,
    x_raw_real: torch.Tensor,
    raw_mean: torch.Tensor,
    raw_std: torch.Tensor,
    labels: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    物理一致性损失：只对正常样本施加较强约束。
    目的：让网络学习AUV正常物理关系；故障样本则允许偏离正常物理关系。
    """
    normal_mask = (labels == 0)
    if normal_mask.sum() == 0:
        return torch.tensor(0.0, device=motion_rec_norm.device)

    motion = motion_rec_norm[normal_mask]  # normalized [B0, T, 9]
    x_real = x_raw_real[normal_mask]       # real-scale [B0, T, 13]

    # 将重构motion反标准化到真实量纲
    motion_mean = raw_mean[:, :, 4:13].to(motion.device)
    motion_std = raw_std[:, :, 4:13].to(motion.device)
    motion_real = motion * motion_std + motion_mean

    cmd = x_real[:, :, 0]

    vx = motion_real[:, :, 0]
    vz = motion_real[:, :, 2]
    yaw = motion_real[:, :, 5]
    q = motion_real[:, :, 7]
    depth = motion_real[:, :, 8]

    k_thruster = 0.75
    damping_vx = 0.55

    vx_next_phy = vx[:, :-1] + dt * (k_thruster * cmd[:, :-1] - damping_vx * vx[:, :-1])
    yaw_next_phy = yaw[:, :-1] + dt * q[:, :-1]
    depth_next_phy = depth[:, :-1] + dt * vz[:, :-1]

    loss_vx = F.mse_loss(vx[:, 1:], vx_next_phy)
    loss_yaw = F.mse_loss(yaw[:, 1:], yaw_next_phy)
    loss_depth = F.mse_loss(depth[:, 1:], depth_next_phy)

    return loss_vx + loss_yaw + loss_depth


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    num_classes = cm.shape[0]
    acc = np.trace(cm) / (cm.sum() + eps)

    precision_list, recall_list, f1_list = [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": float(acc),
        "macro_precision": float(np.mean(precision_list)),
        "macro_recall": float(np.mean(recall_list)),
        "macro_f1": float(np.mean(f1_list)),
    }


def print_confusion_matrix(cm: np.ndarray):
    print("\n[Confusion Matrix] 行是真实类别，列是预测类别")
    header = "true\\pred".ljust(16) + "".join([name[:12].rjust(14) for name in CLASS_NAMES])
    print(header)
    for i, row in enumerate(cm):
        print(CLASS_NAMES[i].ljust(16) + "".join([str(v).rjust(14) for v in row]))


def save_confusion_matrix_plot(cm: np.ndarray, save_path: str):
    if not HAS_MPL:
        print("[WARN] matplotlib不可用，跳过混淆矩阵图片保存。")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================
# 6. 训练与测试
# =========================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device,
    raw_mean_t: torch.Tensor,
    raw_std_t: torch.Tensor,
    cfg: TrainConfig,
    train: bool = True
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true, all_pred = [], []

    for batch in loader:
        x_feat = batch["x_feat"].to(device)
        x_raw_norm = batch["x_raw_norm"].to(device)
        x_raw_real = batch["x_raw_real"].to(device)
        y = batch["label"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits, motion_rec_norm = model(x_feat)

            loss_cls = F.cross_entropy(logits, y)
            # motion通道是原始13通道中的4:13
            target_motion_norm = x_raw_norm[:, :, 4:13]
            loss_recon = F.mse_loss(motion_rec_norm, target_motion_norm)
            loss_phys = physics_consistency_loss(
                motion_rec_norm=motion_rec_norm,
                x_raw_real=x_raw_real,
                raw_mean=raw_mean_t,
                raw_std=raw_std_t,
                labels=y,
                dt=cfg.dt
            )

            loss = loss_cls + cfg.lambda_recon * loss_recon + cfg.lambda_phys * loss_phys

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.item()) * len(y)

        pred = torch.argmax(logits, dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES))
    metrics = metrics_from_confusion(cm)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, metrics, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="真实数据目录。可不填，默认使用模拟数据。")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n_per_class", type=int, default=250)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--no_physical_features", action="store_true", help="关闭物理残差输入，只用原始13通道。")
    parser.add_argument("--lambda_recon", type=float, default=0.15)
    parser.add_argument("--lambda_phys", type=float, default=0.05)
    parser.add_argument("--out_dir", type=str, default="./auv_phys_outputs")
    args = parser.parse_args()

    cfg = TrainConfig(
        seq_len=args.seq_len,
        n_per_class=args.n_per_class,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        noise_std=args.noise_std,
        use_physical_features=not args.no_physical_features,
        lambda_recon=args.lambda_recon,
        lambda_phys=args.lambda_phys,
        out_dir=args.out_dir,
    )

    set_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 1) 读取或生成数据
    X_raw, y = load_real_or_synthetic_data(args, cfg)

    if X_raw.ndim != 3 or X_raw.shape[-1] != 13:
        raise ValueError(f"X必须为[N, T, 13]，当前shape={X_raw.shape}")

    print(f"[INFO] 原始数据 X_raw shape={X_raw.shape}, y shape={y.shape}")
    print(f"[INFO] 类别: {CLASS_NAMES}")

    # 2) 划分训练/测试
    X_train_raw, y_train, X_test_raw, y_test = train_test_split(
        X_raw, y, test_ratio=cfg.test_ratio, seed=cfg.seed
    )

    # 3) 构造物理残差特征
    if cfg.use_physical_features:
        R_train = compute_physical_residuals(X_train_raw, dt=cfg.dt)
        R_test = compute_physical_residuals(X_test_raw, dt=cfg.dt)
        X_train_feat = np.concatenate([X_train_raw, R_train], axis=-1)
        X_test_feat = np.concatenate([X_test_raw, R_test], axis=-1)
        print(f"[INFO] 使用物理残差输入: 13原始通道 + 4物理残差 = {X_train_feat.shape[-1]}通道")
    else:
        X_train_feat = X_train_raw
        X_test_feat = X_test_raw
        print("[INFO] 关闭物理残差输入，只使用原始13通道。")

    # 4) 标准化：网络输入特征单独标准化；原始13通道也标准化以便重构损失
    X_train_feat_norm, X_test_feat_norm, feat_mean, feat_std = standardize_train_test(X_train_feat, X_test_feat)
    X_train_raw_norm, X_test_raw_norm, raw_mean, raw_std = standardize_train_test(X_train_raw, X_test_raw)

    raw_mean_t = torch.tensor(raw_mean, dtype=torch.float32, device=device)
    raw_std_t = torch.tensor(raw_std, dtype=torch.float32, device=device)

    # 5) Dataset / DataLoader
    train_ds = AUVDataset(X_train_feat_norm, X_train_raw_norm, X_train_raw, y_train)
    test_ds = AUVDataset(X_test_feat_norm, X_test_raw_norm, X_test_raw, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # 6) 模型
    input_dim = X_train_feat_norm.shape[-1]
    model = AUVPhysicsConsistentNet(input_dim=input_dim, num_classes=len(CLASS_NAMES)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_path = Path(cfg.out_dir) / "best_auv_physics_consistent_net.pt"

    # 7) 训练
    print("\n[INFO] 开始训练...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_metrics, _ = run_epoch(
            model, train_loader, optimizer, device, raw_mean_t, raw_std_t, cfg, train=True
        )
        test_loss, test_metrics, test_cm = run_epoch(
            model, test_loader, optimizer, device, raw_mean_t, raw_std_t, cfg, train=False
        )

        if test_metrics["macro_f1"] > best_f1:
            best_f1 = test_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "class_names": CLASS_NAMES,
                "channel_names": CHANNEL_NAMES,
                "use_physical_features": cfg.use_physical_features,
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "feat_mean": feat_mean,
                "feat_std": feat_std,
            }, best_path)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_metrics['accuracy']:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_metrics['accuracy']:.4f}, "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )

    # 8) 最终测试
    print("\n[INFO] 加载最佳模型并测试...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, test_cm = run_epoch(
        model, test_loader, optimizer=None, device=device,
        raw_mean_t=raw_mean_t, raw_std_t=raw_std_t, cfg=cfg, train=False
    )

    print("\n[Final Test Metrics]")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    print_confusion_matrix(test_cm)

    cm_path = Path(cfg.out_dir) / "confusion_matrix.png"
    save_confusion_matrix_plot(test_cm, str(cm_path))

    print(f"\n[INFO] 最佳模型已保存: {best_path}")
    print(f"[INFO] 混淆矩阵图片已保存: {cm_path}")
    print("\n[提示]")
    print("1. 对照实验可运行：python auv_physics_consistency_fault_diagnosis.py --no_physical_features")
    print("2. 如果要用你的真实数据，请准备 X.npy/y.npy，或 envpower.npy/motion.npy/labels.npy。")


if __name__ == "__main__":
    main()
