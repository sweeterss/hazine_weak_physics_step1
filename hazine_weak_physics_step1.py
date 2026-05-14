#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 for the HAZINE AUV dataset:
Weak-physics-residual enhanced fault diagnosis.

This script keeps your original 13-channel preprocessing idea, but replaces the
synthetic-data residuals with weak physical residuals suitable for the available columns:

    1) roll_dot  <-> w_row consistency
    2) pitch_dot <-> w_pitch consistency
    3) yaw_dot   <-> w_yaw consistency
    4) pressure  <-> depth consistency
    5) press_dot <-> depth_dot consistency
    6) pwm/voltage <-> acceleration response consistency
    7) pwm <-> voltage electrical consistency

It does not require a precise AUV hydrodynamic model. The weak consistency relations are
identified from Normal training samples using ridge regression, then converted into residual
sequences and fused with raw temporal features.

Expected folder structure:

root_dir/
  train/
    Normal/*.csv
    PropellerDamage_slight/*.csv
    PropellerDamage_bad/*.csv
    PressureGain_constant/*.csv
    AddWeight/*.csv
  val/     optional, same subfolders
  test/    optional, same subfolders

Run example:
    python hazine_weak_physics_step1.py --root-dir ./HAZINE --epochs 30

If there is no test folder, the script splits val into validation/test when possible.
If there is no val folder, it splits train into train/val/test.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# 0. Constants
# -----------------------------------------------------------------------------

SENSOR_COLS = [
    "pwm1",
    "a_z",
    "w_row",
    "w_pitch",
    "press",
    "voltage",
    "roll",
    "pitch",
    "yaw",
    "a_x",
    "a_y",
    "w_yaw",
    "depth",
]

LABEL_MAP = {
    "Normal": 0,
    "PropellerDamage_slight": 1,
    "PropellerDamage_bad": 2,
    "PressureGain_constant": 3,
    "AddWeight": 4,
}

FAULT_NAMES = [
    "Normal",
    "PropellerDamage_slight",
    "PropellerDamage_bad",
    "PressureGain_constant",
    "AddWeight",
]


# -----------------------------------------------------------------------------
# 1. Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use stratified split when every class has at least two samples; otherwise fallback."""
    counts = np.bincount(y, minlength=len(FAULT_NAMES))
    stratify = y if np.all(counts[counts > 0] >= 2) else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)


def folder_exists(root_dir: str, mode: str) -> bool:
    return os.path.isdir(os.path.join(root_dir, mode))


# -----------------------------------------------------------------------------
# 2. Data loading and preprocessing
# -----------------------------------------------------------------------------


class DataAugmentation:
    """
    Lightweight sequence augmentation for tabular AUV time series.

    The augmentation is intentionally weak. It should increase robustness without
    destroying the physical consistency too much.
    """

    def __init__(
        self,
        noise_factor: float = 0.002,
        max_noise_factor: float = 0.01,
        scale_range: Tuple[float, float] = (0.98, 1.02),
        shift_max: int = 4,
    ) -> None:
        self.noise_factor = noise_factor
        self.max_noise_factor = max_noise_factor
        self.scale_range = scale_range
        self.shift_max = shift_max

    def add_noise(self, sample: np.ndarray) -> np.ndarray:
        # Feature-wise noise scaled by the sample standard deviation.
        std = np.std(sample, axis=0, keepdims=True) + 1e-6
        factor = np.random.uniform(self.noise_factor, self.max_noise_factor)
        noise = np.random.normal(0.0, factor, size=sample.shape) * std
        return (sample + noise).astype(np.float32)

    def random_scale(self, sample: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(1, sample.shape[1]))
        return (sample * scale).astype(np.float32)

    def time_shift(self, sample: np.ndarray) -> np.ndarray:
        shift = np.random.randint(-self.shift_max, self.shift_max + 1)
        if shift == 0:
            return sample.astype(np.float32)
        return np.roll(sample, shift=shift, axis=0).astype(np.float32)

    def augment(self, sample: np.ndarray) -> np.ndarray:
        out = sample.copy().astype(np.float32)
        if np.random.rand() < 0.8:
            out = self.add_noise(out)
        if np.random.rand() < 0.5:
            out = self.random_scale(out)
        if np.random.rand() < 0.3:
            out = self.time_shift(out)
        return out.astype(np.float32)


class HazineRawLoader:
    """
    Loads CSV files and returns raw clipped/padded sequences.

    This mirrors your original preprocessing:
        - use the same 13 columns;
        - IQR clipping per file;
        - pad short sequences with edge values;
        - otherwise take the last seq_length samples.
    """

    def __init__(self, root_dir: str, seq_length: int = 190) -> None:
        self.root_dir = root_dir
        self.seq_length = seq_length

    def load_mode(self, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        samples: List[np.ndarray] = []
        labels: List[int] = []
        data_dir = os.path.join(self.root_dir, mode)
        print(f"Loading data from: {data_dir}")

        if not os.path.isdir(data_dir):
            raise RuntimeError(f"Missing split folder: {data_dir}")

        for fault_type, label in LABEL_MAP.items():
            fault_dir = os.path.join(data_dir, fault_type)
            if not os.path.isdir(fault_dir):
                print(f"  Missing class folder: {fault_dir}")
                continue

            csv_files = [f for f in os.listdir(fault_dir) if f.lower().endswith(".csv")]
            csv_files.sort()
            print(f"  {fault_type}: {len(csv_files)} files")

            for file_name in csv_files:
                file_path = os.path.join(fault_dir, file_name)
                try:
                    df = pd.read_csv(file_path)
                    sample = self._preprocess(df, file_path)
                    if sample is not None:
                        samples.append(sample)
                        labels.append(label)
                except Exception as exc:
                    print(f"  Error reading {file_path}: {exc}")

        if len(samples) == 0:
            raise RuntimeError(f"No valid CSV samples found in {data_dir}")

        X = np.stack(samples).astype(np.float32)  # [N, T, F]
        y = np.asarray(labels, dtype=np.int64)
        print(f"Loaded {mode}: X={X.shape}, y={y.shape}, class_counts={np.bincount(y, minlength=len(FAULT_NAMES)).tolist()}")
        return X, y

    def _preprocess(self, df: pd.DataFrame, file_path: str = "") -> Optional[np.ndarray]:
        missing = [col for col in SENSOR_COLS if col not in df.columns]
        if missing:
            print(f"  Missing required columns in {file_path}: {missing}")
            return None

        df = df[SENSOR_COLS].copy()
        df = df.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill")

        # IQR clipping, same idea as your original code.
        for col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if np.isfinite(iqr) and iqr > 0:
                df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        values = df.values.astype(np.float32)
        if len(values) == 0:
            return None
        if len(values) < self.seq_length:
            values = np.pad(values, ((0, self.seq_length - len(values)), (0, 0)), mode="edge")
        else:
            values = values[-self.seq_length :]
        return values.astype(np.float32)


def augment_training_data(X: np.ndarray, y: np.ndarray, use_augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if not use_augment:
        return X, y

    augmenter = DataAugmentation()
    aug_samples: List[np.ndarray] = []
    aug_labels: List[int] = []

    for sample, label in zip(X, y):
        aug_samples.append(sample)
        aug_labels.append(int(label))

        # Keep your original idea: minority / difficult classes get more augmentation.
        if int(label) in [2, 4]:
            repeat = 2
        elif int(label) in [1, 3]:
            repeat = 1
        else:
            repeat = 0

        for _ in range(repeat):
            aug_samples.append(augmenter.augment(sample))
            aug_labels.append(int(label))

    Xa = np.stack(aug_samples).astype(np.float32)
    ya = np.asarray(aug_labels, dtype=np.int64)
    print(f"After augmentation: X={Xa.shape}, class_counts={np.bincount(ya, minlength=len(FAULT_NAMES)).tolist()}")
    return Xa, ya


class HazineTorchDataset(Dataset):
    """Returns [T, F] raw feature sequence, optional [T, R] residual sequence, and label."""

    def __init__(self, X: np.ndarray, y: np.ndarray, R: Optional[np.ndarray] = None) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.R = None if R is None else torch.tensor(R, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        if self.R is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.R[idx], self.y[idx]


# -----------------------------------------------------------------------------
# 3. Weak physical residuals for HAZINE 13 channels
# -----------------------------------------------------------------------------


class HazineWeakPhysicsResidualizer:
    """
    Weak physical residuals matched to SENSOR_COLS.

    All relations are fitted from Normal training data. This avoids assuming exact
    physical coefficients or exact units.

    Residuals per timestep:
        r_roll_rate      = d(roll)/dt  - fitted(w_row)
        r_pitch_rate     = d(pitch)/dt - fitted(w_pitch)
        r_yaw_rate       = d(yaw)/dt   - fitted(w_yaw)
        r_press_depth    = press       - fitted(depth)
        r_press_rate     = d(press)/dt - fitted(d(depth)/dt)
        r_ax_response    = a_x         - fitted(pwm1, voltage, pitch, w_yaw)
        r_az_response    = a_z         - fitted(pwm1, voltage, depth, press, pitch)
        r_voltage_pwm    = voltage     - fitted(pwm1, |pwm1|)
    """

    residual_names = [
        "r_roll_rate",
        "r_pitch_rate",
        "r_yaw_rate",
        "r_press_depth",
        "r_press_rate",
        "r_ax_response",
        "r_az_response",
        "r_voltage_pwm",
    ]

    def __init__(self, feature_names: List[str], dt: float = 0.1, ridge: float = 1e-3, eps: float = 1e-6) -> None:
        self.feature_names = feature_names
        self.idx = {name: i for i, name in enumerate(feature_names)}
        self.dt = dt
        self.ridge = ridge
        self.eps = eps
        self.beta: Dict[str, np.ndarray] = {}
        self.res_mean: Optional[np.ndarray] = None
        self.res_std: Optional[np.ndarray] = None

    def _f(self, X: np.ndarray, name: str) -> np.ndarray:
        return X[..., self.idx[name]].astype(np.float32)

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        dx = np.zeros_like(x, dtype=np.float32)
        if x.shape[1] > 1:
            dx[:, 1:] = (x[:, 1:] - x[:, :-1]) / self.dt
            dx[:, 0] = dx[:, 1]
        return dx

    def _ridge_fit(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        A = A.astype(np.float32)
        b = b.astype(np.float32)
        AtA = A.T @ A
        reg = self.ridge * np.eye(AtA.shape[0], dtype=np.float32)
        return np.linalg.solve(AtA + reg, A.T @ b).astype(np.float32)

    @staticmethod
    def _predict(A: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return (A @ beta).astype(np.float32)

    def _design_matrices(self, X: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        N, T, _ = X.shape

        pwm1 = self._f(X, "pwm1")
        a_z = self._f(X, "a_z")
        w_row = self._f(X, "w_row")
        w_pitch = self._f(X, "w_pitch")
        press = self._f(X, "press")
        voltage = self._f(X, "voltage")
        roll = self._f(X, "roll")
        pitch = self._f(X, "pitch")
        yaw = self._f(X, "yaw")
        a_x = self._f(X, "a_x")
        w_yaw = self._f(X, "w_yaw")
        depth = self._f(X, "depth")

        droll = self._derivative(roll)
        dpitch = self._derivative(pitch)
        dyaw = self._derivative(yaw)
        dpress = self._derivative(press)
        ddepth = self._derivative(depth)

        ones = np.ones((N, T), dtype=np.float32)

        # Rate consistency. Fitted coefficient absorbs unit/sign convention differences.
        A_roll = np.stack([ones, w_row], axis=-1).reshape(-1, 2)
        b_roll = droll.reshape(-1)

        A_pitch = np.stack([ones, w_pitch], axis=-1).reshape(-1, 2)
        b_pitch = dpitch.reshape(-1)

        A_yaw = np.stack([ones, w_yaw], axis=-1).reshape(-1, 2)
        b_yaw = dyaw.reshape(-1)

        # Hydrostatic consistency: pressure should be strongly related to depth.
        A_press_depth = np.stack([ones, depth], axis=-1).reshape(-1, 2)
        b_press_depth = press.reshape(-1)

        A_press_rate = np.stack([ones, ddepth], axis=-1).reshape(-1, 2)
        b_press_rate = dpress.reshape(-1)

        # Weak actuator/response consistency.
        A_ax = np.stack(
            [ones, pwm1, np.abs(pwm1), voltage, pitch, w_yaw], axis=-1
        ).reshape(-1, 6)
        b_ax = a_x.reshape(-1)

        A_az = np.stack(
            [ones, pwm1, np.abs(pwm1), voltage, depth, press, pitch], axis=-1
        ).reshape(-1, 7)
        b_az = a_z.reshape(-1)

        A_voltage = np.stack([ones, pwm1, np.abs(pwm1)], axis=-1).reshape(-1, 3)
        b_voltage = voltage.reshape(-1)

        return {
            "roll": (A_roll, b_roll, (N, T)),
            "pitch": (A_pitch, b_pitch, (N, T)),
            "yaw": (A_yaw, b_yaw, (N, T)),
            "press_depth": (A_press_depth, b_press_depth, (N, T)),
            "press_rate": (A_press_rate, b_press_rate, (N, T)),
            "ax": (A_ax, b_ax, (N, T)),
            "az": (A_az, b_az, (N, T)),
            "voltage": (A_voltage, b_voltage, (N, T)),
        }

    def fit(self, X_train_raw: np.ndarray, y_train: Optional[np.ndarray] = None, normal_label: int = 0) -> "HazineWeakPhysicsResidualizer":
        if y_train is not None and np.any(y_train == normal_label):
            X_fit = X_train_raw[y_train == normal_label]
            print(f"Fitting weak physical relations on Normal samples: {X_fit.shape[0]}")
        else:
            X_fit = X_train_raw
            print("No Normal samples found. Fitting weak physical relations on all training samples.")

        mats = self._design_matrices(X_fit)
        for key, (A, b, _) in mats.items():
            self.beta[key] = self._ridge_fit(A, b)

        R = self._raw_residuals(X_fit)
        flat = R.reshape(-1, R.shape[-1])
        self.res_mean = flat.mean(axis=0, keepdims=True)
        self.res_std = flat.std(axis=0, keepdims=True) + self.eps
        return self

    def _raw_residuals(self, X: np.ndarray) -> np.ndarray:
        if not self.beta:
            raise RuntimeError("Call fit before transform.")

        N, T, _ = X.shape
        mats = self._design_matrices(X)

        roll = self._f(X, "roll")
        pitch = self._f(X, "pitch")
        yaw = self._f(X, "yaw")
        press = self._f(X, "press")
        depth = self._f(X, "depth")
        a_x = self._f(X, "a_x")
        a_z = self._f(X, "a_z")
        voltage = self._f(X, "voltage")

        droll = self._derivative(roll)
        dpitch = self._derivative(pitch)
        dyaw = self._derivative(yaw)
        dpress = self._derivative(press)

        pred_roll = self._predict(mats["roll"][0], self.beta["roll"]).reshape(N, T)
        pred_pitch = self._predict(mats["pitch"][0], self.beta["pitch"]).reshape(N, T)
        pred_yaw = self._predict(mats["yaw"][0], self.beta["yaw"]).reshape(N, T)
        pred_press_depth = self._predict(mats["press_depth"][0], self.beta["press_depth"]).reshape(N, T)
        pred_press_rate = self._predict(mats["press_rate"][0], self.beta["press_rate"]).reshape(N, T)
        pred_ax = self._predict(mats["ax"][0], self.beta["ax"]).reshape(N, T)
        pred_az = self._predict(mats["az"][0], self.beta["az"]).reshape(N, T)
        pred_voltage = self._predict(mats["voltage"][0], self.beta["voltage"]).reshape(N, T)

        R = np.stack(
            [
                droll - pred_roll,
                dpitch - pred_pitch,
                dyaw - pred_yaw,
                press - pred_press_depth,
                dpress - pred_press_rate,
                a_x - pred_ax,
                a_z - pred_az,
                voltage - pred_voltage,
            ],
            axis=-1,
        ).astype(np.float32)
        return R

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        if self.res_mean is None or self.res_std is None:
            raise RuntimeError("Call fit before transform.")
        R = self._raw_residuals(X_raw)
        R = (R - self.res_mean) / self.res_std
        return np.clip(R, -10.0, 10.0).astype(np.float32)

    def fit_transform(self, X_train_raw: np.ndarray, y_train: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X_train_raw, y_train).transform(X_train_raw)


# -----------------------------------------------------------------------------
# 4. Models
# -----------------------------------------------------------------------------


class TCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        h = self.net(x.transpose(1, 2))
        h_avg = self.pool_avg(h).squeeze(-1)
        h_max = self.pool_max(h).squeeze(-1)
        return torch.cat([h_avg, h_max], dim=-1)


class BaselineTCN(nn.Module):
    def __init__(self, raw_dim: int, num_classes: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = TCNEncoder(raw_dim, hidden, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class WeakPhysicsTCN(nn.Module):
    def __init__(self, raw_dim: int, residual_dim: int, num_classes: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.raw_encoder = TCNEncoder(raw_dim, hidden, dropout)
        self.res_encoder = TCNEncoder(residual_dim, max(hidden // 2, 16), dropout)
        res_hidden = max(hidden // 2, 16)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2 + res_hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        h_raw = self.raw_encoder(x)
        h_res = self.res_encoder(r)
        return self.classifier(torch.cat([h_raw, h_res], dim=-1))


# -----------------------------------------------------------------------------
# 5. Training and evaluation
# -----------------------------------------------------------------------------


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, is_physics_model: bool = False) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        if is_physics_model:
            x, r, y = batch
            x = x.to(device)
            r = r.to(device)
            logits = model(x, r)
        else:
            x, y = batch
            x = x.to(device)
            logits = model(x)

        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    is_physics_model: bool = False,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_state = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if is_physics_model:
                x, r, y = batch
                x, r, y = x.to(device), r.to(device), y.to(device)
                logits = model(x, r)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)

            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total_count += y.size(0)

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, is_physics_model=is_physics_model)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d}/{epochs} | loss={total_loss / max(total_count, 1):.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def print_report(title: str, metrics: Dict[str, object]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            metrics["y_true"],
            metrics["y_pred"],
            labels=list(range(len(FAULT_NAMES))),
            target_names=FAULT_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix: rows=true, cols=pred")
    print(confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(FAULT_NAMES)))))


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------


@dataclass
class SplitData:
    X_train_raw: np.ndarray
    y_train: np.ndarray
    X_val_raw: np.ndarray
    y_val: np.ndarray
    X_test_raw: np.ndarray
    y_test: np.ndarray


def load_hazine_splits(root_dir: str, seq_length: int, seed: int) -> SplitData:
    loader = HazineRawLoader(root_dir=root_dir, seq_length=seq_length)

    X_train_raw, y_train = loader.load_mode("train")

    if folder_exists(root_dir, "val") and folder_exists(root_dir, "test"):
        X_val_raw, y_val = loader.load_mode("val")
        X_test_raw, y_test = loader.load_mode("test")
    elif folder_exists(root_dir, "val"):
        X_val_all, y_val_all = loader.load_mode("val")
        if len(y_val_all) >= 4:
            X_val_raw, X_test_raw, y_val, y_test = safe_stratified_split(X_val_all, y_val_all, test_size=0.5, seed=seed)
            print("No test folder found. Split val into validation/test.")
        else:
            X_val_raw, y_val = X_val_all, y_val_all
            X_test_raw, y_test = X_val_all, y_val_all
            print("No test folder found and val is too small. Reusing val as test.")
    else:
        print("No val/test folders found. Splitting train into train/val/test.")
        X_train_raw, X_tmp, y_train, y_tmp = safe_stratified_split(X_train_raw, y_train, test_size=0.3, seed=seed)
        X_val_raw, X_test_raw, y_val, y_test = safe_stratified_split(X_tmp, y_tmp, test_size=0.5, seed=seed)

    return SplitData(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test)


def standardize_raw_features(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    out_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    n_train, seq_len, num_features = X_train_raw.shape
    scaler.fit(X_train_raw.reshape(-1, num_features))

    X_train = scaler.transform(X_train_raw.reshape(-1, num_features)).reshape(n_train, seq_len, num_features).astype(np.float32)
    X_val = scaler.transform(X_val_raw.reshape(-1, num_features)).reshape(X_val_raw.shape).astype(np.float32)
    X_test = scaler.transform(X_test_raw.reshape(-1, num_features)).reshape(X_test_raw.shape).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "hazine_scaler.save"))
    return X_train, X_val, X_test, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="HAZINE weak-physics-residual AUV fault diagnosis step 1")
    parser.add_argument("--root-dir", type=str, required=True, help="Dataset root directory containing train/val/test folders")
    parser.add_argument("--seq-length", type=int, default=190)
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling interval used for numerical derivatives")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs_hazine_step1")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time augmentation")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    os.makedirs(args.out_dir, exist_ok=True)

    splits = load_hazine_splits(args.root_dir, args.seq_length, args.seed)

    # Augment train only. Validation/test remain untouched.
    X_train_raw, y_train = augment_training_data(splits.X_train_raw, splits.y_train, use_augment=not args.no_augment)
    X_val_raw, y_val = splits.X_val_raw, splits.y_val
    X_test_raw, y_test = splits.X_test_raw, splits.y_test

    # Raw feature standardization for TCN input.
    X_train, X_val, X_test, _ = standardize_raw_features(X_train_raw, X_val_raw, X_test_raw, args.out_dir)

    # Weak physical residuals are computed from raw physical quantities, not standardized features.
    residualizer = HazineWeakPhysicsResidualizer(feature_names=SENSOR_COLS, dt=args.dt)
    R_train = residualizer.fit_transform(X_train_raw, y_train)
    R_val = residualizer.transform(X_val_raw)
    R_test = residualizer.transform(X_test_raw)

    print(f"Raw feature shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Residual shape: train={R_train.shape}, residuals={residualizer.residual_names}")

    # DataLoaders.
    train_raw_ds = HazineTorchDataset(X_train, y_train)
    val_raw_ds = HazineTorchDataset(X_val, y_val)
    test_raw_ds = HazineTorchDataset(X_test, y_test)

    train_phy_ds = HazineTorchDataset(X_train, y_train, R_train)
    val_phy_ds = HazineTorchDataset(X_val, y_val, R_val)
    test_phy_ds = HazineTorchDataset(X_test, y_test, R_test)

    train_raw_loader = DataLoader(train_raw_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_raw_loader = DataLoader(val_raw_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_raw_loader = DataLoader(test_raw_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_phy_loader = DataLoader(train_phy_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_phy_loader = DataLoader(val_phy_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_phy_loader = DataLoader(test_phy_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_weights = compute_class_weights(y_train, num_classes=len(FAULT_NAMES))

    # 1) Baseline: raw 13-channel temporal data only.
    print("\nTraining baseline TCN...")
    baseline = BaselineTCN(raw_dim=len(SENSOR_COLS), num_classes=len(FAULT_NAMES), hidden=args.hidden, dropout=args.dropout)
    baseline = train_one_model(
        baseline,
        train_raw_loader,
        val_raw_loader,
        class_weights,
        device,
        epochs=args.epochs,
        lr=args.lr,
        is_physics_model=False,
    )
    baseline_metrics = evaluate_model(baseline, test_raw_loader, device, is_physics_model=False)
    print_report("Test result: baseline TCN", baseline_metrics)

    # 2) Weak-physics model: raw 13-channel temporal data + weak physical residual sequences.
    print("\nTraining weak-physics-residual TCN...")
    phy_model = WeakPhysicsTCN(
        raw_dim=len(SENSOR_COLS),
        residual_dim=R_train.shape[-1],
        num_classes=len(FAULT_NAMES),
        hidden=args.hidden,
        dropout=args.dropout,
    )
    phy_model = train_one_model(
        phy_model,
        train_phy_loader,
        val_phy_loader,
        class_weights,
        device,
        epochs=args.epochs,
        lr=args.lr,
        is_physics_model=True,
    )
    phy_metrics = evaluate_model(phy_model, test_phy_loader, device, is_physics_model=True)
    print_report("Test result: TCN + weak physical residuals", phy_metrics)

    # Save outputs.
    torch.save(baseline.state_dict(), os.path.join(args.out_dir, "baseline_tcn.pt"))
    torch.save(phy_model.state_dict(), os.path.join(args.out_dir, "weak_physics_tcn.pt"))
    joblib.dump(residualizer, os.path.join(args.out_dir, "hazine_weak_physics_residualizer.save"))

    summary = {
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
        },
        "weak_physics": {
            "accuracy": phy_metrics["accuracy"],
            "macro_f1": phy_metrics["macro_f1"],
        },
        "sensor_cols": SENSOR_COLS,
        "residual_names": residualizer.residual_names,
        "fault_names": FAULT_NAMES,
        "seq_length": args.seq_length,
        "dt": args.dt,
    }
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Ablation summary")
    print("=" * 80)
    print(f"Baseline      : acc={baseline_metrics['accuracy']:.4f}, macro_f1={baseline_metrics['macro_f1']:.4f}")
    print(f"Weak-physics  : acc={phy_metrics['accuracy']:.4f}, macro_f1={phy_metrics['macro_f1']:.4f}")
    print(f"Saved outputs to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
