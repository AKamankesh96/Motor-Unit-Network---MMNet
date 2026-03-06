"""
Motor-unit Mode Network (MMNet)

Implementation of the MMNet framework for extracting motor-unit modes from
motor-unit (MU) discharge-rate signals.

This script trains a variational autoencoder with a CNN–LSTM encoder and
BiLSTM decoder to model motor-unit discharge-rate activity and extract
low-dimensional latent representations (motor-unit modes).

Pipeline:
- Load MU discharge-rate signals from a MATLAB file
- Segment the signal into overlapping windows
- Train the MMNet model
- Optionally reconstruct the full signal and extract latent time series (motor-unit modes)

Author: Alireza Kamankesh
"""

# ---------------------------- Imports ----------------------------

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers

# ---------------------------- Configuration ----------------------------

@dataclass(frozen=True)
class Config:
    """Container for model and preprocessing settings."""

    seed: int = 42
    fs_hz: int = 2000

    # preprocessing
    trim_s: float = 0.4
    eps: float = 1e-8

    # windowing
    window: int = 1000
    step: int = 250

    # training
    latent_dim: int = 3
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    dropout: float = 0.2
    noise_std: float = 0.05

    # VAE
    beta_kl: float = 0.1

    # train–validation split fraction
    val_frac: float = 0.2


def set_global_determinism(seed: int) -> None:
    """Best-effort reproducibility across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ---------------------------- Data utilities ----------------------------

def load_mat_concatenated_data(mat_path: str | Path, key: str = "concatenated_data") -> np.ndarray:
    """Load the MU matrix from a MATLAB file.
    Returns
    -------
    np.ndarray
        Array with shape [n_mus, n_samples].
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = scipy.io.loadmat(str(mat_path))
    if key not in mat:
        raise KeyError(f"Variable '{key}' not found in {mat_path.name}. Available keys include: {list(mat.keys())[:20]}")

    data = np.asarray(mat[key], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected '{key}' to be 2D [n_mus, n_samples], got shape {data.shape}")

    return data


def zscore_per_mu(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each MU independently across time."""
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + eps)


def trim_edges(data_TxM: np.ndarray, fs_hz: int, trim_s: float) -> np.ndarray:
    """Trim samples from the start and end of a [time, channels] array."""
    n_trim = int(round(trim_s * fs_hz))
    if 2 * n_trim >= data_TxM.shape[0]:
        raise ValueError("trim_s is too large for the available signal length.")
    return data_TxM[n_trim:-n_trim]


def make_overlapping_windows(data_TxM: np.ndarray, window: int, step: int) -> np.ndarray:
    """Convert a continuous signal into overlapping windows.
    Parameters
    ----------
    data_TxM : np.ndarray
        Input signal with shape [T, M].
    window : int
        Window length in samples.
    step : int
        Step size in samples.
    Returns
    -------
    np.ndarray
        Array with shape [n_windows, window, M].
    """
    T, _ = data_TxM.shape
    if T < window:
        raise ValueError(f"Signal too short for requested window size: T={T}, window={window}")

    starts = range(0, T - window + 1, step)
    windows = np.stack([data_TxM[s:s + window] for s in starts], axis=0)
    return windows.astype(np.float32)


def train_val_split(windows: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/validation split without sklearn."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    
    n = windows.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return windows[train_idx], windows[val_idx]


def make_tf_datasets(train_x: np.ndarray, val_x: np.ndarray, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create TensorFlow datasets for autoencoder training."""
    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_x)
        .shuffle(buffer_size=min(4096, max(1, train_x.shape[0])), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_x)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds

# ---------------------------- Model components ----------------------------

class LearnablePositionalEncoding(layers.Layer):
    """Trainable positional encoding for sequence data.

    The layer stores a [1, max_len, d_model] tensor. If an input sequence is
    shorter than max_len, it is sliced; if it is longer, the encoding is
    interpolated along the sequence axis.
    """

    def __init__(self, max_len: int, d_model: int, name: str = "pos_enc"):
        super().__init__(name=name)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(1, self.max_len, self.d_model),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]

        def _slice() -> tf.Tensor:
            return self.pos_encoding[:, :seq_len, :]

        def _interp() -> tf.Tensor:
            pe = self.pos_encoding[..., tf.newaxis]  # [1, max_len, d_model, 1]
            pe = tf.image.resize(pe, size=(seq_len, self.d_model), method="bilinear")
            return tf.squeeze(pe, axis=-1)

        pe = tf.cond(seq_len <= self.max_len, _slice, _interp)
        return x + pe


class SequenceLatentVAE(tf.keras.Model):
    """Sequence-latent VAE for MU discharge-rate signals.

    Architecture
    ------------
    Encoder: Conv1D -> BatchNorm -> Dropout -> Positional Encoding -> LSTM -> LayerNorm
    Latent:  Dense(mean), Dense(logvar)
    Decoder: Dense expansion -> BiLSTM -> TimeDistributed Dense
    """

    def __init__(
        self,
        n_mus: int,
        window: int,
        latent_dim: int,
        dropout: float = 0.2,
        l2_reg: float = 1e-4,
        name: str = "mmnet_vaelstm",
    ):
        super().__init__(name=name)
        self.n_mus = int(n_mus)
        self.window = int(window)
        self.latent_dim = int(latent_dim)

        self.encoder = tf.keras.Sequential(
            [
                layers.Conv1D(32, kernel_size=5, padding="same", activation="gelu"),
                layers.BatchNormalization(),
                layers.Dropout(dropout),
                LearnablePositionalEncoding(max_len=window, d_model=32),
                layers.LSTM(64, return_sequences=True),
                layers.LayerNormalization(),
            ],
            name="encoder",
        )

        self.to_mean = layers.Dense(self.latent_dim, name="to_mean")
        self.to_logvar = layers.Dense(self.latent_dim, name="to_logvar")

        self.expand = tf.keras.Sequential([
            layers.Dense(32, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
        ], name="expand")

        self.decoder = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.TimeDistributed(layers.Dense(self.n_mus)),
        ], name="decoder")

    @staticmethod
    def reparameterize(mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        h = self.encoder(x, training=training)
        mean = self.to_mean(h)
        logvar = self.to_logvar(h)
        z = self.reparameterize(mean, logvar)
        z_up = self.expand(z, training=training)
        recon = self.decoder(z_up, training=training)
        return mean, logvar, z, recon

# ---------------------------- Loss + training ----------------------------

def balanced_recon_mse(
    x_true: tf.Tensor,
    x_pred: tf.Tensor,
    mu_groups: Optional[Dict[str, Iterable[int]]] = None,
) -> tf.Tensor:
    """Reconstruction MSE.

    If `mu_groups` is given, the loss is computed within each group and then
    averaged across groups. This can reduce bias when some muscles contain more
    MUs than others.
    """
    if not mu_groups:
        return tf.reduce_mean(tf.square(x_true - x_pred))

    group_losses = []
    for idx in mu_groups.values():
        idx = tf.constant(list(idx), dtype=tf.int32)
        xt = tf.gather(x_true, idx, axis=-1)
        xp = tf.gather(x_pred, idx, axis=-1)
        group_losses.append(tf.reduce_mean(tf.square(xt - xp)))

    return tf.add_n(group_losses) / tf.cast(len(group_losses), tf.float32)


def kl_divergence(mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    """KL[q(z|x) || N(0, I)] averaged over batch, time, and latent dimensions."""
    return -0.5 * tf.reduce_mean(1.0 + logvar - tf.square(mean) - tf.exp(logvar))


@tf.function

def train_step(
    model: SequenceLatentVAE,
    optimizer: tf.keras.optimizers.Optimizer,
    x: tf.Tensor,
    beta_kl: float,
    noise_std: float,
    mu_groups: Optional[Dict[str, Iterable[int]]] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """One training step."""
    x = tf.cast(x, tf.float32)
    x_in = x + tf.random.normal(tf.shape(x), stddev=noise_std) if noise_std > 0 else x

    with tf.GradientTape() as tape:
        mean, logvar, _, recon = model(x_in, training=True)
        recon_loss = balanced_recon_mse(x, recon, mu_groups)
        kl = kl_divergence(mean, logvar)
        loss = recon_loss + beta_kl * kl

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, recon_loss, kl


@tf.function

def val_step(
    model: SequenceLatentVAE,
    x: tf.Tensor,
    beta_kl: float,
    mu_groups: Optional[Dict[str, Iterable[int]]] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """One validation step."""
    x = tf.cast(x, tf.float32)
    mean, logvar, _, recon = model(x, training=False)
    recon_loss = balanced_recon_mse(x, recon, mu_groups)
    kl = kl_divergence(mean, logvar)
    loss = recon_loss + beta_kl * kl
    return loss, recon_loss, kl


def fit(
    model: SequenceLatentVAE,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg: Config,
    mu_groups: Optional[Dict[str, Iterable[int]]] = None,
    ckpt_path: Optional[str | Path] = None,
) -> Dict[str, list]:
    """Train the model with best-validation checkpointing."""
    optimizer = tf.keras.optimizers.Adam(cfg.lr)
    history = {"loss": [], "recon": [], "kl": [], "val_loss": [], "val_recon": [], "val_kl": []}
    best_val = np.inf

    for epoch in range(1, cfg.epochs + 1):
        tr = [0.0, 0.0, 0.0]
        n_tr = 0
        for x in train_ds:
            loss, recon, kl = train_step(model, optimizer, x, cfg.beta_kl, cfg.noise_std, mu_groups)
            tr[0] += float(loss)
            tr[1] += float(recon)
            tr[2] += float(kl)
            n_tr += 1
        tr = [v / max(n_tr, 1) for v in tr]

        va = [0.0, 0.0, 0.0]
        n_va = 0
        for x in val_ds:
            loss, recon, kl = val_step(model, x, cfg.beta_kl, mu_groups)
            va[0] += float(loss)
            va[1] += float(recon)
            va[2] += float(kl)
            n_va += 1
        va = [v / max(n_va, 1) for v in va]

        history["loss"].append(tr[0])
        history["recon"].append(tr[1])
        history["kl"].append(tr[2])
        history["val_loss"].append(va[0])
        history["val_recon"].append(va[1])
        history["val_kl"].append(va[2])

        if va[0] < best_val:
            best_val = va[0]
            if ckpt_path is not None:
                model.save_weights(str(ckpt_path))

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"loss={tr[0]:.6f} (recon={tr[1]:.6f}, kl={tr[2]:.6f}) | "
                f"val={va[0]:.6f} (recon={va[1]:.6f}, kl={va[2]:.6f})"
            )

    return history

# ---------------------------- Inference utilities ----------------------------

def reconstruct_full_signal(
    model: SequenceLatentVAE,
    data_TxM: np.ndarray,
    window: int,
    step: int,
) -> np.ndarray:
    """Reconstruct a full-length signal by overlap-adding window reconstructions."""
    T, M = data_TxM.shape
    recon_accum = np.zeros((T, M), dtype=np.float32)
    weight = np.zeros((T, M), dtype=np.float32)

    for start in range(0, T - window + 1, step):
        seg = data_TxM[start:start + window][None, ...].astype(np.float32)
        _, _, _, recon = model(seg, training=False)
        recon = recon.numpy()[0]
        recon_accum[start:start + window] += recon
        weight[start:start + window] += 1.0

    return recon_accum / np.maximum(weight, 1.0)


def extract_latent_full(
    model: SequenceLatentVAE,
    data_TxM: np.ndarray,
    window: int,
    step: int,
) -> np.ndarray:
    """Extract a full-length latent time series (MU mode) using overlap-averaged latent means."""
    T, _ = data_TxM.shape
    latent_dim = model.latent_dim

    latent_accum = np.zeros((T, latent_dim), dtype=np.float32)
    count = np.zeros((T, latent_dim), dtype=np.float32)

    for start in range(0, T - window + 1, step):
        seg = data_TxM[start:start + window][None, ...].astype(np.float32)
        mean, _, _, _ = model(seg, training=False)
        mean = mean.numpy()[0]
        latent_accum[start:start + window] += mean
        count[start:start + window] += 1.0

    return latent_accum / np.maximum(count, 1.0)


def variance_explained_percent(x_true: np.ndarray, x_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute variance explained (%) across all channels and time points."""
    mse = np.mean((x_true - x_pred) ** 2)
    var = np.var(x_true)
    return 100.0 * (1.0 - mse / (var + eps))


# ---------------------------- CLI ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MMNet VAE-LSTM on MU discharge-rate signals.")
    parser.add_argument("--mat_path", type=str, required=True, help="Path to MAT file containing concatenated_data")
    parser.add_argument("--mat_key", type=str, default="concatenated_data", help="MAT variable name")
    parser.add_argument("--latent_dim", type=int, default=4, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--window", type=int, default=1000, help="Window length in samples")
    parser.add_argument("--step", type=int, default=250, help="Window step in samples")
    parser.add_argument("--beta_kl", type=float, default=0.1, help="KL-divergence weight")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--save_full_recon", action="store_true", help="Save overlap-added full reconstruction")
    parser.add_argument("--save_full_latent", action="store_true", help="Save overlap-added latent time series")
    parser.add_argument("--save_config", action="store_true", help="Save configuration as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window=args.window,
        step=args.step,
        beta_kl=args.beta_kl,
    )

    set_global_determinism(cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_weights_latent{cfg.latent_dim}.weights.h5"

    data_MxT = load_mat_concatenated_data(args.mat_path, key=args.mat_key)
    data_MxT = zscore_per_mu(data_MxT, eps=cfg.eps)
    data_TxM = trim_edges(data_MxT.T, fs_hz=cfg.fs_hz, trim_s=cfg.trim_s)

    windows = make_overlapping_windows(data_TxM, window=cfg.window, step=cfg.step)
    train_x, val_x = train_val_split(windows, val_frac=cfg.val_frac, seed=cfg.seed)
    train_ds, val_ds = make_tf_datasets(train_x, val_x, batch_size=cfg.batch_size)

    n_mus = data_TxM.shape[1]
    model = SequenceLatentVAE(
        n_mus=n_mus,
        window=cfg.window,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    )

    _ = model(tf.zeros((1, cfg.window, n_mus), dtype=tf.float32), training=False)

    print(f"Loaded data with {n_mus} motor units and {data_TxM.shape[0]} time samples after trimming.")
    print(f"Windowed dataset shape: {windows.shape}")

    history = fit(model, train_ds, val_ds, cfg, mu_groups=None, ckpt_path=ckpt_path)

    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))

    np.save(out_dir / "history.npy", history, allow_pickle=True)

    if args.save_config:
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

    if args.save_full_recon:
        recon_full = reconstruct_full_signal(model, data_TxM, window=cfg.window, step=cfg.step)
        np.save(out_dir / "recon_full.npy", recon_full)
        ve = variance_explained_percent(data_TxM, recon_full, eps=cfg.eps)
        print(f"Full-signal variance explained: {ve:.2f}%")

    if args.save_full_latent:
        latent_full = extract_latent_full(model, data_TxM, window=cfg.window, step=cfg.step)
        np.save(out_dir / "latent_full.npy", latent_full)

    print(f"\nDone. Outputs saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
