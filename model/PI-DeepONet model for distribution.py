
# -*- coding: utf-8 -*-
"""
Physics-informed DeepONet for Ti-xZr-8Nb-2Sn cyclic data (40Zr default).

- Double-head DeepONet predicting martensitic fraction xi and plastic strain ep
- Lagoudas-type reduced stress law:
      sigma = (Ea + xi * (Em - Ea)) * (eps - el * xi - ep) * 1e3   [MPa]
- Boundary conditions on xi from minimum-stress, peak-stress, and early-unloading segments
- Two-stage training: xi-only pretraining, followed by (xi, ep) with constraints

Expected CSV formats (relative to this script):
  data/Ti-40Zr-8Nb-2Sn_cyclic.csv
      columns: "Strain", "Stress", "Second"

  data/Ti-40Zr-8Nb-2Sn_BC.csv
      columns containing at least:
        - strain (or "Strain")
        - time   (or "Time")
        - mf     (martensite fraction at equality anchors)
        - mf bound (upper bound; column name may contain "bound")

Change COMPOSITION or file names below to use 45Zr / 50Zr.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Material properties and default filenames for each composition
ALLOYS = {
    # values from Table 1 (EA, EM, el) for Ti-xZr-8Nb-2Sn (x = 40, 45, 50)
    "40Zr": {
        "Ea": 49.726,   # GPa
        "Em": 27.692,   # GPa
        "el": 0.0796,
        "cyclic_file": "Ti-40Zr-8Nb-2Sn_cyclic.csv",
        "bc_file":     "Ti-40Zr-8Nb-2Sn_BC.csv",
    },
    "45Zr": {
        "Ea": 64.884,
        "Em": 36.810,
        "el": 0.0790,
        "cyclic_file": "Ti-45Zr-8Nb-2Sn_cyclic.csv",
        "bc_file":     "Ti-45Zr-8Nb-2Sn_BC.csv",
    },
    "50Zr": {
        "Ea": 59.339,
        "Em": 28.282,
        "el": 0.0810,
        "cyclic_file": "Ti-50Zr-8Nb-2Sn_cyclic.csv",
        "bc_file":     "Ti-50Zr-8Nb-2Sn_BC.csv",
    },
}

# Select composition here
COMPOSITION = "40Zr"

# Subsampling factor for raw time series (use 1 for no subsampling)
SAMPLING_NUMBER = 10


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_cyclic_data(data_dir="data"):
    cfg = ALLOYS[COMPOSITION]
    path = os.path.join(data_dir, cfg["cyclic_file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cyclic data CSV not found: {path}")

    df = pd.read_csv(path)
    # Expect columns: "Strain", "Stress", "Second"
    strain = df["Strain"].values.astype(np.float32)[::SAMPLING_NUMBER]
    stress = df["Stress"].values.astype(np.float32)[::SAMPLING_NUMBER]
    time = df["Second"].values.astype(np.float32)[::SAMPLING_NUMBER]

    # Normalization
    strain_mean, strain_std = float(np.mean(strain)), float(np.std(strain) + 1e-8)
    stress_mean, stress_std = float(np.mean(stress)), float(np.std(stress) + 1e-8)
    time_mean, time_std = float(np.mean(time)), float(np.std(time) + 1e-8)

    strain_norm = (strain - strain_mean) / strain_std
    stress_norm = (stress - stress_mean) / stress_std
    time_norm = (time - time_mean) / time_std

    return {
        "strain": strain,
        "stress": stress,
        "time": time,
        "strain_norm": strain_norm,
        "stress_norm": stress_norm,
        "time_norm": time_norm,
        "strain_mean": strain_mean,
        "strain_std": strain_std,
        "stress_mean": stress_mean,
        "stress_std": stress_std,
        "time_mean": time_mean,
        "time_std": time_std,
    }


def smooth3(x):
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    y = np.convolve(x, k, mode="same")
    if len(x) >= 2:
        y[0] = 0.5 * (x[0] + x[1])
        y[-1] = 0.5 * (x[-1] + x[-2])
    return y.astype(np.float32)


def build_history_derivative(strain_norm, time_norm):
    dot_eps = np.gradient(strain_norm, time_norm, edge_order=2).astype(np.float32)
    dot_eps = smooth3(dot_eps)
    return dot_eps


# ---------------------------------------------------------------------------
# Physics constants for chosen composition
# ---------------------------------------------------------------------------

Ea = ALLOYS[COMPOSITION]["Ea"]
Em = ALLOYS[COMPOSITION]["Em"]
el = ALLOYS[COMPOSITION]["el"]


# ---------------------------------------------------------------------------
# DeepONet input builders
# ---------------------------------------------------------------------------

def get_branch_input(series, target_index, num_samples=20, eps=1e-3):
    """
    Build non-uniformly sampled history for DeepONet branch input
    (logarithmic clustering near the target_index).
    """
    if target_index < 1:
        raise ValueError("target_index must be >= 1")
    ratios = np.logspace(np.log10(eps), 0, num_samples)
    idx = target_index - (ratios * target_index)
    idx = np.clip(np.round(idx).astype(int), 0, target_index)
    idx = np.unique(idx)
    # Pad with backward indices if needed
    while len(idx) < num_samples:
        extra = np.setdiff1d(np.arange(target_index, -1, -1), idx)
        idx = np.sort(np.concatenate([idx, extra[: num_samples - len(idx)]]))
    return series[idx].astype(np.float32)


def get_trunk_input(series, target_index):
    return np.array([series[target_index]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stress law (MPa)
# ---------------------------------------------------------------------------

def stress_pred(mf, eps_raw, ep):
    """
    σ = (Ea + mf*(Em - Ea)) * (ε - el*mf - ep) * 1e3   [MPa]
    """
    return (Ea + mf * (Em - Ea)) * (eps_raw - el * mf - ep) * 1e3


# ---------------------------------------------------------------------------
# DeepONet architecture
# ---------------------------------------------------------------------------

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim=20, trunk_input_dim=1, hidden_dim=50):
        super().__init__()

        # Branch: history of strain
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Trunk: query time
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Head for martensitic fraction xi
        self.head_mf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Head for plastic strain ep
        self.head_ep = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, branch_in, trunk_in):
        phi = self.branch(branch_in) * self.trunk(trunk_in)
        mf_hat = torch.sigmoid(self.head_mf(phi))               # xi in [0, 1]
        ep_hat = torch.nn.functional.softplus(self.head_ep(phi))  # ep >= 0
        return mf_hat, ep_hat


# ---------------------------------------------------------------------------
# Schedules and utilities
# ---------------------------------------------------------------------------

def ramp(epoch, total_epochs, start=0.0, end=1.0, t0=0.20, t1=0.70):
    """Linear ramp between [t0, t1]·total_epochs."""
    a, b = int(total_epochs * t0), int(total_epochs * t1)
    if epoch <= a:
        return start
    if epoch >= b:
        return end
    r = (epoch - a) / max(1, b - a)
    return start + r * (end - start)


class EMAMeter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


def robust_delta_from_residuals(residuals, eps=1e-8):
    """MAD-based robust scale for Huber delta."""
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med)) + eps
    return 1.4826 * mad


class HuberLoss(nn.Module):
    def __init__(self, delta=0.02):
        super().__init__()
        self.delta = delta

    def set_delta(self, delta):
        self.delta = float(delta)

    def forward(self, pred, target):
        r = pred - target
        abs_r = torch.abs(r)
        d = self.delta
        quad = 0.5 * (r ** 2)
        lin = d * (abs_r - 0.5 * d)
        return torch.mean(torch.where(abs_r <= d, quad, lin))


def soft_hinge(x, alpha):
    """
    Smooth upper-bound penalty on x <= 0.
    Penalty ~0 when x << 0, grows smoothly as x > 0.
    """
    return torch.nn.functional.softplus(x / alpha) * alpha


# ---------------------------------------------------------------------------
# BC table loading and index mapping
# ---------------------------------------------------------------------------

def load_bc_table(path):
    """
    Load BC table for xi bound/equality constraints.

    The loader is intentionally tolerant to slightly different header styles.
    It looks for columns containing:
      - 'strain'
      - 'time'
      - 'mf'
      - 'bound'
    """
    if not os.path.exists(path):
        print(f"[Warning] BC CSV not found, continuing without BCs: {path}")
        return None

    tab = pd.read_csv(path)

    # Handle layout where the first row contains custom headers in columns 1..4
    if ("Unnamed: 1" in tab.columns) and (str(tab.iloc[0, 1]).strip().lower() == "strain"):
        headers = [str(tab.iloc[0, i]).strip() for i in range(1, 5)]
        sub = tab.iloc[1:, 1:5].copy()
        sub.columns = headers
    else:
        sub = tab.copy()

    cols = {c.lower().strip(): c for c in sub.columns}

    def pick_exact(name):
        for key in cols:
            if key == name:
                return cols[key]
        return None

    def pick_contains(token):
        for key in cols:
            if token in key:
                return cols[key]
        return None

    c_strain = pick_exact("strain") or pick_contains("strain")
    c_time = pick_exact("time") or pick_contains("time")
    c_mf = pick_exact("mf") or pick_contains("mf")
    c_bound = None
    for key in cols:
        if "mf" in key and "bound" in key:
            c_bound = cols[key]
            break
    if c_bound is None:
        c_bound = pick_contains("bound")

    use_cols = [c for c in [c_strain, c_time, c_mf, c_bound] if c is not None]
    if not use_cols:
        print(f"[Warning] No recognizable BC columns in {path}")
        return None

    sub = sub[use_cols].copy()
    sub = sub.apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(how="all").reset_index(drop=True)
    return sub


def build_bc_index(time_array, bc_table):
    """
    Map BC entries onto nearest time index in the cyclic data.
    Returns:
      bound_idx[k] = upper bound on xi at time index k
      eq_idx[k]    = equality anchor xi at time index k
    """
    bound_idx = {}
    eq_idx = {}
    if bc_table is None:
        return bound_idx, eq_idx

    # Identify columns again (robust to casing)
    cols = {c.lower().strip(): c for c in bc_table.columns}
    def find(col_name, token=None):
        if col_name is not None and col_name in cols:
            return cols[col_name]
        if token is not None:
            for key in cols:
                if token in key:
                    return cols[key]
        return None

    tcol = find("time", token="time")
    scol = find("mf", token="mf")
    bcol = None
    for key in cols:
        if "bound" in key:
            bcol = cols[key]
            break

    if tcol is None:
        print("[Warning] BC table has no time column; skipping BCs.")
        return bound_idx, eq_idx

    for _, row in bc_table.iterrows():
        if not np.isfinite(row[tcol]):
            continue
        t_star = float(row[tcol])
        k = int(np.argmin(np.abs(time_array - t_star)))

        # upper bound
        if bcol is not None and np.isfinite(row.get(bcol, np.nan)):
            u = float(row[bcol])
            bound_idx[k] = min(u, bound_idx.get(k, 1.0))

        # equality anchor
        if scol is not None and np.isfinite(row.get(scol, np.nan)):
            s = float(row[scol])
            eq_idx[k] = s if k not in eq_idx else 0.5 * (eq_idx[k] + s)

    return bound_idx, eq_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pinn():
    # Load data
    data = load_cyclic_data(data_dir="data")
    strain = data["strain"]
    stress = data["stress"]
    time = data["time"]
    strain_norm = data["strain_norm"]
    stress_norm = data["stress_norm"]
    time_norm = data["time_norm"]
    stress_mean = data["stress_mean"]
    stress_std = data["stress_std"]

    dot_eps = build_history_derivative(strain_norm, time_norm)

    # Load BCs
    bc_path = os.path.join("data", ALLOYS[COMPOSITION]["bc_file"])
    bc_table = load_bc_table(bc_path)
    bound_idx, eq_idx = build_bc_index(time, bc_table)

    # Model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONet().to(device)

    # Bias ep head to negative so softplus(ep) ~ 0 initially
    with torch.no_grad():
        last_ep = model.head_ep[-1]
        if isinstance(last_ep, nn.Linear):
            last_ep.bias.fill_(-5.0)

    ep_params = list(model.head_ep.parameters())
    mf_params = [p for n, p in model.named_parameters() if not n.startswith("head_ep")]

    base_lr = 1e-5
    opt = optim.Adam(
        [
            {"params": mf_params, "lr": base_lr},
            {"params": ep_params, "lr": base_lr},
        ]
    )

    epochs = 3000
    eta_min = 2e-6
    eta_max = 3e-4
    lr_now = base_lr
    lr_growth = 1.01
    lr_decay = 0.9
    patience = 5

    huber = HuberLoss(delta=0.02)
    ema_loss = EMAMeter(alpha=0.1)

    t0 = 20  # first index used for DeepONet histories
    freeze_until = int(0.2 * epochs)  # stage-I: xi-only
    clip_norm = 1.0

    prev_smoothed = None
    worse_streak = 0
    better_streak = 0

    def get_schedules(epoch):
        lam_bound = ramp(epoch, epochs, start=0.0, end=0.60, t0=0.00, t1=0.20)
        lam_mf = ramp(epoch, epochs, start=0.0, end=0.30, t0=0.20, t1=0.70)
        lam_ep = ramp(epoch, epochs, start=0.0, end=0.10, t0=0.20, t1=0.70)
        alpha_b = 0.03 - 0.02 * ramp(
            epoch, epochs, start=0.0, end=1.0, t0=0.00, t1=0.60
        )  # sharper hinge over epochs
        gamma_eq = 0.03
        return lam_bound, lam_mf, lam_ep, alpha_b, gamma_eq

    for epoch in range(epochs):
        total_loss = 0.0
        residual_bucket = []

        # Stage-I: freeze ep head
        if epoch < freeze_until:
            for p in ep_params:
                p.requires_grad = False
        else:
            for p in ep_params:
                p.requires_grad = True

        # Initialize previous state at t0
        with torch.no_grad():
            b0 = torch.tensor(
                get_branch_input(strain_norm, t0, num_samples=20),
                device=device,
            ).unsqueeze(0)
            t0_in = torch.tensor(
                get_trunk_input(time_norm, t0), device=device
            ).unsqueeze(0)
            mf_prev, ep_prev = model(b0, t0_in)
            ep_prev = torch.clamp(ep_prev, min=0.0)

        g_ep = ramp(epoch, epochs, start=0.0, end=1.0, t0=0.20, t1=0.70)
        lam_bound, lam_mf, lam_ep, alpha_b, gamma_eq = get_schedules(epoch)

        for k in range(t0, len(strain_norm)):
            b = torch.tensor(
                get_branch_input(strain_norm, k, num_samples=20),
                device=device,
            ).unsqueeze(0)
            t_in = torch.tensor(
                get_trunk_input(time_norm, k), device=device
            ).unsqueeze(0)
            eps_raw = torch.tensor([[strain[k]]], device=device)
            y_meas_n = torch.tensor([[stress_norm[k]]], device=device)

            opt.zero_grad()

            mf_hat, ep_hat = model(b, t_in)

            de = torch.tensor([[dot_eps[k]]], device=device)
            load_mask = torch.relu(torch.sign(de))

            # Monotone ep during loading, hold during unloading
            if epoch < freeze_until:
                ep_t = ep_prev
            else:
                ep_t = torch.where(
                    load_mask.bool(),
                    torch.maximum(ep_prev, ep_hat),
                    ep_prev,
                )

            y_pred = stress_pred(mf_hat, eps_raw, g_ep * ep_t)
            y_pred_n = (y_pred - stress_mean) / stress_std

            resid = (y_pred_n - y_meas_n).detach().cpu().numpy().ravel()[0]
            residual_bucket.append(resid)

            loss = huber(y_pred_n, y_meas_n)

            # (1) xi upper bound inequality BC
            if lam_bound > 0 and k in bound_idx:
                u = torch.tensor([[bound_idx[k]]], device=device)
                L_bound = soft_hinge(mf_hat - u, alpha_b)
                loss = loss + lam_bound * L_bound

            # (2) xi equality anchor (only active when below bound)
            if lam_mf > 0 and k in eq_idx:
                s = torch.tensor([[eq_idx[k]]], device=device)
                if k in bound_idx:
                    u = torch.tensor([[bound_idx[k]]], device=device)
                    gate = torch.sigmoid((u - mf_hat) / gamma_eq)
                else:
                    gate = torch.ones_like(mf_hat)
                L_eq = HuberLoss(delta=0.02)(mf_hat, s)
                loss = loss + lam_mf * gate * L_eq

            # (3) ep temporal smoothing
            if lam_ep > 0:
                L_ep = torch.abs(ep_t - ep_prev)
                loss = loss + lam_ep * L_ep

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()

            total_loss += float(loss.item())

            with torch.no_grad():
                mf_prev = mf_hat.detach()
                ep_prev = ep_t.detach()

        # Update Huber delta from residuals
        if residual_bucket:
            delta_new = robust_delta_from_residuals(np.array(residual_bucket)) + 1e-5
            delta_new = max(0.5 * huber.delta, min(3.0 * huber.delta, delta_new))
            huber.set_delta(delta_new)

        avg_loss = total_loss / max(1, len(strain_norm) - t0)
        smoothed = ema_loss.update(avg_loss)

        # Simple EMA-based LR scheduling
        if epoch > 0:
            rel = (prev_smoothed - smoothed) / max(prev_smoothed, 1e-12)
            if rel > 0.005:
                better_streak += 1
                worse_streak = 0
                if better_streak >= patience:
                    lr_now = min(eta_max, lr_now * lr_growth)
                    better_streak = 0
            elif rel < -0.002:
                worse_streak += 1
                better_streak = 0
                if worse_streak >= 1:
                    lr_now = max(eta_min, lr_now * lr_decay)
                    worse_streak = 0
            for g in opt.param_groups:
                g["lr"] = lr_now

        prev_smoothed = smoothed

        if epoch % 10 == 0:
            print(
                f"[{COMPOSITION}] Epoch {epoch:5d}  "
                f"Loss {avg_loss:.6f}  EMA {smoothed:.6f}  "
                f"g_ep={g_ep:.3f}  "
                f"lam_bound={lam_bound:.2f} lam_mf={lam_mf:.2f} lam_ep={lam_ep:.2f}  "
                f"alpha_b={alpha_b:.3f}  delta={huber.delta:.4f}  lr={lr_now:.2e}  "
                f"freeze<{freeze_until}"
            )

    print("Training complete.")
    return model, data, dot_eps, bound_idx, eq_idx, t0


# ---------------------------------------------------------------------------
# Inference and export
# ---------------------------------------------------------------------------

def run_inference_and_export(model, data, dot_eps, t0):
    device = next(model.parameters()).device

    strain = data["strain"]
    stress = data["stress"]
    time = data["time"]
    strain_norm = data["strain_norm"]
    time_norm = data["time_norm"]

    mf_list, ep_list, sp_list = [], [], []

    model.eval()
    with torch.no_grad():
        # initialize previous state at t0
        b0 = torch.tensor(
            get_branch_input(strain_norm, t0, num_samples=20),
            device=device,
        ).unsqueeze(0)
        t0_in = torch.tensor(
            get_trunk_input(time_norm, t0), device=device
        ).unsqueeze(0)
        mf_prev, ep_prev = model(b0, t0_in)
        ep_prev = torch.clamp(ep_prev, min=0.0)

        for k in range(t0, len(strain_norm)):
            b = torch.tensor(
                get_branch_input(strain_norm, k, num_samples=20),
                device=device,
            ).unsqueeze(0)
            t_in = torch.tensor(
                get_trunk_input(time_norm, k), device=device
            ).unsqueeze(0)

            mf_hat, ep_hat = model(b, t_in)

            de = torch.tensor([[dot_eps[k]]], device=device)
            load_mask = torch.relu(torch.sign(de))
            ep_t = torch.where(
                load_mask.bool(), torch.maximum(ep_prev, ep_hat), ep_prev
            )

            s_pred = stress_pred(
                mf_hat,
                torch.tensor([[strain[k]]], device=device),
                ep_t,
            )

            mf_list.append(float(mf_hat.item()))
            ep_list.append(float(ep_t.item()))
            sp_list.append(float(s_pred.item()))

            mf_prev, ep_prev = mf_hat, ep_t

    # Prepare arrays for plotting/export
    time_valid = time[t0:]
    stress_valid = stress[t0:]
    strain_valid = strain[t0:]

    # Plot (optional)
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 2, 1)
    plt.plot(time_valid, mf_list, label="xi")
    plt.xlabel("Time (s)")
    plt.ylabel("Martensite fraction")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time_valid, sp_list, label="σ pred")
    plt.xlabel("Time (s)")
    plt.ylabel("Stress (MPa)")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(time, stress, label="σ meas")
    plt.xlabel("Time (s)")
    plt.ylabel("Stress (MPa)")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time, strain, label="ε")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(strain_valid, stress_valid, label="σ-ε meas")
    plt.plot(strain_valid, sp_list, "--", label="σ-ε pred")
    plt.xlabel("Strain")
    plt.ylabel("Stress (MPa)")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(time_valid, ep_list, label="εp")
    plt.xlabel("Time (s)")
    plt.ylabel("Plastic strain")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Saving models and CSVs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("models", COMPOSITION, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Save weights and full model
    torch.save(
        model.state_dict(), os.path.join(save_dir, "deeponet_weights_ep_adaptive.pt")
    )
    torch.save(model, os.path.join(save_dir, "deeponet_model_ep_adaptive.pth"))
    print(f"Model saved to: {save_dir}")

    # Time-series export
    df_valid = pd.DataFrame(
        {
            "time_valid_s": np.asarray(time_valid, dtype=np.float32),
            "xi": np.asarray(mf_list, dtype=np.float32),
            "ep": np.asarray(ep_list, dtype=np.float32),
            "sigma_pred_MPa": np.asarray(sp_list, dtype=np.float32),
            "stress_meas_MPa": np.asarray(stress_valid, dtype=np.float32),
            "strain": np.asarray(strain_valid, dtype=np.float32),
        }
    )

    df_raw = pd.DataFrame(
        {
            "time_s": np.asarray(time, dtype=np.float32),
            "stress_MPa": np.asarray(stress, dtype=np.float32),
            "strain": np.asarray(strain, dtype=np.float32),
        }
    )

    df_ss = pd.DataFrame(
        {
            "strain_valid": np.asarray(strain_valid, dtype=np.float32),
            "stress_meas_MPa": np.asarray(stress_valid, dtype=np.float32),
            "stress_pred_MPa": np.asarray(sp_list, dtype=np.float32),
        }
    )

    csv_valid = os.path.join(save_dir, "timeseries_valid.csv")
    csv_raw = os.path.join(save_dir, "raw_full.csv")
    csv_ss = os.path.join(save_dir, "stress_strain_meas_pred.csv")

    df_valid.to_csv(csv_valid, index=False)
    df_raw.to_csv(csv_raw, index=False)
    df_ss.to_csv(csv_ss, index=False)

    print("Saved CSVs:")
    print("  ", csv_valid)
    print("  ", csv_raw)
    print("  ", csv_ss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, data, dot_eps, _, _, t0 = train_pinn()
    run_inference_and_export(model, data, dot_eps, t0)
