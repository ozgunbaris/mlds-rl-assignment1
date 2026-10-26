"""Shared helpers for the policy-gradient experiments (no plotting here)."""
import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discounted_returns(rewards, gamma: float, reset_on_nonzero: bool = False) -> np.ndarray:
    """G_t = sum_{k >= t} gamma^(k - t) r_k, computed with a backward recursion.

    reset_on_nonzero=True restarts the running sum at every nonzero reward
    (a Pong-specific trick that treats every point as its own episode). The
    assignment asks for the discounted reward over the whole episode, so the
    default is False; the flag exists only for the ablation in the report.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        if reset_on_nonzero and rewards[t] != 0.0:
            running = 0.0
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def moving_average(x, window: int = 100) -> np.ndarray:
    """Trailing moving average; the first window-1 entries average what is available."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(len(x))
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


class CsvLogger:
    """Append-only CSV writer that flushes after every row (safe to plot mid-run)."""

    def __init__(self, path: str, fieldnames, resume: bool = False):
        self.path = path
        self.fieldnames = list(fieldnames)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        mode = "a" if (resume and exists) else "w"
        self.f = open(path, mode, newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        if mode == "w":
            self.w.writeheader()
            self.f.flush()

    def write(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self.f.flush()

    def close(self):
        self.f.close()


def read_csv_column(path: str, column: str, dtype=float):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return np.array([dtype(row[column]) for row in r])
