"""
utils.py — Shared constants, model architecture, and helpers for MIMIC-3 LOS experiments.
"""

import random
import numpy as np
import torch
import torch.nn as nn

# ── Section constants ──────────────────────────────────────────────────────────

NURSING_SECTIONS = [
    "nursing_assessment",
    "nursing_action",
    "nursing_response",
    "nursing_plan",
]

RADIOLOGY_SECTIONS = [
    "radiology_wet_read",
    "radiology_indication",
    "radiology_technique",
    "radiology_impression",
]

NURSING_OTHER_SECTIONS = [
    "nursing_other_other_nursing_other",
    "nursing_other_respiratory_care",
]

SECTIONS_TO_USE = NURSING_SECTIONS + RADIOLOGY_SECTIONS + NURSING_OTHER_SECTIONS


# ── Model ──────────────────────────────────────────────────────────────────────

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=768, n_sections=10, sec_emb_dim=32, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.sec_emb = nn.Embedding(n_sections, sec_emb_dim)
        self.lstm = nn.LSTM(
            input_dim + sec_emb_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x, sec_idx):
        sec = self.sec_emb(sec_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, sec], dim=-1))
        w = torch.softmax(self.attn(out), dim=1)
        return (w * out).sum(dim=1)

    def forward(self, x, sec_idx):
        return self.head(self.drop(self.encode(x, sec_idx))).squeeze(-1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def pad_seq(arr, max_notes=8):
    arr = arr[:max_notes]
    if len(arr) < max_notes:
        arr = np.vstack([arr, np.zeros((max_notes - len(arr), 768), dtype=np.float32)])
    return arr


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
