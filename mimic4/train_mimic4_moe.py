"""
train_mimic4_moe.py

MIMIC-IV Hospital LOS prediction — cross-dataset generalization from MIMIC-III.

Architecture (mirrors MIMIC-III M5):
  Structured LightGBM  →  pred_struct  (baseline)
  Radiology LSTM (retrained on MIMIC-IV)  →  rad_feat (4 sec × hidden-dim)
  Static history ClinicalBERT  →  PCA-64  →  hist_feat
  Expert A: bulk (LOS < P90), Expert B: tail (LOS ≥ P90)
  Gate clf → tail probability
  Final = pred_struct + (1-gate_prob)*resid_A + gate_prob*resid_B

Models:
  m1  structured_only
  m2  struct + static history (PMH/family ClinicalBERT, PCA-64)
  m3  struct + radiology LSTM (frozen MIMIC-III weights — zero-shot transfer)
  m4  struct + radiology LSTM (retrained on MIMIC-IV — upper bound)
  m5  MoE residual [OURS]: struct + hist + rad_lstm (retrained)
  m6  MoE residual [transfer]: struct + hist + rad_lstm (frozen MIMIC-III)

Usage:
  # Single run (all models, seed=42)
  python mimic4_cross_dataset/train_mimic4_moe.py

  # Single model
  python mimic4_cross_dataset/train_mimic4_moe.py --model m5

  # 5-seed run for a specific model
  python mimic4_cross_dataset/train_mimic4_moe.py --model m5 --mode seeds --seeds 0,1,2,3,4

  # 5-seed run for all models
  python mimic4_cross_dataset/train_mimic4_moe.py --model all --mode seeds --seeds 0,1,2,3,4
"""

import os
import hashlib
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import lightgbm as lgb
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--output",        default="outputs_mimic4_moe")
parser.add_argument("--mimic3-output", default="outputs_mimic3_lstm_full_gated_residual",
                    help="Path to MIMIC-III outputs (for frozen LSTM transfer)")
parser.add_argument("--embed-cache",   default="embed_cache_m4",
                    help="Directory for BERT embedding caches (shared across seeds)")
parser.add_argument("--mode",          choices=["train", "seeds"], default="train",
                    help="train=single seed, seeds=multi-seed with summary")
parser.add_argument("--model",         choices=["m1","m2","m3","m4","m5","m6","all"],
                    default="all",
                    help="Which model(s) to run")
parser.add_argument("--seed",          type=int, default=42,
                    help="Seed for --mode train")
parser.add_argument("--seeds",         default="0,1,2,3,4",
                    help="Comma-separated seeds for --mode seeds")
parser.add_argument("--lstm-epochs",   type=int, default=10)
parser.add_argument("--lstm-hidden",   type=int, default=64)
parser.add_argument("--lstm-lr",       type=float, default=1e-3)
parser.add_argument("--hist-pca-dim",  type=int, default=64)
parser.add_argument("--max-notes",     type=int, default=8)
parser.add_argument("--batch-size",    type=int, default=256)
args = parser.parse_args()

OUT       = args.output
M3_OUT    = args.mimic3_output
CACHE_DIR = args.embed_cache
HIDDEN    = args.lstm_hidden
GATE_PCT  = 90

ALL_MODELS = ["m1", "m2", "m3", "m4", "m5", "m6"]
models_to_run = ALL_MODELS if args.model == "all" else [args.model]

# Which features are needed for the selected models
NEEDS_BERT   = bool({"m2", "m5", "m6"} & set(models_to_run))
NEEDS_FROZEN = bool({"m3", "m6"} & set(models_to_run))
NEEDS_LSTM   = bool({"m4", "m5"} & set(models_to_run))
NEEDS_RAD    = NEEDS_FROZEN or NEEDS_LSTM

for d in [OUT, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Device: {device}")
print(f"Models to run: {models_to_run}")

# ── Section definitions ───────────────────────────────────────────────────────

MIMIC4_RAD_SECTIONS = [
    "radiology_indication", "radiology_technique",
    "radiology_impression", "radiology_findings",
]
M3_SECTIONS = [
    "nursing_assessment", "nursing_action", "nursing_response", "nursing_plan",
    "radiology_wet_read", "radiology_indication",
    "radiology_technique", "radiology_impression",
    "nursing_other_other_nursing_other", "nursing_other_respiratory_care",
]
M3_SEC_IDX  = {s: i for i, s in enumerate(M3_SECTIONS)}
MIMIC4_TO_M3 = {
    "radiology_indication": "radiology_indication",
    "radiology_technique":  "radiology_technique",
    "radiology_impression": "radiology_impression",
    "radiology_findings":   "radiology_wet_read",
}
M4_SEC_IDX = {s: i for i, s in enumerate(MIMIC4_RAD_SECTIONS)}

# ── Model definition ──────────────────────────────────────────────────────────

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=768, n_sections=10, sec_emb_dim=32,
                 hidden_dim=64, dropout=0.0):
        super().__init__()
        self.sec_emb = nn.Embedding(n_sections, sec_emb_dim)
        self.lstm    = nn.LSTM(input_dim + sec_emb_dim, hidden_dim, batch_first=True)
        self.attn    = nn.Linear(hidden_dim, 1)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def encode(self, x, sec_idx):
        sec = self.sec_emb(sec_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, sec], dim=-1))
        w = torch.softmax(self.attn(out), dim=1)
        return (w * out).sum(dim=1)

    def forward(self, x, sec_idx):
        return self.head(self.drop(self.encode(x, sec_idx))).squeeze(-1)


class SeqDataset(Dataset):
    def __init__(self, xs, secs, ys):
        self.X   = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.sec = torch.tensor(secs,          dtype=torch.long)
        self.y   = torch.tensor(ys,            dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.sec[i], self.y[i]

# ── Helpers ───────────────────────────────────────────────────────────────────

def pad_seq(arr):
    arr = arr[:args.max_notes]
    if len(arr) < args.max_notes:
        arr = np.vstack([arr, np.zeros((args.max_notes - len(arr), 768), np.float32)])
    return arr


@torch.no_grad()
def encode_texts(texts, tokenizer, bert, batch_size=128):
    vecs = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        vecs.append(bert(**tok).last_hidden_state[:, 0, :].cpu().numpy())
        if i % 10000 == 0 and i > 0:
            print(f"    Encoded {i}/{len(texts)}")
    return np.vstack(vecs)


@torch.no_grad()
def extract_lstm_feats(rad_seqs, hadm_ids, model, sec_map, sec_idx_map, batch_size=512):
    model.eval()
    feats = np.zeros((len(hadm_ids), len(MIMIC4_RAD_SECTIONS) * HIDDEN), np.float32)
    items = []
    for i, hid in enumerate(hadm_ids):
        for j, sec in enumerate(MIMIC4_RAD_SECTIONS):
            arr = rad_seqs[sec].get(int(hid))
            if arr is not None:
                items.append((i, j, pad_seq(arr), sec_idx_map[sec_map[sec]]))
    for start in range(0, len(items), batch_size):
        batch   = items[start:start + batch_size]
        row_ids = [b[0] for b in batch]
        col_ids = [b[1] for b in batch]
        sec_ids = [b[3] for b in batch]
        xb  = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32).to(device)
        sb  = torch.tensor(sec_ids, dtype=torch.long).to(device)
        enc = model.encode(xb, sb).cpu().numpy()
        for k, (ri, cj) in enumerate(zip(row_ids, col_ids)):
            feats[ri, cj * HIDDEN:(cj + 1) * HIDDEN] = enc[k]
    return feats


def train_lgbm(X_tr, y_tr, X_va, y_va, seed, iters=300, lr=0.05, leaves=31):
    m = lgb.LGBMRegressor(
        n_estimators=iters, learning_rate=lr, num_leaves=leaves,
        min_child_samples=20, reg_lambda=1.0, colsample_bytree=0.5,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def eval_metrics(name, y_true, y_pred, p90_thresh):
    y_true = np.asarray(y_true, np.float32)
    y_pred = np.asarray(y_pred, np.float32)
    tail   = y_true >= p90_thresh
    return {
        "model":       name,
        "MAE":         float(mean_absolute_error(y_true, y_pred)),
        "RMSE":        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":          float(r2_score(y_true, y_pred)),
        "Tail_MAE":    float(mean_absolute_error(y_true[tail],  y_pred[tail])),
        "NonTail_MAE": float(mean_absolute_error(y_true[~tail], y_pred[~tail])),
    }


def sep(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Structured data  (always loaded — deterministic across seeds)
# ═══════════════════════════════════════════════════════════════════════════════

sep("Loading structured data")

struct = pd.read_csv("mimic4_cross_dataset/processed/mimic4_structured_hosp_los_24h.csv")
struct["hadm_id"]   = struct["hadm_id"].astype(int)
struct["admittime"] = pd.to_datetime(struct["admittime"])
struct = struct[
    struct["los_days"].notna()
    & (struct["los_days"] >= 0.5)
    & (struct["los_days"] <= 365)
].copy()

DROP_COLS = {"subject_id", "hadm_id", "stay_id",
             "admittime", "dischtime", "first_icu_intime", "first_icu_outtime",
             "los_days"}
FEAT_COLS = [c for c in struct.columns if c not in DROP_COLS]

for col in struct[FEAT_COLS].select_dtypes(include="object").columns:
    struct[col] = LabelEncoder().fit_transform(
        struct[col].fillna("unknown").astype(str)
    )
for col in struct[FEAT_COLS].select_dtypes(include=[np.number]).columns:
    struct[col] = pd.to_numeric(struct[col], errors="coerce")
    struct[col] = struct[col].fillna(struct[col].median())

print(f"  Total: {len(struct):,}  |  features: {len(FEAT_COLS)}")

# Temporal split 70 / 10 / 20
struct = struct.sort_values("admittime").reset_index(drop=True)
n      = len(struct)
n_tr   = int(0.70 * n)
n_va   = int(0.10 * n)
tr_idx = np.arange(n_tr)
va_idx = np.arange(n_tr, n_tr + n_va)
te_idx = np.arange(n_tr + n_va, n)

yr = struct["admittime"].dt.year
print(f"  Train: {len(tr_idx):,}  ({yr.iloc[tr_idx[0]]}–{yr.iloc[tr_idx[-1]]})")
print(f"  Val:   {len(va_idx):,}  ({yr.iloc[va_idx[0]]}–{yr.iloc[va_idx[-1]]})")
print(f"  Test:  {len(te_idx):,}  ({yr.iloc[te_idx[0]]}–{yr.iloc[te_idx[-1]]})")

hadm_arr    = struct["hadm_id"].values
y_all       = struct["los_days"].values.astype(np.float32)
X_struct    = struct[FEAT_COLS].values.astype(np.float32)
hadm_set    = set(hadm_arr.tolist())
hadm_to_los = dict(zip(struct["hadm_id"], struct["los_days"]))

hadm_tr, hadm_va, hadm_te = hadm_arr[tr_idx], hadm_arr[va_idx], hadm_arr[te_idx]
y_tr,    y_va,    y_te     = y_all[tr_idx],    y_all[va_idx],    y_all[te_idx]
Xs_tr,   Xs_va,   Xs_te    = X_struct[tr_idx], X_struct[va_idx], X_struct[te_idx]

P90_LOS   = float(np.percentile(y_tr, GATE_PCT))
bulk_mask = y_tr < P90_LOS
tail_mask = ~bulk_mask
print(f"  P90_LOS = {P90_LOS:.1f}d  |  tail train = {tail_mask.sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ClinicalBERT  (loaded if any text model is selected)
# ═══════════════════════════════════════════════════════════════════════════════

tokenizer  = None
bert_model = None

if NEEDS_BERT or NEEDS_RAD:
    sep("Loading ClinicalBERT")
    tokenizer  = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    bert_model.eval()
    print("  Loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Static history embeddings  (m2, m5, m6 — cached, deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

hist_tr = hist_va = hist_te = None

if NEEDS_BERT:
    sep("Static history embeddings (PMH + family_history)")

    hist_df = pd.read_parquet("mimic4_cross_dataset/processed/discharge_history.parquet")
    hist_df["hadm_id"] = hist_df["hadm_id"].astype(int)
    hist_df = hist_df[hist_df["hadm_id"].isin(hadm_set)]

    _h_hash  = hashlib.md5(str(sorted(hadm_set)).encode()).hexdigest()[:12]
    _h_cache = os.path.join(CACHE_DIR, f"hist_vecs_{_h_hash}.npy")

    if os.path.exists(_h_cache):
        print(f"  [cache] {_h_cache}")
        hist_vecs_raw = np.load(_h_cache)
    else:
        hid_idx   = hist_df.set_index("hadm_id")
        all_texts = []
        text_map  = {}
        for hid in struct["hadm_id"].tolist():
            if hid not in hid_idx.index:
                continue
            rows  = hid_idx.loc[[hid]]
            parts = []
            for sec in ["pmh", "family_history"]:
                t_rows = rows[rows["section"] == sec]["text"]
                if not t_rows.empty:
                    parts.append(str(t_rows.iloc[0])[:512])
            if parts:
                text_map[int(hid)] = len(all_texts)
                all_texts.append(" ".join(parts))

        print(f"  Encoding {len(all_texts):,} history texts...")
        hist_raw = (encode_texts(all_texts, tokenizer, bert_model)
                    if all_texts else np.zeros((0, 768), np.float32))
        hist_vecs_raw = np.zeros((len(struct), 768), np.float32)
        for i, hid in enumerate(struct["hadm_id"].tolist()):
            idx = text_map.get(int(hid))
            if idx is not None:
                hist_vecs_raw[i] = hist_raw[idx]
        np.save(_h_cache, hist_vecs_raw)
        print(f"  Saved → {_h_cache}")

    print(f"  With history: {(hist_vecs_raw.any(axis=1)).sum():,} / {len(struct):,}")

    # PCA fit is deterministic (fixed random_state) — same across seeds
    pca_hist = PCA(n_components=args.hist_pca_dim, random_state=0)
    hist_tr  = pca_hist.fit_transform(hist_vecs_raw[tr_idx]).astype(np.float32)
    hist_va  = pca_hist.transform(hist_vecs_raw[va_idx]).astype(np.float32)
    hist_te  = pca_hist.transform(hist_vecs_raw[te_idx]).astype(np.float32)
    print(f"  History dim after PCA: {hist_tr.shape[1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Radiology section sequences  (m3, m4, m5, m6 — cached, deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

rad_seqs = None

if NEEDS_RAD:
    sep("Radiology section sequences")

    rad_df = pd.read_parquet("mimic4_cross_dataset/processed/radiology_sections_24h.parquet")
    rad_df["hadm_id"] = rad_df["hadm_id"].astype(int)
    rad_df = rad_df[
        rad_df["hadm_id"].isin(hadm_set)
        & rad_df["section_name"].isin(MIMIC4_RAD_SECTIONS)
        & (rad_df["text"].str.strip().str.len() > 5)
    ].copy()

    unique_texts = rad_df["text"].unique().tolist()
    _r_hash  = hashlib.md5("|".join(sorted(unique_texts)).encode()).hexdigest()[:12]
    _r_cache = os.path.join(CACHE_DIR, f"rad_vecs_{_r_hash}.npz")

    if os.path.exists(_r_cache):
        print(f"  [cache] {_r_cache}")
        d = np.load(_r_cache, allow_pickle=True)
        text_to_vec = {t: v for t, v in zip(d["texts"].tolist(), d["vecs"])}
    else:
        print(f"  Encoding {len(unique_texts):,} unique radiology texts...")
        vecs = encode_texts(unique_texts, tokenizer, bert_model)
        np.savez(_r_cache, texts=np.array(unique_texts), vecs=vecs)
        text_to_vec = {t: v for t, v in zip(unique_texts, vecs)}
        print(f"  Saved → {_r_cache}")

    rad_seqs = {s: {} for s in MIMIC4_RAD_SECTIONS}
    sort_cols = ["hadm_id", "section_name"]
    if "hours_from_admit" in rad_df.columns:
        sort_cols.append("hours_from_admit")
    for row in rad_df.sort_values(sort_cols).itertuples(index=False):
        hid, sec, text = int(row.hadm_id), row.section_name, row.text
        v = text_to_vec.get(text)
        if v is not None:
            rad_seqs[sec].setdefault(hid, []).append(v)
    for sec in MIMIC4_RAD_SECTIONS:
        for hid in list(rad_seqs[sec].keys()):
            rad_seqs[sec][hid] = np.stack(rad_seqs[sec][hid]).astype(np.float32)

    n_pairs = sum(len(rad_seqs[s]) for s in MIMIC4_RAD_SECTIONS)
    print(f"  Non-empty patient-section pairs: {n_pairs:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Frozen MIMIC-III LSTM features  (m3, m6 — deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

rad_frozen_tr = rad_frozen_va = rad_frozen_te = None

if NEEDS_FROZEN:
    sep("Frozen MIMIC-III LSTM (zero-shot transfer)")

    m3_path = os.path.join(M3_OUT, "lstm_model.pt")
    lstm_frozen = LSTMWithAttention(
        input_dim=768, n_sections=len(M3_SECTIONS),
        sec_emb_dim=32, hidden_dim=HIDDEN,
    )
    lstm_frozen.load_state_dict(torch.load(m3_path, map_location="cpu"))
    lstm_frozen.to(device).eval()
    for p in lstm_frozen.parameters():
        p.requires_grad = False
    print(f"  Loaded from {m3_path}")

    rad_frozen_tr = extract_lstm_feats(rad_seqs, hadm_tr, lstm_frozen, MIMIC4_TO_M3, M3_SEC_IDX)
    rad_frozen_va = extract_lstm_feats(rad_seqs, hadm_va, lstm_frozen, MIMIC4_TO_M3, M3_SEC_IDX)
    rad_frozen_te = extract_lstm_feats(rad_seqs, hadm_te, lstm_frozen, MIMIC4_TO_M3, M3_SEC_IDX)
    print(f"  Frozen LSTM feat dim: {rad_frozen_tr.shape[1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# run_seed — train all selected models for one random seed
# ═══════════════════════════════════════════════════════════════════════════════

def run_seed(seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"\n{'='*70}\nSeed {seed}  →  {out_dir}\n{'='*70}")

    # Save data splits (same across seeds — written once per out_dir)
    np.save(f"{out_dir}/train_y.npy",     y_tr)
    np.save(f"{out_dir}/val_y.npy",       y_va)
    np.save(f"{out_dir}/test_y.npy",      y_te)
    np.save(f"{out_dir}/train_struct.npy", Xs_tr)
    np.save(f"{out_dir}/test_struct.npy",  Xs_te)

    # ── Retrained LSTM (seed-dependent, m4 / m5) ─────────────────────────────

    rad_ret_tr = rad_ret_va = rad_ret_te = None
    if NEEDS_LSTM:
        print(f"  Training MIMIC-IV LSTM (seed={seed})...")
        train_hset = set(hadm_tr.tolist())
        val_hset   = set(hadm_va.tolist())
        xs_tr, secs_tr, ys_tr = [], [], []
        xs_va, secs_va, ys_va = [], [], []
        for j, sec in enumerate(MIMIC4_RAD_SECTIONS):
            for hid, arr in rad_seqs[sec].items():
                if hid in train_hset:
                    xs_tr.append(pad_seq(arr))
                    secs_tr.append(j)
                    ys_tr.append(hadm_to_los[hid])
                elif hid in val_hset:
                    xs_va.append(pad_seq(arr))
                    secs_va.append(j)
                    ys_va.append(hadm_to_los[hid])

        print(f"    Training samples: {len(xs_tr):,}")
        loader = DataLoader(SeqDataset(xs_tr, secs_tr, ys_tr),
                            batch_size=args.batch_size, shuffle=True)
        lstm = LSTMWithAttention(
            input_dim=768, n_sections=len(MIMIC4_RAD_SECTIONS),
            sec_emb_dim=32, hidden_dim=HIDDEN,
        ).to(device)
        opt     = torch.optim.Adam(lstm.parameters(), lr=args.lstm_lr)
        loss_fn = nn.MSELoss()
        best_state, best_val = None, float("inf")

        for epoch in range(args.lstm_epochs):
            lstm.train()
            tr_losses = []
            for xb, sb, yb in loader:
                xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(lstm(xb, sb), yb)
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())
            if xs_va:
                lstm.eval()
                with torch.no_grad():
                    vx = torch.tensor(np.stack(xs_va), dtype=torch.float32).to(device)
                    vs = torch.tensor(secs_va, dtype=torch.long).to(device)
                    vy = torch.tensor(ys_va,   dtype=torch.float32).to(device)
                    val_loss = loss_fn(lstm(vx, vs), vy).item()
                if val_loss < best_val:
                    best_val   = val_loss
                    best_state = {k: v.cpu().clone() for k, v in lstm.state_dict().items()}
                print(f"    Epoch {epoch+1}/{args.lstm_epochs}: "
                      f"train={np.mean(tr_losses):.4f}  val={val_loss:.4f}")
            else:
                print(f"    Epoch {epoch+1}/{args.lstm_epochs}: "
                      f"train={np.mean(tr_losses):.4f}")

        if best_state:
            lstm.load_state_dict(best_state)
        lstm.eval()
        torch.save(lstm.state_dict(), f"{out_dir}/lstm_retrained.pt")

        rad_ret_tr = extract_lstm_feats(rad_seqs, hadm_tr, lstm, {s: s for s in MIMIC4_RAD_SECTIONS}, M4_SEC_IDX)
        rad_ret_va = extract_lstm_feats(rad_seqs, hadm_va, lstm, {s: s for s in MIMIC4_RAD_SECTIONS}, M4_SEC_IDX)
        rad_ret_te = extract_lstm_feats(rad_seqs, hadm_te, lstm, {s: s for s in MIMIC4_RAD_SECTIONS}, M4_SEC_IDX)
        print(f"    Retrained LSTM feat dim: {rad_ret_tr.shape[1]}")

    # ── Structured baseline (always, seed-dependent GBM) ─────────────────────

    lgbm_struct = train_lgbm(Xs_tr, y_tr, Xs_va, y_va, seed)
    pred_s_tr   = lgbm_struct.predict(Xs_tr).astype(np.float32)
    pred_s_va   = lgbm_struct.predict(Xs_va).astype(np.float32)
    pred_s_te   = lgbm_struct.predict(Xs_te).astype(np.float32)
    np.save(f"{out_dir}/pred_struct_train.npy", pred_s_tr)
    np.save(f"{out_dir}/pred_struct_test.npy",  pred_s_te)
    joblib.dump(lgbm_struct, f"{out_dir}/lgbm_struct.pkl")

    resid_tr = y_tr - pred_s_tr

    # ── Per-model results ─────────────────────────────────────────────────────

    main_results = []

    # m1: structured only
    if "m1" in models_to_run:
        main_results.append(eval_metrics("m1_structured_only", y_te, pred_s_te, P90_LOS))

    # m2: struct + static history
    if "m2" in models_to_run:
        lgbm_h    = train_lgbm(np.hstack([Xs_tr, hist_tr]), y_tr,
                               np.hstack([Xs_va, hist_va]), y_va, seed)
        pred_h_te = lgbm_h.predict(np.hstack([Xs_te, hist_te])).astype(np.float32)
        np.save(f"{out_dir}/pred_hist_test.npy", pred_h_te)
        main_results.append(eval_metrics("m2_struct+history", y_te, pred_h_te, P90_LOS))

    # m3: struct + frozen M3 LSTM (transfer)
    if "m3" in models_to_run:
        lgbm_f       = train_lgbm(np.hstack([Xs_tr, rad_frozen_tr]), y_tr,
                                   np.hstack([Xs_va, rad_frozen_va]), y_va, seed)
        pred_f_te    = lgbm_f.predict(np.hstack([Xs_te, rad_frozen_te])).astype(np.float32)
        np.save(f"{out_dir}/pred_rad_frozen_test.npy", pred_f_te)
        main_results.append(eval_metrics("m3_struct+rad_frozen", y_te, pred_f_te, P90_LOS))

    # m4: struct + retrained LSTM
    if "m4" in models_to_run:
        lgbm_r    = train_lgbm(np.hstack([Xs_tr, rad_ret_tr]), y_tr,
                               np.hstack([Xs_va, rad_ret_va]), y_va, seed)
        pred_r_te = lgbm_r.predict(np.hstack([Xs_te, rad_ret_te])).astype(np.float32)
        np.save(f"{out_dir}/pred_rad_retrained_test.npy", pred_r_te)
        main_results.append(eval_metrics("m4_struct+rad_retrained", y_te, pred_r_te, P90_LOS))

    # m5: MoE residual [OURS] — struct + hist + retrained LSTM
    pred_moe_te  = None
    expert_A = expert_B = gate_clf = gate_prob_te = None
    X_exp_tr = X_exp_va = X_exp_te = None

    if "m5" in models_to_run:
        X_exp_tr = np.hstack([Xs_tr, hist_tr, rad_ret_tr])
        X_exp_va = np.hstack([Xs_va, hist_va, rad_ret_va])
        X_exp_te = np.hstack([Xs_te, hist_te, rad_ret_te])

        expert_A = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                      colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        expert_B = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                      colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        gate_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                       colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)

        expert_A.fit(X_exp_tr[bulk_mask], resid_tr[bulk_mask])
        expert_B.fit(X_exp_tr[tail_mask], resid_tr[tail_mask])
        gate_clf.fit(X_exp_tr, tail_mask.astype(int))

        gate_prob_te = gate_clf.predict_proba(X_exp_te)[:, 1].astype(np.float32)
        pred_resid_A = expert_A.predict(X_exp_te).astype(np.float32)
        pred_resid_B = expert_B.predict(X_exp_te).astype(np.float32)
        pred_moe_te  = pred_s_te + (1 - gate_prob_te) * pred_resid_A + gate_prob_te * pred_resid_B

        _true_tail = y_te >= P90_LOS
        _pred_tail = gate_prob_te > 0.5
        prec = (_pred_tail & _true_tail).sum() / max(_pred_tail.sum(), 1)
        rec  = (_pred_tail & _true_tail).sum() / max(_true_tail.sum(), 1)
        print(f"  Gate: precision={prec:.3f}  recall={rec:.3f}  "
              f"(pred={_pred_tail.sum()}, actual={_true_tail.sum()})")

        joblib.dump(expert_A, f"{out_dir}/expert_A.pkl")
        joblib.dump(expert_B, f"{out_dir}/expert_B.pkl")
        joblib.dump(gate_clf, f"{out_dir}/gate_clf.pkl")
        np.save(f"{out_dir}/pred_moe_test.npy",  pred_moe_te)
        np.save(f"{out_dir}/gate_prob_test.npy", gate_prob_te)
        main_results.append(eval_metrics("m5_moe_residual [OURS]", y_te, pred_moe_te, P90_LOS))

    # m6: MoE residual [transfer] — struct + hist + frozen LSTM
    if "m6" in models_to_run:
        X_f_tr = np.hstack([Xs_tr, hist_tr, rad_frozen_tr])
        X_f_va = np.hstack([Xs_va, hist_va, rad_frozen_va])
        X_f_te = np.hstack([Xs_te, hist_te, rad_frozen_te])

        eA_f = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                   colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        eB_f = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                   colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        gc_f = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                   colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        eA_f.fit(X_f_tr[bulk_mask], resid_tr[bulk_mask])
        eB_f.fit(X_f_tr[tail_mask], resid_tr[tail_mask])
        gc_f.fit(X_f_tr, tail_mask.astype(int))
        gp_f         = gc_f.predict_proba(X_f_te)[:, 1].astype(np.float32)
        pred_moe_f   = (pred_s_te
                        + (1 - gp_f) * eA_f.predict(X_f_te).astype(np.float32)
                        + gp_f       * eB_f.predict(X_f_te).astype(np.float32))
        np.save(f"{out_dir}/pred_moe_frozen_test.npy", pred_moe_f)
        main_results.append(eval_metrics("m6_moe_frozen_transfer", y_te, pred_moe_f, P90_LOS))

    # ── Ablation table (only when m5 is run) ─────────────────────────────────

    if "m5" in models_to_run:
        abl = []
        abl.append(eval_metrics("00_struct_only (anchor)", y_te, pred_s_te, P90_LOS))

        single_exp = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                        colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        single_exp.fit(X_exp_tr, resid_tr)
        pred_single = pred_s_te + single_exp.predict(X_exp_te).astype(np.float32)
        abl.append(eval_metrics("01_single_expert", y_te, pred_single, P90_LOS))

        eA2 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                  colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        eB2 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                  colsample_bytree=0.5, random_state=seed, n_jobs=-1, verbose=-1)
        eA2.fit(X_exp_tr, resid_tr)
        eB2.fit(X_exp_tr, resid_tr)
        pred_no_spec = (pred_s_te
                        + (1 - gate_prob_te) * eA2.predict(X_exp_te).astype(np.float32)
                        + gate_prob_te       * eB2.predict(X_exp_te).astype(np.float32))
        abl.append(eval_metrics("02_moe_no_specialisation", y_te, pred_no_spec, P90_LOS))

        abl.append(eval_metrics("03_expert_A_only", y_te, pred_s_te + pred_resid_A, P90_LOS))
        abl.append(eval_metrics("04_expert_B_only", y_te, pred_s_te + pred_resid_B, P90_LOS))
        abl.append(eval_metrics("05_moe_residual [OURS]", y_te, pred_moe_te, P90_LOS))

        oracle_g    = (y_te >= P90_LOS).astype(np.float32)
        pred_oracle = pred_s_te + (1 - oracle_g) * pred_resid_A + oracle_g * pred_resid_B
        abl.append(eval_metrics("06_oracle_gate [UB]", y_te, pred_oracle, P90_LOS))

        df_abl = pd.DataFrame(abl)
        anc_mae  = df_abl.loc[df_abl["model"].str.startswith("00"), "MAE"].iloc[0]
        anc_tail = df_abl.loc[df_abl["model"].str.startswith("00"), "Tail_MAE"].iloc[0]
        df_abl["Delta_MAE"]      = (df_abl["MAE"]      - anc_mae).round(4)
        df_abl["Delta_Tail_MAE"] = (df_abl["Tail_MAE"] - anc_tail).round(4)
        df_abl.to_csv(f"{out_dir}/results_ablation.csv", index=False)
        print("\nAblation:")
        print(df_abl[["model", "MAE", "Tail_MAE", "Delta_MAE", "Delta_Tail_MAE"]]
              .to_string(index=False, float_format="%.4f"))

    # ── Save main results ─────────────────────────────────────────────────────

    if main_results:
        df_main = pd.DataFrame(main_results)
        anc_mae = df_main.iloc[0]["MAE"]
        df_main["Delta_MAE"] = (df_main["MAE"] - anc_mae).round(4)
        df_main.to_csv(f"{out_dir}/results_main.csv", index=False)
        print(f"\nMain results (seed={seed}):")
        print(df_main.to_string(index=False, float_format="%.4f"))
        return df_main

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-dataset comparison  (optional — runs only when MIMIC-III outputs exist)
# ═══════════════════════════════════════════════════════════════════════════════

def cross_dataset_compare(m4_out_dir):
    m3_y_path = os.path.join(M3_OUT, "test_y.npy")
    if not os.path.exists(m3_y_path):
        print(f"\n  [skip] Cross-dataset compare: {M3_OUT}/test_y.npy not found.")
        print("  Run MIMIC-III training first, then re-run with --mimic3-output <path>")
        return

    sep("Cross-Dataset Comparison (MIMIC-III vs MIMIC-IV)")

    m3_y_te = np.load(m3_y_path)
    m3_p90  = float(np.percentile(np.load(os.path.join(M3_OUT, "train_y.npy")), 90))

    m3_preds = {}
    for fname, label in [("pred_m1.npy", "m1_structured_only"),
                          ("pred_m5.npy", "m5_moe_residual"),
                          ("pred_moe_test.npy", "m5_moe_residual")]:
        path = os.path.join(M3_OUT, fname)
        if os.path.exists(path) and label not in m3_preds:
            m3_preds[label] = np.load(path).astype(np.float32)

    m4_preds = {}
    for fname, label in [("pred_struct_test.npy",  "m1_structured_only"),
                          ("pred_moe_test.npy",     "m5_moe_residual")]:
        path = os.path.join(m4_out_dir, fname)
        if os.path.exists(path):
            m4_preds[label] = np.load(path).astype(np.float32)

    rows = []
    for label in sorted(set(m3_preds) | set(m4_preds)):
        row = {"model": label}
        if label in m3_preds:
            m = eval_metrics("", m3_y_te, m3_preds[label], m3_p90)
            row["M3_MAE"]      = round(m["MAE"],      4)
            row["M3_Tail_MAE"] = round(m["Tail_MAE"], 4)
            row["M3_R2"]       = round(m["R2"],        4)
        if label in m4_preds:
            m = eval_metrics("", y_te, m4_preds[label], P90_LOS)
            row["M4_MAE"]      = round(m["MAE"],      4)
            row["M4_Tail_MAE"] = round(m["Tail_MAE"], 4)
            row["M4_R2"]       = round(m["R2"],        4)
        rows.append(row)

    df_cross = pd.DataFrame(rows)
    df_cross.to_csv(f"{m4_out_dir}/cross_dataset_compare.csv", index=False)
    print(df_cross.to_string(index=False, float_format="%.4f"))
    print(f"\n  Saved → {m4_out_dir}/cross_dataset_compare.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if args.mode == "train":
    run_seed(args.seed, OUT)
    cross_dataset_compare(OUT)

else:  # seeds
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    all_dfs   = []
    for s in seed_list:
        df = run_seed(s, os.path.join(OUT, f"seed_{s}"))
        if df is not None:
            df = df.copy()
            df["seed"] = s
            all_dfs.append(df)

    if all_dfs:
        sep("Multi-Seed Summary")
        combined = pd.concat(all_dfs, ignore_index=True)
        numeric  = [c for c in combined.columns
                    if c not in ("model", "seed") and pd.api.types.is_numeric_dtype(combined[c])]
        grp = combined.groupby("model")[numeric]
        mean_df = grp.mean().round(4)
        std_df  = grp.std().round(4)
        summary = mean_df.copy()
        for col in numeric:
            summary[f"{col}_std"] = std_df[col]
        summary = summary.reset_index()
        summary.to_csv(f"{OUT}/results_seeds.csv", index=False)
        print(f"  Saved → {OUT}/results_seeds.csv")
        print(summary.to_string(index=False, float_format="%.4f"))

sep("Done")
print(f"  Outputs: {OUT}/")
