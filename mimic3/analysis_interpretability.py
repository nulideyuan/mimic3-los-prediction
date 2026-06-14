"""
analysis_interpretability.py

Interpretability figures for EMNLP paper.
Run AFTER: train_section_lstm_full.py --mode train

Produces <output>/figs/:
  fig1_recency_violin.pdf   — Attention recency index by LOS quartile
  fig2_case_study.pdf       — 24h note trajectory for short-stay vs long-stay patient
  fig3_los_bucket.pdf       — MAE per LOS bucket
  fig4_expert_importance.pdf — Expert A vs B section importance

Usage:
  python mimic3_dataset/analysis_interpretability.py \
      --output outputs_mimic3_lstm_full_gated_residual
"""

import os
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error

from utils import LSTMWithAttention, pad_seq, SECTIONS_TO_USE, \
    NURSING_SECTIONS, RADIOLOGY_SECTIONS, NURSING_OTHER_SECTIONS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs_mimic3_lstm_full_gated_residual")
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--n-importance-repeats", type=int, default=5)
args = parser.parse_args()

OUT  = args.output
SEED = args.seed
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)
np.random.seed(SEED)

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ─────────────────────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

BLUE   = "#2166AC"
ORANGE = "#D6604D"
GREEN  = "#4DAC26"
GRAY   = "#888888"

SEC_ABBR = {
    "nursing_assessment":                "Nurs.\nAssess",
    "nursing_action":                    "Nurs.\nAction",
    "nursing_response":                  "Nurs.\nResp",
    "nursing_plan":                      "Nurs.\nPlan",
    "radiology_wet_read":                "Rad.\nWetRead",
    "radiology_indication":              "Rad.\nIndic",
    "radiology_technique":               "Rad.\nTech",
    "radiology_impression":              "Rad.\nImpr",
    "nursing_other_other_nursing_other": "Nurs.\nOther",
    "nursing_other_respiratory_care":    "Nurs.\nRespCare",
}

# Color per note source (for case study)
SEC_COLOR = {}
for s in NURSING_SECTIONS:
    SEC_COLOR[s] = "#4CAF50"   # green
for s in RADIOLOGY_SECTIONS:
    SEC_COLOR[s] = "#FF9800"   # orange
for s in NURSING_OTHER_SECTIONS:
    SEC_COLOR[s] = "#E91E63"   # pink

# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nLoading artifacts from {OUT}/")

y_test         = np.load(f"{OUT}/test_y.npy")
y_train        = np.load(f"{OUT}/train_y.npy")
lstm_feat_test = np.load(f"{OUT}/test_lstm.npy")
X_struct_test  = np.load(f"{OUT}/test_struct.npy")
pred_m1        = np.load(f"{OUT}/pred_m1.npy")
pred_m5        = np.load(f"{OUT}/pred_m5.npy")
test_hadm      = np.load(f"{OUT}/test_hadm.npy").tolist()

with open(f"{OUT}/model_config.json") as f:
    model_cfg = json.load(f)

HIDDEN    = model_cfg["hidden_dim"]
SEC_EMB   = model_cfg.get("sec_emb_dim", 32)
MAX_NOTES = model_cfg.get("max_notes", 8)
N_SEC     = len(SECTIONS_TO_USE)

expert_A = joblib.load(f"{OUT}/expert_A.pkl")
expert_B = joblib.load(f"{OUT}/expert_B.pkl")
gate_clf = joblib.load(f"{OUT}/gate_clf.pkl")

print("  Loading seqs.pkl...")
with open(f"{OUT}/seqs.pkl", "rb") as f:
    seqs = pickle.load(f)

print("  Loading LSTM model...")
lstm_model = LSTMWithAttention(
    input_dim=768, n_sections=N_SEC,
    sec_emb_dim=SEC_EMB, hidden_dim=HIDDEN,
)
lstm_model.load_state_dict(torch.load(f"{OUT}/lstm_model.pt", map_location="cpu"))
lstm_model.to(device).eval()

P90_LOS  = float(np.percentile(y_train, 90))
tail_test = y_test >= P90_LOS

print(f"  test n={len(y_test):,}  P90_LOS={P90_LOS:.1f}d  "
      f"tail n={tail_test.sum():,} ({tail_test.mean():.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# Attention weight helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def attn_weights_for(hid, sec_idx):
    """
    Returns (weights, n_real) for patient hid in section SECTIONS_TO_USE[sec_idx].
    weights: (n_real,) normalised attention over real (non-padded) notes.
    Returns (None, 0) if patient has no notes for that section.
    """
    sec = SECTIONS_TO_USE[sec_idx]
    arr = seqs[sec].get(int(hid))
    if arr is None:
        return None, 0
    n_real = min(len(arr), MAX_NOTES)
    x  = torch.tensor(pad_seq(arr, MAX_NOTES)[None], dtype=torch.float32).to(device)
    si = torch.tensor([sec_idx], dtype=torch.long).to(device)
    sec_e = lstm_model.sec_emb(si).unsqueeze(1).expand(-1, x.size(1), -1)
    out, _ = lstm_model.lstm(torch.cat([x, sec_e], dim=-1))
    w = torch.softmax(lstm_model.attn(out), dim=1).squeeze().cpu().numpy()
    w[n_real:] = 0.0
    s = w.sum()
    if s > 1e-8:
        w = w / s
    return w[:n_real], n_real


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 0 — Within-section Attention Weight by Note Position
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Fig 0] Within-section attention by note position...")

MIN_NOTES_POS = 4   # only include patient-sections with ≥4 notes

attn_sum   = {j: {"all": np.zeros(MAX_NOTES), "tail": np.zeros(MAX_NOTES),
                   "nontail": np.zeros(MAX_NOTES)} for j in range(N_SEC)}
attn_count = {j: {"all": 0, "tail": 0, "nontail": 0} for j in range(N_SEC)}

for i, hid in enumerate(test_hadm):
    is_tail = bool(tail_test[i])
    for j in range(N_SEC):
        w, n_real = attn_weights_for(hid, j)
        if w is None or n_real < MIN_NOTES_POS:
            continue
        padded = np.zeros(MAX_NOTES)
        padded[:n_real] = w
        for grp in ("all", "tail" if is_tail else "nontail"):
            attn_sum[j][grp]   += padded
            attn_count[j][grp] += 1

SEC_SHORT = {
    "nursing_assessment":                "Nurs-Assess",
    "nursing_action":                    "Nurs-Action",
    "nursing_response":                  "Nurs-Response",
    "nursing_plan":                      "Nurs-Plan",
    "radiology_wet_read":                "Rad-WetRead",
    "radiology_indication":              "Rad-Indication",
    "radiology_technique":               "Rad-Technique",
    "radiology_impression":              "Rad-Impression",
    "nursing_other_other_nursing_other": "NursOther-Other",
    "nursing_other_respiratory_care":    "NursOther-Resp",
}

fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
axes = axes.flatten()
positions = np.arange(1, MAX_NOTES + 1)

for j, sec in enumerate(SECTIONS_TO_USE):
    ax = axes[j]
    for grp, style, col, label in [
        ("all",     "-",  "black",   "All"),
        ("tail",    "--", "#D6604D", "Tail"),
        ("nontail", ":",  "#4393C3", "Non-tail"),
    ]:
        c = attn_count[j][grp]
        if c == 0:
            continue
        ax.plot(positions, attn_sum[j][grp] / c,
                linestyle=style, color=col, linewidth=1.6,
                label=f"{label} (n={c:,})")

    ax.set_title(SEC_SHORT[sec], fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Note position", fontsize=8)
    ax.set_ylabel("Attn weight", fontsize=8)
    ax.tick_params(labelsize=7.5)
    if j == 0:
        ax.legend(fontsize=7.5, loc="upper right")

fig.suptitle(
    "Within-section Attention Weight by Note Position (MIMIC-III)\n"
    f"(patients with ≥{MIN_NOTES_POS} notes per section)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out_path = f"{FIGS}/fig0_attn_by_position.pdf"
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path}")
for j, sec in enumerate(SECTIONS_TO_USE):
    n_all = attn_count[j]["all"]
    print(f"  {SEC_SHORT[sec]:20s}  n={n_all:4d} ({n_all/len(test_hadm)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Attention Recency Index by LOS Quartile
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Fig 1] Computing attention recency index (all notes, all sections pooled)...")

# Pool all per-note attention weights across ALL sections for each patient.
# Per-section position is normalised to [0, 1] (0=earliest, 1=latest note).
# n_real=1 → position=0.5 (single note, no ordering signal).
# Recency = weighted mean of positions, weights = attention values (globally renormalised).
# No per-section minimum: include any patient with ≥1 note in any section.
recency_per_patient = []
for i, hid in enumerate(test_hadm):
    all_positions = []
    all_weights   = []
    for j in range(N_SEC):
        w, n_real = attn_weights_for(hid, j)
        if w is None or n_real == 0:
            continue
        if n_real == 1:
            pos = np.array([0.5])
        else:
            pos = np.arange(n_real) / (n_real - 1)   # [0, ..., 1]
        all_positions.extend(pos.tolist())
        all_weights.extend(w.tolist())

    if not all_weights:
        continue
    total_w = sum(all_weights)
    if total_w < 1e-8:
        continue
    recency = sum(p * wt for p, wt in zip(all_positions, all_weights)) / total_w
    recency_per_patient.append({
        "hadm_id":  hid,
        "los_days": y_test[i],
        "recency":  float(recency),
        "n_notes":  len(all_weights),
    })

df_rec = pd.DataFrame(recency_per_patient)
print(f"  Patients with ≥1 note: {len(df_rec):,} / {len(test_hadm):,}")

# LOS quartile labels using actual quantile boundaries
q_bounds = np.quantile(y_test, [0.25, 0.50, 0.75])
def los_group(los):
    if los <= q_bounds[0]:  return f"Q1 (≤{q_bounds[0]:.0f}d)"
    if los <= q_bounds[1]:  return f"Q2 ({q_bounds[0]:.0f}–{q_bounds[1]:.0f}d)"
    if los <= q_bounds[2]:  return f"Q3 ({q_bounds[1]:.0f}–{q_bounds[2]:.0f}d)"
    return                         f"Q4 (>{q_bounds[2]:.0f}d)"

df_rec["group"] = df_rec["los_days"].apply(los_group)
group_order = [
    f"Q1 (≤{q_bounds[0]:.0f}d)",
    f"Q2 ({q_bounds[0]:.0f}–{q_bounds[1]:.0f}d)",
    f"Q3 ({q_bounds[1]:.0f}–{q_bounds[2]:.0f}d)",
    f"Q4 (>{q_bounds[2]:.0f}d)",
]
palette = ["#66C2A5", "#FDB863", "#F4A582", "#B2ABD2"]

fig, ax = plt.subplots(figsize=(10, 5.5))

parts = ax.violinplot(
    [df_rec.loc[df_rec["group"] == g, "recency"].values for g in group_order],
    positions=range(len(group_order)),
    widths=0.6, showmedians=False, showextrema=False,
)
for pc, col in zip(parts["bodies"], palette):
    pc.set_facecolor(col)
    pc.set_alpha(0.75)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.8)

# Median + IQR whiskers
for k, g in enumerate(group_order):
    vals = df_rec.loc[df_rec["group"] == g, "recency"].values
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    ax.plot([k, k], [q25, q75], color="black", linewidth=2.5, solid_capstyle="round")
    ax.plot(k, med, "_", color="black", markersize=14, markeredgewidth=2.5)

ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.2, label="uniform attention")
ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order, fontsize=10)
ax.set_ylabel("Recency Index\n(0 = always attend to first note, 1 = always attend to last)", fontsize=10)
ax.set_title(
    "Attention Recency Index by LOS Group\n"
    "(all notes across all sections pooled per patient)",
    fontsize=12,
)
ax.legend(loc="upper left")
ax.set_ylim(0.28, 0.82)
ax.grid(axis="y", linewidth=0.4, alpha=0.4)

out_path = f"{FIGS}/fig1_recency_violin.pdf"
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path}")

# Summary stats
print("  Median recency by group:")
for g in group_order:
    vals = df_rec.loc[df_rec["group"] == g, "recency"].values
    print(f"    {g:20s}  median={np.median(vals):.3f}  n={len(vals):,}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Case Study: 24h Note Trajectory
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Fig 2] Building case study...")

NURSING_PARQUET       = "processed/section_trajectory_long/nursing_sections_24h_long.parquet"
RADIOLOGY_PARQUET     = "processed/section_trajectory_long/radiology_24h_long.parquet"
NURSING_OTHER_PARQUET = "processed/section_trajectory_long/nursing_other_sections_24h_long.parquet"


def load_notes_for_patient(hid):
    """Load all notes for one patient from parquets, return sorted DataFrame."""
    rows = []
    for path, allowed in [
        (NURSING_PARQUET,       NURSING_SECTIONS),
        (RADIOLOGY_PARQUET,     RADIOLOGY_SECTIONS),
        (NURSING_OTHER_PARQUET, NURSING_OTHER_SECTIONS),
    ]:
        if not os.path.exists(path):
            continue
        df_p = pd.read_parquet(path)
        df_p.columns = df_p.columns.str.lower()
        df_p["hadm_id"] = df_p["hadm_id"].astype(int)
        df_p = df_p[
            (df_p["hadm_id"] == int(hid))
            & df_p["section_name"].isin(allowed)
            & (df_p["text"].fillna("").str.strip().str.len() > 5)
        ].copy()
        if "hours_from_admit" in df_p.columns:
            df_p = df_p.sort_values("hours_from_admit")
        rows.append(df_p[["section_name", "hours_from_admit", "text"]])
    if not rows:
        return pd.DataFrame(columns=["section_name", "hours_from_admit", "text"])
    return pd.concat(rows).sort_values("hours_from_admit").reset_index(drop=True)


def pick_case_patient(candidates_hids, candidates_los, kind="short"):
    """
    Pick a representative patient (not the extreme outlier).
      kind='short': LOS in [Q10, Q25], sorted ascending — near the short-stay median
      kind='long':  LOS in [P90, P95], sorted ascending — representative tail case
    Requires notes in ≥3 distinct sections.
    """
    los_arr = np.array(candidates_los)
    if kind == "short":
        lo, hi = np.percentile(los_arr, 10), np.percentile(los_arr, 25)
    else:
        lo, hi = np.percentile(los_arr, 90), np.percentile(los_arr, 95)

    candidates = [(hid, los) for hid, los in zip(candidates_hids, candidates_los)
                  if lo <= los <= hi]
    candidates.sort(key=lambda x: x[1])   # ascending by LOS

    for hid, los in candidates:
        n_secs = sum(1 for j in range(N_SEC)
                     if seqs[SECTIONS_TO_USE[j]].get(int(hid)) is not None)
        if n_secs >= 3:
            return hid, los
    return None, None


short_hid, short_los = pick_case_patient(test_hadm, y_test.tolist(), kind="short")
long_hid,  long_los  = pick_case_patient(test_hadm, y_test.tolist(), kind="long")

print(f"  Short-stay patient: hadm_id={short_hid}  LOS={short_los:.1f}d")
print(f"  Long-stay  patient: hadm_id={long_hid}   LOS={long_los:.1f}d")


def get_note_attentions(hid, notes_df):
    """
    For each note row in notes_df, compute the attention weight the LSTM assigns
    to it within its section sequence.
    Returns a list of attention weights aligned with notes_df rows.
    """
    # Build per-section note order (same order as seqs)
    sec_note_idx = {sec: 0 for sec in SECTIONS_TO_USE}
    attns = []
    for _, row in notes_df.iterrows():
        sec = row["section_name"]
        if sec not in SECTIONS_TO_USE:
            attns.append(0.0)
            continue
        j = SECTIONS_TO_USE.index(sec)
        w, _ = attn_weights_for(hid, j)
        if w is None:
            attns.append(0.0)
            continue
        note_pos = sec_note_idx[sec]
        sec_note_idx[sec] += 1
        attns.append(float(w[note_pos]) if note_pos < len(w) else 0.0)
    return attns


def fmt_hours(h):
    """Convert float hours-from-admit to HH:MM string."""
    h = max(0.0, float(h))
    hh = int(h)
    mm = int(round((h - hh) * 60))
    return f"{hh:02d}:{mm:02d}"


def plot_patient_timeline(ax, notes_df, attentions, title):
    if notes_df.empty:
        ax.set_title(f"{title}\n(no notes found)")
        return

    n = len(notes_df)
    max_attn = max(attentions) if max(attentions) > 0 else 1.0
    ys = np.arange(n)[::-1]   # top row = earliest note

    ytick_labels = []
    for idx, (_, row) in enumerate(notes_df.iterrows()):
        y        = ys[idx]
        attn     = attentions[idx]
        sec      = row["section_name"]
        col      = SEC_COLOR.get(sec, GRAY)
        time_str = fmt_hours(row.get("hours_from_admit", 0))

        # Horizontal bar — width proportional to attention weight
        ax.barh(y, attn / max_attn, height=0.65,
                color=col, alpha=0.80, edgecolor="white", linewidth=0.4)

        # Attention value annotated at bar end
        ax.text(attn / max_attn + 0.015, y,
                f"{attn:.3f}", va="center", fontsize=7, color="#444444")

        sec_short = (sec.replace("nursing_other_", "nurs_other/")
                        .replace("nursing_", "nursing/")
                        .replace("radiology_", "rad/"))
        ytick_labels.append(f"{time_str}  {sec_short}")

    ax.set_xlim(0, 1.22)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks(ys)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel("Normalised attention weight", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axvline(1.0, color=GRAY, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


fig, axes = plt.subplots(1, 2, figsize=(14, 10))

for ax, hid, los, kind in [
    (axes[0], short_hid, short_los, "Short-stay"),
    (axes[1], long_hid,  long_los,  "Long-stay"),
]:
    if hid is None:
        ax.set_title(f"{kind} — no suitable patient found")
        continue
    notes_df = load_notes_for_patient(hid)
    # Keep at most first MAX_NOTES notes per section (same as training)
    notes_df = (notes_df.groupby("section_name", group_keys=False)
                .head(MAX_NOTES)
                .sort_values("hours_from_admit")
                .reset_index(drop=True))
    attentions = get_note_attentions(hid, notes_df)
    title = f"{kind} Patient  (LOS = {los:.1f} days)"
    plot_patient_timeline(ax, notes_df, attentions, title)

fig.suptitle(
    "24h Clinical Note Trajectory with Learned Attention Weights\n"
    "(bar width = attention weight; HH:MM = hours from admission)",
    fontsize=11, y=1.01,
)

legend_handles = [
    mpatches.Patch(color="#4CAF50", label="Nursing"),
    mpatches.Patch(color="#FF9800", label="Radiology"),
    mpatches.Patch(color="#E91E63", label="Nursing other"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out_path = f"{FIGS}/fig2_case_study.pdf"
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — MAE per LOS Bucket
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Fig 3] MAE per LOS bucket...")

bins   = [0, 3, 7, 14, 30, np.inf]
labels = ["0–3d", "3–7d", "7–14d", "14–30d", "30+d"]
bucket = np.clip(np.digitize(y_test, bins) - 1, 0, len(labels) - 1)

models     = {"Struct Only": pred_m1, "MoE (Ours)": pred_m5}
colors_bar = [GRAY, ORANGE]
bucket_ns  = [int((bucket == b).sum()) for b in range(len(labels))]
xlabels    = [f"{l}\n(n={n:,})" for l, n in zip(labels, bucket_ns)]

fig, ax = plt.subplots(figsize=(9, 4.5))
x, bw = np.arange(len(labels)), 0.35

for i, (name, pred, col) in enumerate(zip(models.keys(), models.values(), colors_bar)):
    offset = (i - 0.5) * bw
    maes   = [mean_absolute_error(y_test[bucket == b], pred[bucket == b])
              if (bucket == b).sum() > 5 else np.nan
              for b in range(len(labels))]
    ax.bar(x + offset, maes, bw, color=col, alpha=0.85, label=name)

ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel("MAE (days)")
ax.set_title("MAE by LOS Bucket: MoE Gains Concentrated in Long-Stay Patients",
             fontweight="bold")
ax.legend()

out_path = f"{FIGS}/fig3_los_bucket.pdf"
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Expert A vs B Section Importance
# ═════════════════════════════════════════════════════════════════════════════

print("\n[Fig 4] Expert A vs B section importance...")

# Experts trained on X_exp = [struct | lstm].  LSTM starts at col n_struct.
n_struct = X_struct_test.shape[1]
X_exp    = np.hstack([X_struct_test, lstm_feat_test])
y_resid  = y_test - pred_m1   # residual ground truth


# Section coverage mask: which test patients have ≥1 note in each section
sec_covered = np.array([
    np.array([seqs[SECTIONS_TO_USE[j]].get(int(hid)) is not None
               for hid in test_hadm], dtype=bool)
    for j in range(N_SEC)
])   # shape (N_SEC, n_test)

tail_mask = y_test >= P90_LOS


def section_permutation_importance(expert, X, y_true, covered_rows, n_repeats=5):
    """
    Compute importance of each section on the subset of patients who
    actually have notes in that section (covered_rows[j] = bool mask).
    Returns (mean, std) arrays of shape (N_SEC,).
    """
    rng = np.random.default_rng(SEED)
    imp = np.zeros((N_SEC, n_repeats))
    for j in range(N_SEC):
        mask = covered_rows[j]
        if mask.sum() < 5:
            continue
        Xj     = X[mask]
        yj     = y_true[mask]
        base   = mean_absolute_error(yj, expert.predict(Xj))
        s, e   = n_struct + j * HIDDEN, n_struct + (j + 1) * HIDDEN
        for r in range(n_repeats):
            perm       = rng.permutation(len(Xj))
            Xp         = Xj.copy()
            Xp[:, s:e] = Xp[perm, s:e]
            imp[j, r]  = mean_absolute_error(yj, expert.predict(Xp)) - base
    return imp.mean(axis=1), imp.std(axis=1)


# Expert A evaluated on bulk covered patients; Expert B on tail covered patients
bulk_mask = ~tail_mask
imp_A_mean, imp_A_std = section_permutation_importance(
    expert_A, X_exp, y_resid,
    covered_rows=sec_covered & bulk_mask[np.newaxis, :],
    n_repeats=args.n_importance_repeats)
imp_B_mean, imp_B_std = section_permutation_importance(
    expert_B, X_exp, y_resid,
    covered_rows=sec_covered & tail_mask[np.newaxis, :],
    n_repeats=args.n_importance_repeats)

# Coverage % for annotation (all test patients)
coverage_pct = sec_covered.mean(axis=1) * 100   # shape (N_SEC,)

order = np.argsort(imp_B_mean)[::-1]
x     = np.arange(N_SEC)
w     = 0.38

fig, ax = plt.subplots(figsize=(11, 5.0))
ax.bar(x - w/2, imp_A_mean[order], w, yerr=imp_A_std[order],
       color=BLUE,   alpha=0.85, label="Expert A (bulk, covered patients)",
       error_kw=dict(elinewidth=1, capsize=3))
ax.bar(x + w/2, imp_B_mean[order], w, yerr=imp_B_std[order],
       color=ORANGE, alpha=0.85, label="Expert B (tail, covered patients)",
       error_kw=dict(elinewidth=1, capsize=3))

# Annotate coverage % as secondary x-tick labels
sec_labels_with_cov = [
    f"{SEC_ABBR[SECTIONS_TO_USE[i]]}\n{coverage_pct[i]:.0f}% cov"
    for i in order
]

ax.set_xticks(x)
ax.set_xticklabels([SEC_ABBR[SECTIONS_TO_USE[i]] for i in order], fontsize=8.5)
ax.set_ylabel("MAE increase when section permuted\n(on covered patients only)")
ax.set_title("Section Importance: Expert A (bulk) vs Expert B (tail)\n"
             "(coverage % shown below each section)", fontweight="bold")
ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax.legend()

out_path = f"{FIGS}/fig4_expert_importance.pdf"
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\nAll figures saved to {FIGS}/")
print("  fig1_recency_violin.pdf")
print("  fig2_case_study.pdf")
print("  fig3_los_bucket.pdf")
print("  fig4_expert_importance.pdf")
