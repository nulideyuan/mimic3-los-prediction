# MIMIC-III LOS Prediction — Section-Aware LSTM with MoE Residual Fusion

Code for our EMNLP paper on length-of-stay (LOS) prediction from clinical notes using a section-aware LSTM with mixture-of-experts (MoE) residual fusion.

## Overview

The pipeline has four stages:

```
1. extract/                          — parse raw MIMIC-III notes into section-level parquets
2. train_section_lstm_full.py embed  — encode sections with ClinicalBERT
3. train_section_lstm_full.py train  — train LSTM + LightGBM MoE fusion (main model)
4. ablation_moe.py                   — reproduce ablation table (multi-seed)
5. run_interpretability.py           — reproduce interpretability figures
```

## Requirements

```bash
pip install -r requirements.txt
```

For data extraction only, also install the spaCy sentencizer model:
```bash
python -m spacy download en_core_web_sm
```

## Data

This code requires access to [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) (credentialed access via PhysioNet).

Raw tables needed: `NOTEEVENTS`, `ADMISSIONS`, `PATIENTS`, `ICUSTAYS`, `LABEVENTS`, `PRESCRIPTIONS`, `PROCEDURES_ICD`.

Place the raw data in BigQuery or update paths in `extract/build_mimic3_los_bigquery.py`.

## Reproduction

All commands are run from the **root of this repository** (i.e., the `mimic3_dataset/` folder after cloning).

### Step 1 — Extract and preprocess notes

```bash
# Pull structured features + raw note parquets from BigQuery
python extract/build_mimic3_los_bigquery.py

# Clean structured features
python extract/build_structured_clean.py

# Parse section fields from each note type
python extract/extract_nursing_fields.py
python extract/extract_nursing_other_fields.py
python extract/extract_radiology_fields.py
python extract/extract_physician_fields.py
```

Outputs go to `processed/`.

### Step 2 — Embed sections with ClinicalBERT

```bash
python train_section_lstm_full.py --mode embed
```

Cached embeddings are saved to `processed/embed_cache/` and
`processed/section_trajectory_long/seqs_clinicalbert_24h.pkl`.

### Step 3 — (Optional) Hyperparameter search

```bash
# Tune M1 — structured only (fast, ~5 min)
python hparam_search.py --model m1 --trials 40

# Tune M5 — MoE residual (expensive, ~2-4 h for 30 trials)
python hparam_search.py --model m5 --trials 30 --sample 3000
```

Best params are saved to `outputs_mimic3_lstm_full_gated_residual/best_params_<model>.json`.

### Step 4 — Train main model

```bash
python train_section_lstm_full.py \
    --mode train \
    --params outputs_mimic3_lstm_full_gated_residual/best_params_m5.json
```

Outputs saved to `outputs_mimic3_lstm_full_gated_residual/`.

### Step 5 — Reproduce ablation table

```bash
# Multi-seed (5 seeds) for mean ± std
python ablation_moe.py \
    --output outputs_mimic3_lstm_full_gated_residual \
    --seeds 0 1 2 3 4
```

### Step 6 — Reproduce interpretability figures

```bash
python run_interpretability.py \
    --output outputs_mimic3_lstm_full_gated_residual \
    --interp-out outputs_interpretability
```

Produces 4 figures in `outputs_interpretability/`:
- `fig1_struct_shap.png` — TreeSHAP on M1 structured features (top drivers of average LOS)
- `fig1b_section_recency.png` — Attention recency index: bulk vs tail patients per section
- `fig2_text_perm.png` — Section permutation importance: Expert A (bulk) vs Expert B (tail)
- `fig4_recency.png` — Recency violin: bulk vs tail

## File Structure

```
mimic3_dataset/
  extract/
    build_mimic3_los_bigquery.py       # pull structured features + notes from BigQuery
    build_structured_clean.py          # clean structured CSV
    extract_nursing_fields.py          # parse nursing note sections
    extract_nursing_other_fields.py    # parse nursing_other note sections
    extract_radiology_fields.py        # parse radiology note sections
    extract_physician_fields.py        # parse physician note sections
  processed/                           # intermediate data — NOT committed (requires MIMIC access)
  utils.py                             # LSTMWithAttention, pad_seq, set_seed, section constants
  train_section_lstm_full.py           # main training script (embed / train / seeds)
  hparam_search.py                     # per-model hyperparameter tuning (m1/m2/m3/m5)
  ablation_moe.py                      # ablation table: model and gate variants
  run_interpretability.py              # interpretability figures (SHAP, permutation, recency)
  analysis_interpretability.py         # legacy interpretability script
  analyze_shap_contributions.py        # SHAP structured vs text feature importance
  requirements.txt
  README.md
```

## Model

The core model (`utils.LSTMWithAttention`) is a section-aware LSTM:

- **Section embeddings**: 10 clinical note section types, each encoded with ClinicalBERT (768-dim), compressed via soft attention into a fixed-size representation
- **LSTM encoder**: processes the note sequence within each section (max 8 notes), outputs a 64-dim hidden state per section
- **Expert A (bulk)**: LightGBM trained on structured + LSTM features for patients with LOS < P90
- **Expert B (tail)**: LightGBM trained on structured + LSTM features for patients with LOS ≥ P90
- **Soft gate**: LightGBM classifier on structured features predicts tail probability, weights Expert A vs B
- **MoE residual**: final prediction = structured baseline + gated residual correction from Expert A/B

## Results (MIMIC-III)

| Model | MAE (days) | R² | Tail MAE P99 |
|-------|-----------|-----|--------------|
| M1: Structured only | 5.98 | 0.251 | 55.6 |
| M5: MoE Residual (ours) | 5.84 | 0.319 | 52.2 |

Overall MAE reduced by 2.3%, R² improved by 6.8%, P99 tail MAE reduced by 3.4 days.
