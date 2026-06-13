import re
import json
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

INPUT_PATH = "processed/notes_nursing.parquet"
LOS_PATH = "processed/mimic3_los_aggregated.csv"

OUTPUT_DIR = Path("processed/section_trajectory_long")
OUTPUT_WIDE_CSV = OUTPUT_DIR / "nursing_sections_24h_wide.csv"
OUTPUT_WIDE_PARQUET = OUTPUT_DIR / "nursing_sections_24h_wide.parquet"
OUTPUT_LONG_CSV = OUTPUT_DIR / "nursing_sections_24h_long.csv"
OUTPUT_LONG_PARQUET = OUTPUT_DIR / "nursing_sections_24h_long.parquet"
OUTPUT_JSONL = OUTPUT_DIR / "nursing_sections_24h_long.jsonl"


NURSING_HEADERS = [
    "ASSESSMENT",
    "ACTION",
    "RESPONSE",
    "PLAN",
    "DEMOGRAPHICS",
    "ADMIT DIAGNOSIS",
    "CODE STATUS",
    "HEIGHT",
    "ADMISSION WEIGHT",
    "DAILY WEIGHT",
    "ALLERGIES/REACTIONS",
    "PRECAUTIONS",
    "PMH",
    "ADDITIONAL HISTORY",
    "SURGERY / PROCEDURE AND DATE",
    "LATEST VITAL SIGNS AND I/O",
    "PERTINENT LAB RESULTS",
    "VALUABLES / SIGNATURE",
    "TRANSFERRED FROM",
    "TRANSFERRED TO",
    "DATE & TIME OF TRANSFER",
]

HEADER_MAP = {
    "ASSESSMENT": "assessment",
    "ACTION": "action",
    "RESPONSE": "response",
    "PLAN": "plan",
    "DEMOGRAPHICS": "demographics",
    "ADMIT DIAGNOSIS": "admit_diagnosis",
    "CODE STATUS": "code_status",
    "HEIGHT": "height",
    "ADMISSION WEIGHT": "admission_weight",
    "DAILY WEIGHT": "daily_weight",
    "ALLERGIES/REACTIONS": "allergies_reactions",
    "PRECAUTIONS": "precautions",
    "PMH": "pmh",
    "ADDITIONAL HISTORY": "additional_history",
    "SURGERY / PROCEDURE AND DATE": "surgery_procedure_date",
    "LATEST VITAL SIGNS AND I/O": "latest_vital_signs_io",
    "PERTINENT LAB RESULTS": "pertinent_lab_results",
    "VALUABLES / SIGNATURE": "valuables_signature",
    "TRANSFERRED FROM": "transferred_from",
    "TRANSFERRED TO": "transferred_to",
    "DATE & TIME OF TRANSFER": "date_time_transfer",
}

_NURSING_RE = re.compile(
    r"(?im)^\s*("
    + "|".join(map(re.escape, sorted(NURSING_HEADERS, key=len, reverse=True)))
    + r")\s*(?::|\n|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)


USEFUL_COLS = [
    "assessment", "action", "response", "plan",
    "admit_diagnosis", "pmh", "additional_history",
    "latest_vital_signs_io", "pertinent_lab_results",
    "transferred_from", "transferred_to", "date_time_transfer",
]


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = text.replace("#NAME?", " ")
    text = re.sub(r"_{5,}", " ", text)
    text = re.sub(r"-{5,}", " ", text)
    text = re.sub(r"\bPRBC\s+s\b", "PRBCs", text, flags=re.I)
    text = re.sub(r"\bPIV\s+s\b", "PIVs", text, flags=re.I)
    text = re.sub(r"\bIV\s+s\b", "IVs", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_segment(text):
    text = clean_text(text)
    if not text:
        return ""
    doc = nlp(text)
    return " ".join(
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.strip()) >= 3
    )


def normalize_header(header):
    h = header.upper().strip()
    return HEADER_MAP.get(
        h,
        h.lower().replace("/", "_").replace("&", "and").replace(" ", "_"),
    )


def split_nursing_sections(text):
    if pd.isna(text):
        return {}

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_NURSING_RE.finditer(text))
    sections = {}

    if not matches:
        return {}

    for i, m in enumerate(matches):
        key = normalize_header(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = sentence_segment(text[start:end])

        if content:
            sections[key] = (sections.get(key, "") + " " + content).strip()

    return sections


def extract_nursing_fields(text):
    all_sections = split_nursing_sections(text)
    return pd.Series({
        **{col: all_sections.get(col, "") for col in USEFUL_COLS},
        "all_sections": all_sections,
    })


def assign_time_bin(hours):
    if 0 <= hours < 6:
        return "0_6h"
    elif 6 <= hours < 12:
        return "6_12h"
    elif 12 <= hours < 18:
        return "12_18h"
    elif 18 <= hours <= 24:
        return "18_24h"
    else:
        return None


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    df.columns = df.columns.str.lower()

    print("Loaded:", df.shape, flush=True)
    print("Columns:", df.columns.tolist(), flush=True)

    text_col = "text" if "text" in df.columns else "TEXT"

    required = ["hadm_id", "charttime", text_col.lower() if text_col.lower() in df.columns else text_col]
    for c in ["hadm_id", "charttime"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if text_col not in df.columns:
        if text_col.lower() in df.columns:
            text_col = text_col.lower()
        else:
            raise ValueError("Missing text column.")

    duplicate_cols = [c for c in ["hadm_id", "charttime", text_col] if c in df.columns]
    df = df.drop_duplicates(subset=duplicate_cols).copy()

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.dropna(subset=["charttime"])

    los_df = pd.read_csv(
        LOS_PATH,
        parse_dates=["admittime"],
        usecols=["hadm_id", "admittime"],
    )

    los_df["hadm_id"] = los_df["hadm_id"].astype(int)
    df["hadm_id"] = df["hadm_id"].astype(int)

    df = df.merge(los_df, on="hadm_id", how="left")
    df = df.dropna(subset=["admittime"])

    df["hours_from_admit"] = (
        df["charttime"] - df["admittime"]
    ).dt.total_seconds() / 3600

    df = df[
        (df["hours_from_admit"] >= 0) &
        (df["hours_from_admit"] <= 24)
    ].copy()

    df["time_bin"] = df["hours_from_admit"].apply(assign_time_bin)
    df = df.dropna(subset=["time_bin"])

    print("After 24h filtering:", df.shape, flush=True)
    print(df["time_bin"].value_counts().sort_index(), flush=True)

    print("Extracting Nursing sections...", flush=True)

    extracted = df[text_col].progress_apply(extract_nursing_fields)

    df_out = pd.concat(
        [df.reset_index(drop=True), extracted.reset_index(drop=True)],
        axis=1,
    )

    df_wide = df_out[
        df_out[USEFUL_COLS]
        .fillna("")
        .apply(lambda row: any(str(x).strip() for x in row), axis=1)
    ].copy()

    output_cols = [
        c for c in [
            "hadm_id", "admittime", "charttime", "category",
            "hours_from_admit", "time_bin",
            *USEFUL_COLS,
            "all_sections",
        ]
        if c in df_wide.columns
    ]

    df_wide = df_wide[output_cols].copy()

    # wide output
    df_wide.to_csv(OUTPUT_WIDE_CSV, index=False, encoding="utf-8-sig")
    df_wide.to_parquet(OUTPUT_WIDE_PARQUET, index=False)

    # long output
    long_records = []

    for _, row in tqdm(df_wide.iterrows(), total=len(df_wide), desc="Building long format"):
        for sec in USEFUL_COLS:
            if sec not in df_wide.columns:
                continue

            text = str(row.get(sec, "")).strip()

            if not text:
                continue

            long_records.append({
                "hadm_id": int(row["hadm_id"]),
                "admittime": row["admittime"],
                "charttime": row["charttime"],
                "category": row.get("category", "Nursing"),
                "section_name": f"nursing_{sec}",
                "text": text,
                "hours_from_admit": float(row["hours_from_admit"]),
                "time_bin": row["time_bin"],
            })

    df_long = pd.DataFrame(long_records)

    df_long.to_csv(OUTPUT_LONG_CSV, index=False, encoding="utf-8-sig")
    df_long.to_parquet(OUTPUT_LONG_PARQUET, index=False)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in df_long.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print("\nSaved wide CSV:    ", OUTPUT_WIDE_CSV, flush=True)
    print("Saved wide Parquet:", OUTPUT_WIDE_PARQUET, flush=True)
    print("Saved long CSV:    ", OUTPUT_LONG_CSV, flush=True)
    print("Saved long Parquet:", OUTPUT_LONG_PARQUET, flush=True)
    print("Saved JSONL:       ", OUTPUT_JSONL, flush=True)

    print("\nLong format shape:", df_long.shape, flush=True)
    print("\nSection coverage:", flush=True)

    if len(df_long) > 0:
        print(df_long["section_name"].value_counts(), flush=True)
        print("\nTime-bin coverage:", flush=True)
        print(df_long["time_bin"].value_counts().sort_index(), flush=True)