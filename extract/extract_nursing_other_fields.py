import re
import json
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

INPUT_PATH = "processed/notes_nursing_other.parquet"
LOS_PATH = "processed/mimic3_los_aggregated.csv"

OUTPUT_DIR = Path("processed/section_trajectory_long")

OUTPUT_WIDE_CSV = OUTPUT_DIR / "nursing_other_sections_24h_wide.csv"
OUTPUT_WIDE_PARQUET = OUTPUT_DIR / "nursing_other_sections_24h_wide.parquet"

OUTPUT_LONG_CSV = OUTPUT_DIR / "nursing_other_sections_24h_long.csv"
OUTPUT_LONG_PARQUET = OUTPUT_DIR / "nursing_other_sections_24h_long.parquet"
OUTPUT_JSONL = OUTPUT_DIR / "nursing_other_sections_24h_long.jsonl"


NOTE_TYPE_PATTERNS = {
    "respiratory_care": [
        r"\bresp\s*care\b",
        r"\bresp\.?\s*note\b",
        r"\brespiratory\s+care\b",
        r"\brespiratory\s+therapy\b",
        r"\brt\s+note\b",
        r"\bvent\s+note\b",
    ],
    "nursing_progress_note": [
        r"\bnursing\s+progress\s+note\b",
        r"\bmicu\s+progress\s+note\b",
        r"\bsicu\s+progress\s+note\b",
        r"\bnicu\s+progress\s+note\b",
        r"\bprogress\s+note\b",
        r"\b7p\s+to\s+7a\b",
        r"\b7a\s+to\s+7p\b",
        r"\b3p\s+to\s+11p\b",
        r"\b11p\s+to\s+7a\b",
    ],
    "narrative_nursing_note": [
        r"\bnarrative\s+nursing\s+note\b",
        r"\bnarrative\s+note\b",
    ],
    "admission_note": [
        r"\badmit\s+note\b",
        r"\badmission\s+note\b",
        r"\bmicu\s+admission\s+note\b",
        r"\bsicu\s+admission\s+note\b",
        r"\bnursing\s+admission\b",
    ],
    "micu_nursing": [
        r"\bmicu\s+nsg\b",
        r"\bmicu\s+nursing\b",
        r"\bmicu\s+note\b",
        r"\bpmicu\b",
    ],
    "sicu_nursing": [
        r"\bsicu\s+nsg\b",
        r"\bsicu\s+nursing\b",
        r"\bsicu\s+note\b",
    ],
    "transfer_discharge_note": [
        r"\btransfer\s+note\b",
        r"\btransfer\s+summary\b",
        r"\bdischarge\s+note\b",
        r"\bcall[- ]?out\b",
        r"\btransfer\s+to\s+floor\b",
    ],
    "care_plan": [
        r"\bcare\s+plan\b",
        r"\bplan\s+of\s+care\b",
    ],
    "event_note": [
        r"\bevent\s+note\b",
        r"\bsignificant\s+event\b",
        r"\baddendum\b",
        r"\badden\b",
    ],
    "free_text_nursing": [
        r"\bnursing\b",
        r"\bnsg\b",
    ],
}

PRIORITY = [
    "respiratory_care",
    "admission_note",
    "nursing_progress_note",
    "narrative_nursing_note",
    "micu_nursing",
    "sicu_nursing",
    "transfer_discharge_note",
    "care_plan",
    "event_note",
    "free_text_nursing",
]


NURSING_OTHER_HEADERS = [
    "CURRENT ROS",
    "NEURO", "NEUROLOGIC",
    "RESP", "RESPIRATORY", "RESP CARE",
    "CV", "C-V", "CARD", "CARDIAC",
    "GI", "GU", "GI/GU",
    "F/E", "FEN",
    "ID", "INFECTIOUS",
    "ENDO",
    "ACCESS", "IVF",
    "SOCIAL",
    "PLAN", "A+P", "A/P",
    "ASSESSMENT", "ACTION", "RESPONSE",
]

_NURSING_OTHER_HEADER_MAP = {
    "CURRENT ROS": "current_ros",
    "NEURO": "neuro",
    "NEUROLOGIC": "neuro",
    "RESP": "resp",
    "RESPIRATORY": "resp",
    "RESP CARE": "resp",
    "CV": "cv",
    "C-V": "cv",
    "CARD": "cv",
    "CARDIAC": "cv",
    "GI": "gi_gu",
    "GU": "gi_gu",
    "GI/GU": "gi_gu",
    "F/E": "fluids_electrolytes",
    "FEN": "fluids_electrolytes",
    "ID": "infectious",
    "INFECTIOUS": "infectious",
    "ENDO": "endo",
    "ACCESS": "access",
    "IVF": "ivf",
    "SOCIAL": "social",
    "PLAN": "plan",
    "A+P": "plan",
    "A/P": "plan",
    "ASSESSMENT": "assessment",
    "ACTION": "action",
    "RESPONSE": "response",
}

_NURSING_OTHER_RE = re.compile(
    r"(?im)^\s*("
    + "|".join(map(re.escape, sorted(NURSING_OTHER_HEADERS, key=len, reverse=True)))
    + r")\s*(?::|\n|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = text.replace("#NAME?", " ")
    text = re.sub(r"_{5,}", " ", text)
    text = re.sub(r"-{5,}", " ", text)
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


def get_note_title(text, max_lines=5):
    if pd.isna(text):
        return ""

    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    candidates = []

    for line in lines[:max_lines]:
        line = clean_text(line)

        if not line:
            continue

        if len(line.split()) <= 12:
            candidates.append(line)

    if candidates:
        return candidates[0]

    return ""


def normalize_for_matching(text):
    text = clean_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9/\-\+\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_note_type(text):
    if pd.isna(text):
        return "unknown"

    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")

    head_text = "\n".join(lines[:8])
    head_norm = normalize_for_matching(head_text)
    full_norm = normalize_for_matching(raw)

    for note_type in PRIORITY:
        for pat in NOTE_TYPE_PATTERNS[note_type]:
            if re.search(pat, head_norm, flags=re.IGNORECASE):
                return note_type

    for note_type in PRIORITY:
        for pat in NOTE_TYPE_PATTERNS[note_type]:
            if re.search(pat, full_norm, flags=re.IGNORECASE):
                return note_type

    return "other_nursing_other"


def split_nursing_other_sections(text):
    if pd.isna(text):
        return {}

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_NURSING_OTHER_RE.finditer(text))
    sections = {}

    if not matches:
        return {}

    for i, m in enumerate(matches):
        raw = m.group(1).upper().strip()
        key = _NURSING_OTHER_HEADER_MAP.get(
            raw,
            raw.lower().replace("/", "_").replace(" ", "_"),
        )

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = sentence_segment(text[start:end])

        if content:
            sections[key] = (sections.get(key, "") + " " + content).strip()

    return sections


def extract_note_type_fields(text):
    note_title = get_note_title(text)
    note_type = detect_note_type(text)
    clean_note_text = sentence_segment(text)
    sections = split_nursing_other_sections(text)

    return pd.Series({
        "note_title": note_title,
        "note_type": note_type,
        "clean_text": clean_note_text,
        "sections": sections,
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

    if text_col not in df.columns:
        if text_col.lower() in df.columns:
            text_col = text_col.lower()
        else:
            raise ValueError("Missing text column.")

    for c in ["hadm_id", "charttime"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    duplicate_cols = [
        c for c in ["hadm_id", "charttime", text_col]
        if c in df.columns
    ]

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

    print("Detecting Nursing/other note types and sections...", flush=True)

    extracted = df[text_col].progress_apply(extract_note_type_fields)

    df_out = pd.concat(
        [df.reset_index(drop=True), extracted.reset_index(drop=True)],
        axis=1,
    )

    output_cols = [
        c for c in [
            "hadm_id",
            "admittime",
            "charttime",
            "category",
            "hours_from_admit",
            "time_bin",
            "note_title",
            "note_type",
            "clean_text",
            "sections",
        ]
        if c in df_out.columns
    ]

    df_wide = df_out[output_cols].copy()

    # wide output
    df_wide.to_csv(
        OUTPUT_WIDE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    df_wide.to_parquet(
        OUTPUT_WIDE_PARQUET,
        index=False,
    )

    # long output
    long_records = []

    for _, row in tqdm(df_wide.iterrows(), total=len(df_wide), desc="Building long format"):
        sections = row.get("sections", {})

        # 1. 如果能切出 subsection，用 subsection
        if isinstance(sections, dict) and len(sections) > 0:
            for sec_name, sec_text in sections.items():
                sec_text = str(sec_text).strip()

                if not sec_text:
                    continue

                long_records.append({
                    "hadm_id": int(row["hadm_id"]),
                    "admittime": row["admittime"],
                    "charttime": row["charttime"],
                    "category": row.get("category", "Nursing/other"),
                    "section_name": f"nursing_other_{row['note_type']}_{sec_name}",
                    "text": sec_text,
                    "hours_from_admit": float(row["hours_from_admit"]),
                    "time_bin": row["time_bin"],
                })

        # 2. 如果切不出 subsection，用 note_type 作为 section
        else:
            clean_text = str(row.get("clean_text", "")).strip()

            if clean_text:
                long_records.append({
                    "hadm_id": int(row["hadm_id"]),
                    "admittime": row["admittime"],
                    "charttime": row["charttime"],
                    "category": row.get("category", "Nursing/other"),
                    "section_name": f"nursing_other_{row['note_type']}",
                    "text": clean_text,
                    "hours_from_admit": float(row["hours_from_admit"]),
                    "time_bin": row["time_bin"],
                })

    df_long = pd.DataFrame(long_records)

    df_long.to_csv(
        OUTPUT_LONG_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    df_long.to_parquet(
        OUTPUT_LONG_PARQUET,
        index=False,
    )

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in df_long.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print("\nSaved wide CSV:    ", OUTPUT_WIDE_CSV, flush=True)
    print("Saved wide Parquet:", OUTPUT_WIDE_PARQUET, flush=True)
    print("Saved long CSV:    ", OUTPUT_LONG_CSV, flush=True)
    print("Saved long Parquet:", OUTPUT_LONG_PARQUET, flush=True)
    print("Saved JSONL:       ", OUTPUT_JSONL, flush=True)

    print("\nLong format shape:", df_long.shape, flush=True)

    if len(df_long) > 0:
        print("\nSection coverage:", flush=True)
        print(df_long["section_name"].value_counts().head(50), flush=True)

        print("\nTime-bin coverage:", flush=True)
        print(df_long["time_bin"].value_counts().sort_index(), flush=True)