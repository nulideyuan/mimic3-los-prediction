import re
import json
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

INPUT_PATH = "processed/notes_radiology.parquet"
LOS_PATH = "processed/mimic3_los_aggregated.csv"

OUTPUT_DIR = Path("processed/section_trajectory_long")
OUTPUT_CSV = OUTPUT_DIR / "radiology_24h_long.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "radiology_24h_long.parquet"
OUTPUT_JSONL = OUTPUT_DIR / "radiology_24h_long.jsonl"


SUBHEADERS = [
    "INDICATION",
    "CLINICAL HISTORY",
    "HISTORY",
    "MEDICAL CONDITION",
    "REASON FOR THIS EXAMINATION",
    "REASON FOR EXAM",
    "REASON",
    "COMPARISON",
    "TECHNIQUE",
    "FINDINGS",
    "IMPRESSION",
    "CONCLUSION",
    "COMMENT",
    "RECOMMENDATION",
    "RECOMMENDATIONS",
]

HEADER_MAP = {
    "INDICATION": "indication",
    "CLINICAL HISTORY": "indication",
    "HISTORY": "indication",
    "MEDICAL CONDITION": "clinical_reason",
    "REASON FOR THIS EXAMINATION": "clinical_reason",
    "REASON FOR EXAM": "clinical_reason",
    "REASON": "clinical_reason",
    "COMPARISON": "comparison",
    "TECHNIQUE": "technique",
    "FINDINGS": "findings",
    "IMPRESSION": "impression",
    "CONCLUSION": "impression",
    "COMMENT": "comment",
    "RECOMMENDATION": "recommendation",
    "RECOMMENDATIONS": "recommendation",
}

SUBHEADER_PATTERN = "|".join(
    re.escape(h) for h in sorted(SUBHEADERS, key=len, reverse=True)
)

SUBHEADER_RE = re.compile(
    rf"^\s*({SUBHEADER_PATTERN})\s*:\s*(.*)$",
    flags=re.IGNORECASE,
)

HLINE_RE = re.compile(r"^\s*[_\-]{5,}\s*$")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = text.replace("#NAME?", " ")
    text = re.sub(r"_{5,}", " ", text)
    text = re.sub(r"-{5,}", " ", text)
    text = re.sub(r"\(\s*Over\s*\)", " ", text, flags=re.I)
    text = re.sub(r"\(\s*Cont\s*\)", " ", text, flags=re.I)
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
    return HEADER_MAP.get(h, h.lower().replace(" ", "_"))


def split_by_horizontal_lines(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    blocks = []
    buf = []

    for line in lines:
        if HLINE_RE.match(line):
            content = "\n".join(buf).strip()
            if content:
                blocks.append(content)
            buf = []
        else:
            buf.append(line)

    content = "\n".join(buf).strip()
    if content:
        blocks.append(content)

    return blocks


def split_subsections(text):
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections = {}
    current = None
    buf = []

    def flush():
        nonlocal current, buf

        if current and buf:
            content = sentence_segment(" ".join(buf))
            if content:
                sections[current] = (
                    sections.get(current, "") + " " + content
                ).strip()

        buf = []

    for line in lines:
        raw = line.strip()
        m = SUBHEADER_RE.match(raw)

        if m:
            flush()
            current = normalize_header(m.group(1))
            after = m.group(2).strip()

            if after:
                buf.append(after)
        else:
            if current:
                buf.append(line)

    flush()
    return sections


def classify_radiology_blocks(text):
    blocks = split_by_horizontal_lines(text)

    sections = {
        "exam_order_info": "",
        "clinical_reason": "",
        "wet_read": "",
        "final_report": "",
        "indication": "",
        "comparison": "",
        "technique": "",
        "findings": "",
        "impression": "",
        "comment": "",
        "recommendation": "",
    }

    for i, block in enumerate(blocks):
        upper = block.upper()

        if "FINAL REPORT" in upper:
            final_text = re.split(
                r"\bFINAL REPORT\b",
                block,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[-1].strip()

            sections["final_report"] = (
                sections["final_report"] + " " + sentence_segment(final_text)
            ).strip()

            sub_sections = split_subsections(final_text)

            for k, v in sub_sections.items():
                if k in sections and v:
                    sections[k] = (sections[k] + " " + v).strip()

        elif (
            "WET READ" in upper
            or "PFI" in upper
            or "PROVISIONAL FINDINGS IMPRESSION" in upper
        ):
            wet_text = re.sub(
                r"^\s*(WET READ|PFI|PROVISIONAL FINDINGS IMPRESSION)\s*:?",
                "",
                block,
                flags=re.IGNORECASE,
            ).strip()

            sections["wet_read"] = (
                sections["wet_read"] + " " + sentence_segment(wet_text)
            ).strip()

            if wet_text:
                sections["impression"] = (
                    sections["impression"] + " " + sentence_segment(wet_text)
                ).strip()

        elif (
            "MEDICAL CONDITION" in upper
            or "REASON FOR THIS EXAMINATION" in upper
            or "REASON FOR EXAM" in upper
        ):
            sub_sections = split_subsections(block)

            if sub_sections:
                for k, v in sub_sections.items():
                    if k in sections and v:
                        sections[k] = (sections[k] + " " + v).strip()
            else:
                sections["clinical_reason"] = (
                    sections["clinical_reason"] + " " + sentence_segment(block)
                ).strip()

        else:
            if i == 0:
                sections["exam_order_info"] = (
                    sections["exam_order_info"] + " " + sentence_segment(block)
                ).strip()
            else:
                sub_sections = split_subsections(block)
                if sub_sections:
                    for k, v in sub_sections.items():
                        if k in sections and v:
                            sections[k] = (sections[k] + " " + v).strip()

    return {k: v for k, v in sections.items() if str(v).strip()}


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
        text_col = text_col.lower()

    for c in ["hadm_id", "charttime", text_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

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

    print("Extracting Radiology sections...", flush=True)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        raw_text = row[text_col]
        sections = classify_radiology_blocks(raw_text)

        for sec_name, sec_text in sections.items():
            sec_text = str(sec_text).strip()

            if not sec_text:
                continue

            records.append({
                "hadm_id": int(row["hadm_id"]),
                "admittime": row["admittime"],
                "charttime": row["charttime"],
                "category": row.get("category", "Radiology"),
                "section_name": f"radiology_{sec_name}",
                "text": sec_text,
                "hours_from_admit": float(row["hours_from_admit"]),
                "time_bin": row["time_bin"],
            })

    df_long = pd.DataFrame(records)

    df_long.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df_long.to_parquet(OUTPUT_PARQUET, index=False)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in df_long.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print("\nSaved CSV:    ", OUTPUT_CSV, flush=True)
    print("Saved Parquet:", OUTPUT_PARQUET, flush=True)
    print("Saved JSONL:  ", OUTPUT_JSONL, flush=True)

    print("\nLong format shape:", df_long.shape, flush=True)

    if len(df_long) > 0:
        print("\nSection coverage:", flush=True)
        print(df_long["section_name"].value_counts().head(30), flush=True)

        print("\nTime-bin coverage:", flush=True)
        print(df_long["time_bin"].value_counts().sort_index(), flush=True)