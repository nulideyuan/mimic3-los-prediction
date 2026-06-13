import re
import json
import pandas as pd
import spacy
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

INPUT_PATH = "processed/notes_physician_all.parquet"
LOS_PATH = "processed/mimic3_los_aggregated.csv"

OUTPUT_DIR = Path("processed/section_trajectory_long")
OUTPUT_CSV = OUTPUT_DIR / "physician_24h_long.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "physician_24h_long.parquet"
OUTPUT_JSONL = OUTPUT_DIR / "physician_24h_long.jsonl"


PHYSICIAN_HEADERS = [
    "CHIEF COMPLAINT", "HPI", "24 HOUR EVENTS", "HOSPITAL COURSE",
    "ALLERGIES", "LAST DOSE OF ANTIBIOTICS", "INFUSIONS",
    "OTHER ICU MEDICATIONS", "OTHER MEDICATIONS",
    "MEDS AT HOME", "CURRENT MEDICATIONS", "MEDS",
    "PAST MEDICAL HISTORY", "PMH", "PAST SURGICAL HISTORY", "PSH",
    "FAMILY HISTORY", "SOCIAL HISTORY", "REVIEW OF SYSTEMS",
    "CHANGES TO MEDICAL AND FAMILY HISTORY",
    "FLOWSHEET DATA", "VITAL SIGNS", "HEMODYNAMIC MONITORING",
    "FLUID BALANCE", "RESPIRATORY SUPPORT", "RESPIRATORY",
    "PHYSICAL EXAMINATION", "PHYSICAL EXAM",
    "LABS / RADIOLOGY", "LABS", "IMAGING", "MICROBIOLOGY",
    "ASSESSMENT AND PLAN", "ASSESSMENT/PLAN", "ASSESSMENT", "PLAN",
    "ICU CARE", "NUTRITION", "GLYCEMIC CONTROL",
    "LINES / INTUBATION", "LINES", "ACCESS", "PROPHYLAXIS",
    "COMMUNICATION", "CODE STATUS", "CODE", "DISPOSITION",
    "TOTAL TIME SPENT", "PATIENT IS CRITICALLY ILL",
    "PROTECTED SECTION", "MICU ATTENDING ADDENDUM",
    "CONSULTS", "FLUIDS", "BILLING DIAGNOSIS",
]

HEADER_MAP = {
    "CHIEF COMPLAINT": "chief_complaint",
    "HPI": "hpi",
    "24 HOUR EVENTS": "events_24h",
    "HOSPITAL COURSE": "hospital_course",
    "ALLERGIES": "allergies",
    "LAST DOSE OF ANTIBIOTICS": "last_dose_of_antibiotics",
    "INFUSIONS": "infusions",
    "OTHER ICU MEDICATIONS": "other_icu_medications",
    "OTHER MEDICATIONS": "other_medications",
    "MEDS": "meds",
    "MEDS AT HOME": "meds_at_home",
    "CURRENT MEDICATIONS": "current_medications",
    "PAST MEDICAL HISTORY": "past_medical_history",
    "PMH": "past_medical_history",
    "PAST SURGICAL HISTORY": "past_surgical_history",
    "PSH": "past_surgical_history",
    "FAMILY HISTORY": "family_history",
    "SOCIAL HISTORY": "social_history",
    "REVIEW OF SYSTEMS": "review_of_systems",
    "CHANGES TO MEDICAL AND FAMILY HISTORY": "changes_to_medical_and_family_history",
    "FLOWSHEET DATA": "flowsheet_data",
    "VITAL SIGNS": "vital_signs",
    "HEMODYNAMIC MONITORING": "hemodynamic_monitoring",
    "FLUID BALANCE": "fluid_balance",
    "RESPIRATORY SUPPORT": "respiratory_support",
    "RESPIRATORY": "respiratory",
    "PHYSICAL EXAMINATION": "physical_examination",
    "PHYSICAL EXAM": "physical_examination",
    "LABS / RADIOLOGY": "labs_radiology",
    "LABS": "labs",
    "IMAGING": "imaging",
    "MICROBIOLOGY": "microbiology",
    "ASSESSMENT AND PLAN": "assessment_plan",
    "ASSESSMENT/PLAN": "assessment_plan",
    "ASSESSMENT": "assessment",
    "PLAN": "plan",
    "ICU CARE": "icu_care",
    "NUTRITION": "nutrition",
    "GLYCEMIC CONTROL": "glycemic_control",
    "LINES / INTUBATION": "lines_intubation",
    "LINES": "lines",
    "ACCESS": "access",
    "PROPHYLAXIS": "prophylaxis",
    "COMMUNICATION": "communication",
    "CODE STATUS": "code_status",
    "CODE": "code",
    "DISPOSITION": "disposition",
    "TOTAL TIME SPENT": "total_time_spent",
    "PATIENT IS CRITICALLY ILL": "patient_is_critically_ill",
    "PROTECTED SECTION": "protected_section",
    "MICU ATTENDING ADDENDUM": "micu_attending_addendum",
    "CONSULTS": "consults",
    "FLUIDS": "fluids",
    "BILLING DIAGNOSIS": "billing_diagnosis",
}

HEADER_PATTERN = "|".join(
    re.escape(h) for h in sorted(PHYSICIAN_HEADERS, key=len, reverse=True)
)

HEADER_RE = re.compile(
    rf"^\s*#?\s*({HEADER_PATTERN})\s*:?\s*(.*)$",
    flags=re.IGNORECASE
)


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
        h.lower().replace("/", "_").replace("&", "and").replace(" ", "_")
    )


def detect_header_line(line):
    raw = str(line).strip()
    if not raw:
        return None, None

    m = HEADER_RE.match(raw)
    if not m:
        return None, None

    header_raw = m.group(1).strip()
    after_header = m.group(2).strip()

    if len(after_header.split()) > 80:
        return None, None

    return normalize_header(header_raw), after_header


def split_physician_sections(text):
    if pd.isna(text):
        return {}

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    sections = {}
    current_header = None
    buffer = []

    def flush():
        nonlocal buffer, current_header

        if current_header and buffer:
            content = sentence_segment(" ".join(buffer))
            if content:
                sections[current_header] = (
                    sections.get(current_header, "") + " " + content
                ).strip()

        buffer = []

    for line in lines:
        header, after_header = detect_header_line(line)

        if header:
            flush()
            current_header = header

            if after_header:
                buffer.append(after_header)
        else:
            if current_header:
                buffer.append(line)

    flush()
    return sections


def merge_physician_sections(sections):
    return {
        "chief_complaint": clean_text(sections.get("chief_complaint", "")),

        "hpi_events": clean_text(" ".join([
            sections.get("hpi", ""),
            sections.get("events_24h", ""),
            sections.get("hospital_course", ""),
        ])),

        "past_history": clean_text(" ".join([
            sections.get("past_medical_history", ""),
            sections.get("past_surgical_history", ""),
            sections.get("family_history", ""),
            sections.get("social_history", ""),
            sections.get("meds", ""),
            sections.get("meds_at_home", ""),
            sections.get("current_medications", ""),
            sections.get("other_medications", ""),
        ])),

        "exam_labs": clean_text(" ".join([
            sections.get("flowsheet_data", ""),
            sections.get("vital_signs", ""),
            sections.get("hemodynamic_monitoring", ""),
            sections.get("fluid_balance", ""),
            sections.get("respiratory_support", ""),
            sections.get("respiratory", ""),
            sections.get("physical_examination", ""),
            sections.get("labs_radiology", ""),
            sections.get("labs", ""),
            sections.get("imaging", ""),
            sections.get("microbiology", ""),
        ])),

        "assessment_plan": clean_text(" ".join([
            sections.get("assessment_plan", ""),
            sections.get("assessment", ""),
            sections.get("plan", ""),
            sections.get("icu_care", ""),
            sections.get("nutrition", ""),
            sections.get("glycemic_control", ""),
            sections.get("lines_intubation", ""),
            sections.get("lines", ""),
            sections.get("access", ""),
            sections.get("prophylaxis", ""),
            sections.get("communication", ""),
            sections.get("code_status", ""),
            sections.get("code", ""),
            sections.get("disposition", ""),
            sections.get("total_time_spent", ""),
            sections.get("patient_is_critically_ill", ""),
            sections.get("protected_section", ""),
            sections.get("micu_attending_addendum", ""),
            sections.get("consults", ""),
            sections.get("fluids", ""),
            sections.get("billing_diagnosis", ""),
        ])),

        "other_sections": clean_text(" ".join([
            sections.get("allergies", ""),
            sections.get("last_dose_of_antibiotics", ""),
            sections.get("infusions", ""),
            sections.get("other_icu_medications", ""),
            sections.get("review_of_systems", ""),
            sections.get("changes_to_medical_and_family_history", ""),
        ])),
    }


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

# =========================================================
# admittime
# =========================================================

    df["hadm_id"] = df["hadm_id"].astype(int)

    if "admittime" in df.columns:
        print("Using existing admittime from physician parquet")

        df["admittime"] = pd.to_datetime(
            df["admittime"],
            errors="coerce"
        )

    else:
        print("Loading admittime from LOS file")

        los_df = pd.read_csv(
            LOS_PATH,
            parse_dates=["admittime"],
            usecols=["hadm_id", "admittime"],
        )

        los_df["hadm_id"] = los_df["hadm_id"].astype(int)

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

    print("Extracting Physician sections...", flush=True)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        raw_text = row[text_col]

        all_sections = split_physician_sections(raw_text)

        for sec_name, sec_text in all_sections.items():
            sec_text = str(sec_text).strip()

            if not sec_text:
                continue

            records.append({
                "hadm_id": int(row["hadm_id"]),
                "admittime": row["admittime"],
                "charttime": row["charttime"],
                "category": row.get("category", "Physician"),
                "section_name": f"physician_{sec_name}",
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