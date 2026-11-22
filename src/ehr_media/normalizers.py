"""Deterministic clinical data normalizers.

All functions are pure, side-effect free and return enough context (a rule
name and an optional warning) for the pipeline to write a meaningful audit
trail. Determinism matters: re-running ingestion on the same input must
produce byte-identical cleaned records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from dateutil import parser as dateutil_parser

from .models import Anomaly, DocumentType, EncounterClass, Gender, LabInterpretation

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_SENTINELS = {
    "",
    "-",
    "--",
    "---",
    "n/a",
    "na",
    "n.a.",
    "none",
    "null",
    "nil",
    "unknown",
    "unk",
    "not documented",
    "not available",
    "not recorded",
    "no data",
    "missing",
    "0",
}

_HONORIFICS = re.compile(r"^(dr|mr|mrs|ms|miss|prof)\.?\s+", re.IGNORECASE)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]")
_PAGE_MARKER = re.compile(r"\[\s*page\s+\d+(\s+of\s+\d+)?\s*\]", re.IGNORECASE)


def is_blank(value: Any) -> bool:
    """True for None, empty/whitespace strings and common sentinel values."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _SENTINELS
    return False


def clean_text(value: Any) -> str:
    """Trim, collapse internal whitespace and strip control characters."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\xa0", " ").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_document_text(value: Any) -> str:
    """Normalize OCR'd / pasted clinical text while preserving paragraphs."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text).replace("\xa0", " ").replace("\f", "\n")
    text = _PAGE_MARKER.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_name(value: Any) -> str | None:
    """Title-case a person name, stripping honorifics and stray punctuation."""
    text = clean_text(value)
    if is_blank(text):
        return None
    text = _HONORIFICS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:.")
    if not text:
        return None
    if text.isupper() or text.islower():
        text = text.title()
    text = re.sub(r"\s*,\s*", ", ", text)
    # Common name particles that should stay lowercase in the middle of a name.
    text = re.sub(r"\b(Mc|Mac)(\w)", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\bO'(\w)", lambda m: "O'" + m.group(1).upper(), text)
    text = re.sub(
        r"\b(De|Del|Della|Van|Von|Da|Di|La|Le)\b",
        lambda m: m.group(1).lower(),
        text,
    )
    # Re-capitalize the first character (the particle rule can hit position 0).
    return text[:1].upper() + text[1:]


def split_comma_name(value: str | None) -> tuple[str | None, str | None]:
    """Split ``"Bell, Marcus"`` into ``("Marcus", "Bell")``."""
    if not value:
        return None, None
    if "," in value:
        family, _, given = value.partition(",")
        return normalize_name(given), normalize_name(family)
    parts = value.split()
    if len(parts) == 1:
        return None, normalize_name(parts[0])
    return normalize_name(" ".join(parts[:-1])), normalize_name(parts[-1])


def normalize_title(value: Any, fallback: str = "Clinical document") -> str:
    text = clean_text(value)
    if not text:
        return fallback
    text = text.strip(" .;:-")
    if text.isupper():
        text = text.title()
    return text[:140]


# ---------------------------------------------------------------------------
# MRN
# ---------------------------------------------------------------------------

_MRN_PREFIX = re.compile(
    r"^(mrn|mr|medical\s*record(\s*(number|no\.?|#))?|patient\s*id|acct|account)\s*[:#\-]?\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MrnParse:
    value: str | None
    rule: str
    changed: bool
    warning: str | None = None


def normalize_mrn(value: Any) -> MrnParse:
    """Normalize MRNs to a canonical 8-digit form when numeric.

    ``" mrn 1234 "`` -> ``"00001234"``. Alphanumeric MRNs are upper-cased and
    stripped of separators (``"a-7781"`` -> ``"A7781"``).
    """
    if is_blank(value):
        return MrnParse(None, "mrn.empty", False, "missing MRN")
    original = str(value).strip()
    text = _MRN_PREFIX.sub("", original)
    text = re.sub(r"[\s\-_.]", "", text)
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    if not text:
        return MrnParse(None, "mrn.empty_after_strip", True, "MRN contained no alphanumerics")
    rules: list[str] = []
    if text != original:
        rules.append("mrn.separators_stripped")
    if text.isdigit():
        stripped = text.lstrip("0") or "0"
        if len(stripped) <= 8:
            text = stripped.zfill(8)
            rules.append("mrn.zero_padded")
        else:
            text = stripped
    else:
        text = text.upper()
        rules.append("mrn.uppercased")
    return MrnParse(text, "+".join(rules) or "mrn.already_canonical", text != original)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_DATE_SENTINELS = _SENTINELS | {"0000-00-00", "00000000", "1900-01-01"}
_US_SLASH = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:\s+.*)?$")
_ISO_PREFIX = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s].*)?$")
_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


@dataclass(frozen=True)
class DateParse:
    value: date | None
    rule: str
    warning: str | None = None


def parse_date(
    value: Any,
    *,
    today: date | None = None,
    allow_future: bool = False,
) -> DateParse:
    """Parse heterogeneous date strings into a :class:`datetime.date`.

    Handles ISO-8601, US ``MM/DD/YYYY``, European ``DD/MM/YYYY`` (when the
    first component cannot be a month), ``DD-MMM-YYYY``, compact ``YYYYMMDD``
    and free-form dates. Ambiguous numeric dates are interpreted month-first
    (US convention) and flagged in the returned warning.
    """
    today = today or date.today()
    if value is None:
        return DateParse(None, "date.empty")
    if isinstance(value, datetime):
        return _sanity_check(value.date(), "date.iso8601", None, today, allow_future)
    if isinstance(value, date):
        return _sanity_check(value, "date.object", None, today, allow_future)

    raw = str(value).strip()
    if raw.lower() in _DATE_SENTINELS or is_blank(raw):
        return DateParse(None, "date.empty")

    # Strip a time component we do not need; keep the date.
    raw = re.sub(r"\s+(UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|Z)$", "", raw, flags=re.IGNORECASE)
    raw = raw.strip()

    warning: str | None = None

    compact = _COMPACT.match(raw)
    if compact:
        try:
            parsed = date(int(compact.group(1)), int(compact.group(2)), int(compact.group(3)))
            return _sanity_check(parsed, "date.compact", warning, today, allow_future)
        except ValueError:
            return DateParse(None, "date.invalid", f"invalid compact date {value!r}")

    iso = _ISO_PREFIX.match(raw)
    if iso:
        try:
            parsed = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            return _sanity_check(parsed, "date.iso8601", warning, today, allow_future)
        except ValueError:
            return DateParse(None, "date.invalid", f"invalid ISO date {value!r}")

    slash = _US_SLASH.match(raw)
    if slash:
        first, second, year = int(slash.group(1)), int(slash.group(2)), int(slash.group(3))
        if year < 100:
            year += 2000 if year <= 68 else 1900
        if first > 12 and second <= 12:
            month, day, rule = second, first, "date.day_first"
        elif second > 12 and first <= 12:
            month, day, rule = first, second, "date.month_first"
        else:
            month, day, rule = first, second, "date.month_first"
            warning = "ambiguous numeric date interpreted as MM/DD/YYYY"
        try:
            parsed = date(year, month, day)
        except ValueError:
            return DateParse(None, "date.invalid", f"invalid calendar date {value!r}")
        return _sanity_check(parsed, rule, warning, today, allow_future)

    # Free-form fallback (e.g. "March 4, 2021", "4 Mar 2021", "Mar 4 2021").
    try:
        parsed_dt = dateutil_parser.parse(raw, dayfirst=False, fuzzy=False)
        return _sanity_check(parsed_dt.date(), "date.freeform", warning, today, allow_future)
    except (ValueError, OverflowError, TypeError):
        pass
    try:
        parsed_dt = dateutil_parser.parse(raw, dayfirst=True, fuzzy=False)
        return _sanity_check(
            parsed_dt.date(),
            "date.freeform_day_first",
            "ambiguous free-form date interpreted day-first",
            today,
            allow_future,
        )
    except (ValueError, OverflowError, TypeError):
        return DateParse(None, "date.invalid", f"could not parse date {value!r}")


def _sanity_check(
    parsed: date,
    rule: str,
    warning: str | None,
    today: date,
    allow_future: bool,
) -> DateParse:
    if parsed.year < 1900:
        return DateParse(None, "date.out_of_range", f"implausible year {parsed.year}")
    if not allow_future and parsed > today:
        return DateParse(None, "date.future", f"date {parsed.isoformat()} is in the future")
    return DateParse(parsed, rule, warning)


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

_GENDER_MAP: dict[str, Gender] = {
    "m": "male",
    "male": "male",
    "man": "male",
    "boy": "male",
    "1": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
    "girl": "female",
    "2": "female",
    "o": "other",
    "other": "other",
    "nonbinary": "other",
    "non-binary": "other",
    "non binary": "other",
    "a": "other",
    "ambiguous": "other",
    "d": "other",
    "u": "unknown",
    "unknown": "unknown",
    "unk": "unknown",
    "n": "unknown",
    "not applicable": "unknown",
    "undisclosed": "unknown",
    "": "unknown",
}

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC", "PR",
}

_STATE_NAMES = {
    "new york": "NY",
    "new jersey": "NJ",
    "connecticut": "CT",
    "california": "CA",
    "texas": "TX",
    "florida": "FL",
    "massachusetts": "MA",
    "pennsylvania": "PA",
    "illinois": "IL",
    "washington": "WA",
}


@dataclass(frozen=True)
class GenderParse:
    value: Gender
    recognized: bool
    rule: str
    warning: str | None = None


def normalize_gender(value: Any) -> GenderParse:
    if value is None or (isinstance(value, str) and not value.strip()):
        return GenderParse("unknown", False, "gender.missing", "gender not documented")
    key = re.sub(r"[\s_\-]+", " ", str(value).strip().lower())
    key = key.replace("sex:", "").strip()
    if key in _GENDER_MAP:
        gender = _GENDER_MAP[key]
        return GenderParse(gender, gender != "unknown" or key in {"u", "unknown", "unk", "n"}, "gender.lookup")
    # "Male (M)" / "F - female" style values
    for token in re.split(r"[^a-z0-9\-]+", key):
        if token in _GENDER_MAP and token not in {"1", "2", "a", "d", "o", "u", "n"}:
            return GenderParse(_GENDER_MAP[token], True, "gender.token_lookup")
    return GenderParse("unknown", False, "gender.unrecognized", f"unrecognized gender code {value!r}")


def normalize_phone(value: Any) -> tuple[str | None, str | None]:
    """Return ``(E.164-ish phone, warning)`` for US numbers."""
    if is_blank(value):
        return None, None
    raw = str(value)
    digits = re.sub(r"\D", "", re.sub(r"\s*(ext\.?|x)\s*\d+$", "", raw, flags=re.IGNORECASE))
    if len(digits) == 10:
        return f"+1{digits}", None
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}", None
    if raw.strip().startswith("+") and 8 <= len(digits) <= 15:
        return f"+{digits}", None
    return None, f"unrecognized phone number {value!r}"


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def normalize_email(value: Any) -> tuple[str | None, str | None]:
    if is_blank(value):
        return None, None
    email = str(value).strip().lower()
    if _EMAIL_RE.match(email):
        return email, None
    return None, f"invalid email {value!r}"


def normalize_state(value: Any) -> str | None:
    if is_blank(value):
        return None
    state = str(value).strip()
    if state.lower() in _STATE_NAMES:
        return _STATE_NAMES[state.lower()]
    state = state.upper()
    return state if state in _US_STATES else None


# ---------------------------------------------------------------------------
# Document / encounter classification
# ---------------------------------------------------------------------------

_DOC_TYPE_PATTERNS: list[tuple[re.Pattern[str], DocumentType]] = [
    (re.compile(r"discharge", re.I), "discharge_summary"),
    (re.compile(r"consult", re.I), "consultation"),
    (re.compile(r"progress|office visit|clinic note|follow.?up", re.I), "progress_note"),
    (re.compile(r"imag|radiol|ct\b|mri\b|x.?ray|ultrasound|sonograph|echo|mammo|pet\b", re.I), "imaging_report"),
    (re.compile(r"lab|laborator|patholog|blood work|panel", re.I), "lab_report"),
    (re.compile(r"scan|ocr|fax|handwritten", re.I), "scanned_note"),
]


def normalize_doc_type(value: Any) -> DocumentType:
    text = clean_text(value)
    if not text:
        return "other"
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if pattern.search(text):
            return doc_type
    return "other"


def normalize_encounter_class(value: Any) -> EncounterClass:
    text = clean_text(value).lower()
    if re.search(r"inpatient|\bip\b|\bimp\b|admit", text):
        return "IMP"
    if re.search(r"emergency|\bed\b|\ber\b|emer", text):
        return "EMER"
    if re.search(r"observation|\bobs", text):
        return "OBSENC"
    return "AMB"


# ---------------------------------------------------------------------------
# Laboratory results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabCatalogEntry:
    name: str
    loinc: str
    unit: str | None
    low: float | None
    high: float | None
    critical_low: float | None = None
    critical_high: float | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)


LAB_CATALOG: dict[str, LabCatalogEntry] = {
    "hemoglobin": LabCatalogEntry(
        "Hemoglobin", "718-7", "g/dL", 12.0, 17.5, 7.0, 20.0,
        ("hgb", "hb", "hemoglobin", "haemoglobin", "hgb blood"),
    ),
    "hematocrit": LabCatalogEntry(
        "Hematocrit", "4544-3", "%", 36.0, 52.0, 20.0, 60.0,
        ("hct", "hematocrit", "haematocrit", "pcv"),
    ),
    "wbc": LabCatalogEntry(
        "White Blood Cell Count", "6690-2", "10^3/uL", 4.0, 11.0, 2.0, 30.0,
        ("wbc", "white blood cell count", "white count", "leukocytes", "wbc count"),
    ),
    "platelets": LabCatalogEntry(
        "Platelet Count", "777-3", "10^3/uL", 150.0, 400.0, 50.0, 1000.0,
        ("plt", "platelet", "platelets", "platelet count"),
    ),
    "glucose": LabCatalogEntry(
        "Glucose", "2345-7", "mg/dL", 70.0, 140.0, 40.0, 500.0,
        ("glu", "glucose", "blood glucose", "fasting glucose", "random glucose"),
    ),
    "hba1c": LabCatalogEntry(
        "Hemoglobin A1c", "4548-4", "%", 4.0, 5.6, None, None,
        ("a1c", "hba1c", "hemoglobin a1c", "glycated hemoglobin", "glycohemoglobin"),
    ),
    "creatinine": LabCatalogEntry(
        "Creatinine", "2160-0", "mg/dL", 0.6, 1.3, None, 4.0,
        ("creat", "creatinine", "serum creatinine"),
    ),
    "egfr": LabCatalogEntry(
        "eGFR", "33914-3", "mL/min/1.73m2", 60.0, None, 15.0, None,
        ("egfr", "gfr", "estimated gfr", "glomerular filtration rate"),
    ),
    "sodium": LabCatalogEntry(
        "Sodium", "2951-2", "mmol/L", 135.0, 145.0, 120.0, 160.0,
        ("na", "sodium", "na+", "serum sodium"),
    ),
    "potassium": LabCatalogEntry(
        "Potassium", "2823-3", "mmol/L", 3.5, 5.1, 2.5, 6.5,
        ("k", "potassium", "k+", "serum potassium"),
    ),
    "chloride": LabCatalogEntry(
        "Chloride", "2075-0", "mmol/L", 98.0, 107.0,
        aliases=("cl", "chloride", "cl-"),
    ),
    "bicarbonate": LabCatalogEntry(
        "Bicarbonate", "2028-9", "mmol/L", 22.0, 29.0,
        aliases=("co2", "bicarbonate", "hco3", "bicarb", "total co2"),
    ),
    "bun": LabCatalogEntry(
        "Blood Urea Nitrogen", "3094-0", "mg/dL", 7.0, 20.0,
        aliases=("bun", "blood urea nitrogen", "urea nitrogen", "urea"),
    ),
    "alt": LabCatalogEntry(
        "ALT", "1742-6", "U/L", 7.0, 56.0,
        aliases=("alt", "sgpt", "alanine aminotransferase", "alanine transaminase"),
    ),
    "ast": LabCatalogEntry(
        "AST", "1920-8", "U/L", 10.0, 40.0,
        aliases=("ast", "sgot", "aspartate aminotransferase", "aspartate transaminase"),
    ),
    "alp": LabCatalogEntry(
        "Alkaline Phosphatase", "6768-6", "U/L", 44.0, 147.0,
        aliases=("alp", "alk phos", "alkaline phosphatase"),
    ),
    "bilirubin": LabCatalogEntry(
        "Total Bilirubin", "1975-2", "mg/dL", 0.1, 1.2, None, 5.0,
        aliases=("bili", "bilirubin", "total bilirubin", "t bili", "tbil"),
    ),
    "troponin": LabCatalogEntry(
        "Troponin I", "10839-9", "ng/mL", None, 0.04, None, 0.4,
        aliases=("trop", "troponin", "troponin i", "trop i", "hs troponin"),
    ),
    "tsh": LabCatalogEntry(
        "TSH", "3016-3", "mIU/L", 0.4, 4.0,
        aliases=("tsh", "thyroid stimulating hormone", "thyrotropin"),
    ),
    "ldl": LabCatalogEntry(
        "LDL Cholesterol", "2089-1", "mg/dL", None, 100.0,
        aliases=("ldl", "ldl cholesterol", "ldl-c"),
    ),
    "hdl": LabCatalogEntry(
        "HDL Cholesterol", "2085-9", "mg/dL", 40.0, None,
        aliases=("hdl", "hdl cholesterol", "hdl-c"),
    ),
    "triglycerides": LabCatalogEntry(
        "Triglycerides", "2571-8", "mg/dL", None, 150.0,
        aliases=("trig", "triglycerides", "tg"),
    ),
    "inr": LabCatalogEntry(
        "INR", "6301-6", "ratio", 0.8, 1.2, None, 5.0,
        aliases=("inr", "international normalized ratio", "pt/inr"),
    ),
    "psa": LabCatalogEntry(
        "PSA", "2857-1", "ng/mL", None, 4.0,
        aliases=("psa", "prostate specific antigen"),
    ),
    "bnp": LabCatalogEntry(
        "BNP", "30934-4", "pg/mL", None, 100.0, None, 1000.0,
        aliases=("bnp", "b-type natriuretic peptide", "brain natriuretic peptide"),
    ),
    "lactate": LabCatalogEntry(
        "Lactate", "2524-7", "mmol/L", 0.5, 2.0, None, 4.0,
        aliases=("lactate", "lactic acid"),
    ),
    "magnesium": LabCatalogEntry(
        "Magnesium", "19123-9", "mg/dL", 1.7, 2.2,
        aliases=("mg", "magnesium", "mg2+"),
    ),
    "calcium": LabCatalogEntry(
        "Calcium", "17861-6", "mg/dL", 8.6, 10.2, 7.0, 13.0,
        aliases=("ca", "calcium", "ca2+"),
    ),
    "albumin": LabCatalogEntry(
        "Albumin", "1751-7", "g/dL", 3.5, 5.0,
        aliases=("alb", "albumin"),
    ),
    "ferritin": LabCatalogEntry(
        "Ferritin", "2276-4", "ng/mL", 15.0, 200.0,
        aliases=("ferritin",),
    ),
    "vitamin_d": LabCatalogEntry(
        "Vitamin D", "1989-3", "ng/mL", 30.0, 100.0,
        aliases=("vit d", "vitamin d", "25-oh vitamin d", "25 hydroxy vitamin d"),
    ),
    "crp": LabCatalogEntry(
        "C-Reactive Protein", "1988-5", "mg/L", None, 10.0,
        aliases=("crp", "c-reactive protein", "c reactive protein"),
    ),
    "procalcitonin": LabCatalogEntry(
        "Procalcitonin", "33959-8", "ng/mL", None, 0.1,
        aliases=("procalcitonin", "pct"),
    ),
    "urine_protein": LabCatalogEntry(
        "Urine Protein", "2888-6", "mg/dL", None, 20.0,
        aliases=("urine protein", "protein urine", "urine protein qualitative"),
    ),
    "uacr": LabCatalogEntry(
        "Urine Albumin-Creatinine Ratio", "9318-7", "mg/g", None, 30.0,
        aliases=(
            "uacr",
            "urine albumin-creatinine ratio",
            "urine albumin creatinine ratio",
            "albumin creatinine ratio",
            "albumin/creatinine ratio",
            "microalbumin creatinine ratio",
        ),
    ),
    "free_t4": LabCatalogEntry(
        "Free T4", "3024-7", "ng/dL", 0.8, 1.8,
        aliases=("free t4", "ft4", "free thyroxine"),
    ),
    "ammonia": LabCatalogEntry(
        "Ammonia", "16362-6", "umol/L", 15.0, 45.0,
        aliases=("ammonia", "nh3"),
    ),
    "ldh": LabCatalogEntry(
        "LDH", "2532-0", "U/L", 140.0, 280.0,
        aliases=("ldh", "lactate dehydrogenase"),
    ),
}

_LAB_ALIAS_INDEX: dict[str, LabCatalogEntry] = {
    alias: entry for entry in LAB_CATALOG.values() for alias in entry.aliases
}


def _test_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9%/\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lookup_lab(test_name: Any) -> LabCatalogEntry | None:
    key = _test_key(test_name)
    if key in _LAB_ALIAS_INDEX:
        return _LAB_ALIAS_INDEX[key]
    for alias, entry in _LAB_ALIAS_INDEX.items():
        if len(alias) >= 5 and (key.startswith(alias + " ") or key.endswith(" " + alias)):
            return entry
    return None


def normalize_lab_name(value: Any) -> tuple[str, str | None]:
    """Return ``(canonical display name, LOINC code)``."""
    entry = lookup_lab(value)
    if entry:
        return entry.name, entry.loinc
    text = clean_text(value)
    text = re.sub(r"\s+", " ", text).strip(" :*")
    if text.isupper():
        text = text.title()
    return (text or "Unknown test"), None


@dataclass(frozen=True)
class ParsedValue:
    value: float | None
    value_text: str | None
    qualifier: str | None
    unit: str | None
    flag: str | None = None
    warning: str | None = None


_VALUE_RE = re.compile(
    r"^\s*(?P<qual>[<>]=?|≤|≥)?\s*(?P<num>[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>[A-Za-z%µμ/^0-9.\-]+)?\s*(?P<flag>[HLN])?\s*$"
)


def parse_lab_value(value: Any) -> ParsedValue:
    """Parse numeric values with qualifiers/units, falling back to text."""
    if value is None:
        return ParsedValue(None, None, None, None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ParsedValue(float(value), None, None, None)
    raw = clean_text(value)
    if is_blank(raw):
        return ParsedValue(None, None, None, None)
    raw = re.sub(r"\s*\((low|high|normal|abnormal|critical)\)\s*$", "", raw, flags=re.I)
    match = _VALUE_RE.match(raw)
    if not match:
        return ParsedValue(None, raw.upper()[:60], None, None, warning="non-numeric lab value")
    qual = match.group("qual")
    if qual in {"≤"}:
        qual = "<="
    if qual in {"≥"}:
        qual = ">="
    number = match.group("num")
    if "," in number and "." not in number:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", number):
            number = number.replace(",", "")
        else:
            number = number.replace(",", ".")
    unit = match.group("unit")
    flag = match.group("flag")
    try:
        parsed = float(number)
    except ValueError:
        return ParsedValue(None, raw.upper()[:60], None, None, warning="non-numeric lab value")
    return ParsedValue(parsed, None, qual, unit, flag=flag)


_RANGE_RE = re.compile(
    r"^\s*(?P<low>[-+]?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?P<high>[-+]?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_UPPER_RE = re.compile(r"^\s*(?:<|<=|≤|less than|up to|max(?:imum)?)\s*(?P<high>\d+(?:\.\d+)?)", re.I)
_LOWER_RE = re.compile(r"^\s*(?:>|>=|≥|greater than|at least|min(?:imum)?)\s*(?P<low>\d+(?:\.\d+)?)", re.I)


def parse_reference_range(value: Any) -> tuple[float | None, float | None, str | None]:
    """Parse ``"3.5-5.1"``, ``"<100"``, ``">40"`` into numeric bounds."""
    if is_blank(value):
        return None, None, None
    # Strip a trailing unit only when the remainder still parses as a range.
    candidates = [clean_text(value)]
    stripped = re.sub(r"\s+[A-Za-z%µμ/^0-9.\-]+$", "", clean_text(value)).strip()
    if stripped and stripped != candidates[0]:
        candidates.append(stripped)
    for candidate in candidates:
        match = _RANGE_RE.match(candidate)
        if match:
            low, high = float(match.group("low")), float(match.group("high"))
            return (min(low, high), max(low, high), "range.low_high")
        match = _UPPER_RE.match(candidate)
        if match:
            return (None, float(match.group("high")), "range.upper_bound")
        match = _LOWER_RE.match(candidate)
        if match:
            return (float(match.group("low")), None, "range.lower_bound")
    return None, None, None


def interpret_lab(
    test_name: str,
    value: float | None,
    reference_low: float | None,
    reference_high: float | None,
    *,
    qualifier: str | None = None,
    explicit_flag: str | None = None,
) -> LabInterpretation:
    if value is None:
        return "unknown"
    entry = lookup_lab(test_name)
    if entry:
        if entry.critical_high is not None and value > entry.critical_high:
            return "critical"
        if entry.critical_low is not None and value < entry.critical_low:
            return "critical"
    if explicit_flag == "H":
        return "high"
    if explicit_flag == "L":
        return "low"
    if qualifier in {"<", "<="} and reference_high is not None and value > reference_high:
        return "high"
    if qualifier in {">", ">="} and reference_low is not None and value < reference_low:
        return "low"
    if reference_low is not None and value < reference_low:
        return "low"
    if reference_high is not None and value > reference_high:
        return "high"
    if reference_low is None and reference_high is None:
        return "unknown"
    return "normal"


# ---------------------------------------------------------------------------
# Clinical text anomaly detection
# ---------------------------------------------------------------------------

_ANOMALY_PATTERNS: list[tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(r"\b(critical|panic value|critically low|critically high)\b", re.I),
     "critical_language", "Document contains critical/panic-value language", "critical"),
    (re.compile(r"\bsepsis\b|\bseptic\b|\bseptic shock\b", re.I),
     "sepsis_mention", "Document mentions sepsis", "critical"),
    (re.compile(r"\bpositive blood culture\b", re.I),
     "positive_blood_culture", "Positive blood culture documented", "critical"),
    (re.compile(r"\bpneumothorax\b", re.I),
     "pneumothorax", "Pneumothorax documented", "critical"),
    (re.compile(r"\b(hemorrhage|haemorrhage)\b", re.I),
     "hemorrhage", "Hemorrhage documented", "critical"),
    (re.compile(r"\b(acute infarct|ischemic stroke|intracranial hemorrhage)\b", re.I),
     "acute_neuro_finding", "Acute neurological finding documented", "critical"),
    (re.compile(r"\bintubat(ed|ion)\b|\bmechanical ventilation\b", re.I),
     "mechanical_ventilation", "Mechanical ventilation documented", "critical"),
    (re.compile(r"\bcode blue\b", re.I),
     "code_blue", "Code blue documented", "critical"),
    (re.compile(r"\bpneumonia\b|\binfiltrate\b|\bconsolidation\b", re.I),
     "infection_imaging_finding", "Possible infection on imaging", "warning"),
    (re.compile(r"\bpleural effusion\b", re.I),
     "pleural_effusion", "Pleural effusion documented", "warning"),
    (re.compile(r"\b(nodule|mass|neoplasm|malignan\w*)\b", re.I),
     "mass_or_nodule", "Mass/nodule/malignancy language documented", "warning"),
    (re.compile(r"\bfracture\b", re.I),
     "fracture", "Fracture documented", "warning"),
    (re.compile(r"\b(STAT|urgent(ly)?)\b"),
     "urgent_language", "Urgent/STAT language in document", "warning"),
    (re.compile(r"\bDNR\b|\bDNI\b|\bcode status\b", re.I),
     "code_status", "Code status documented", "info"),
]

_NEGATION = re.compile(r"\b(no|without|negative for|free of|denies)\s+(?:\w+\s+){0,2}$", re.I)


def detect_anomalies(text: str) -> list[Anomaly]:
    """Flag clinically meaningful phrases, respecting simple negation."""
    found: list[Anomaly] = []
    seen: set[str] = set()
    for pattern, code, description, severity in _ANOMALY_PATTERNS:
        if code in seen:
            continue
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 40): match.start()]
            if _NEGATION.search(prefix):
                continue
            found.append(Anomaly(code=code, description=description, severity=severity))  # type: ignore[arg-type]
            seen.add(code)
            break
        if len(found) >= 6:
            break
    return found


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
