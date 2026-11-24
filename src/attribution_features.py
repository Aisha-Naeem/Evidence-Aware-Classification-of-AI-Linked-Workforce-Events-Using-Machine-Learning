"""
Feature engineering for the AI-attribution classifier.

This module re-implements, in one importable place, the Part 1 transformations that turn
a raw AI Layoffs Register row into the model input columns listed in
`data/processed/feature_spec.json`. It is used by:

* `Part2_Modelling_and_Evaluation.ipynb`, which checks that the module reproduces the
  Part 1 feature columns exactly from the raw CSV (so the demo cannot drift from training), and
* `app.py` (Streamlit demo), which builds the same features for a new, user-entered event.

Every regex and mapping below is copied from Part 1 (Sections 5.5, 6.1-6.4 and 7.1).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants copied from Part 1
# ---------------------------------------------------------------------------
PLACEHOLDER_RE = r"^(?:not (?:specified|disclosed)(?: by the company)?|unknown|n/?a|none)\.?$"

# Earliest event date in the snapshot (used for `months_since_start`, kept for completeness)
DATA_START = pd.Timestamp("2023-02-22")

JOB_FAMILIES = [
    "Consulting and professional services", "Content moderation", "Corporate functions",
    "Customer service and support", "Frontline and manual work", "Management layers",
    "Operations and back office", "Product and design", "Sales and marketing",
    "Software engineering and IT", "Writing, editorial and translation",
]

KIND_OPTIONS = ["cut", "workforce statement"]
EXECUTION_OPTIONS = ["executed", "announced", "partial", "unknown", "reversed"]

# --- Country -> region (Part 1, Section 5.5) ---
REGION_MAP = {
    "united states": "north_america", "canada": "north_america",
    "argentina": "latin_america",
    "united kingdom": "europe", "germany": "europe", "france": "europe", "netherlands": "europe",
    "italy": "europe", "sweden": "europe", "poland": "europe", "belgium": "europe",
    "switzerland": "europe", "norway": "europe", "spain": "europe", "ireland": "europe",
    "finland": "europe", "denmark": "europe", "portugal": "europe", "austria": "europe",
    "israel": "middle_east",
    "india": "asia_pacific", "australia": "asia_pacific", "singapore": "asia_pacific",
    "japan": "asia_pacific", "malaysia": "asia_pacific",
    "south africa": "africa", "nigeria": "africa",
    "global": "global",
}
MULTI_COUNTRY_RE = r"global|,|/|;| and |countries|worldwide|abroad"

# --- Sector -> sector group (Part 1, Section 5.5). Ordered rules, first match wins. ---
SECTOR_RULES = [
    ("gaming", r"game|gaming|gambl|betting"),
    ("edtech", r"educat|edtech|course|upskilling|academic"),
    ("health_pharma", r"health|pharma|drug|vaccin"),
    ("fintech_crypto", r"fintech|payment|crypto|bitcoin|blockchain|digital asset|neobank|custody|credit scoring"),
    ("banking_insurance", r"bank|insurance|financial services"),
    ("telecom", r"telecom"),
    ("big_tech_internet", r"big tech|social|consumer internet|^internet|ridesharing|hardware|semiconductor|"
                          r"consumer technology|metaverse|ai models|consumer apps"),
    ("media_marketing", r"media|news|publish|journalism|broadcast|radio|entertainment|music|audience|"
                        r"advertising|marketing services|creator"),
    ("it_services_bpo_consulting", r"it services|it solutions|consult|bpo|call cent|contact centres|business process|"
                                   r"outsourc|reseller|^legal$|services$|technology / services|travel management|"
                                   r"digital operations"),
    ("commerce_travel", r"e-commerce|ecommerce|marketplace|retail|grocery|delivery|fashion|travel|airline|aviation|"
                        r"ticketing|resale|interiors|styling|tour"),
    ("industrial_consumer_goods", r"automotive|chemical|logistic|freight|consumer goods|tobacco|appliance|cpg|materials"),
    ("software_saas", r"software|saas|tech|cyber|data|cloud|dev|platform|\bai\b|artificial intelligence|analytics|"
                      r"intelligence|web|internet|digital|online|mobile|app|geolocation|maps|recruit|\bhr\b|infrastructure"),
]
SECTOR_GROUPS = [g for g, _ in SECTOR_RULES] + ["other"]

# --- Company-name masking (Part 1, Section 6.1) ---
NAME_SUFFIX_RE = (r"(?:,?\s+(?:Inc\.?|plc|PLC|Corporation|Corp\.?|N\.V\.|Holdings?|Holding Corp\.|Group|"
                  r"Worldwide|& Company|Software|Technologies|Systems|International))+$")
MANUAL_ALIASES = {
    "Bank of America": ["BofA"], "JPMorgan Chase": ["JPMorgan", "JP Morgan", "J.P. Morgan"],
    "General Motors": ["GM"], "Procter & Gamble": ["P&G"], "National Australia Bank": ["NAB"],
    "Commonwealth Bank": ["CBA", "CommBank"], "Google": ["Alphabet"], "Meta": ["Facebook"],
    "British American Tobacco": ["BAT"], "Tata Consultancy Services (TCS)": ["Tata"],
    "Commerce.com (BigCommerce)": ["Commerce.com"], "Wix.com": ["Wix"], "Gizmodo en Español": ["Gizmodo"],
    "Bendigo and Adelaide Bank": ["Bendigo", "Bendigo Bank"], "Bolt Financial": ["Bolt"],
    "Just Eat Takeaway": ["Just Eat", "Takeaway.com"], "Papaya Gaming": ["Papaya"], "Starling Bank": ["Starling"],
    "Telenor Norge": ["Telenor"], "Vodafone Germany": ["Vodafone"], "Rackspace Technology": ["Rackspace"],
    "WiseTech Global": ["WiseTech"],
}
GENERIC_ALIASES = {"People", "Freight", "Commerce", "General", "National"}
CASE_SENSITIVE_ALIASES = {"Block", "Square", "Bird", "Snap", "Chime", "Visa", "Sunrise", "Apple", "Dune", "Bolt",
                          "Gemini", "Papaya", "Starling", "Just Eat", "Meta", "Uber", "Angi", "Klue", "Mews", "Skai",
                          "Envato", "ING", "BT", "GM", "BAT", "NAB", "CBA", "HP", "UPS", "IAC", "EXL", "WPP"}

# --- Register verdict phrases (Part 1, Section 6.1) ---
VERDICT_RE = (r"(?:does|did|do|has|have|had) not (?:name|mention|cite|attribute|give|link)\w*[^.;:]{0,40}?"
              r"\b(?:ai|artificial intelligence|automation)\b(?: as (?:the|a) (?:cause|reason))?"
              r"|never (?:named|mentioned|cited) (?:ai|artificial intelligence)|(?:gave|gives|named|cited) no ai reason"
              r"|no ai reason|makes no mention of (?:ai|artificial intelligence)|no mention of (?:ai|artificial intelligence)"
              r"|did not attribute[^.;:]{0,60}?\bai\b")

# --- Engineered text signals (Part 1, Section 7.1) ---
AI_TERMS_RE = (r"\b(?:ai|a\.i\.|artificial intelligence|automation|automate[ds]?|automating|gen ?ai|generative|"
               r"llms?|large language models?|machine learning|chatbots?|agentic|ai agents?|copilot|chatgpt)\b")
BUSINESS_RE = (r"\b(?:costs?|cost-cutting|restructur\w*|efficien\w*|reorgani[sz]\w*|simplif\w*|margins?|"
               r"profitab\w*|economic|macro\w*|streamlin\w*|realign\w*|savings|budgets?|demand|"
               r"tariffs?|downturn|headwinds|priorit\w*|consolidat\w*)\b")
THIRD_PARTY_RE = (r"\b(?:according to|reported|reports|union|unions|sources?|people familiar|"
                  r"familiar with|analysts?|leaked|internal memo|told employees|works council)\b")
AI_NEGATION_RE = (r"(?:not (?:because of|due to|related to|driven by|caused by|about|a result of|replaced by) "
                  r"(?:ai|a\.i\.|artificial intelligence))|(?:(?:ai|artificial intelligence) (?:was|is|were) not)|"
                  r"(?:unrelated to (?:ai|artificial intelligence))|(?:nothing to do with (?:ai|artificial intelligence))|"
                  r"(?:not ai[- ]related)|\bdenied\b|\bdenies\b|\bdisputed\b")
FILING_RE = (r"8-k|10-q|10-k|6-k|20-f|results|earnings|press release|annual report|investor|shareholder|"
             r"asx announcement|filing|transcript|quarterly|\bq[1-4]\b|\bfy ?\d|newsroom|blog")

FRACTION_WORDS = [("two-thirds", 66.7), ("two thirds", 66.7), ("half", 50.0), ("third", 33.3),
                  ("quarter", 25.0), ("fifth", 20.0), ("tenth", 10.0)]


# ---------------------------------------------------------------------------
# Helper functions (identical logic to Part 1)
# ---------------------------------------------------------------------------
def jf_col(family: str) -> str:
    """Column name for a job family, e.g. 'Software engineering and IT' -> 'jf_software_engineering_and_it'."""
    return "jf_" + re.sub(r"[^0-9a-z]+", "_", family.lower()).strip("_")


JF_COLS = [jf_col(f) for f in JOB_FAMILIES]


def primary_country(text: str) -> str:
    """First country mentioned, ignoring parenthetical notes and list separators."""
    head = re.split(r"\(", text)[0]
    head = re.split(r",|/|;| and ", head)[0]
    return head.strip().lower()


def sector_group(text: str) -> str:
    """Map a free-text sector to a sector group using the first matching keyword rule."""
    s = str(text).lower()
    for group, pattern in SECTOR_RULES:
        if re.search(pattern, s):
            return group
    return "other"


def company_aliases(name: str) -> list[str]:
    """All surface forms of a company name to mask, longest first."""
    if not name:
        return []
    aliases = {name, re.sub(r"\s*\(.*?\)", "", name).strip()}
    for inner in re.findall(r"\((.*?)\)", name):
        aliases.update(p.strip() for p in re.split(r"/|,|;| and ", inner))
    for a in list(aliases):
        aliases.update(p.strip() for p in re.split(r"\s*/\s*", a))
    for a in list(aliases):
        stripped = re.sub(NAME_SUFFIX_RE, "", a).strip()
        aliases.add(stripped)
        if stripped.startswith("The "):
            aliases.add(stripped[4:])
        words = stripped.split()
        if len(words) > 1 and re.fullmatch(r"[A-Z0-9&]{2,}", words[0]):
            aliases.add(words[0])
    aliases.update(MANUAL_ALIASES.get(name, []))
    return sorted({a for a in aliases if len(a) >= 2 and a not in GENERIC_ALIASES}, key=len, reverse=True)


def mask_company(text: str, aliases: list[str]) -> str:
    """Replace every alias (plus possessive 's) with the token 'companyname'."""
    if not text:
        return text
    for a in aliases:
        flags = 0 if (a in CASE_SENSITIVE_ALIASES or len(a) < 5) else re.IGNORECASE
        text = re.sub(r"(?<!\w)" + re.escape(a) + r"(?:['’]s)?(?!\w)", " companyname ", text, flags=flags)
    return text


def clean_text(text: str) -> str:
    """Lowercase, drop URLs, numbers -> 'num', strip punctuation, collapse whitespace."""
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\d+(?:[.,]\d+)*", " num ", t)
    t = re.sub(r"[^\w\s]|_", " ", t)
    return " ".join(t.split())


def parse_pct(text) -> float:
    """Convert a free-text workforce share into a percentage (float) or NaN."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == "":
        return np.nan
    t = str(text).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*%", t)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m:
        return float(m.group(1))
    for word, value in FRACTION_WORDS:
        if re.search(r"\b" + word, t):
            return value
    m = re.search(r"(\d[\d,]*) of (?:about |roughly |~)?(\d[\d,]*) (?:staff|employees|workers|people)", t)
    if m:
        return 100 * float(m.group(1).replace(",", "")) / float(m.group(2).replace(",", ""))
    return np.nan


def _count(pattern: str, s: pd.Series) -> pd.Series:
    return s.str.count(re.compile(pattern, flags=re.IGNORECASE))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def engineer_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build the model input columns from raw-style event records.

    Expected columns (snake_case, as after Part 1 Section 5.1):
        company, date, claim, source_title, sector, country, kind, execution,
        headcount, pct_of_workforce, job_families
    `job_families` is a ';'-separated string of family names (or empty).
    Returns a new DataFrame with every feature column used by Experiments 1-3, plus the
    text variants `text_clean` and `text_clean_noverdict`.
    """
    df = events.copy().reset_index(drop=True)
    for c in ["company", "claim", "source_title", "sector", "country", "kind", "execution",
              "pct_of_workforce", "job_families"]:
        if c not in df:
            df[c] = ""
    for c in ["claim", "source_title", "company", "sector", "country"]:
        df[c] = df[c].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["kind"] = df["kind"].fillna("cut").astype(str).str.strip().str.lower()
    df["execution"] = df["execution"].fillna("unknown").astype(str).str.strip().str.lower()

    # --- Country and sector harmonisation ---
    df["country_primary"] = df["country"].map(primary_country)
    df["multi_country"] = df["country"].str.lower().str.contains(MULTI_COUNTRY_RE, regex=True).astype(int)
    df["region"] = df["country_primary"].map(REGION_MAP).fillna("other")   # unseen country -> 'other'
    df["is_us"] = df["country_primary"].eq("united states").astype(int)
    df["sector_group"] = df["sector"].map(sector_group)

    # --- Headcount (log1p, outliers kept) and % of workforce ---
    df["headcount"] = pd.to_numeric(df.get("headcount"), errors="coerce")
    df["log_headcount"] = np.log1p(df["headcount"])
    df["has_headcount"] = df["headcount"].notna().astype(int)
    df["pct_workforce"] = df["pct_of_workforce"].map(parse_pct)
    df["has_pct_workforce"] = df["pct_workforce"].notna().astype(int)

    # --- Dates ---
    date = pd.to_datetime(df["date"])
    df["year"] = date.dt.year
    df["month"] = date.dt.month
    df["months_since_start"] = (df["year"] - DATA_START.year) * 12 + (df["month"] - DATA_START.month)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Job families (multi-hot) ---
    jf_lists = df["job_families"].fillna("").astype(str).str.split(";").map(
        lambda parts: [p.strip() for p in parts if p.strip() and not re.match(PLACEHOLDER_RE, p.strip(), re.I)])
    for fam, col in zip(JOB_FAMILIES, JF_COLS):
        df[col] = jf_lists.map(lambda lst, fam=fam: int(fam in lst))
    df["n_job_families"] = df[JF_COLS].sum(axis=1)

    # --- Text: company masking + cleaning ---
    aliases = df["company"].map(company_aliases)
    claim_masked = [mask_company(t, a) for t, a in zip(df["claim"], aliases)]
    source_masked = [mask_company(t, a) for t, a in zip(df["source_title"], aliases)]
    claim_clean = [clean_text(t) for t in claim_masked]
    source_clean = [clean_text(t) for t in source_masked]
    df["text_clean"] = [(c + " " + s).strip() for c, s in zip(claim_clean, source_clean)]
    df["claim_has_verdict_phrase"] = pd.Series(claim_masked).str.lower().str.contains(VERDICT_RE, regex=True).astype(int)
    claim_noverdict = [re.sub(VERDICT_RE, " ", t, flags=re.IGNORECASE) for t in claim_masked]
    df["text_clean_noverdict"] = [(clean_text(c) + " " + s).strip() for c, s in zip(claim_noverdict, source_clean)]

    # --- Engineered text statistics ---
    claim_l = df["claim"].str.lower()
    df["claim_word_count"] = df["claim"].str.split().str.len().fillna(0).astype(int)
    df["claim_number_count"] = df["claim"].str.count(r"\d[\d,.]*")
    df["claim_quote_count"] = df["claim"].str.count(r'["“][^"”]{3,}["”]')
    df["ai_term_count"] = _count(AI_TERMS_RE, claim_l)
    df["business_reason_count"] = _count(BUSINESS_RE, claim_l)
    df["third_party_count"] = _count(THIRD_PARTY_RE, claim_l)
    df["ai_negation_count"] = _count(AI_NEGATION_RE, claim_l)
    df["source_is_company"] = [int(any(re.match(re.escape(a) + r"(?:\W|$)", s) for a in al))
                               for s, al in zip(df["source_title"], aliases)]
    df["source_is_filing"] = df["source_title"].str.lower().str.contains(FILING_RE, regex=True).astype(int)
    return df


def raw_register_to_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw register columns to the snake_case names `engineer_features` expects."""
    df = raw.rename(columns={
        "Company": "company", "Date": "date", "Claim": "claim", "Source": "source_title",
        "Sector": "sector", "Country": "country", "Kind": "kind", "Execution": "execution",
        "% of workforce": "pct_of_workforce", "Job families": "job_families", "Event ID": "event_id",
    })
    rc = pd.to_numeric(raw["Roles counted"], errors="coerce")
    rn = pd.to_numeric(raw["Roles reported (not counted)"], errors="coerce")
    df["headcount"] = rc.fillna(rn)
    return df
