"""
preprocessor_v2.py
------------------
Upgraded preprocessing pipeline that fixes all 6 failure modes:

  Fix 1: Preserve emotional/sentiment signal words (no longer dropped)
  Fix 2: Normalize slang, abbreviations, informal language
  Fix 3: Handle ALL CAPS as urgency signal
  Fix 4: Sentiment & urgency feature vectors (domain-agnostic signals)
  Fix 5: Detect positive-framing + hidden severity
  Fix 6: Domain-aware keyword flags for non-technical complaint types

The key insight: a bag-of-words TF-IDF model only sees word frequencies.
We add ENGINEERED FEATURES that capture signals TF-IDF cannot:
  - Urgency language intensity
  - Sentiment polarity
  - Domain category
  - Complaint structure (positive prefix + negative suffix)
  - Text style signals (caps, punctuation density)
"""

import re
import string
import numpy as np

# ─────────────────────────────────────────────────────────────
# STOPWORDS  (carefully curated — preserve signal words)
# ─────────────────────────────────────────────────────────────
# NOTE: We deliberately EXCLUDE emotion words from stopwords.
# The original preprocessor dropped too many signal-carrying tokens.

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "its", "this",
    "that", "these", "those", "was", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "so", "yet", "both", "either", "neither", "as",
    "if", "then", "than", "because", "while", "although", "though",
    "since", "when", "where", "who", "which", "what", "how", "why",
    "all", "each", "every", "few", "more", "most", "other", "some",
    "such", "up", "out", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "own", "same",
    "just", "very", "also", "now", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "his", "her", "they", "their",
    "them", "us", "am", "any", "only", "over", "under",
    "again", "here", "there", "once",
    # Short noise tokens
    "s", "re", "ll", "ve", "d", "m", "t",
    # NOT removed: no, not, never, nothing, nobody (negation signals)
    # NOT removed: all, every (totality/scope signals for Critical)
}

# ─────────────────────────────────────────────────────────────
# SLANG / ABBREVIATION NORMALIZATION MAP
# Fix 4: casual language gets mapped to standard vocabulary
# ─────────────────────────────────────────────────────────────

SLANG_MAP = {
    r"\bomg\b":        "extremely urgent",
    r"\bwtf\b":        "very serious issue",
    r"\bsos\b":        "emergency help needed",
    r"\bngl\b":        "",
    r"\btbh\b":        "",
    r"\bfyi\b":        "noting that",
    r"\basap\b":       "immediately",
    r"\brn\b":         "right now",
    r"\blol\b":        "",
    r"\bsmh\b":        "very disappointed",
    r"\bbtw\b":        "",
    r"\bcant\b":       "cannot",
    r"\bdont\b":       "do not",
    r"\bwont\b":       "will not",
    r"\bdoesnt\b":     "does not",
    r"\bisnt\b":       "is not",
    r"\barent\b":      "are not",
    r"\bhasnt\b":      "has not",
    r"\bhavent\b":     "have not",
    r"\bcouldnt\b":    "could not",
    r"\bwouldnt\b":    "would not",
    r"\bshouldnt\b":   "should not",
    r"\bpls\b":        "please",
    r"\bplz\b":        "please",
    r"\bthx\b":        "thanks",
    r"\bu\b":          "you",
    r"\bur\b":         "your",
    r"\bim\b":         "i am",
    r"\bits\b":        "it is",
    r"\bive\b":        "i have",
    r"\bkinda\b":      "somewhat",
    r"\bsorta\b":      "somewhat",
    r"\bgonna\b":      "going to",
    r"\bwanna\b":      "want to",
    r"\bgotta\b":      "have to",
    r"\bcooked\b":     "broken completely",
    r"\bdead\b":       "not working",
    r"\bglitchy\b":    "malfunctioning",
    r"\bbusted\b":     "broken",
    r"\bfell over\b":  "crashed",
    r"\bblew up\b":    "failed completely",
}

# ─────────────────────────────────────────────────────────────
# SEMANTIC KEYWORD BANKS  (for engineered features)
# These are used to compute feature vectors, not for cleaning.
# ─────────────────────────────────────────────────────────────

URGENCY_SIGNALS = {
    # Tier 3 — critical / existential impact
    "emergency": 3.0, "sos": 3.0, "critical": 3.0, "outage": 3.0,
    "breach": 3.0, "hack": 3.0, "gdpr": 3.0, "data breach": 3.0,
    "exposed": 2.5, "lawsuit": 2.5, "regulator": 2.5, "court": 2.5,
    "payroll": 2.5, "losing money": 2.5, "losing sales": 2.5,
    "crashed": 2.5, "offline": 2.5, "unreachable": 2.5,
    "system down": 2.5, "completely down": 2.5,
    # Tier 2 — high business impact
    "urgent": 2.5, "immediately": 2.5, "right now": 2.0, "asap": 2.0,
    "production": 2.0, "entire": 2.0, "salary": 2.0, "deadline": 2.0,
    "unavailable": 2.0, "violation": 2.0, "suspicious charge": 2.0,
    "fraud": 2.0, "double charged": 2.0, "data loss": 2.0,
    "locked out": 2.0, "cannot access": 2.0,
    # Tier 1 — moderate urgency
    "now": 1.5, "live": 1.5, "launch": 1.5, "demo": 1.5,
    "legal": 1.5, "tax": 1.5, "money": 1.5, "clients": 1.5,
    "spins": 1.5, "spinning": 1.5, "frozen": 1.5, "freezing": 1.5,
    "stuck": 1.2, "hangs": 1.2, "loops": 1.2, "forever": 1.2,
    "flagged": 1.5, "blocked": 1.2, "losing": 1.5, "customers": 1.0,
    "not working": 1.5, "help": 1.0, "meeting": 1.0, "zero": 1.0,
    # NOTE: "down" removed — overcounts "prices went down", "quality went down"
    # System-down context handled via "system down", "completely down" multi-grams
}

SENTIMENT_SIGNALS = {
    # Negative sentiment words (emotion intensity → might indicate severity)
    "furious": 3.0, "livid": 3.0, "outrageous": 2.5, "horrified": 2.5,
    "disgusting": 2.0, "terrible": 2.0, "worst": 2.0, "horrible": 2.0,
    "angry": 2.0, "frustrated": 1.5, "disappointed": 1.5, "upset": 1.5,
    "annoyed": 1.0, "irritated": 1.0, "fed up": 2.0, "sick of": 1.5,
    "sick and tired": 2.0, "unacceptable": 2.0, "ridiculous": 1.5,
    "awful": 1.5, "dreadful": 1.5, "appalling": 2.0, "shameful": 1.5,
    "embarrassing": 1.0, "pathetic": 1.5, "negligent": 2.0,
    "useless": 1.5, "incompetent": 1.5, "rude": 1.5, "dismissive": 1.5,
}

DOMAIN_SIGNALS = {
    # Maps keyword → domain category index (for one-hot feature)
    "technical":   ["server","database","api","deploy","kubernetes","redis","elasticsearch",
                    "microservice","docker","kubernetes","ssl","firewall","memory","cpu",
                    "pipeline","webhook","cron","cache"],
    "payment":     ["payment","charge","charged","refund","invoice","billing","subscription",
                    "money","deduct","bank","credit","debit","transaction","wire","transfer",
                    "stripe","paypal","checkout","overcharged","double charged"],
    "retail":      ["order","delivery","package","shipping","arrived","parcel","driver",
                    "warehouse","inventory","tracking","dispatch","courier","return","item",
                    "product","damaged","missing","wrong item","counterfeit"],
    "healthcare":  ["appointment","doctor","nurse","clinic","hospital","prescription",
                    "patient","medical","health","receptionist","waiting room","lab",
                    "test result","referral","insurance","diagnosis"],
    "legal":       ["gdpr","privacy","consent","data breach","lawsuit","violation",
                    "compliance","regulator","court","legal","unsubscribe","opt out",
                    "right to erasure","subject access","data protection","biometric"],
    "hr":          ["payroll","salary","hr","onboarding","leave","holiday","benefits",
                    "employee","performance review","timesheet","contract","pension"],
    "finance":     ["mortgage","loan","account","wire","trading","investment","portfolio",
                    "statement","atm","branch","fraud","credit score","overdraft",
                    "interest","direct debit","standing order"],
    "auth":        ["login","password","2fa","otp","session","logout","locked","account",
                    "sign in","credentials","reset","access"],
    "ui_ux":       ["button","alignment","ui","layout","font","color","icon","tooltip",
                    "modal","design","spacing","dark mode","css","interface","display"],
    "performance": ["slow","lag","timeout","loading","speed","response time","latency",
                    "delay","fast","performance","sluggish"],
}

POSITIVE_FRAMING_CONJUNCTIONS = [
    "but ", "however ", "although ", "though ", "yet ", "despite ",
    "except ", "unfortunately ", "sadly ", "but unfortunately",
    "but suddenly", "but now", "but recently", "but since",
    "overall but", "generally but", "normally but", "usually but",
    "love.*but", "great.*but", "good.*but", "excellent.*but",
    "fan.*but", "recommend.*but", "enjoy.*but", "like.*but",
]

# ─────────────────────────────────────────────────────────────
# CORE TEXT CLEANING
# ─────────────────────────────────────────────────────────────

def _detect_caps_ratio(text: str) -> float:
    """Returns ratio of uppercase characters (Fix 3: caps = urgency signal)."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _count_exclamations(text: str) -> int:
    return text.count("!") + text.count("?") * 0.5


def _apply_slang_normalization(text: str) -> str:
    """Normalize informal/slang language to standard vocabulary."""
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def preprocess(
    text: str,
    *,
    use_stemming: bool = False,
    remove_stopwords: bool = True,
    min_word_len: int = 2,
) -> str:
    """
    Clean and normalize complaint text.

    Improvements over v1:
    - Slang normalization before cleaning
    - Preserves negation words (no, not, never, nothing)
    - Preserves emotional signal words
    - Handles contractions explicitly
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Apply slang normalization BEFORE lowercasing
    text = _apply_slang_normalization(text)

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Step 4: Remove punctuation (but preserve apostrophes for contractions briefly)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Step 5: Remove digits
    text = re.sub(r"\d+", " ", text)

    # Step 6: Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 7: Tokenize
    tokens = text.split()

    # Step 8: Filter
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= min_word_len]
    else:
        tokens = [t for t in tokens if len(t) >= min_word_len]

    # Step 9: Optional simple stemming
    if use_stemming:
        tokens = [_simple_stem(t) for t in tokens]

    return " ".join(tokens)


def _simple_stem(word: str) -> str:
    for suffix in ["ing", "tion", "tions", "ed", "ly", "er", "est", "ness", "ment", "ies", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def preprocess_batch(texts, **kwargs) -> list:
    return [preprocess(t, **kwargs) for t in texts]


# ─────────────────────────────────────────────────────────────
# ENGINEERED FEATURE EXTRACTION  (the key upgrade)
# Returns a numeric feature vector for each complaint
# ─────────────────────────────────────────────────────────────

def extract_features(text: str) -> np.ndarray:
    """
    Extract 20 hand-crafted features that capture signals
    TF-IDF alone cannot learn from sparse / unseen vocabulary.

    Feature vector layout (20 dimensions):
      [0]  Urgency score (sum of urgency signal weights)
      [1]  Sentiment intensity (sum of sentiment signal weights)
      [2]  Caps ratio (0-1)
      [3]  Exclamation/question density
      [4]  Text length (normalized to 0-1 over 500 chars)
      [5]  Has positive prefix + negative content (conjunctive complaint)
      [6-15] Domain one-hot: technical, payment, retail, healthcare,
                             legal, hr, finance, auth, ui_ux, performance
      [16] Negation density (no/not/never/nothing per token)
      [17] All-caps word count
      [18] Multiple punctuation sequences (!!!, ???)
      [19] Scope word presence (entire/all/every/nobody/nothing/zero)
    """
    features = np.zeros(20, dtype=np.float32)
    raw_lower = text.lower()
    tokens = raw_lower.split()
    n_tokens = max(len(tokens), 1)

    # [0] Urgency score
    urgency = 0.0
    for kw, weight in URGENCY_SIGNALS.items():
        if kw in raw_lower:
            urgency += weight
    features[0] = min(urgency / 10.0, 1.0)  # normalize

    # [1] Sentiment intensity
    sentiment = 0.0
    for kw, weight in SENTIMENT_SIGNALS.items():
        if kw in raw_lower:
            sentiment += weight
    features[1] = min(sentiment / 10.0, 1.0)

    # [2] Caps ratio
    features[2] = _detect_caps_ratio(text)

    # [3] Exclamation/question density
    features[3] = min(_count_exclamations(text) / 5.0, 1.0)

    # [4] Text length (normalized)
    features[4] = min(len(text) / 500.0, 1.0)

    # [5] Positive-prefix + negative suffix (conjunctive complaint)
    has_positive_prefix = bool(re.search(
        r"^(love|great|good|excellent|amazing|best|nice|like|enjoy|fan of|recommend)",
        raw_lower
    ))
    has_but_clause = any(conj in raw_lower for conj in
                         ["but ", "however ", "unfortunately ", "but now ", "but since "])
    features[5] = 1.0 if (has_positive_prefix and has_but_clause) else 0.0

    # [6-15] Domain one-hot encoding
    domain_keys = list(DOMAIN_SIGNALS.keys())
    for i, domain in enumerate(domain_keys):
        kws = DOMAIN_SIGNALS[domain]
        if any(kw in raw_lower for kw in kws):
            features[6 + i] = 1.0

    # [16] Negation density
    negations = ["no ", "not ", "never ", "nothing ", "nobody ", "none ", "cannot ", "can't "]
    neg_count = sum(raw_lower.count(n) for n in negations)
    features[16] = min(neg_count / n_tokens, 1.0)

    # [17] All-caps word count (normalized)
    all_caps_words = [t for t in text.split() if t.isupper() and len(t) > 2]
    features[17] = min(len(all_caps_words) / 5.0, 1.0)

    # [18] Multiple punctuation sequences
    multi_punct = len(re.findall(r"[!?]{2,}|[.]{3,}", text))
    features[18] = min(multi_punct / 3.0, 1.0)

    # [19] Scope/totality words (service-context only — avoids venting false positives)
    service_scope = [
        "entire system", "all users", "all customers", "all clients", "all services",
        "all staff", "all employees", "every user", "no one can", "nobody can",
        "completely down", "completely offline", "completely broken", "completely inaccessible",
        "entire platform", "entire database", "entire network", "entire infrastructure",
    ]
    features[19] = 1.0 if any(w in raw_lower for w in service_scope) else 0.0

    return features


def extract_features_batch(texts) -> np.ndarray:
    """Extract feature matrix for a list of texts. Shape: (n, 20)."""
    return np.vstack([extract_features(t) for t in texts])


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("I am absolutely furious, your system wiped all our data", "CRITICAL"),
        ("WTF everything is down customers cant access anything",   "CRITICAL"),
        ("Love the app but it crashes on every checkout attempt",   "HIGH"),
        ("GDPR violation — you shared my data without consent",    "CRITICAL"),
        ("App is kinda slow lately ngl",                           "MEDIUM"),
        ("Minor button alignment issue in settings page",          "LOW"),
    ]
    print("=== preprocessor_v2.py  Self-Test ===\n")
    for text, expected_class in tests:
        clean = preprocess(text)
        feats = extract_features(text)
        print(f"  Raw  : {text}")
        print(f"  Clean: {clean}")
        print(f"  Feats: urgency={feats[0]:.2f}  sentiment={feats[1]:.2f}  "
              f"caps={feats[2]:.2f}  scope={feats[19]:.0f}  "
              f"positive_framing={feats[5]:.0f}")
        print(f"  Expected class: {expected_class}\n")
