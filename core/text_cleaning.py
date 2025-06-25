from __future__ import annotations

"""Text cleaning utilities (Phase-16 FEATURE-03)

Pure functions that strip HTML/URLs, lower-case, and remove stop-words.
Fail-fast: any missing resources (e.g. NLTK stop-word corpus) will raise.
"""

import os
import re
from typing import FrozenSet, List, Set, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS

try:
    from nltk.corpus import stopwords as _nltk_stopwords
except LookupError as e:  # pragma: no cover – fail-fast in runtime code
    raise RuntimeError(
        "NLTK stop-word corpus not available. Run `python -m nltk.downloader stopwords` in the `timeline` env.") from e

# ---------------------------------------------------------------------------
# Stop-word handling
# ---------------------------------------------------------------------------

_DEFAULT_STOPWORDS: FrozenSet[str] = frozenset(SKLEARN_STOP_WORDS).union(
    frozenset(_nltk_stopwords.words("english"))
)

_EXTRA_STOPWORDS_FILE = "resources/stopwords_extra.txt"


def _load_extra_stopwords(path: str = _EXTRA_STOPWORDS_FILE) -> FrozenSet[str]:
    if not os.path.exists(path):
        return frozenset()
    with open(path, "r", encoding="utf-8") as f:
        words: List[str] = [w.strip().lower() for w in f if w.strip()]
    return frozenset(words)


STOP_WORDS: FrozenSet[str] = _DEFAULT_STOPWORDS.union(_load_extra_stopwords())

# ---------------------------------------------------------------------------
# Cleaning function
# ---------------------------------------------------------------------------

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def clean_text(text: str, stop_set: FrozenSet[str] | None = None) -> str:
    """Minimal text cleaning used before TF-IDF.

    Args:
        text: Raw text (abstract, keywords, etc.).
        stop_set: Set of stop-words to remove.  If *None*, uses the default union set.

    Returns:
        Whitespace-joined string of cleaned tokens.
    """
    if text is None:
        return ""

    stop_set = stop_set or STOP_WORDS

    # Remove HTML tags & URLs
    cleaned = _HTML_RE.sub(" ", text)
    cleaned = _URL_RE.sub(" ", cleaned)

    # Lowercase & tokenize (keep alphanumerics)
    tokens: List[str] = _TOKEN_RE.findall(cleaned.lower())

    # Stop-word removal
    filtered = [t for t in tokens if t not in stop_set]

    return " ".join(filtered) 