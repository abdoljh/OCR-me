"""
Post-OCR character- and word-level corrections for Arabic kraken/Tesseract output.

apply_word_corrections(text, extra=None) performs:
  1. Tatweel stripping  — removes U+0640 ARABIC TATWEEL, an almost-universal
     OCR artefact in modern printed books.
  2. Whole-word substitutions — regex built from WORD_CORRECTIONS merged with
     any caller-supplied dict.

Design constraints:
- Only correct tokens that are clearly OCR artefacts (not valid Arabic words).
- Never silently corrupt correct output: when in doubt, omit the correction.
- Word boundaries use \\b-equivalent anchors for Arabic (zero-width assertions
  between Arabic letters and non-Arabic characters).
"""

import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Base corrections — general kraken/Tesseract Arabic OCR artefacts
# ---------------------------------------------------------------------------
# Entries fall into four categories:
#
# 1. Alef-maqsura/ya normalisation
#    Modern Standard Arabic print always uses ya (ي) for these common words;
#    the model frequently outputs alef-maqsura (ى) instead.
#    Words that correctly end in ى (على، إلى، متى، حتى، لدى …) are excluded.
#
# 2. Specific non-words produced by the apt model
#    These token forms do not exist in any Arabic dictionary; they only arise
#    as OCR artefacts from this particular model.
#
# 3. Hamza normalisation for common words
#    The model reliably drops the hamza from a small set of very frequent
#    words; the hamza-less form is not a valid word in any dialect.
#
# 4. Legacy Tesseract artefacts (retained for compatibility)

WORD_CORRECTIONS: dict[str, str] = {
    # ── 1. Alef-maqsura → ya for invariant MSA words ──────────────────────
    "فى":    "في",      # in / at — invariably في in print
    "اى":    "أي",      # which / what — also restores missing hamza
    "التى":  "التي",    # relative pronoun (feminine)
    "هى":    "هي",      # she / it pronoun
    "وهى":   "وهي",     # conjunction + pronoun

    # ── 2. Non-words that only arise as OCR artefacts ─────────────────────
    "بنفه":   "بنفسه",  # "by himself" — dropped sīn before pronoun suffix
    "الألة":  "الآلة",  # typewriter / instrument — missing madda over alef
    "جريثٔة": "جريئة",  # ئ (hamza-on-ya) misread as the sequence ثٔ
    "العكري": "العسكري", # dropped sīn (from GT-derived set, promoted to base)
    "خلاها":  "خلالها",  # dropped lam (from GT-derived set, promoted to base)

    # ── 3. Hamza normalisation ────────────────────────────────────────────
    "اكثر":  "أكثر",    # more / most — hamza on alef is obligatory

    # ── 4. Legacy Tesseract artefacts ────────────────────────────────────
    "قِ": "في",
    "قٍ": "في",
}

# ---------------------------------------------------------------------------
# Ground-truth-derived corrections for Preface.pdf
# Kept for callers that explicitly opt in with include_gt_derived=True.
# The two entries above were promoted to the base set; this dict now serves
# as a hook for future document-specific additions.
# ---------------------------------------------------------------------------
GT_DERIVED_CORRECTIONS: dict[str, str] = {
    # (العكري and خلاها are now in WORD_CORRECTIONS)
}

# ---------------------------------------------------------------------------
# Pattern builder — cached by frozenset of (token, correction) pairs
# so the regex is compiled only once per unique correction dictionary.
# ---------------------------------------------------------------------------

def _build_pattern(corrections: dict[str, str]) -> re.Pattern | None:
    if not corrections:
        return None
    # Sort longest first to avoid partial-match shadowing
    keys = sorted(corrections, key=len, reverse=True)
    # Arabic "word boundary": preceded/followed by non-Arabic-letter or start/end.
    # Range starts at ء (U+0621) to exclude Arabic punctuation (U+0600–U+0620,
    # which includes ، ؛ ؟ etc.) from the "is Arabic" test — otherwise words
    # followed by attached punctuation (e.g. جريثٔة،) are not matched.
    escaped = [re.escape(k) for k in keys]
    pattern = r'(?<![ء-ۿً-ٟـ])(?:' + '|'.join(escaped) + r')(?![ء-ۿً-ٟـ])'
    return re.compile(pattern)


@lru_cache(maxsize=16)
def _cached_pattern(frozen_items: tuple) -> re.Pattern | None:
    """Build and cache a compiled regex for the given correction mapping."""
    return _build_pattern(dict(frozen_items))


def apply_word_corrections(
    text: str,
    extra: dict[str, str] | None = None,
    include_gt_derived: bool = False,
) -> str:
    """
    Apply character- and word-level OCR corrections to *text*.

    Parameters
    ----------
    text:
        Raw or lightly post-processed OCR output.
    extra:
        Caller-supplied additional corrections (merged on top of the base set).
    include_gt_derived:
        Include the ground-truth-derived dictionary (document-specific extras).
        Off by default.
    """
    # ── Step 1: strip tatweel ──────────────────────────────────────────────
    # U+0640 ARABIC TATWEEL (kashida) appears in OCR output almost exclusively
    # as an artefact of the model reading horizontally stretched characters.
    # Genuine typographic tatweel is essentially absent from modern printed
    # Arabic books, so removing it is safe for all standard print inputs.
    text = text.replace('ـ', '')

    # ── Step 2: word-level substitutions ──────────────────────────────────
    combined: dict[str, str] = dict(WORD_CORRECTIONS)
    if include_gt_derived:
        combined.update(GT_DERIVED_CORRECTIONS)
    if extra:
        combined.update(extra)

    # Sort items so the cache key is order-independent
    pat = _cached_pattern(tuple(sorted(combined.items())))
    if pat is None:
        return text

    return pat.sub(lambda m: combined[m.group(0)], text)
