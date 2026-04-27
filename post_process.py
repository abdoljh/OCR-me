import re
from typing import NamedTuple

_WESTERN_TO_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# Footnote marker: )1( or (1) or )١( with optional leading bidi/space
_FOOTNOTE_RE = re.compile(
    r'^[‎‏‪-‮⁦-⁩\s]*'
    r'[\(\)]\s*[\d٠-٩]+\s*[\(\)]'
)

_SENTENCE_ENDERS = frozenset('.؟!…۔')

_BIDI = re.compile(r'[‎‏‪-‮⁦-⁩]')

# Block is noise if it contains only digits, spaces, bidi marks and basic punctuation
_NOISE_RE = re.compile(r'^[‎‏‪-‮⁦-⁩\d\s،؛؟]+$')


class CleanResult(NamedTuple):
    body: str
    footnotes: list[str]


def _split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r'\n{2,}', text) if b.strip()]


def _join_lines(block: str) -> str:
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', block).strip()


def _is_footnote(block: str) -> bool:
    first_line = block.split('\n')[0].strip()
    return bool(_FOOTNOTE_RE.match(first_line))


def _is_page_noise(block: str) -> bool:
    clean = _BIDI.sub('', block).strip()
    if len(clean) <= 5:
        return True
    return bool(_NOISE_RE.match(clean))


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_ENDERS


def clean_pages(
    pages: list[str],
    move_footnotes: bool = True,
    arabic_indic_numerals: bool = False,
) -> CleanResult:
    """Post-process raw OCR page texts into a clean document.

    Joins broken lines, merges cross-page sentence continuations, optionally
    extracts footnotes to a separate list, and converts Western→Arabic-Indic
    numerals on request.
    """
    body_blocks: list[str] = []
    footnotes: list[str] = []

    for page_text in pages:
        for block in _split_blocks(page_text):
            if _is_page_noise(block):
                continue
            joined = _join_lines(block)
            if arabic_indic_numerals:
                joined = joined.translate(_WESTERN_TO_INDIC)
            if move_footnotes and _is_footnote(block):
                footnotes.append(joined)
            else:
                body_blocks.append(joined)

    # Merge cross-page sentence continuations
    merged: list[str] = []
    for block in body_blocks:
        if merged and not _ends_sentence(merged[-1]):
            merged[-1] = merged[-1] + ' ' + block
        else:
            merged.append(block)

    return CleanResult(body='\n\n'.join(merged), footnotes=footnotes)
