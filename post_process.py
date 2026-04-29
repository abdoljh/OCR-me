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

# Block is noise if it contains only digits, spaces, bidi marks and basic Arabic punctuation
_NOISE_RE = re.compile(r'^[‎‏‪-‮⁦-⁩\d\s،؛؟]+$')

_HAS_DIGIT = re.compile(r'[\d٠-٩]')
# Blocks ≤ this many chars that contain a digit are page-number / header fragments
_PAGE_NOISE_MAX_CHARS = 20


class CleanResult(NamedTuple):
    body: str
    footnotes: list[str]


def _split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r'\n{2,}', text) if b.strip()]


def _join_lines(block: str) -> str:
    return block.replace('\n', ' ').strip()


def _is_footnote(block: str) -> bool:
    first_line = block.split('\n')[0].strip()
    return bool(_FOOTNOTE_RE.match(first_line))


def _is_page_noise(block: str) -> bool:
    clean = _BIDI.sub('', block).strip()
    if len(clean) <= 5:
        return True
    if _NOISE_RE.match(clean):
        return True
    if len(clean) <= _PAGE_NOISE_MAX_CHARS and _HAS_DIGIT.search(clean):
        return True
    return False


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_ENDERS


def _process_page(
    page_text: str,
    move_footnotes: bool,
    arabic_indic_numerals: bool,
) -> tuple[list[str], list[str]]:
    """Return (body_blocks, footnote_blocks) for one page."""
    body: list[str] = []
    notes: list[str] = []
    for block in _split_blocks(page_text):
        if _is_page_noise(block):
            continue
        joined = _join_lines(block)
        if arabic_indic_numerals:
            joined = joined.translate(_WESTERN_TO_INDIC)
        if move_footnotes and _is_footnote(block):
            notes.append(joined)
        else:
            body.append(joined)
    return body, notes


def clean_pages(
    pages: list[str],
    move_footnotes: bool = True,
    arabic_indic_numerals: bool = False,
) -> CleanResult:
    """Post-process raw OCR page texts into a clean document.

    Joins broken lines within paragraphs, merges cross-page sentence
    continuations (last block of page N with first block of page N+1 when
    no sentence terminator), optionally extracts footnotes, and converts
    Western→Arabic-Indic numerals on request.
    """
    pages_blocks: list[list[str]] = []
    all_footnotes: list[str] = []

    for page_text in pages:
        body, notes = _process_page(page_text, move_footnotes, arabic_indic_numerals)
        pages_blocks.append(body)
        all_footnotes.extend(notes)

    # Cross-page sentence continuation: merge last block of page N with
    # first block of page N+1 only when the last block has no sentence terminator
    for i in range(len(pages_blocks) - 1):
        if pages_blocks[i] and pages_blocks[i + 1]:
            if not _ends_sentence(pages_blocks[i][-1]):
                pages_blocks[i][-1] += ' ' + pages_blocks[i + 1][0]
                pages_blocks[i + 1] = pages_blocks[i + 1][1:]

    all_blocks = [block for page in pages_blocks for block in page]
    return CleanResult(body='\n\n'.join(all_blocks), footnotes=all_footnotes)
