"""
Compare OCR output against human ground truth.
Outputs:
  1. Character confusion table (informational)
  2. Word-level correction dictionary (safe to apply)
Usage: python analyse_confusables.py
"""
import re
import unicodedata
from collections import Counter
import difflib

# Sentinel returned by edit_distance when words differ too much to align.
_EDIT_INFINITY = 99
# Skip the full DP if word lengths differ by more than this — fast reject.
_MAX_LENGTH_DIFF = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_bidi(text):
    return re.sub(r'[‎‏‪-‮⁦-⁩]', '', text)

def normalise(text):
    text = strip_bidi(text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def tokenise_arabic(text):
    """Extract Arabic word tokens (letters + tatweel, no punctuation)."""
    return re.findall(r'[؀-ۿً-ٟـ]+', text)


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    m, n = len(a), len(b)
    if abs(m - n) > _MAX_LENGTH_DIFF:
        return _EDIT_INFINITY
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n]


def char_substitutions(ocr_word, gt_word):
    """Levenshtein traceback → list of (ocr_char, gt_char) substitution pairs."""
    m, n = len(ocr_word), len(gt_word)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ocr_word[i-1] == gt_word[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    subs = []
    i, j = m, n
    while i > 0 and j > 0:
        if ocr_word[i-1] == gt_word[j-1]:
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            subs.append((ocr_word[i-1], gt_word[j-1]))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1
    return subs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(ocr_text, gt_text, max_edit=2, verbose=True):
    """Align OCR output against ground truth and report character confusions.

    Parameters
    ----------
    ocr_text, gt_text:
        Raw text strings to compare.
    max_edit:
        Maximum edit distance for a word pair to be counted as a substitution.
    verbose:
        Print the confusion table and correction dictionary to stdout.
        Set to False for programmatic/batch use.

    Returns
    -------
    dict[str, str]
        Word correction mapping for all pairs that appear ≥2 times.
    """
    ocr_words = tokenise_arabic(normalise(ocr_text))
    gt_words  = tokenise_arabic(normalise(gt_text))

    matcher = difflib.SequenceMatcher(None, ocr_words, gt_words, autojunk=False)

    confusion: Counter = Counter()
    word_corrections: Counter = Counter()   # (ocr_word, gt_word) -> count
    total_equal = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            total_equal += i2 - i1
        elif tag == 'replace' and (i2 - i1) == (j2 - j1):
            for ocr_w, gt_w in zip(ocr_words[i1:i2], gt_words[j1:j2]):
                if ocr_w == gt_w:
                    continue
                dist = edit_distance(ocr_w, gt_w)
                if dist <= max_edit:
                    word_corrections[(ocr_w, gt_w)] += 1
                    for pair in char_substitutions(ocr_w, gt_w):
                        if pair[0] != pair[1]:
                            confusion[pair] += 1

    total_words = len(ocr_words)
    multi  = {k: v for k, v in word_corrections.items() if v >= 2}
    single = {k: v for k, v in word_corrections.items() if v == 1}

    if verbose:
        print(f"OCR Arabic words total  : {total_words}")
        print(f"Identical to GT         : {total_equal}  ({100*total_equal/total_words:.1f}%)")
        print(f"Close replacements (≤{max_edit}ed): {sum(word_corrections.values())}")
        print()

        print("CHARACTER SUBSTITUTION TABLE (OCR → Ground Truth)")
        print(f"{'OCR':<12} → {'GT':<12} {'Count':>6}")
        print("-" * 40)
        for (oc, gc), cnt in confusion.most_common(30):
            try: on = unicodedata.name(oc)
            except: on = '?'
            try: gn = unicodedata.name(gc)
            except: gn = '?'
            print(f"  {repr(oc):<8} → {repr(gc):<8} {cnt:>5}   {on} → {gn}")

        print()
        print("WORD CORRECTION DICTIONARY")
        print(f"  (OCR form → correct form, edit distance ≤ {max_edit}, sorted by frequency)")
        print()
        print(f"  Multi-occurrence pairs ({len(multi)}):")
        for (ow, gw), cnt in sorted(multi.items(), key=lambda x: -x[1]):
            print(f"    {repr(ow):<25} → {repr(gw):<25} ×{cnt}")
        print()
        print(f"  Single-occurrence pairs ({len(single)}) — sample:")
        for (ow, gw), _ in list(sorted(single.items(), key=lambda x: x[0]))[:30]:
            print(f"    {repr(ow):<25} → {repr(gw):<25}")

    return {ow: gw for (ow, gw) in multi}


if __name__ == "__main__":
    import sys, os
    HERE = os.path.dirname(__file__)

    with open(os.path.join(HERE, "ocr_pages_5_10.txt"), encoding="utf-8") as f:
        ocr_text = f.read()
    with open(os.path.join(HERE, "ground_truth.txt"), encoding="utf-8") as f:
        gt_text = f.read()

    corr = analyse(ocr_text, gt_text, max_edit=2)

    print()
    print("=" * 60)
    print("PROPOSED confusables.py WORD CORRECTION DICT")
    print("=" * 60)
    print("WORD_CORRECTIONS = {")
    for ow, gw in sorted(corr.items()):
        print(f'    {repr(ow)}: {repr(gw)},')
    print("}")
