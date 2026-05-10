"""
kraken vs Tesseract accuracy comparison on Arabic PDF pages.

Usage:
    python3 test_kraken.py <model.mlmodel> [pdf_path]

The script:
  1. Renders pages 5-10 of samples/arabic01.pdf at 400 DPI
  2. Binarizes each page with kraken.binarization.nlbin (same as the app)
  3. Runs kraken OCR (blla segmentation + rpred recognition) using the supplied model
  4. Runs Tesseract OCR (tessdata_best, PSM 4) on the same images
  5. Compares both outputs against ground_truth.txt
  6. Prints a side-by-side accuracy report

Requirements (see requirements-dev.txt):
    pip install pytesseract
    apt install tesseract-ocr tesseract-ocr-ara
"""

import sys
import os
import re
import warnings
import io

import numpy as np
from PIL import Image, ImageOps
from pdf2image import convert_from_path

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python3 test_kraken.py <arabic_model.mlmodel> [pdf_path]")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
PDF_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "samples", "arabic01.pdf")
GT_PATH = os.path.join(HERE, "ground_truth.txt")
DPI = 400
FIRST_PAGE = 5
LAST_PAGE = 10

print(f"Model   : {MODEL_PATH}")
print(f"PDF     : {PDF_PATH}")
print(f"Pages   : {FIRST_PAGE}–{LAST_PAGE}")
print(f"DPI     : {DPI}")
print()


# ---------------------------------------------------------------------------
# Render pages
# ---------------------------------------------------------------------------
print(f"Rendering pages {FIRST_PAGE}–{LAST_PAGE} at {DPI} DPI…")
images = convert_from_path(PDF_PATH, dpi=DPI, first_page=FIRST_PAGE, last_page=LAST_PAGE)
print(f"  {len(images)} pages rendered.")
print()

# ---------------------------------------------------------------------------
# Tesseract OCR (tessdata_best, PSM 4, OEM 1)
# ---------------------------------------------------------------------------
try:
    import pytesseract
except ImportError:
    print("pytesseract not installed — skipping Tesseract comparison.")
    print("Install it with: pip install pytesseract")
    pytesseract = None

TESSDATA_CACHE = os.path.expanduser("~/.tessdata_custom")
TESS_CONFIG = f'--oem 1 --psm 4 --tessdata-dir "{TESSDATA_CACHE}"'
TESS_LANG = "ara"

if pytesseract and not os.path.exists(os.path.join(TESSDATA_CACHE, "ara.traineddata")):
    print("tessdata_best not found — falling back to system Tesseract ara model")
    TESS_CONFIG = "--oem 1 --psm 4"

tess_pages = []
if pytesseract:
    print("Running Tesseract OCR…")
    for i, img in enumerate(images, start=FIRST_PAGE):
        padded = ImageOps.expand(img, border=100, fill=(255, 255, 255))
        text = pytesseract.image_to_string(padded, lang=TESS_LANG, config=TESS_CONFIG)
        tess_pages.append(text)
        print(f"  Page {i}: {len(text)} chars")
    print()

tess_combined = "\n\n".join(tess_pages)

# ---------------------------------------------------------------------------
# kraken OCR  (kraken 7 API: blla.segment + rpred.rpred)
# ---------------------------------------------------------------------------
from kraken import blla, rpred as krpred, binarization as kraken_bin
from kraken.lib.models import load_any

print(f"Loading kraken model: {MODEL_PATH}")
model = load_any(MODEL_PATH)
print(f"  Model loaded.")
print()

print("Running kraken OCR…")
kraken_pages = []
for i, img in enumerate(images, start=FIRST_PAGE):
    bw = kraken_bin.nlbin(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = blla.segment(bw, text_direction="horizontal-rl")
        lines = [
            r.prediction
            for r in krpred.rpred(model, bw, seg)
            if r.prediction.strip()
        ]
    page_text = "\n".join(lines)
    kraken_pages.append(page_text)
    print(f"  Page {i}: {len(page_text)} chars, {len(lines)} lines")

kraken_combined = "\n\n".join(kraken_pages)
print()

# ---------------------------------------------------------------------------
# Write OCR outputs for manual inspection
# ---------------------------------------------------------------------------
tess_out = os.path.join(HERE, "test_kraken_tess_output.txt")
krak_out = os.path.join(HERE, "test_kraken_kraken_output.txt")
if tess_combined:
    with open(tess_out, "w", encoding="utf-8") as f:
        f.write(tess_combined)
    print(f"Tesseract output → {tess_out}")
with open(krak_out, "w", encoding="utf-8") as f:
    f.write(kraken_combined)
print(f"kraken output    → {krak_out}")
print()

# ---------------------------------------------------------------------------
# Compare both against ground truth
# ---------------------------------------------------------------------------
from analyse_confusables import analyse

if not os.path.exists(GT_PATH):
    print(f"Ground truth not found at {GT_PATH} — skipping accuracy comparison.")
    sys.exit(0)

with open(GT_PATH, encoding="utf-8") as f:
    gt_text = f.read()

if tess_combined:
    print("=" * 60)
    print("TESSERACT accuracy vs ground truth")
    print("=" * 60)
    tess_corr = analyse(tess_combined, gt_text)
    print()

print("=" * 60)
print("KRAKEN accuracy vs ground truth")
print("=" * 60)
krak_corr = analyse(kraken_combined, gt_text)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def word_accuracy(ocr_text, gt_text):
    """Exact Arabic word match rate against ground truth."""
    arabic_re = re.compile(r'[؀-ۿً-ٟـ]+')
    ocr_words = arabic_re.findall(ocr_text)
    gt_words  = arabic_re.findall(gt_text)
    if not gt_words:
        return 0.0
    import difflib
    m = difflib.SequenceMatcher(None, ocr_words, gt_words, autojunk=False)
    equal = sum(i2 - i1 for tag, i1, i2, j1, j2 in m.get_opcodes() if tag == 'equal')
    return equal / len(gt_words) * 100

krak_acc = word_accuracy(kraken_combined, gt_text)

print()
print("=" * 60)
print("SUMMARY (exact Arabic word match, pages 5–10)")
print("=" * 60)
if tess_combined:
    tess_acc = word_accuracy(tess_combined, gt_text)
    print(f"  Tesseract (tessdata_best, PSM 4) : {tess_acc:.1f}%")
    print(f"  kraken ({os.path.basename(MODEL_PATH):<25})  : {krak_acc:.1f}%")
    winner = "kraken" if krak_acc > tess_acc else "Tesseract"
    diff = abs(krak_acc - tess_acc)
    print(f"  Winner: {winner} (+{diff:.1f} pp)")
else:
    print(f"  kraken ({os.path.basename(MODEL_PATH)}) : {krak_acc:.1f}%")
