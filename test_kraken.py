"""
Kraken vs Tesseract accuracy comparison on Arabic PDF pages.

Usage:
    python3 test_kraken.py <model.mlmodel> [pages 5-10 of Preface.pdf]

The script:
  1. Renders pages 5-10 of samples/arabic01.pdf at 400 DPI
  2. Runs kraken OCR using the supplied model
  3. Runs Tesseract OCR (tessdata_best) on the same images
  4. Compares both outputs against ground_truth.txt using analyse_confusables.py
  5. Prints a side-by-side accuracy report

Requirements: kraken, pytesseract, pdf2image, PIL
"""

import sys
import os
import re

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
from pdf2image import convert_from_path
print(f"Rendering pages {FIRST_PAGE}–{LAST_PAGE} at {DPI} DPI…")
images = convert_from_path(PDF_PATH, dpi=DPI, first_page=FIRST_PAGE, last_page=LAST_PAGE)
print(f"  {len(images)} pages rendered.")
print()

# ---------------------------------------------------------------------------
# Tesseract OCR (tessdata_best, PSM 4, OEM 1)
# ---------------------------------------------------------------------------
import pytesseract
from PIL import ImageOps

TESSDATA_CACHE = os.path.expanduser("~/.tessdata_custom")
TESS_CONFIG = f'--oem 1 --psm 4 --tessdata-dir "{TESSDATA_CACHE}"'
TESS_LANG = "ara"

# Verify tessdata_best is available
if not os.path.exists(os.path.join(TESSDATA_CACHE, "ara.traineddata")):
    print("tessdata_best not found — falling back to system Tesseract ara model")
    TESS_CONFIG = "--oem 1 --psm 4"

print("Running Tesseract OCR…")
tess_pages = []
for i, img in enumerate(images, start=FIRST_PAGE):
    # Apply the same preprocessing the app uses (grayscale + white border)
    padded = ImageOps.expand(img, border=100, fill=(255, 255, 255))
    text = pytesseract.image_to_string(padded, lang=TESS_LANG, config=TESS_CONFIG)
    tess_pages.append(text)
    print(f"  Page {i}: {len(text)} chars")

tess_combined = "\n\n".join(tess_pages)
print()

# ---------------------------------------------------------------------------
# Kraken OCR
# ---------------------------------------------------------------------------
from kraken import binarization, pageseg, rpred
from kraken.lib import models as kraken_models

print(f"Loading kraken model: {MODEL_PATH}")
model = kraken_models.load_any(MODEL_PATH)
print(f"  Model loaded: {model}")
print()

print("Running Kraken OCR…")
kraken_pages = []
for i, img in enumerate(images, start=FIRST_PAGE):
    bw = binarization.nlbin(img)
    seg = pageseg.segment(bw)
    lines = []
    for record in rpred.rpred(model, bw, seg):
        if record.prediction.strip():
            lines.append(record.prediction)
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
with open(tess_out, "w", encoding="utf-8") as f:
    f.write(tess_combined)
with open(krak_out, "w", encoding="utf-8") as f:
    f.write(kraken_combined)
print(f"Tesseract output → {tess_out}")
print(f"Kraken output    → {krak_out}")
print()

# ---------------------------------------------------------------------------
# Compare both against ground truth
# ---------------------------------------------------------------------------
from analyse_confusables import analyse

with open(GT_PATH, encoding="utf-8") as f:
    gt_text = f.read()

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
    """Quick exact-match word accuracy for summary."""
    import difflib
    arabic_re = re.compile(r'[؀-ۿً-ٟـ]+')
    ocr_words = arabic_re.findall(ocr_text)
    gt_words  = arabic_re.findall(gt_text)
    if not gt_words:
        return 0.0
    m = difflib.SequenceMatcher(None, ocr_words, gt_words, autojunk=False)
    equal = sum(i2 - i1 for tag, i1, i2, j1, j2 in m.get_opcodes() if tag == 'equal')
    return equal / len(gt_words) * 100

tess_acc = word_accuracy(tess_combined, gt_text)
krak_acc = word_accuracy(kraken_combined, gt_text)

print()
print("=" * 60)
print("SUMMARY (exact Arabic word match, pages 5–10)")
print("=" * 60)
print(f"  Tesseract (tessdata_best, PSM 4) : {tess_acc:.1f}%")
print(f"  Kraken ({os.path.basename(MODEL_PATH):<25}) : {krak_acc:.1f}%")
winner = "Kraken" if krak_acc > tess_acc else "Tesseract"
diff = abs(krak_acc - tess_acc)
print(f"  Winner: {winner} (+{diff:.1f} pp)")
