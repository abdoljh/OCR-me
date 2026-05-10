"""
Extract OCR pages 5-10 from the full raw output and the matching
ground-truth text, then write them to separate files for analysis.

Run once: python3 prep_inputs.py <ocr_raw.txt> <ground_truth.txt>
"""
import re
import sys
import os

HERE = os.path.dirname(__file__)

# ------------------------------------------------------------------
# Read full OCR text passed on stdin or as first arg
# ------------------------------------------------------------------
ocr_full_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ocr_output_raw.txt")
gt_full_path  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "Preface_1-10.txt")

with open(ocr_full_path, encoding="utf-8") as f:
    ocr_full = f.read()

with open(gt_full_path, encoding="utf-8") as f:
    gt_full = f.read()

# ------------------------------------------------------------------
# Extract OCR pages 5-10 using the === page header markers
# ------------------------------------------------------------------
page_blocks = re.split(r'={3,}[^=]+={3,}', ocr_full)
# page_blocks[0] is text before first marker (empty/junk)
# page_blocks[1] is page 1, page_blocks[2] is page 2, etc.
pages = page_blocks[1:]  # index 0 = page 1

ocr_5_to_10 = "\n\n".join(pages[4:10])  # pages[4] = page 5, pages[9] = page 10

# ------------------------------------------------------------------
# Extract matching section from ground truth
# Ground truth starts with title pages then flows into the preface.
# We want the preface body: starting from "مقدمة" section.
# The GT is one continuous text; strip title page / TOC noise
# (everything up to and including the last TOC entry line)
# and keep from "مقدمة\nنجدة" onwards.
# ------------------------------------------------------------------
# Find the start of the preface body in the ground truth
gt_marker = "مقدمة\nنجدة فتحى صفوة"
gt_start = gt_full.find(gt_marker)
if gt_start == -1:
    # Try without newline
    gt_start = gt_full.find("مقدمة")
    print(f"Warning: exact marker not found; starting from 'مقدمة' at {gt_start}")

gt_preface = gt_full[gt_start:]

# Write outputs
with open(os.path.join(HERE, "ocr_pages_5_10.txt"), "w", encoding="utf-8") as f:
    f.write(ocr_5_to_10)

with open(os.path.join(HERE, "ground_truth.txt"), "w", encoding="utf-8") as f:
    f.write(gt_preface)

print(f"OCR pages 5-10: {len(ocr_5_to_10)} chars, {len(ocr_5_to_10.split())} tokens")
print(f"Ground truth  : {len(gt_preface)} chars, {len(gt_preface.split())} tokens")
print("Files written: ocr_pages_5_10.txt, ground_truth.txt")
