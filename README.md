# Arabic PDF2OCR

A Streamlit web application that preprocesses scanned Arabic PDF books for OCR.
It automatically detects and strips running headers, footers, and footnote separators,
then exports clean colour-image ZIPs ready for any downstream OCR engine —
or runs OCR directly inside the app using **kraken** or **Claude Haiku**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ocr-me.streamlit.app/)

---

## What it does

Modern OCR engines achieve their best accuracy on clean page images that contain
only the main body text. Scanned Arabic books routinely carry:

- Running headers (book title, chapter name, page number at the top)
- Footnote separators (a horizontal rule partway down the page)
- Page-number footers (often in a different typeface or size)

This app removes all of these automatically. The output is a set of
colour PNG images — one per page — cropped to the body text region and
ready to feed into Apple Live Text, Adobe Acrobat OCR, Tesseract, or any
other tool of your choice.

---

## Modes

The sidebar **Mode** selector exposes four workflows, ordered by typical usage:

### Single Book

Upload one PDF. The app:

1. Renders every page at 300 DPI in grayscale and binarizes it with Otsu's
   method to build an ink-density profile.
2. Detects and strips running headers (narrow strip near the top with a large
   gap below) and footnote regions (horizontal rule detected by morphological
   opening + whitespace-isolation gate; or a narrow strip near the bottom).
3. Sets a `CropBox` on each page of a stripped PDF (lossless — no re-encoding
   of the source PDF text layer).
4. Re-renders the stripped pages at your chosen export DPI (default 400 DPI)
   and streams them directly into a split ZIP without writing intermediate
   PNG files to disk.
5. Optionally assembles all detected footnote regions into a labeled PDF and
   a separate footnote-images ZIP.

Three per-stage progress bars show exactly which page is being processed during
margin detection, rendering, and footnote extraction.

### Batch

Identical to Single Book but accepts any number of uploaded files and processes
them one after another. Each file produces its own independent ZIP(s) and
footers archive.

### Raw Export

Exports the original, un-stripped pages as colour images (no header/footer
removal). Useful when you want the full page for reference or when the
automatic detection is not needed.

### Visual

An interactive page-by-page wizard:

- Scrub through individual pages using a page-number input.
- Set **Top crop %** and **Bottom crop %** sliders manually, or let
  **Auto-detect** pre-fill them:
  - *Fast (page 1)*: scans ink density on page 1 and applies the same crop
    to all pages.
  - *Smart (per page)*: runs full morphological analysis (OpenCV) per page —
    detects header, footnote rule, and page-number footer independently.
- Optional per-page **OCR** via:
  - **Claude Haiku** (API) — near-perfect Arabic accuracy, handles Quranic
    Uthmanic script with full *tashkeel*. Requires `ANTHROPIC_API_KEY`.
  - **kraken** (offline) — fully offline, no API key needed, uses the
    OpenITI `apt-20221130` printed Arabic model.
- Download cropped images and/or OCR text per page or as a full ZIP.

---

## Features at a glance

| Feature | Detail |
|---------|--------|
| Header detection | Narrow/short ink run in the top 12% of the page with a gap ≥ 60% of the median inter-line gap |
| Footnote rule detection | Morphological OPEN with a horizontal kernel + whitespace-isolation gate; rejects calligraphic strokes that happen to pass morphology |
| Footer detection | Narrow/short ink run in the bottom 10% of the page when no rule is found |
| Export DPI | 150–600, step 50; default 400 DPI (optimal for Apple Live Text) |
| ZIP splitting | Configurable part size (50–1 000 MB, default 250 MB); streams directly into ZIP — no intermediate files |
| Footnotes PDF | Every footer region assembled into one labeled PDF (`Page N — file.pdf` caption) |
| Footnotes ZIP | Standalone per-page footer images at export DPI, optionally separate from the main pages ZIP |
| Memory safety | Download selector shows one file at a time — only the selected ZIP is held in RAM |
| OCR (Visual) | kraken 7 BLLA segmenter + CTC greedy decoder; or Claude Haiku via Anthropic API |
| Post-OCR corrections | `confusables.py` dictionary of common Arabic OCR confusions, opt-in per session |
| Config snapshot | Sidebar shows a JSON summary of the current settings for reproducibility |

---

## Technical pipeline

```
Input PDF
    │
    ▼  _render_gray()            300 DPI grayscale via PyMuPDF
    │  _binarize()               Otsu global threshold (THRESH_BINARY_INV | THRESH_OTSU)
    │
    ├─ _active_columns()         Column ink-density profile → strip scanner-margin noise
    ├─ _line_runs()              Row-wise smoothed ink density → text-line segments
    ├─ _detect_horizontal_rules() morphologyEx OPEN + whitespace isolation → footnote rule
    │
    ├─ header_strip?             First run ∈ top 12%, narrow/short, large gap below
    ├─ rule_y?                   Lowest valid horizontal rule below body top + 5%
    └─ footer_strip?             Last run ∈ bottom 10%, narrow/short, large gap above
             │
             ▼  strip_pdf()      Set CropBox (cropbox mode) or re-render (raster mode)
             │
             ▼  _stream_pages_to_zips()   PyMuPDF render → cv2.imencode PNG → ZipFile.writestr()
             │
             ▼  extract_footers_pdf()     Re-detect footer band → labeled composite PDF + images
```

### Key design choices

**Lossless CropBox mode** — `strip_pdf(mode="cropbox")` sets PyMuPDF's `CropBox`
on each page rather than re-encoding. The source PDF's text layer and image
quality are preserved; the crop is reversible.

**Streaming ZIP** — pages are rendered one at a time and written directly into
the ZIP via `ZipFile.writestr()`. No PNG files accumulate on disk. Peak disk
usage equals at most one ZIP part, not the entire output.

**Whitespace-isolation gate** — a horizontal morphological opening can match
calligraphic strokes as well as separator rules. The isolation gate rejects
any candidate row that is not flanked by at least `rule_isolation_px` (18 px)
of nearly ink-free rows on both sides, eliminating false positives.

**Single download selector** — Streamlit's `st.download_button` holds the
button data in the session-scoped download store for the lifetime of the
render. Having five 250 MB buttons simultaneously would load 1.25 GB into
RAM. A `st.selectbox` + one button ensures only the selected file is in
memory at any time.

---

## OCR engines (Visual mode)

### kraken 7.0.1 — offline

- Model: `apt-20221130.mlmodel` from the
  [OpenITI AOCP print models](https://github.com/OpenITI/AOCP_print_models)
  repository. Downloaded once on first startup, cached at
  `~/.kraken_models/apt-20221130.mlmodel`.
- Segmenter: BLLA (`blla.segment`) — the only segmenter available in kraken 5+.
- Decoder: CTC greedy (no beam search in kraken 7).
- Tunable parameters (all exposed in the sidebar):
  DPI, binarization threshold, text direction, autocast (fp16),
  line padding, bidi reordering, legacy polygon extractor, softmax temperature.

### Claude Haiku — API

- Model: `claude-haiku-4-5-20251001` via the Anthropic API.
- Prompt includes a sample of common Arabic ligatures and Quranic bracket
  notation to calibrate the model for printed Arabic book typography.
- Cost: approximately $0.004 per page at default settings.
- Requires `ANTHROPIC_API_KEY` set in Streamlit Cloud **Settings → Secrets**.

---

## Running locally

### Prerequisites

```bash
sudo apt install poppler-utils   # pdfinfo + pdftoppm for pdf2image
```

### Install Python dependencies

```bash
pip install -r requirements-dev.txt   # includes pytesseract for test_kraken.py
```

(`requirements.txt` is the production set; `requirements-dev.txt` adds
`pytesseract` for the accuracy benchmark script.)

### Start the app

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

### Optional: set the Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run streamlit_app.py
```

Or create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Deploying to Streamlit Community Cloud

> Full details including rollback instructions are in [DEPLOYMENT.md](DEPLOYMENT.md).

### Critical: Python version

Streamlit Cloud **ignores `runtime.txt` and `.python-version`**. You must
select Python 3.12 in the **Advanced settings** dialog when you first deploy
the app. This cannot be changed after deployment — delete and redeploy to
change it.

### Critical: the `lightning` shim

The `lightning` package (a transitive dependency of kraken) was quarantined
on PyPI on 30 April 2026. The `lightning-compat/` directory in this repo is
a minimal local package that:

- Declares itself to pip as `lightning==2.6.1` (satisfies kraken's version
  constraint).
- Delegates `lightning.fabric` to `lightning_fabric` (shipped inside
  `pytorch-lightning==2.6.1`, which was **not** quarantined).

`requirements.txt` references it as `lightning @ ./lightning-compat`. Do not
change this back to `lightning~=2.6.0` without first verifying the quarantine
has been lifted.

### Quick deployment checklist

- [ ] Python **3.12** selected in Advanced settings
- [ ] `lightning-compat/` directory present in the repo
- [ ] `packages.txt` contains `poppler-utils`
- [ ] `ANTHROPIC_API_KEY` added in Settings → Secrets (optional, for Claude OCR)
- [ ] Model URL reachable: `curl -I https://raw.githubusercontent.com/OpenITI/AOCP_print_models/refs/heads/main/transcription/apt-20221130.mlmodel`

---

## Repository layout

```
streamlit_app.py          Main Streamlit application
header_footer.py          Margin-detection & CropBox pipeline (Params, detect_margins, strip_pdf)
page_export.py            Page/footer image export utilities (export_pages_as_images, extract_footers_pdf)
image_extract.py          Photograph extraction pipeline (extract_images)
confusables.py            Post-OCR Arabic word-correction dictionary
packages.txt              apt packages for Streamlit Cloud (poppler-utils)
requirements.txt          Production Python dependencies
requirements-dev.txt      Adds pytesseract for local testing
lightning-compat/         Local pip shim for the quarantined lightning package
  pyproject.toml
  lightning/__init__.py
  lightning/fabric/__init__.py
DEPLOYMENT.md             Full deployment guide (pitfalls, shim docs, alternative platforms)
samples/                  Arabic PDF test files and ground-truth text
  arabic01.pdf … arabic05.pdf
  Preface.pdf, Preface_3_22.pdf
  Preface_1-10.txt          Ground truth for Preface.pdf pages 1–10
kraken_docs/
  KRAKEN_ARTICLE.md       Technical article: kraken architecture + Arabic evaluation
  input.jpg               Sample input image for the article
misc/                     Dev/research tools not used by the production app
  analyse_confusables.py  CLI: align OCR output vs ground truth, report character confusions
  test_kraken.py          CLI: compare kraken vs Tesseract accuracy on sample pages
  post_process.py         OCR text post-processing pipeline (future integration)
  prep_inputs.py          One-time script for ground-truth data extraction
  textcleaner             Bash script for text cleaning (unrelated to main pipeline)
  Sample_cropped.pdf      Sample cropped output for reference
```

---

## Accuracy benchmarking

```bash
python misc/test_kraken.py ~/.kraken_models/apt-20221130.mlmodel samples/arabic01.pdf
```

Compares kraken against Tesseract on the same pages and prints a CER/WER
table. Requires `requirements-dev.txt`.

```bash
python misc/analyse_confusables.py
```

Aligns raw kraken output against `ground_truth.txt`, produces a character-confusion
frequency table, and regenerates the correction dictionary used by `confusables.py`.

---

## Known constraints

| Constraint | Impact |
|-----------|--------|
| kraken `requires-python <3.14` | Cannot deploy on Python 3.14+ without patching the wheel |
| `lightning` PyPI quarantine | Resolved by the `lightning-compat/` shim in this repo |
| Greedy CTC decoding only | No beam search; character accuracy is model-limited |
| Cold-start model download | First startup takes 30–60 s on slow connections; subsequent starts use the cached model |
| Streamlit Cloud ephemeral FS | Model re-downloaded on each cold start; `@st.cache_resource` avoids repeat downloads per session |

---

## License

MIT — see `LICENSE` for details. The `apt-20221130.mlmodel` model is
distributed by the OpenITI project under its own terms.
