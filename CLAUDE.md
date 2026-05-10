# CLAUDE.md — Project context for AI assistant sessions

## What this project is

A Streamlit app (`streamlit_app.py`) that performs OCR on Arabic PDF files using
**kraken 7.0.1** with the OpenITI `apt-20221130.mlmodel` Arabic printed-text model.
The app is deployed on **Streamlit Cloud** and hosted at the `abdoljh/ocr-me` GitHub
repository. The primary development branch is **`main`**.

---

## Repository layout

```
streamlit_app.py          Main Streamlit application
packages.txt              apt packages installed by Streamlit Cloud
requirements.txt          Python dependencies (production)
requirements-dev.txt      Extra deps for local testing (adds pytesseract)
lightning-compat/         Local pip package that shims the quarantined `lightning`
  pyproject.toml
  lightning/__init__.py
  lightning/fabric/__init__.py
confusables.py            Post-OCR Arabic word-correction dictionary + apply fn
DEPLOYMENT.md             Full deployment guide (Streamlit Cloud pitfalls, shim docs)
samples/                  Arabic PDF test files + ground-truth text
  arabic01.pdf … arabic05.pdf
  Preface.pdf, Preface_3_22.pdf
  Preface_1-10.txt          (ground truth for Preface.pdf pages 1–10)
kraken_docs/
  KRAKEN_ARTICLE.md       Technical article: kraken architecture + Arabic evaluation
  input.jpg               Sample input image for the article
misc/                     Dev/research tools not used by the production app
  analyse_confusables.py  CLI tool: align OCR output vs ground truth, report errors
  test_kraken.py          CLI: compare kraken vs Tesseract accuracy on sample pages
  post_process.py         OCR text post-processing pipeline (future integration)
  prep_inputs.py          One-time script: extract pages 5-10 from raw OCR output
  textcleaner             Bash script for text cleaning (unrelated to main pipeline)
  Sample_cropped.pdf      Sample cropped output for reference
```

---

## Critical deployment constraints

### Python version
- Streamlit Cloud **must** run **Python 3.12**.
- `kraken==7.0.1` declares `requires-python = "<3.14"`, but its transitive deps
  (coremltools, scikit-image) fail to build on 3.13+.
- `runtime.txt` (`python-3.12`) and `.python-version` (`3.12`) are present for
  documentation; Streamlit Cloud **ignores** both files.
- The Python version is set only once, in the **"Advanced settings"** dialog at
  first deployment. It **cannot** be changed after deploy — the app must be deleted
  and redeployed to change it.

### The `lightning` quarantine shim
- PyPI quarantined the `lightning` package on 2026-04-30. All versions of `lightning`
  are uninstallable via pip or uv.
- **Fix**: `lightning-compat/` is a local pip package that registers itself as
  `lightning==2.6.1` and proxies to `pytorch-lightning==2.6.1`, which ships
  `lightning_fabric` internally (the only sub-package kraken actually imports).
- `requirements.txt` references it as `lightning @ ./lightning-compat`.
- **Do not** change this back to `lightning~=2.6.0` — that will break deployment.
- See `DEPLOYMENT.md` for full rollback instructions and alternative platforms.

### System packages (`packages.txt`)
```
poppler-utils
```
`poppler-utils` provides `pdfinfo` (used for fast page counts) and `pdftoppm`
(used by pdf2image). No tesseract packages are needed for production — kraken
handles OCR.

---

## Key design decisions in `streamlit_app.py`

### Model loading
- `@st.cache_resource` — loaded once, shared across all sessions.
- Model URL: `https://raw.githubusercontent.com/OpenITI/AOCP_print_models/refs/heads/main/transcription/apt-20221130.mlmodel`
- Cached locally at `~/.kraken_models/apt-20221130.mlmodel`.

### Caching strategy
- `@st.cache_data` on all pure functions: `_get_page_count`, `_render_page`,
  `_binarize_page`, `_ocr_page`, and the four download builders.
- Download builders (`_build_txt`, `_build_pdf`, `_build_tiff`, `_build_zip`)
  accept **`tuple[bytes, ...]`** (not lists) — required because `@st.cache_data`
  hashes arguments and lists are not hashable.
- Call sites in `main()` must always pass `tuple(all_bw_bytes)` / `tuple(all_texts)`.

### Binarization (`_nlbin`)
- scipy port of kraken's non-linear binarization.
- Tunable: `threshold` (0–100 slider → divided by 100) and `dpi`.
- Returns PNG bytes (stored in `all_bw_bytes` and passed to all downstream functions).

### OCR (`_ocr_page`)
Signature:
```python
def _ocr_page(
    bw_bytes: bytes,
    text_direction: str = "horizontal-rl",
    autocast: bool = False,
    pad: int = 16,
    bidi_key: str = "auto",
    no_legacy_polygons: bool = False,
    temperature: float = 1.0,
) -> tuple[str, list[float]]:
```
Returns `(text, per_line_confidences)`.

Temperature scales logits before softmax but does **not** change greedy decoding
results — it only shifts confidence scores. The mutation `model.temperature = temperature`
is applied to the `@st.cache_resource` model object, so it persists across calls.

### Bidi handling
```python
_BIDI_OPTIONS = {
    "Auto — let kraken decide (True)": "auto",
    "Force RTL — override to right-to-left ('R')": "R",
    "Force LTR — override to left-to-right ('L')": "L",
    "Off — raw display order (False)": "off",
}
_BIDI_TO_RPRED = {"auto": True, "R": "R", "L": "L", "off": False}
_BIDI_SHORT    = {v: k.split(" —")[0] for k, v in _BIDI_OPTIONS.items()}
```
The config JSON snapshot uses `_BIDI_SHORT[bidi_key]` — not the raw selectbox label.

### Sidebar parameters exposed for evaluation
All of these are wired to `_ocr_page` and cached:
| Control | Default | Range / Options |
|---------|---------|-----------------|
| DPI | 300 | 150–600, step 50 |
| Binarization threshold | 50 | 1–99 |
| Text direction | `horizontal-rl` | `horizontal-rl`, `horizontal-lr`, `vertical-lr`, `vertical-rl` |
| Autocast (fp16) | False | checkbox |
| Line padding | 16 | 0–64, step 4 |
| Bidi reordering | Auto | selectbox (4 options above) |
| Force new polygon extractor | False | checkbox |
| Softmax temperature | 1.0 | 0.1–3.0, step 0.1 |

---

## Known kraken 7.0.1 constraints

- **No beam search**: only CTC greedy decoding is available.
- **`valid_norm`**: auto-set to `False` for BLLA segmentation; not user-tunable.
- **Temperature**: affects confidence scores only, not character predictions.
- **Segmentation model**: BLLA (`blla.segment`) is the only supported segmenter;
  legacy `pageseg` was removed in kraken 5.
- **API**: `blla.segment(img, text_direction=..., no_legacy_polygons=...)` then
  `rpred.rpred(model, img, seg, bidi_reorder=..., pad=..., autocast=...)`.

---

## Working with the ground-truth files

- `ground_truth.txt`: pages 5–10 of `samples/arabic01.pdf` (Arabic text).
- `samples/Preface_1-10.txt`: pages 1–10 of `samples/Preface.pdf`.
- `ocr_pages_5_10.txt`: raw kraken output for arabic01.pdf pages 5–10.
- `analyse_confusables.py`: run standalone to produce a character confusion table
  and word correction dictionary. Accepts `verbose=True/False`.
- `confusables.py`: apply corrections with `apply_word_corrections(text)`.
  Uses `@lru_cache` on the compiled regex; call `include_gt_derived=True` for
  Preface.pdf-specific corrections.

---

## Running locally

```bash
# Install system deps
sudo apt install poppler-utils

# Install Python deps
pip install -r requirements-dev.txt   # includes pytesseract for test_kraken.py

# Run the app
streamlit run streamlit_app.py

# Test kraken vs Tesseract accuracy
python test_kraken.py path/to/arabic.mlmodel [path/to/file.pdf]
```

The devcontainer (`.devcontainer/devcontainer.json`) runs `packages.txt` + `requirements.txt`
automatically and starts the Streamlit server on port 8501.

---

## Git workflow

- Production branch: **`main`** — all recent work has landed here directly.
- Feature branches follow `claude/<description>` naming (e.g. `claude/arabic-pdf-ocr-app-4Lfzy`).
- Large binary files (`.mlmodel`) are in `.gitignore`; the model is fetched at runtime.
- Commit messages use imperative mood and end with the Claude session URL.

---

## What was done in the last two sessions

1. Built the initial Streamlit OCR app (Tesseract → kraken).
2. Diagnosed and fixed Streamlit Cloud deployment failures:
   - Python 3.14 default → must select 3.12 in Advanced settings.
   - `lightning` PyPI quarantine → created `lightning-compat/` shim.
3. Analysed OCR errors on `arabic01.pdf` vs ground truth.
4. Exposed all tunable kraken parameters as sidebar controls.
5. Wrote `DEPLOYMENT.md` and `kraken/KRAKEN_ARTICLE.md`.
6. Ran a full code review and fixed every identified issue:
   - `@lru_cache` on regex compilation in `confusables.py`
   - `verbose` param and named constants in `analyse_confusables.py`
   - kraken 7 API rewrite in `test_kraken.py`
   - `tuple` call-site fixes for `@st.cache_data` builders in `streamlit_app.py`
   - `_BIDI_SHORT` lookup replacing fragile string split
   - `pillow` lowercase in `requirements.txt`
   - `requirements-dev.txt` created

---

## Possible next steps

- Accuracy benchmarking: run `test_kraken.py` across parameter combinations,
  build a results table in `results/`.
- UI polish: RTL rendering in the text area, per-page confidence chart.
- Multi-model support: allow uploading a custom `.mlmodel` file.
- Post-processing pipeline: wire `apply_word_corrections()` into the download
  path as an opt-in toggle.
- Preface.pdf evaluation: extend ground truth and `GT_DERIVED_CORRECTIONS`.
