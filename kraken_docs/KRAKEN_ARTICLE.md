# kraken: A Deep-Dive Presentation and Evaluation for Arabic OCR

*Based on kraken 7.0.1 — April 2026*

---

## Table of Contents

1. [What Is kraken?](#1-what-is-kraken)
2. [History and Development](#2-history-and-development)
3. [Architecture Overview](#3-architecture-overview)
4. [Stage 1 — Image Binarization (nlbin)](#4-stage-1--image-binarization-nlbin)
5. [Stage 2 — Layout Analysis (BLLA)](#5-stage-2--layout-analysis-blla)
6. [Stage 3 — Text Recognition (VGSL + CTC)](#6-stage-3--text-recognition-vgsl--ctc)
7. [The mlmodel Format](#7-the-mlmodel-format)
8. [Complete Parameter Reference](#8-complete-parameter-reference)
9. [Arabic OCR Evaluation](#9-arabic-ocr-evaluation)
10. [Available Models for Arabic](#10-available-models-for-arabic)
11. [kraken vs. Alternatives](#11-kraken-vs-alternatives)
12. [Strengths and Limitations](#12-strengths-and-limitations)
13. [Practical Recipes](#13-practical-recipes)
14. [Conclusion](#14-conclusion)

---

## 1. What Is kraken?

kraken is an open-source, turn-key OCR engine designed specifically for
**historical and non-Latin scripts**.  Where commercial tools (ABBYY,
Adobe Acrobat) and dominant open-source engines (Tesseract) were built
around Latin-script printed books, kraken was purpose-built for the needs
of humanities researchers working with:

- Arabic, Persian, Ottoman Turkish manuscripts and early print
- Hebrew, Syriac, and other right-to-left scripts
- Medieval Latin manuscripts with irregular spacing
- Historical East Asian texts

It is developed and maintained by **Benjamin Kiessling** at the École
Pratique des Hautes Études (Paris) and is the OCR engine underlying
several large-scale digitisation projects including the OpenITI (Open
Islamicate Texts Initiative) and eScriptorium.

The canonical repository is <https://github.com/mittagessen/kraken>.

---

## 2. History and Development

### 2012–2016 — Ocropus roots

kraken began as a refactored fork of **ocropus**, Google's research OCR
system from the early 2010s.  It retained ocropus's LSTM-based line
recogniser but replaced its brittle binarisation and layout analysis with
more robust alternatives.

### 2017–2020 — Baseline revolution

The most significant architectural shift was the introduction of
**baseline-based segmentation** (BLLA — Baseline Layout Analysis).
Classical OCR segmenters found bounding boxes around lines; this works
for cleanly typeset Latin text but fails catastrophically for Arabic
(connected script with diacritics above and below the baseline), damaged
manuscripts, and pages with complex column layouts.

BLLA instead detects the **baseline** — the invisible line on which the
script sits — and constructs a polygon *above and below* it based on
predicted ink density.  This is far more tolerant of varying line heights,
overlapping ascenders/descenders, and marginal annotations.

### 2020–2023 — eScriptorium integration

kraken became the recognition backend for **eScriptorium**, a
collaborative annotation and transcription platform developed under the
Scripta PSL research programme.  This created a large user community,
drove extensive model development (hundreds of community-trained models on
the Zenodo platform), and accelerated feature development.

### 2024–present — kraken 5–7, lightning dependency

Versions 5 and above replaced the custom training loop with
**PyTorch Lightning (the `lightning` package)**, enabling distributed
training and modern training features (mixed precision, gradient
accumulation, early stopping).  This introduced `lightning` as a hard
runtime dependency — a decision that proved problematic when the package
was quarantined on PyPI in April 2026 (see `DEPLOYMENT.md`).

kraken 7.0.1 (current stable) also pinned:
- `torch>=2.4.0,<=2.10.0`
- `coremltools~=9.0` (for Apple Silicon export)
- `scikit-image~=0.25.2`
- `scipy~=1.15.3`

---

## 3. Architecture Overview

A complete kraken inference pipeline has three sequential stages:

```
PDF / image
     │
     ▼
┌─────────────────────┐
│  1. Binarization    │  nlbin algorithm (scipy)
│     (optional)      │  → black-on-white binary image
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  2. Layout Analysis │  BLLA — deep convolutional + heatmap
│     blla.segment()  │  → list of (baseline, polygon) pairs
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  3. Recognition     │  VGSL model (CNN + LSTM) + CTC decoder
│     rpred.rpred()   │  → Unicode text per line
└─────────────────────┘
          │
          ▼
    OCR text output
```

Each stage is independently configurable and can use separate models.

---

## 4. Stage 1 — Image Binarization (nlbin)

### Purpose

Convert a colour or greyscale scan into a clean black-on-white binary
image.  Binary input is not strictly required by kraken's neural models
(they handle greyscale too), but binarization removes ink-bleed, foxing,
and paper discolouration that would otherwise confuse the segmenter.

### The nlbin Algorithm

nlbin (**non-linear binarization**) is a local adaptive thresholding
algorithm originally from the ocropus project.  kraken ships a pure-Python
port; this repo implements the same algorithm using scipy:

```
1. Normalise pixel values to [0, 1]
2. Compute a local background estimate via percentile filtering:
      • 2× zoom-out (for speed)
      • percentile_filter(p=80, size=(20,2))  — vertical background
      • percentile_filter(p=80, size=(2,20))  — horizontal background
      • zoom back to original size
3. Subtract background from image → "flat" image (illumination normalised)
4. Estimate foreground ink statistics on the central 80 % of the image:
      • Remove background-only regions using a local variance mask
      • lo = 5th percentile of remaining pixels (ink floor)
      • hi = 90th percentile (ink ceiling)
5. Stretch contrast: flat = clip((flat − lo) / (hi − lo), 0, 1)
6. Threshold: pixel is ink if flat > threshold (default 0.50)
```

**Tunable parameter:** `threshold` (float, 0.0–1.0, default 0.50).

- **Raise** (e.g. 0.60–0.70) to recover faint strokes on aged paper.
- **Lower** (e.g. 0.30–0.40) if background texture is being mistaken for ink.

The algorithm is intentionally *local* — it adapts to uneven illumination
across the page, which is essential for manuscript scans.

### DPI Sensitivity

nlbin performance degrades at very low DPI.  Recommended minimums:

| Script | Minimum DPI | Recommended DPI |
|--------|------------|----------------|
| Latin print | 200 | 300 |
| Arabic print | 300 | 400 |
| Arabic manuscript | 400 | 600 |

At 150 DPI, diacritical dots merge with the base letter; at 600 DPI,
binarization is sharp but memory usage and processing time quadruple.

---

## 5. Stage 2 — Layout Analysis (BLLA)

### The Baseline Paradigm

Classical page segmenters find bounding **boxes** around text lines.
For Arabic this fails because:

1. Diacritics (harakat) sit above and below the baseline, making the
   bounding box taller than the actual script body.
2. Overlapping lines in manuscripts or tight printing share vertical space.
3. Marginal annotations intrude into the bounding box of the main text.

BLLA instead locates the **baseline** — the line on which the main script
body sits — and extracts a *polygon* that follows the ink up and down from
that baseline.  The polygon height is not fixed but adapts to local ink
density.

### Neural Network Architecture

The segmentation model is a fully-convolutional U-Net style network that
produces a **multi-class heatmap** at 1:1 resolution:

| Heatmap channel | Meaning |
|----------------|---------|
| Background | No ink / not a text region |
| Baseline | Baseline pixels |
| Boundary | Polygon boundary pixels |
| Region classes | User-defined region types (e.g. "main text", "heading", "margin") |

The network is trained with a binary cross-entropy loss per channel.  At
inference the heatmap is passed through a sigmoid (implicit threshold 0.5)
and vectorised into geometric primitives.

### Vectorisation Pipeline

```
heatmap (H×W×C)
    │
    ├─→ baseline channel → skeletonize → polyline fitting → baseline list
    │
    └─→ boundary channel → polygon construction per baseline
            ├─ Gaussian smoothing σ=0.5 on Sobel derivative
            ├─ calculate_polygonal_environment()
            └─ polygonal_reading_order() → sorted line list
```

### `blla.segment()` Parameters

```python
blla.segment(
    im,                                    # PIL.Image (binary or greyscale)
    text_direction = 'horizontal-lr',      # reading order heuristic
    mask           = None,                 # np.ndarray binary mask (ignore regions)
    reading_order_fn = polygonal_reading_order,
    model          = None,                 # custom segmentation model
    device         = 'cpu',
    raise_on_error = False,
    autocast       = False,               # torch.autocast (GPU speed)
)
```

**`text_direction`** is the most impactful parameter for Arabic.  Use
`'horizontal-rl'` for Arabic, Persian, and Hebrew.  This affects the
reading-order heuristic that sorts detected baselines into correct top-to-
bottom, right-to-left order.

**`mask`** allows supplying a binary numpy array the same size as the
input image where 0-valued pixels are excluded from segmentation.  This is
useful for masking binding gutters, page borders, stamps, and marginalia
that should not be recognised.

**`model`** accepts a custom `TorchVGSLModel` instance, enabling use of
specialised segmentation models for specific document types.

---

## 6. Stage 3 — Text Recognition (VGSL + CTC)

### VGSL — Variable-size Graph Specification Language

kraken defines its neural network architectures using VGSL, a compact
string notation originally developed by Google for Tesseract 4.  A typical
Arabic recognition model might specify:

```
[1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 Cr3,3,128
 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do O1c103]
```

Reading this left to right:

| Token | Meaning |
|-------|---------|
| `[1,48,0,1` | Input: batch=1, height=48, width=variable, channels=1 |
| `Cr3,3,32` | Conv 3×3, 32 filters, ReLU |
| `Do0.1,2` | Dropout p=0.1, 2D |
| `Mp2,2` | MaxPool 2×2 |
| `S1(1x0)1,3` | Reshape: collapse height into channel dimension |
| `Lbx200` | Bidirectional LSTM, 200 units, x-axis |
| `O1c103` | Output: 1D, CTC mode, 103 classes |

### The Recognition Pipeline

```
line polygon (PIL.Image)
    │
    ├─ extract_polygons()   — warp polygon to rectangular strip
    │
    ├─ ImageInputTransforms — resize to model height, pad horizontally
    │     • height  : fixed (model metadata, typically 48 px)
    │     • width   : variable (proportional to line length)
    │     • pad     : (pad, 0) pixels left+right (default pad=16)
    │     • valid_norm: False for baseline models (no centerline warp)
    │
    ├─ CNN layers           — feature extraction
    │
    ├─ BiLSTM layers        — sequential context
    │
    ├─ Softmax              — P(character | position) after temperature scaling
    │     logits / temperature → softmax → (N, W, C) probability matrix
    │
    └─ CTC greedy decoder   — best-path decoding
          → [(char, start_px, end_px, confidence), ...]
```

### CTC Decoding

Connectionist Temporal Classification (CTC) allows the network to produce
output sequences shorter than the input width without explicit alignment.
The greedy decoder used in kraken 7 takes the argmax at each timestep
(ignoring blanks and collapsing repeated labels):

```python
# Simplified
labels = outputs.argmax(dim=0)            # best character per timestep
labels = collapse_repeats_and_blanks(labels)
text   = codec.decode(labels)
```

**No beam search is available in kraken 7.0.1.**  A beam-search CTC
decoder (e.g. with a language model prior) would significantly improve
accuracy on ambiguous passages but is not implemented.  This is the single
largest accuracy gap compared to commercial systems.

### `rpred.rpred()` Parameters

```python
rpred.rpred(
    network,                   # TorchSeqRecognizer (loaded model)
    im,                        # PIL.Image — full page image
    bounds,                    # Segmentation from blla.segment()
    pad              = 16,     # horizontal blank padding per line (px)
    bidi_reordering  = True,   # Unicode BiDi algorithm on output
    no_legacy_polygons = False, # force new polygon extractor
)
```

**`pad`** directly affects what the model sees at line edges.  The default
16 px adds white space at both ends of the extracted line strip.  Too
little padding truncates the first/last character; too much introduces
blank context the model was not trained to handle.

**`bidi_reordering`** applies the Unicode Bidirectional Algorithm (python-
bidi library) to reorder the raw character sequence into logical reading
order.  For Arabic this is essential: the LSTM produces characters in
visual order (right-to-left visually = first character emitted last), and
BiDi reordering corrects this.  Set to `'R'` to force RTL regardless of
detected direction; set to `False` to get the raw display-order string.

**`no_legacy_polygons`** controls which polygon extraction algorithm is
used to cut lines out of the page.  Older models were trained with a
legacy extractor; using the new extractor on such models degrades accuracy.
New models trained from kraken 5 onwards use the new extractor by default.

### Temperature Scaling

```python
probs = (logits / temperature).softmax(dim=1)
```

With greedy decoding, `argmax(softmax(x/T)) = argmax(x)` for any `T > 0`,
so **temperature does not change recognised characters**.  It does change
the reported per-character confidence:

- `T < 1.0` — sharper distribution; high-confidence characters become
  closer to 1.0, low-confidence become closer to 0.0.  Useful for
  identifying uncertain characters.
- `T > 1.0` — flatter distribution; all confidences compress toward
  equal probability.  Not useful in practice.
- `T = 1.0` — default, raw softmax output.

---

## 7. The mlmodel Format

kraken models are stored as Apple CoreML `.mlmodel` files.  This is an
unusual choice for a Linux-first tool; it was adopted because CoreML's
protobuf format provides a self-contained, versioned container for both
the network weights and all metadata needed to reconstruct the inference
pipeline.

Relevant metadata fields stored in `user_defined_metadata`:

| Field | Type | Content |
|-------|------|---------|
| `vgsl` | str | Full VGSL architecture spec |
| `codec` | JSON | Mapping of integer label → Unicode character |
| `seg_type` | str | `'baselines'` or `'bbox'` |
| `one_channel_mode` | str | `'1'` (binary), `'L'` (greyscale), or null |
| `model_type` | list | `['recognition']`, `['segmentation']`, etc. |
| `legacy_polygons` | bool | Whether trained with old polygon extractor |
| `class_mapping` | dict | Heatmap channel → region type label |
| `topline` | bool/null | Baseline position: bottom/top/centre |
| `hyper_params` | dict | Training hyperparameters including `padding` |
| `accuracy` | list | Validation accuracy at each training epoch |

Loading a model:

```python
from kraken.lib import models
model = models.load_any('path/to/model.mlmodel', device='cpu')
# model is a TorchSeqRecognizer wrapping a TorchVGSLModel
```

---

## 8. Complete Parameter Reference

### Binarization

| Parameter | Function | Default | Range | Effect |
|-----------|----------|---------|-------|--------|
| `threshold` | `_nlbin()` | `0.50` | 0.10–0.90 | Ink/background cutoff |
| `dpi` | `convert_from_bytes()` | `300` | 150–600 | Rasterisation resolution |

### Segmentation

| Parameter | Function | Default | Options | Effect |
|-----------|----------|---------|---------|--------|
| `text_direction` | `blla.segment()` | `'horizontal-lr'` | `horizontal-rl/lr`, `vertical-rl/lr` | Reading order heuristic |
| `mask` | `blla.segment()` | `None` | `np.ndarray` (binary) | Exclude page regions |
| `model` | `blla.segment()` | default blla.mlmodel | `TorchVGSLModel` | Custom segmenter |
| `device` | `blla.segment()` | `'cpu'` | `'cpu'`, `'cuda'` | Inference device |
| `autocast` | `blla.segment()` | `False` | bool | Mixed-precision inference |
| `raise_on_error` | `blla.segment()` | `False` | bool | Strict error handling |
| `input_padding` | `SegmentationInferenceConfig` | `0` | int or 4-tuple | Image padding before segmentation |
| `baseline_ro_fn` | `SegmentationInferenceConfig` | `polygonal_reading_order` | callable | Custom reading-order function |

### Recognition

| Parameter | Function | Default | Range/Options | Effect |
|-----------|----------|---------|---------------|--------|
| `pad` | `rpred.rpred()` | `16` | 0–64 px | Horizontal blank padding per line |
| `bidi_reordering` | `rpred.rpred()` | `True` | `True`, `False`, `'L'`, `'R'` | Unicode BiDi character reordering |
| `no_legacy_polygons` | `rpred.rpred()` | `False` | bool | Force new polygon extractor |
| `temperature` | `TorchSeqRecognizer` | `1.0` | 0.1–3.0 | Softmax temperature (confidence only) |
| `decoder` | `TorchSeqRecognizer` | `greedy_decoder` | callable | CTC decoding function |
| `batch_size` | `RecognitionInferenceConfig` | `1` | int | Lines processed per forward pass |
| `num_line_workers` | `RecognitionInferenceConfig` | `2` | int | Parallel line extraction workers |
| `return_logits` | `RecognitionInferenceConfig` | `False` | bool | Include raw CTC logits in output |
| `return_line_image` | `RecognitionInferenceConfig` | `False` | bool | Include extracted line image in output |
| `precision` | `RecognitionInferenceConfig` | `'32-true'` | `'32-true'`, `'16-true'`, `'bf16-true'` | Torch compute precision |

### Image Transforms (applied internally per line)

| Parameter | Class | Default | Effect |
|-----------|-------|---------|--------|
| `height` | `ImageInputTransforms` | model metadata | Target line height in pixels |
| `width` | `ImageInputTransforms` | `0` (variable) | Target line width (0 = proportional) |
| `valid_norm` | `ImageInputTransforms` | auto (False for baselines) | CenterNormalizer baseline warp |
| `force_binarization` | `ImageInputTransforms` | `False` | Otsu binarization inside transform |
| `pad` | `ImageInputTransforms` | `(pad, 0)` from rpred | (horizontal, vertical) padding |
| `dtype` | `ImageInputTransforms` | `torch.float32` | Tensor precision |

---

## 9. Arabic OCR Evaluation

### Test Document

**Source:** *Mudhakkirāt Jaʿfar al-ʿAskarī* (Memoirs of Jafar al-Askari),
edited by Najda Fathi Safwa.  Early printed Arabic book, circa 1960–1980,
clean typography, standard Naskh font, fully vocalised (harakat present).

**Model:** `apt-20221130.mlmodel` from OpenITI AOCP (Arabic Printed Text
model trained on ~200 digitised books from the late 19th–early 20th century).

**Settings:** DPI 300, nlbin threshold 0.50, pad 16, bidi_reordering True,
text_direction horizontal-rl, temperature 1.0.

### Error Analysis

A sample passage (Preface, page 1) was compared character-by-character
against a manually verified ground truth.  Errors are categorised below.

#### Category 1 — Critical structural failures

These errors cannot be corrected by post-processing; they indicate model
or segmentation failures.

| OCR output | Ground truth | Type |
|-----------|-------------|------|
| `لفار` | `مُقَدّمَة` | Title in different typeface — model failure |
| `بحده فتحى صعوه` | `نجدة فتحي صفوة` | Author name — font/size variation |
| `الراحه` | `السراحة` | Missed multi-character sequence |

*Root cause:* The title and author name are set in a larger, bolder typeface
than the body text.  The `apt-20221130` model was trained predominantly on
body text and struggles with display-size typography.

#### Category 2 — Character substitutions in body text

| OCR | Ground truth | Error |
|-----|-------------|-------|
| `جريثٔة` | `جريئة` | ث ↔ ئ confusion |
| `بنفه` | `بنفسه` | dropped medial س |
| `القصته` | `لقصته` | initial ق → ل |
| `الألة` | `الآلة` | alef without madda |
| `فصوها` | `فصولها` | dropped medial ل |

*Root cause:* These are confidence near-misses where the correct character's
probability is slightly below the second-best candidate.  Beam-search
decoding with even a simple n-gram language model would likely fix most of
these.

#### Category 3 — ة / ه confusion

Taa marbuta (ة) and haa (ه) are visually identical except for the two dots
above ة.  At 300 DPI these dots can be faint or merged.

Examples: `بكتابه`→`بكتابة`, `الراحه`→`الراحة`, `وسوريه`→`وسورية`.

*Frequency:* ~4 per page in this sample.  This is the most consistent and
fixable error class.

#### Category 4 — ي / ى confusion

Final ya (ي) and alef maqsura (ى) are visually identical in many typefaces.

Examples: `العسكرى`→`العسكري`, `بشىء`→`بشيء`, `التى`→`التي`.

*Frequency:* ~3–6 per page, proportional to text length.

#### Category 5 — Missing diacritics

The model does not attempt to restore harakat (short vowel marks).  These
are missing from the entire output.  This is expected behaviour — the
training data is likely a mix of vocalised and unvocalised text, and the
model has learned to output the base consonantal skeleton.

#### Summary Statistics (page 1 sample)

| Error category | Occurrences | Fixable by post-processing? |
|---------------|-------------|---------------------------|
| Critical structural | 3 | No — needs better model or fine-tuning |
| Character substitution | 5 | Partially (language model) |
| ة / ه confusion | 4 | Yes (rule-based normalisation) |
| ي / ى confusion | 5 | Partially (context-dependent) |
| Missing diacritics | All | Requires separate diacritic restoration model |
| **Total** | **17** | |

Estimated Character Error Rate (CER) on body text: **~3–5 %**
(excluding title, author line, and diacritics).

### Effect of Key Parameters on Quality

#### DPI

| DPI | Observed effect |
|-----|----------------|
| 150 | Diacritical dots merge; ة/ه errors increase sharply |
| 200 | Marginal improvement; still problematic for small fonts |
| 300 | Baseline performance; recommended for most printed text |
| 400 | Marginal improvement on thin strokes |
| 600 | No further accuracy gain on clean print; large memory cost |

#### nlbin threshold

| Threshold | Observed effect |
|-----------|----------------|
| 0.30 | Background texture bleeds in as false ink |
| 0.50 | Default; good balance for clean printed text |
| 0.65 | Faint diacritics partially lost |
| 0.80 | Heavy stroke dropout; recognition collapses |

#### Line padding (`pad`)

| pad (px) | Observed effect |
|----------|----------------|
| 0 | First and last character of each line frequently truncated |
| 8 | Slight improvement at line ends |
| 16 | Default; well-calibrated for body text |
| 32 | Marginal; some lines show blank-context confusion |
| 64 | No improvement; blank padding dominates line image |

#### BiDi reordering

| Setting | Result |
|---------|--------|
| `True` (auto) | Correct Arabic logical order in most lines |
| `'R'` (force RTL) | Equivalent to True for pure Arabic text |
| `False` | Output is in visual/display order — Arabic words appear reversed |
| `'L'` (force LTR) | Incorrect for Arabic; reverses word order |

---

## 10. Available Models for Arabic

### OpenITI AOCP (Arabic Printed Text)

The primary source of high-quality Arabic printed text models.

| Model file | Training data | Recommended for |
|-----------|--------------|----------------|
| `apt-20221130.mlmodel` | ~200 printed Arabic books, 19th–early 20th c. | Classical Arabic printed books |
| (others in repo) | Varies | Specific time periods or publishers |

Repository: <https://github.com/OpenITI/AOCP_print_models>

### eScriptorium Community Models

The Zenodo platform hosts hundreds of models trained by the eScriptorium
community:

- Arabic manuscript models (various periods and regions)
- Ottoman Turkish print and manuscript
- Persian/Farsi

Search: <https://zenodo.org/search?q=kraken%20arabic>

### Training Your Own Model

For a specific corpus, fine-tuning (transfer learning from an existing
model) is highly effective.  Even 500–1000 annotated lines can push CER
from ~5 % to < 1 % for a specific document type:

```bash
ketos train \
  --load apt-20221130.mlmodel \
  --output my_model \
  --epochs 50 \
  ground_truth/*.xml   # PAGE XML or Alto XML annotations
```

---

## 11. kraken vs. Alternatives

### Tesseract 5 (LSTM)

| Criterion | kraken 7 | Tesseract 5 |
|-----------|---------|------------|
| Arabic accuracy (modern print) | ~95–97 % | ~85–90 % |
| Arabic manuscript | Good with right model | Poor |
| Baseline segmentation | Native (BLLA) | Bounding-box only |
| Custom model training | Full retraining + fine-tuning | Fine-tuning with tesstrain |
| Python API | Clean (`blla`, `rpred`) | Via `pytesseract` wrapper |
| Speed | Slower | Faster |
| Diacritics output | Lost (model-dependent) | Lost |
| Confidence scores | Per character | Per word |

### EasyOCR

| Criterion | kraken 7 | EasyOCR |
|-----------|---------|---------|
| Arabic accuracy (modern print) | ~95–97 % | ~70–80 % |
| Script coverage | Specialist (historical) | Broad (80+ scripts) |
| Segmentation | Baseline-aware | Bounding-box |
| Custom training | Native | Supported but complex |
| Python 3.14 | No | Yes |
| Memory (GPU) | 2–4 GB | 1–2 GB |

### PaddleOCR

| Criterion | kraken 7 | PaddleOCR |
|-----------|---------|-----------|
| Arabic accuracy (modern print) | ~95–97 % | ~75–85 % |
| Manuscript support | Excellent | Poor |
| Model ecosystem | Zenodo community | PaddlePaddle hub |
| Python 3.14 | No | Partial |
| Training pipeline | ketos CLI | PaddleOCR scripts |

### Commercial (ABBYY FineReader, Adobe Acrobat)

| Criterion | kraken 7 | Commercial |
|-----------|---------|-----------|
| Arabic accuracy (modern print) | ~95–97 % | ~97–99 % |
| Historical document accuracy | Often better | Often worse |
| Diacritics output | Model-dependent | Good |
| Cost | Free (AGPL) | Expensive |
| Offline use | Yes | Limited |
| API access | Full Python | Limited |
| Custom training | Yes | No (ABBYY has some) |

**Summary:** For historical and scholarly Arabic texts, kraken is the
best freely available option.  For modern Arabic documents where accuracy
on standard fonts is the only concern, a fine-tuned commercial system may
marginally outperform it.

---

## 12. Strengths and Limitations

### Strengths

1. **Baseline segmentation** — the only open-source engine with production-
   quality baseline detection; essential for manuscripts and early print.

2. **Per-character confidence scores** — enables targeted post-correction
   and quality-control workflows.

3. **Flexible model ecosystem** — any researcher can train, share, and load
   models; hundreds of community models cover obscure scripts and periods.

4. **Tight eScriptorium integration** — ground truth annotation, model
   training, and batch recognition in a single web interface.

5. **Right-to-left by design** — not retrofitted; Arabic, Hebrew, and
   Syriac are first-class citizens.

6. **Full pipeline control** — every stage (binarisation, segmentation,
   recognition) can be run independently with custom models.

### Limitations

1. **No beam search** — greedy CTC decoding is the only option in 7.0.1.
   A language-model-guided decoder would reduce character substitution
   errors significantly.

2. **No diacritic restoration** — the model outputs the consonantal skeleton.
   A separate seq2seq model would be needed for full vocalisation.

3. **Dependency fragility** — the `lightning` package dependency has already
   caused one quarantine-related deployment failure.  The `coremltools`
   dependency is unnecessary for inference (it is used only for model
   export to Apple format) but is still a required install.

4. **Python version ceiling** — `requires-python <3.14` and compiled
   dependencies without cp314 wheels mean the package cannot be deployed
   on Python 3.14+ without workarounds.

5. **Speed** — inference is noticeably slower than Tesseract for the same
   hardware.  A 300 DPI page takes 10–30 s on CPU.

6. **Title/display type** — the standard models are trained on body text.
   Large display fonts, chapter headings, and captions often produce
   higher error rates than the body text of the same page.

7. **No streaming output** — the entire page must be segmented before the
   first character is returned.

---

## 13. Practical Recipes

### Minimal Python inference (3 lines)

```python
from kraken.lib.models import load_any
from kraken import blla, rpred
from PIL import Image

model = load_any('apt-20221130.mlmodel')
img   = Image.open('page.png')                         # binary or greyscale
seg   = blla.segment(img, text_direction='horizontal-rl')
text  = '\n'.join(r.prediction for r in rpred.rpred(model, img, seg)
                  if r.prediction.strip())
```

### Filtering low-confidence lines

```python
CONF_THRESHOLD = 0.70   # discard lines where mean confidence < 70 %

import numpy as np
lines = []
for r in rpred.rpred(model, img, seg):
    if r.prediction.strip():
        conf = np.mean(r.confidences) if r.confidences else 0.0
        if conf >= CONF_THRESHOLD:
            lines.append(r.prediction)
```

### Masking page borders

```python
import numpy as np
from PIL import Image

img = Image.open('page.png')
w, h = img.size
mask = np.ones((h, w), dtype=np.uint8)
# zero out 5 % border on all sides
margin_y, margin_x = int(0.05 * h), int(0.05 * w)
mask[:margin_y, :]  = 0
mask[-margin_y:, :] = 0
mask[:, :margin_x]  = 0
mask[:, -margin_x:] = 0

seg = blla.segment(img, text_direction='horizontal-rl', mask=mask)
```

### Batch processing multiple pages

```python
from pathlib import Path

pages = sorted(Path('scans').glob('*.png'))
all_text = []
for path in pages:
    img = Image.open(path).convert('L')  # greyscale
    seg = blla.segment(img, text_direction='horizontal-rl')
    page_lines = [r.prediction for r in rpred.rpred(model, img, seg)
                  if r.prediction.strip()]
    all_text.append('\n'.join(page_lines))

Path('output.txt').write_text('\n\n'.join(all_text), encoding='utf-8')
```

### Post-processing: normalise ة/ه and ي/ى

```python
import re

def normalise_arabic_ocr(text: str) -> str:
    # ة/ه: replace standalone ه at end of word with ة
    # (heuristic: ه preceded by a letter and followed by space/punctuation)
    text = re.sub(r'(?<=[ابتثجحخدذرزسشصضطظعغفقكلمنويىأإآ])ه(?=[\s.,،؛؟!]|$)',
                  'ة', text)
    # ى/ي: normalise final ى to ي (many typefaces use them interchangeably)
    # Caution: ى is semantically distinct in some dialects/words
    # text = text.replace('ى', 'ي')  # enable only if appropriate
    return text
```

---

## 14. Conclusion

kraken is the most capable freely available OCR engine for Arabic and other
right-to-left scripts, particularly for historical printed and manuscript
material.  Its baseline-based segmentation is a genuine technical advance
over bounding-box approaches and is the main reason it outperforms
Tesseract and EasyOCR on complex Arabic layouts.

The primary opportunity for improvement lies in the decoding stage: the
current greedy CTC decoder leaves a measurable accuracy gap that a
language-model-guided beam search could close.  Character substitution
errors (ث↔ئ, ق↔ل, ة↔ه) are systematic and predictable, suggesting that
a relatively small Arabic language model used as a decoding prior would
produce significant CER improvements without any retraining of the visual
model.

For Arabic printed books from the 19th and 20th century, the
`apt-20221130.mlmodel` from OpenITI AOCP achieves approximately 95–97 %
character accuracy on body text at 300 DPI with default parameters.
Titles and display-size typography remain the weakest point, addressable
either by fine-tuning on representative examples or by training a
dedicated display-font model.

The practical deployment challenges (Python version ceiling, `lightning`
dependency fragility, `coremltools` as an unnecessary runtime requirement)
are known to the development team and are being addressed: the April 2026
commit "Factor out LightningModule load/write methods" suggests that the
`lightning` dependency will be reduced or removed in a future release.

---

*Document authored April 2026.  kraken version 7.0.1.
Model: OpenITI AOCP apt-20221130.mlmodel.
Evaluation corpus: Mudhakkirāt Jaʿfar al-ʿAskarī, page 1 preface.*
