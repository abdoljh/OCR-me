"""
Programmatic header/footer detection and stripping for scanned Arabic PDFs.

Design goals
------------
- Work on born-image PDFs (every page is one scan), since text-coordinate methods
  do not apply.
- Be content-aware: do NOT crop fixed percentages. The footer in particular varies
  drastically between pages (footnote-heavy pages vs. pages with no footnote at all).
- Be robust to scanner artifacts (ink streaks at the very top/bottom edges).
- Produce per-page crop boxes plus a cleaned output PDF.

Algorithm overview
------------------
Per page (rendered to grayscale at a fixed DPI, then Otsu-binarized):

1. Strip outer scanner noise. Compute an "active" left/right margin using a robust
   column-density profile, then ignore everything outside it for vertical analysis.

2. Build a *line-strip* profile. Smooth the row-wise ink density with a kernel
   roughly the height of an Arabic line, and extract contiguous runs above a low
   threshold. Each run is a candidate text line.

3. Detect a footnote separator. A footnote separator is a short horizontal rule
   (typically 1/4 to 1/2 of the text-column width). We detect any horizontal line
   via morphological opening with a long horizontal structuring element. Each
   candidate must then pass two filters:

   (a) WHITESPACE ISOLATION. The candidate's actual ink extent is found by
       walking outward from y_mid until row-ink drops below a small threshold;
       beyond that, ~18 rows in each direction must remain near-empty. This is
       the discriminator that separates real footnote rules — which sit alone
       in inter-line whitespace — from long horizontal calligraphic strokes
       (e.g., kashida elongations in chapter-title art) that look like rules
       under morphology but are embedded in tapering ink.

   (b) BODY-CONTEXT. At least two body line strips must lie above the rule, and
       the rule must sit below `body_top + 5% of page height`, to prevent
       header underlines from being mistaken for footnote separators.

   The topmost rule passing both filters becomes the footer boundary. This
   handles the hardest case in the sample, where the footnote on page 2 takes
   70% of the page (rule sits at only 30% from the top), as well as books that
   use slightly thicker (5-10 px) separator rules.

4. Detect the running header. A running header is a SHORT line strip near the top
   (typically within the first ~10% of page height) whose vertical gap to the next
   line strip is significantly larger than the inter-line gap of the body. If both
   conditions hold, classify it as header and exclude it.

5. Detect a non-separator footer (page numbers, running titles at the bottom).
   Same logic mirrored: a short, isolated strip near the bottom separated from the
   last body line by an unusually large gap.

6. Compose the keep-region. The kept area is `[header_bottom + small_pad,
   footer_top - small_pad]` vertically, and `[left_active, right_active]`
   horizontally. We never crop into the body; if a signal is ambiguous we err on
   the side of keeping content.

7. Apply the crop. Either rasterize the cropped region into a new image-PDF, or
   set the page's CropBox in a copy of the original PDF (cheaper, lossless).

This file exposes:
    Params(...)                         configuration dataclass; see fields
    detect_margins(page, p, verbose=False) -> PageMargins
    strip_pdf(in_path, out_path, p=Params(), mode=..., verbose=False)
        mode: "cropbox" (lossless, default) or "raster" (re-render at p.dpi)

Defaults: 300 DPI rendering, 50 px top/bottom padding, original L/R margins
preserved, no diagnostic notes (set `verbose=True` to populate them).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np


# ---------------------------------------------------------------------------
# Tunable parameters. Defaults work for the provided sample at 200 DPI on A4.
# Almost all are expressed as fractions of page height/width so they generalize.
# ---------------------------------------------------------------------------

@dataclass
class Params:
    # Render resolution. 300 DPI is the practical floor for downstream OCR
    # (Tesseract / EasyOCR both prefer ~300 DPI). Higher = better OCR, slower
    # detection.
    dpi: int = 300

    # Column-noise trim used internally by row-density profiling. This is NOT
    # the kept horizontal range; see `preserve_horizontal` below.
    side_margin_frac: float = 0.02

    # If True, the kept horizontal extent is the original page width (i.e.
    # left/right margins of the source PDF are preserved exactly). If False,
    # the active-column band is used as the kept horizontal extent — useful
    # when the source has heavy vertical bleed-through or scan streaks on the
    # outer edges that you want trimmed.
    preserve_horizontal: bool = True

    # Row profile smoothing kernel as a fraction of page height. Scaling by
    # height (rather than a fixed pixel count) keeps the kernel sized to one
    # text line regardless of DPI.
    smooth_frac: float = 0.004

    # Threshold for "this row contains ink" (fraction of the central band).
    ink_row_thresh: float = 0.010

    # Minimum run height (px) to be considered a line strip (drops specks).
    min_line_height_frac: float = 0.0017

    # Width-ratio rule: a strip is treated as a header/footer candidate when
    # its ink-width is below `narrow_width_ratio` × the body's MEDIAN line
    # width. The previous parameter `narrow_strip_frac` (an absolute fraction
    # of column width) was unused; this one replaces it and is the value
    # actually consulted during header/footer classification.
    narrow_width_ratio: float = 0.65

    # Header search band (top fraction of page).
    header_band_frac: float = 0.12

    # Footer search band (bottom fraction of page) — used only when no rule
    # is found.
    footer_band_frac: float = 0.10

    # Footnote rule detection.
    rule_min_len_frac: float = 0.12      # ≥12% of usable width
    rule_max_len_frac: float = 0.95      # ≤95% (avoid full-width edge artifacts)
    # Max thickness (in px at `dpi`) of a candidate horizontal stripe still
    # considered as a rule. Real footnote separators in scanned books range
    # from 1 px (sharp) up to ~10 px (bolder presses, slight ink bleed).
    # The previous value of 6 was too tight — it excluded the page-3 rule of
    # the original sample and the page-5/8 rules of Chapter_10 (all 7-9 px).
    # We now permit up to 12 px and rely on the whitespace-isolation check
    # below to weed out non-rule horizontal ink (e.g., kashida strokes inside
    # calligraphic chapter titles).
    rule_thickness_max_px: int = 12

    # Whitespace-isolation gate for rule candidates. A real footnote rule sits
    # alone in the inter-line gap: when we walk outward from the rule's middle
    # row until we exit its ink, we expect ~empty rows beyond that.
    #
    # `rule_isolation_px`     — number of rows above AND below the rule that
    #                           must stay near-empty (at the configured DPI).
    # `rule_isolation_max_ink_frac` — a row counts as empty if its column-
    #                           ink fraction is below this value. 2% tolerates
    #                           scanner specks but rejects tapering text.
    # `rule_extent_max_walk_px` — how far we walk out from y_mid looking for
    #                           the rule's edge. Rules thicker than this are
    #                           treated as embedded in a textured region and
    #                           rejected (defensive cap).
    rule_isolation_px: int = 18
    rule_isolation_max_ink_frac: float = 0.02
    rule_extent_max_walk_px: int = 30

    # Padding around the kept region (in pixels at `dpi`). Top/bottom default
    # to 50 px so cropped pages do not feel cramped against the text. Sides
    # default to 0 because `preserve_horizontal=True` already keeps the source
    # PDF's left/right margins.
    pad_top_px: int = 50
    pad_bottom_px: int = 50
    pad_side_px: int = 0


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

@dataclass
class PageMargins:
    page_index: int
    page_w: int
    page_h: int
    keep_top: int           # y of first kept row (inclusive)
    keep_bottom: int        # y of last kept row (inclusive)
    keep_left: int
    keep_right: int
    header_strip: Optional[tuple[int, int]]   # (y0, y1) if header detected
    footer_strip: Optional[tuple[int, int]]   # (y0, y1) if footer detected
    rule_y: Optional[int]                     # y of footnote rule if found
    notes: list[str]

    def __repr__(self) -> str:  # short single-line summary
        flags = []
        if self.header_strip:
            flags.append("H")
        if self.rule_y is not None:
            flags.append("R")
        if self.footer_strip:
            flags.append("F")
        tag = "".join(flags) or "-"
        return (
            f"PageMargins(page={self.page_index}, "
            f"keep=x[{self.keep_left}..{self.keep_right}] "
            f"y[{self.keep_top}..{self.keep_bottom}], flags={tag})"
        )


def _render_gray(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Return ink=255, paper=0 (uint8)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def _active_columns(bw: np.ndarray, side_frac: float) -> tuple[int, int]:
    """Find the active text column band by trimming sparse outer columns."""
    H, W = bw.shape
    col_ink = bw.sum(axis=0) / 255 / H
    # Smooth and threshold
    k = max(5, W // 200)
    sm = np.convolve(col_ink, np.ones(k) / k, mode="same")
    thr = max(0.005, sm.mean() * 0.25)
    cols = np.where(sm > thr)[0]
    if not len(cols):
        return int(W * side_frac), int(W * (1 - side_frac))
    left, right = int(cols[0]), int(cols[-1])
    # Never include the outermost 1% (often scanner streaks)
    left = max(left, int(W * 0.01))
    right = min(right, W - 1 - int(W * 0.01))
    return left, right


def _line_runs(
    bw: np.ndarray, left: int, right: int, p: Params
) -> list[tuple[int, int, float]]:
    """
    Return list of (y0, y1, narrow_frac) for each detected line strip.
    `narrow_frac` is the run's ink width / column-band width, used to flag
    narrow lines like page-number headers.
    """
    H, W = bw.shape
    central = bw[:, left : right + 1]
    band_w = central.shape[1]
    row_ink = central.sum(axis=1) / 255 / band_w
    k = max(3, int(round(p.smooth_frac * H)) | 1)  # odd
    sm = np.convolve(row_ink, np.ones(k) / k, mode="same")

    is_line = sm > p.ink_row_thresh
    runs: list[tuple[int, int]] = []
    in_run = False
    s = 0
    for r in range(H):
        if is_line[r] and not in_run:
            in_run, s = True, r
        elif (not is_line[r]) and in_run:
            in_run = False
            runs.append((s, r - 1))
    if in_run:
        runs.append((s, H - 1))

    min_h = max(3, int(round(p.min_line_height_frac * H)))
    runs = [(a, b) for (a, b) in runs if (b - a + 1) >= min_h]

    out: list[tuple[int, int, float]] = []
    for a, b in runs:
        sub = central[a : b + 1]
        col_has_ink = (sub.sum(axis=0) > 0).sum()
        out.append((a, b, col_has_ink / band_w))
    return out


def _is_isolated_horizontal(
    bw_band: np.ndarray,
    y_mid: int,
    p: Params,
) -> bool:
    """Return True iff `y_mid` looks like a free-standing horizontal rule.

    Walks outward from `y_mid` row-by-row until the column-ink fraction of
    that row drops below `p.rule_isolation_max_ink_frac` — that is the rule's
    edge. Beyond each edge we then require `p.rule_isolation_px` consecutive
    rows that stay below the threshold.

    Real footnote separators pass: they sit alone in inter-line whitespace.
    Calligraphic strokes (e.g., the kashida تطويل in Arabic chapter-title art),
    or any bold horizontal element embedded in textured regions, fail because
    the ink tapers in over many rows rather than dropping cleanly to zero.
    """
    H, band_w = bw_band.shape
    if band_w == 0:
        return False
    # Per-row ink density inside the active column band, normalized 0..1.
    row_ink = (bw_band > 0).sum(axis=1) / band_w

    # Walk up: find the first row above y_mid whose ink dips below threshold
    # (= we've exited the rule itself). Cap the walk; very thick stripes are
    # treated as embedded ink and rejected.
    above_top: Optional[int] = None
    for dy in range(1, p.rule_extent_max_walk_px + 1):
        y = y_mid - dy
        if y < 0:
            break
        if row_ink[y] < p.rule_isolation_max_ink_frac:
            above_top = y
            break
    if above_top is None:
        return False

    # Walk down similarly.
    below_bot: Optional[int] = None
    for dy in range(1, p.rule_extent_max_walk_px + 1):
        y = y_mid + dy
        if y >= H:
            break
        if row_ink[y] < p.rule_isolation_max_ink_frac:
            below_bot = y
            break
    if below_bot is None:
        return False

    # Verify the next isolation_px rows beyond each edge stay near-empty.
    # If we run out of page (e.g., rule near top/bottom edge), accept what's
    # available as long as it's at least half the requested isolation.
    half_iso = p.rule_isolation_px // 2
    above_lo = max(0, above_top - p.rule_isolation_px + 1)
    above_hi = above_top + 1
    below_lo = below_bot
    below_hi = min(H, below_bot + p.rule_isolation_px)
    if (above_hi - above_lo) < half_iso or (below_hi - below_lo) < half_iso:
        return False

    return (
        row_ink[above_lo:above_hi].max() < p.rule_isolation_max_ink_frac
        and row_ink[below_lo:below_hi].max() < p.rule_isolation_max_ink_frac
    )


def _detect_horizontal_rules(bw: np.ndarray, left: int, right: int, p: Params) -> list[int]:
    """
    Return a list of y-coordinates for thin horizontal rules inside the active
    column band. A footnote separator usually shows up as one or two such rules.
    Bottom/top edge artifacts and calligraphic horizontal strokes are filtered.
    """
    H, W = bw.shape
    band = bw[:, left : right + 1]
    band_w = band.shape[1]
    min_len = max(15, int(round(p.rule_min_len_frac * band_w)))
    max_len = int(round(p.rule_max_len_frac * band_w))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    horiz = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
    # ink length per row inside the band
    line_len = (horiz > 0).sum(axis=1)
    candidate_rows = np.where((line_len >= min_len) & (line_len <= max_len))[0]
    if not len(candidate_rows):
        return []

    # Cluster contiguous rows; reject thick groups (those are paragraphs of text
    # that happen to span the column, not thin rules).
    groups: list[list[int]] = []
    cur = [int(candidate_rows[0])]
    for r in candidate_rows[1:]:
        if r - cur[-1] <= 2:
            cur.append(int(r))
        else:
            groups.append(cur)
            cur = [int(r)]
    groups.append(cur)

    out: list[int] = []
    for g in groups:
        if (g[-1] - g[0] + 1) > p.rule_thickness_max_px:
            continue
        y_mid = (g[0] + g[-1]) // 2
        # Drop top/bottom edge artifacts (within 1.5% of edges)
        if y_mid < int(0.015 * H) or y_mid > int(0.985 * H):
            continue
        # Whitespace-isolation gate. This is the key filter that separates
        # real footnote separators from calligraphic horizontal strokes
        # (which pass the morphological length+thickness gate but are
        # embedded in tapering ink rather than sitting alone in whitespace).
        if not _is_isolated_horizontal(band, y_mid, p):
            continue
        out.append(y_mid)
    return out


class _NullNotes(list):
    """A drop-in list that ignores append calls. Lets us keep the
    `notes.append(...)` call sites unchanged when running quietly."""
    __slots__ = ()
    def append(self, _item) -> None:  # noqa: D401
        pass


def detect_margins(
    page: fitz.Page,
    p: Params = Params(),
    *,
    verbose: bool = False,
) -> PageMargins:
    """
    Run the full detection pipeline on a single PDF page.

    Parameters
    ----------
    page : fitz.Page
        Source page (typically from a scanned PDF).
    p : Params
        Tunable parameters; see the dataclass definition for details.
    verbose : bool, default False
        If True, populate `PageMargins.notes` with a per-step diagnostic log.
        If False (default), `notes` is empty — useful for batch jobs.
    """
    notes: list[str] = [] if verbose else _NullNotes()
    gray = _render_gray(page, p.dpi)
    bw = _binarize(gray)
    H, W = bw.shape

    left, right = _active_columns(bw, p.side_margin_frac)
    if p.preserve_horizontal:
        keep_left_default, keep_right_default = 0, W - 1
    else:
        keep_left_default, keep_right_default = left, right

    runs = _line_runs(bw, left, right, p)
    if not runs:
        notes.append("no line runs detected; keeping full page")
        return PageMargins(0, W, H, 0, H - 1, keep_left_default, keep_right_default,
                           None, None, None, notes)

    # ---- Drop scanner-edge speck runs ------------------------------------
    # Very short ink runs that sit literally at row 0 or row H-1 are scanner
    # streaks, never real headers. Filter them BEFORE downstream classification.
    edge_band = max(3, int(0.012 * H))
    def _is_edge_artifact(a: int, b: int) -> bool:
        h = b - a + 1
        touches_top = a <= 2
        touches_bot = b >= H - 3
        return (touches_top or touches_bot) and h <= edge_band
    runs = [r for r in runs if not _is_edge_artifact(r[0], r[1])]
    if not runs:
        notes.append("only edge artifacts detected; keeping full page")
        return PageMargins(0, W, H, 0, H - 1, keep_left_default, keep_right_default,
                           None, None, None, notes)

    # Median inter-line gap and median line height inside the body.
    if len(runs) >= 3:
        gaps = sorted(runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1))
        median_gap = gaps[len(gaps) // 2]
    else:
        median_gap = max(8, int(0.012 * H))
    line_widths = sorted(r[2] for r in runs)
    median_width = line_widths[len(line_widths) // 2]

    header_band_end = int(p.header_band_frac * H)
    footer_band_start = int((1 - p.footer_band_frac) * H)

    # ---- Header detection -------------------------------------------------
    # A header is any line strip in the top header-band that is much NARROWER
    # than the body's median line width (page-number + short title), with a
    # plausible gap to the next strip. We rely on width-ratio first because
    # gap-only thresholds are unreliable when body line spacing varies.
    header_strip: Optional[tuple[int, int]] = None
    if runs and runs[0][0] <= header_band_end:
        first = runs[0]
        gap_to_next = runs[1][0] - first[1] if len(runs) >= 2 else H
        is_much_narrower = first[2] < median_width * p.narrow_width_ratio
        is_short = (first[1] - first[0] + 1) < int(0.04 * H)
        plausible_gap = gap_to_next >= max(int(0.6 * median_gap), 20)
        if (is_much_narrower or is_short) and plausible_gap:
            header_strip = (first[0], first[1])
            notes.append(
                f"header: y={first[0]}-{first[1]} "
                f"(width={first[2]:.2f} vs median {median_width:.2f}, "
                f"gap_to_next={gap_to_next})"
            )
            runs = runs[1:]
        else:
            notes.append(
                f"top run y={first[0]}-{first[1]} kept as body "
                f"(width={first[2]:.2f} vs median {median_width:.2f}, "
                f"gap_to_next={gap_to_next})"
            )

    # ---- Footnote-rule detection (strongest footer signal) ----------------
    # A valid footnote rule must sit below the header (or below the first body
    # line if no header) AND have at least a few body lines above it. This
    # handles pages where the footnote takes most of the page (rule near 30%).
    rules = _detect_horizontal_rules(bw, left, right, p)
    rule_y: Optional[int] = None
    body_top = (header_strip[1] if header_strip else runs[0][0] if runs else 0)
    min_rule_y = body_top + max(int(0.05 * H), 50)  # at least one line below body top
    candidate_rules = [r for r in rules if r > min_rule_y]
    # Also require that at least 2 body line strips fall above the rule
    valid_rules = []
    for r in candidate_rules:
        lines_above = sum(1 for run in runs if run[1] < r)
        if lines_above >= 2:
            valid_rules.append(r)
    if valid_rules:
        rule_y = min(valid_rules)  # highest rule = top boundary of footer zone
        notes.append(f"footnote rule at y={rule_y} ({rule_y/H:.1%})")
    elif rules:
        notes.append(f"horizontal rules at {rules} were all filtered out")

    # ---- Footer-strip detection (used when no rule found) -----------------
    footer_strip: Optional[tuple[int, int]] = None
    if rule_y is None and runs:
        last = runs[-1]
        if last[1] >= footer_band_start:
            gap_from_prev = last[0] - runs[-2][1] if len(runs) >= 2 else H
            is_much_narrower = last[2] < median_width * p.narrow_width_ratio
            is_short = (last[1] - last[0] + 1) < int(0.04 * H)
            plausible_gap = gap_from_prev >= max(int(0.6 * median_gap), 20)
            if (is_much_narrower or is_short) and plausible_gap:
                footer_strip = (last[0], last[1])
                notes.append(
                    f"footer: y={last[0]}-{last[1]} "
                    f"(width={last[2]:.2f} vs median {median_width:.2f}, "
                    f"gap_from_prev={gap_from_prev})"
                )
                runs = runs[:-1]

    # ---- Compose keep region ---------------------------------------------
    # Pad outward from the body into whitespace, clamped so we never crop
    # back into a header/footer/rule that we just identified, and never past
    # the page edge.
    #
    # Top: ideal = first_body_line - pad_top_px
    #      lower bound = (header_strip[1] + small safety) if header exists, else 0
    # Bottom: ideal = last_body_line + pad_bottom_px
    #         upper bound = (rule_y - small safety) if rule exists,
    #                       else (footer_strip[0] - small safety) if footer exists,
    #                       else H - 1
    safety_px = 4  # tiny gap so we don't paint over the header pixel itself
    rule_safety_px = 12  # rules can be 1-3 px and we want clear separation
    body_first_y = runs[0][0] if runs else 0
    body_last_y = runs[-1][1] if runs else H - 1

    top_lower_bound = (header_strip[1] + safety_px) if header_strip else 0
    keep_top = max(top_lower_bound, body_first_y - p.pad_top_px)

    if rule_y is not None:
        bottom_upper_bound = rule_y - rule_safety_px
    elif footer_strip is not None:
        bottom_upper_bound = footer_strip[0] - safety_px
    else:
        bottom_upper_bound = H - 1
    keep_bottom = min(bottom_upper_bound, body_last_y + p.pad_bottom_px)
    keep_bottom = max(keep_top + 1, keep_bottom)  # never invert

    # Horizontal: preserve original page width by default; otherwise use the
    # active-column band, optionally padded inward/outward by pad_side_px.
    if p.preserve_horizontal:
        keep_left, keep_right = 0, W - 1
    else:
        keep_left = max(0, left - p.pad_side_px)
        keep_right = min(W - 1, right + p.pad_side_px)

    return PageMargins(
        page_index=0,
        page_w=W,
        page_h=H,
        keep_top=int(keep_top),
        keep_bottom=int(keep_bottom),
        keep_left=int(keep_left),
        keep_right=int(keep_right),
        header_strip=header_strip,
        footer_strip=footer_strip,
        rule_y=rule_y,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API: process a whole PDF
# ---------------------------------------------------------------------------

def strip_pdf(
    in_path: str | Path,
    out_path: str | Path,
    p: Params = Params(),
    mode: str = "cropbox",
    *,
    verbose: bool = False,
) -> list[PageMargins]:
    """
    Crop headers & footers from every page of `in_path` and write `out_path`.

    Parameters
    ----------
    in_path, out_path : str | Path
        Source and destination PDFs.
    p : Params
        Tunable detection parameters.
    mode : {"cropbox", "raster"}
        "cropbox" — set each page's CropBox in a copy of the original PDF.
                    Lossless, fast, small. Best default.
        "raster"  — re-render each page at `p.dpi`, slice out the kept region,
                    and assemble a new PDF from those PNGs. Use this when
                    downstream tools ignore CropBox or you want a flattened
                    image-only output for OCR.
    verbose : bool, default False
        If True, each PageMargins in the returned list carries a step-by-step
        `notes` list. If False, notes are suppressed entirely (no allocations,
        no log strings produced).

    Returns
    -------
    list[PageMargins]
        One entry per page. Always returned, regardless of `verbose`. Use
        `verbose=True` if you want the per-page diagnostic notes too.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    src = fitz.open(in_path)
    results: list[PageMargins] = []

    if mode == "cropbox":
        for i, page in enumerate(src):
            m = detect_margins(page, p, verbose=verbose)
            m.page_index = i
            results.append(m)
            # Convert pixel coordinates (at p.dpi) back to PDF points (1 pt = 1/72 in)
            scale = 72.0 / p.dpi
            page_rect = page.rect
            x0 = page_rect.x0 + m.keep_left * scale
            x1 = page_rect.x0 + (m.keep_right + 1) * scale
            y0 = page_rect.y0 + m.keep_top * scale
            y1 = page_rect.y0 + (m.keep_bottom + 1) * scale
            crop = fitz.Rect(x0, y0, x1, y1)
            crop &= page_rect  # clamp to mediabox just in case
            page.set_cropbox(crop)
        src.save(out_path, garbage=4, deflate=True)
        src.close()
        return results

    if mode == "raster":
        out = fitz.open()
        for i, page in enumerate(src):
            m = detect_margins(page, p, verbose=verbose)
            m.page_index = i
            results.append(m)
            gray = _render_gray(page, p.dpi)
            cropped = gray[m.keep_top : m.keep_bottom + 1, m.keep_left : m.keep_right + 1]
            ok, buf = cv2.imencode(".png", cropped)
            if not ok:
                raise RuntimeError(f"PNG encode failed on page {i}")
            png_bytes = buf.tobytes()
            h, w = cropped.shape
            pt_w = w * 72.0 / p.dpi
            pt_h = h * 72.0 / p.dpi
            new_page = out.new_page(width=pt_w, height=pt_h)
            new_page.insert_image(new_page.rect, stream=png_bytes)
        out.save(out_path, garbage=4, deflate=True)
        out.close()
        src.close()
        return results

    raise ValueError(f"unknown mode: {mode!r}")
