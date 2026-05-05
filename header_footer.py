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
   via morphological opening with a long horizontal structuring element and treat
   the topmost rule that has at least two body line strips above it as the footer
   boundary. This single signal handles the hardest case in the sample, where the
   footnote on page 2 takes 70% of the page (rule sits at only 30% from the top).

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
# Tunable parameters
# ---------------------------------------------------------------------------

@dataclass
class Params:
    dpi: int = 300
    side_margin_frac: float = 0.02
    preserve_horizontal: bool = True
    smooth_frac: float = 0.004
    ink_row_thresh: float = 0.010
    min_line_height_frac: float = 0.0017
    narrow_width_ratio: float = 0.65
    header_band_frac: float = 0.12
    footer_band_frac: float = 0.10
    rule_min_len_frac: float = 0.12
    rule_max_len_frac: float = 0.95
    rule_thickness_max_px: int = 6
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
    keep_top: int
    keep_bottom: int
    keep_left: int
    keep_right: int
    header_strip: Optional[tuple[int, int]]
    footer_strip: Optional[tuple[int, int]]
    rule_y: Optional[int]
    notes: list[str]

    def __repr__(self) -> str:
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
    H, W = bw.shape
    col_ink = bw.sum(axis=0) / 255 / H
    k = max(5, W // 200)
    sm = np.convolve(col_ink, np.ones(k) / k, mode="same")
    thr = max(0.005, sm.mean() * 0.25)
    cols = np.where(sm > thr)[0]
    if not len(cols):
        return int(W * side_frac), int(W * (1 - side_frac))
    left, right = int(cols[0]), int(cols[-1])
    left = max(left, int(W * 0.01))
    right = min(right, W - 1 - int(W * 0.01))
    return left, right


def _line_runs(
    bw: np.ndarray, left: int, right: int, p: Params
) -> list[tuple[int, int, float]]:
    H, W = bw.shape
    central = bw[:, left : right + 1]
    band_w = central.shape[1]
    row_ink = central.sum(axis=1) / 255 / band_w
    k = max(3, int(round(p.smooth_frac * H)) | 1)
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


def _detect_horizontal_rules(bw: np.ndarray, left: int, right: int, p: Params) -> list[int]:
    H, W = bw.shape
    band = bw[:, left : right + 1]
    band_w = band.shape[1]
    min_len = max(15, int(round(p.rule_min_len_frac * band_w)))
    max_len = int(round(p.rule_max_len_frac * band_w))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    horiz = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
    line_len = (horiz > 0).sum(axis=1)
    candidate_rows = np.where((line_len >= min_len) & (line_len <= max_len))[0]
    if not len(candidate_rows):
        return []

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
        if (g[-1] - g[0] + 1) <= p.rule_thickness_max_px:
            y_mid = (g[0] + g[-1]) // 2
            if y_mid < int(0.015 * H) or y_mid > int(0.985 * H):
                continue
            out.append(y_mid)
    return out


class _NullNotes(list):
    """Silently drops append calls when verbose=False."""
    __slots__ = ()
    def append(self, _item) -> None:
        pass


def detect_margins(
    page: fitz.Page,
    p: Params = Params(),
    *,
    verbose: bool = False,
) -> PageMargins:
    """Run the full detection pipeline on a single PDF page."""
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

    if len(runs) >= 3:
        gaps = sorted(runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1))
        median_gap = gaps[len(gaps) // 2]
    else:
        median_gap = max(8, int(0.012 * H))
    line_widths = sorted(r[2] for r in runs)
    median_width = line_widths[len(line_widths) // 2]

    header_band_end = int(p.header_band_frac * H)
    footer_band_start = int((1 - p.footer_band_frac) * H)

    header_strip: Optional[tuple[int, int]] = None
    if runs and runs[0][0] <= header_band_end:
        first = runs[0]
        gap_to_next = runs[1][0] - first[1] if len(runs) >= 2 else H
        is_much_narrower = first[2] < median_width * p.narrow_width_ratio
        is_short = (first[1] - first[0] + 1) < int(0.04 * H)
        plausible_gap = gap_to_next >= max(int(0.6 * median_gap), 20)
        if (is_much_narrower or is_short) and plausible_gap:
            header_strip = (first[0], first[1])
            runs = runs[1:]

    rules = _detect_horizontal_rules(bw, left, right, p)
    rule_y: Optional[int] = None
    body_top = (header_strip[1] if header_strip else runs[0][0] if runs else 0)
    min_rule_y = body_top + max(int(0.05 * H), 50)
    candidate_rules = [r for r in rules if r > min_rule_y]
    valid_rules = []
    for r in candidate_rules:
        lines_above = sum(1 for run in runs if run[1] < r)
        if lines_above >= 2:
            valid_rules.append(r)
    if valid_rules:
        rule_y = min(valid_rules)

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
                runs = runs[:-1]

    safety_px = 4
    rule_safety_px = 12
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
    keep_bottom = max(keep_top + 1, keep_bottom)

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
    """Crop headers & footers from every page of `in_path` and write `out_path`."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    src = fitz.open(in_path)
    results: list[PageMargins] = []

    if mode == "cropbox":
        for i, page in enumerate(src):
            m = detect_margins(page, p, verbose=verbose)
            m.page_index = i
            results.append(m)
            scale = 72.0 / p.dpi
            page_rect = page.rect
            x0 = page_rect.x0 + m.keep_left * scale
            x1 = page_rect.x0 + (m.keep_right + 1) * scale
            y0 = page_rect.y0 + m.keep_top * scale
            y1 = page_rect.y0 + (m.keep_bottom + 1) * scale
            crop = fitz.Rect(x0, y0, x1, y1)
            crop &= page_rect
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
