"""
Programmatic header/footer detection and stripping for scanned Arabic PDFs.

Algorithm overview (per page, rendered to grayscale + Otsu-binarized):
1. Strip outer scanner noise via column-density profile.
2. Build a line-strip profile (smoothed row-wise ink density).
3. Detect footnote separator via morphological horizontal-rule detection +
   whitespace-isolation gate (rejects calligraphic strokes that pass
   morphology but are embedded in tapering ink rather than clean whitespace).
4. Detect running header (narrow/short strip near top with large gap below).
5. Detect non-separator footer (narrow/short strip near bottom with large gap above).
6. Compose keep-region with padding, clamped to body bounds.
7. Apply crop: "cropbox" (lossless CropBox) or "raster" (re-render PNGs).

Public API:
    Params(...)
    detect_margins(page, p, verbose=False) -> PageMargins
    strip_pdf(in_path, out_path, p=Params(), mode="cropbox", verbose=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np


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
    # Raised to 12 from v2's 6 — handles bolder-press rules (7-9 px).
    # The whitespace-isolation gate below rejects non-rule horizontals.
    rule_thickness_max_px: int = 12
    rule_isolation_px: int = 18
    rule_isolation_max_ink_frac: float = 0.02
    rule_extent_max_walk_px: int = 30
    pad_top_px: int = 50
    pad_bottom_px: int = 50
    pad_side_px: int = 0


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
        flags = "".join(
            f for f, v in [("H", self.header_strip), ("R", self.rule_y), ("F", self.footer_strip)]
            if v is not None
        ) or "-"
        return (
            f"PageMargins(page={self.page_index}, "
            f"keep=x[{self.keep_left}..{self.keep_right}] "
            f"y[{self.keep_top}..{self.keep_bottom}], flags={flags})"
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
    left = max(int(cols[0]), int(W * 0.01))
    right = min(int(cols[-1]), W - 1 - int(W * 0.01))
    return left, right


def _line_runs(bw: np.ndarray, left: int, right: int, p: Params) -> list[tuple[int, int, float]]:
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
        elif not is_line[r] and in_run:
            in_run = False
            runs.append((s, r - 1))
    if in_run:
        runs.append((s, H - 1))

    min_h = max(3, int(round(p.min_line_height_frac * H)))
    out: list[tuple[int, int, float]] = []
    for a, b in runs:
        if (b - a + 1) < min_h:
            continue
        col_has_ink = (central[a : b + 1].sum(axis=0) > 0).sum()
        out.append((a, b, col_has_ink / band_w))
    return out


def _is_isolated_horizontal(bw_band: np.ndarray, y_mid: int, p: Params) -> bool:
    """True iff `y_mid` sits alone in whitespace (real rule, not calligraphic stroke)."""
    H, band_w = bw_band.shape
    if band_w == 0:
        return False
    row_ink = (bw_band > 0).sum(axis=1) / band_w

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
        if (g[-1] - g[0] + 1) > p.rule_thickness_max_px:
            continue
        y_mid = (g[0] + g[-1]) // 2
        if y_mid < int(0.015 * H) or y_mid > int(0.985 * H):
            continue
        if not _is_isolated_horizontal(band, y_mid, p):
            continue
        out.append(y_mid)
    return out


class _NullNotes(list):
    __slots__ = ()
    def append(self, _item) -> None:
        pass


def detect_margins(page: fitz.Page, p: Params = Params(), *, verbose: bool = False) -> PageMargins:
    notes: list[str] = [] if verbose else _NullNotes()
    gray = _render_gray(page, p.dpi)
    bw = _binarize(gray)
    H, W = bw.shape

    left, right = _active_columns(bw, p.side_margin_frac)
    keep_left_default = 0 if p.preserve_horizontal else left
    keep_right_default = W - 1 if p.preserve_horizontal else right

    runs = _line_runs(bw, left, right, p)
    if not runs:
        return PageMargins(0, W, H, 0, H - 1, keep_left_default, keep_right_default,
                           None, None, None, notes)

    edge_band = max(3, int(0.012 * H))
    runs = [r for r in runs
            if not ((r[0] <= 2 or r[1] >= H - 3) and (r[1] - r[0] + 1) <= edge_band)]
    if not runs:
        return PageMargins(0, W, H, 0, H - 1, keep_left_default, keep_right_default,
                           None, None, None, notes)

    if len(runs) >= 3:
        gaps = sorted(runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1))
        median_gap = gaps[len(gaps) // 2]
    else:
        median_gap = max(8, int(0.012 * H))
    median_width = sorted(r[2] for r in runs)[len(runs) // 2]

    header_band_end   = int(p.header_band_frac * H)
    footer_band_start = int((1 - p.footer_band_frac) * H)

    header_strip: Optional[tuple[int, int]] = None
    if runs and runs[0][0] <= header_band_end:
        first = runs[0]
        gap_to_next = runs[1][0] - first[1] if len(runs) >= 2 else H
        is_narrow = first[2] < median_width * p.narrow_width_ratio
        is_short  = (first[1] - first[0] + 1) < int(0.04 * H)
        if (is_narrow or is_short) and gap_to_next >= max(int(0.6 * median_gap), 20):
            header_strip = (first[0], first[1])
            runs = runs[1:]

    rules = _detect_horizontal_rules(bw, left, right, p)
    rule_y: Optional[int] = None
    body_top = header_strip[1] if header_strip else (runs[0][0] if runs else 0)
    min_rule_y = body_top + max(int(0.05 * H), 50)
    valid_rules = [r for r in rules
                   if r > min_rule_y and sum(1 for run in runs if run[1] < r) >= 2]
    if valid_rules:
        rule_y = min(valid_rules)

    footer_strip: Optional[tuple[int, int]] = None
    if rule_y is None and runs:
        last = runs[-1]
        if last[1] >= footer_band_start:
            gap_from_prev = last[0] - runs[-2][1] if len(runs) >= 2 else H
            is_narrow = last[2] < median_width * p.narrow_width_ratio
            is_short  = (last[1] - last[0] + 1) < int(0.04 * H)
            if (is_narrow or is_short) and gap_from_prev >= max(int(0.6 * median_gap), 20):
                footer_strip = (last[0], last[1])
                runs = runs[:-1]

    body_first_y = runs[0][0] if runs else 0
    body_last_y  = runs[-1][1] if runs else H - 1

    keep_top = max(
        (header_strip[1] + 4) if header_strip else 0,
        body_first_y - p.pad_top_px,
    )

    if rule_y is not None:
        bottom_bound = rule_y - 12
    elif footer_strip is not None:
        bottom_bound = footer_strip[0] - 4
    else:
        bottom_bound = H - 1
    keep_bottom = max(keep_top + 1, min(bottom_bound, body_last_y + p.pad_bottom_px))

    if p.preserve_horizontal:
        keep_left, keep_right = 0, W - 1
    else:
        keep_left  = max(0, left  - p.pad_side_px)
        keep_right = min(W - 1, right + p.pad_side_px)

    return PageMargins(
        page_index=0, page_w=W, page_h=H,
        keep_top=int(keep_top), keep_bottom=int(keep_bottom),
        keep_left=int(keep_left), keep_right=int(keep_right),
        header_strip=header_strip, footer_strip=footer_strip,
        rule_y=rule_y, notes=notes,
    )


def strip_pdf(
    in_path: str | Path,
    out_path: str | Path,
    p: Params = Params(),
    mode: str = "cropbox",
    *,
    verbose: bool = False,
    on_page=None,
) -> list[PageMargins]:
    in_path, out_path = Path(in_path), Path(out_path)
    src = fitz.open(in_path)
    n = src.page_count
    results: list[PageMargins] = []

    if mode == "cropbox":
        for i, page in enumerate(src):
            m = detect_margins(page, p, verbose=verbose)
            m.page_index = i
            results.append(m)
            scale = 72.0 / p.dpi
            pr = page.rect      # CropBox if set, else MediaBox (screen coords)
            mb = page.mediabox  # always the full MediaBox; set_cropbox validates against this
            crop = fitz.Rect(
                pr.x0 + m.keep_left * scale,
                pr.y0 + m.keep_top * scale,
                pr.x0 + (m.keep_right + 1) * scale,
                pr.y0 + (m.keep_bottom + 1) * scale,
            ) & mb              # clamp to MediaBox, not CropBox, to satisfy set_cropbox
            if not crop.is_empty:
                try:
                    page.set_cropbox(crop)
                except Exception:
                    pass        # leave page uncropped rather than abort the whole document
            if on_page:
                on_page(i + 1, n)
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
            crop = gray[m.keep_top : m.keep_bottom + 1, m.keep_left : m.keep_right + 1]
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                raise RuntimeError(f"PNG encode failed on page {i}")
            h, w = crop.shape
            new_page = out.new_page(width=w * 72.0 / p.dpi, height=h * 72.0 / p.dpi)
            new_page.insert_image(new_page.rect, stream=buf.tobytes())
            if on_page:
                on_page(i + 1, n)
        out.save(out_path, garbage=4, deflate=True)
        out.close()
        src.close()
        return results

    raise ValueError(f"unknown mode: {mode!r}")
