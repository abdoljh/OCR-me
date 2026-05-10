"""
image_extract.py — Phase 1b of the Arabic-OCR pipeline.

Extracts photographic regions (and optionally their captions) from
scanned Arabic book pages. Companion to header_footer.py and page_export.py.

Design
------
Scanned book pages contain photographs surrounded by Arabic captions.
Each PDF page is one big embedded JPEG, so PyMuPDF's get_images() returns
the whole page-scan, useless for extracting individual photos. We segment
in pixel space.

Key insight: photos and captions live in different intensity ranges.
Captions are mid-gray ink on white; photographic regions contain large
swaths of very dark pixels (deep shadows, dark clothing, B&W tones). Otsu
binarization conflates them. So we use two masks:

  bw_dark : gray < OTSU * dark_factor (~0.85)
            → photographic regions only; caption text largely disappears.
            → small closing kernel; one component per photo.

  bw_full : Otsu binarization
            → all ink including caption text; used for caption detection.

The dark-factor is auto-adaptive: Otsu's threshold sits around 140 across
pages of widely different median brightness, so 0.85 * otsu lands at
~117–124, which empirically separates photos from caption text.

Public API
----------
    Params(...)
    ExtractResult(page_index, photos, captions, saved_files, notes)
    extract_images(src_pdf, out_dir, *, dpi, with_captions, use_body_crop,
                   params, zip_path, verbose, on_page) -> list[ExtractResult]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import zipfile

import cv2
import fitz
import numpy as np


# ------------------------------------------------------------------ Params

@dataclass
class Params:
    """Tunable thresholds for photo / caption segmentation.

    Area / length values that scale with the page are expressed as fractions
    of page dimensions so the same defaults work at any DPI.
    """
    dpi: int = 400

    # Photo detection (bw_dark)
    dark_factor: float = 0.85
    close_kernel_frac: float = 0.0015
    close_iters: int = 1
    min_area_frac: float = 0.02
    max_area_frac: float = 0.85
    min_fill_ratio: float = 0.35
    min_dim_frac: float = 0.10
    max_aspect_ratio: float = 4.0
    expand_after_detection: bool = True
    edge_walk_max_frac: float = 0.10
    edge_walk_stop_white_run: int = 15

    # Caption detection (bw_full)
    caption_search_px: int = 600
    caption_min_white_above: int = 5
    caption_text_min_density: float = 0.005
    caption_text_max_density: float = 0.50
    caption_end_white_run: int = 35
    caption_horizontal_expand: bool = True
    caption_horizontal_white_run: int = 30
    caption_try_above: bool = True
    caption_min_total_height: int = 15
    caption_min_peak_density: float = 0.10

    # Output framing
    photo_pad_px: int = 10
    caption_pad_px: int = 8


# ------------------------------------------------------------------ Result

@dataclass
class ExtractResult:
    page_index: int
    photos: list = field(default_factory=list)
    captions: list = field(default_factory=list)
    saved_files: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:
        n_cap = sum(1 for c in self.captions if c is not None)
        return (f"ExtractResult(page={self.page_index}, "
                f"photos={len(self.photos)}, captions={n_cap})")


# ----------------------------------------------------------------- internals

def _render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.height, pix.width, pix.n)


def _build_masks(rgb: np.ndarray, p: Params):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    otsu_t, bw_full = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw_dark = (gray < otsu_t * p.dark_factor).astype(np.uint8) * 255
    return bw_dark, bw_full, otsu_t


def _detect_photo_bboxes(bw_dark: np.ndarray, p: Params, notes: list):
    H, W = bw_dark.shape
    page_area = H * W

    k_size = max(3, int(p.close_kernel_frac * H))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(bw_dark, cv2.MORPH_CLOSE, kernel,
                              iterations=p.close_iters)

    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    photos = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        area_frac = area / page_area
        if area_frac < p.min_area_frac:
            continue
        if area_frac > p.max_area_frac:
            notes.append(f"reject c{i}: area_frac={area_frac:.2f} (full-page blob)")
            continue
        fill = area / (w * h) if w * h else 0.0
        if fill < p.min_fill_ratio:
            notes.append(f"reject c{i}: fill={fill:.2f} (sparse — text?)")
            continue
        if min(w, h) / min(W, H) < p.min_dim_frac:
            notes.append(f"reject c{i}: thin (min_dim_frac)")
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > p.max_aspect_ratio:
            notes.append(f"reject c{i}: aspect={aspect:.1f}")
            continue
        photos.append((int(x), int(y), int(w), int(h)))
        notes.append(f"keep c{i}: bbox=({x},{y},{w},{h}) "
                     f"area_frac={area_frac:.3f} fill={fill:.2f}")

    def _sort_key(b):
        x, y, w, h = b
        return (y // max(1, H // 20), x)
    photos.sort(key=_sort_key)
    return photos


def _expand_photo_bbox(bbox, bw_dark: np.ndarray, p: Params,
                       bw_full: Optional[np.ndarray] = None):
    H, W = bw_dark.shape
    x, y, w, h = bbox
    max_v = int(p.edge_walk_max_frac * H)
    max_h = int(p.edge_walk_max_frac * W)
    stop_run = p.edge_walk_stop_white_run
    mask = bw_full if bw_full is not None else bw_dark

    def _empty(arr):
        return (arr > 0).sum() / max(1, arr.size) < 0.005

    for direction in ("top", "bottom", "left", "right"):
        empty_run = 0
        for _ in range(max_v if direction in ("top", "bottom") else max_h):
            if direction == "top":
                if y - 1 < 0: break
                row = mask[y - 1, x:x + w]
                if _empty(row): empty_run += 1
                else: empty_run = 0
                if empty_run >= stop_run: break
                y -= 1; h += 1
            elif direction == "bottom":
                if y + h >= H: break
                row = mask[y + h, x:x + w]
                if _empty(row): empty_run += 1
                else: empty_run = 0
                if empty_run >= stop_run: break
                h += 1
            elif direction == "left":
                if x - 1 < 0: break
                col = mask[y:y + h, x - 1]
                if _empty(col): empty_run += 1
                else: empty_run = 0
                if empty_run >= stop_run: break
                x -= 1; w += 1
            else:  # right
                if x + w >= W: break
                col = mask[y:y + h, x + w]
                if _empty(col): empty_run += 1
                else: empty_run = 0
                if empty_run >= stop_run: break
                w += 1
        # trim trailing empty rows/cols
        if direction == "top" and empty_run:
            y += empty_run; h -= empty_run
        elif direction == "bottom" and empty_run:
            h -= empty_run
        elif direction == "left" and empty_run:
            x += empty_run; w -= empty_run
        elif direction == "right" and empty_run:
            w -= empty_run

    return (x, y, w, h)


def _scan_caption_band(row_density: np.ndarray, p: Params,
                       reverse: bool = False):
    seq = list(enumerate(row_density))
    if reverse:
        seq = list(reversed(seq))

    cap_start = cap_end = None
    consec_white = 0
    saw_white = False

    def _passes(s, e):
        if s is None or e is None:
            return False
        lo, hi = (min(s, e), max(s, e))
        if hi - lo < p.caption_min_total_height:
            return False
        sub = row_density[lo:hi + 1]
        return sub.size > 0 and sub.max() >= p.caption_min_peak_density

    for _, (i, d) in enumerate(seq):
        is_text = p.caption_text_min_density < d < p.caption_text_max_density
        if is_text:
            if cap_start is None and (saw_white or consec_white >= p.caption_min_white_above):
                cap_start = i
            cap_end = i
            consec_white = 0
        else:
            if cap_start is not None:
                consec_white += 1
                if consec_white >= p.caption_end_white_run:
                    if _passes(cap_start, cap_end):
                        break
                    cap_start = cap_end = None
                    saw_white = True
            else:
                consec_white += 1
                if consec_white >= p.caption_min_white_above:
                    saw_white = True

    if not _passes(cap_start, cap_end):
        return None, None
    if cap_start > cap_end:
        cap_start, cap_end = cap_end, cap_start
    return cap_start, cap_end + 1


def _expand_caption_horizontally(bw_full: np.ndarray, cy: int, ch: int,
                                 px: int, pw: int, p: Params, other_photos):
    H, W = bw_full.shape
    cap_strip = bw_full[cy:cy + ch, :]
    if cap_strip.size == 0:
        return px, pw
    col_density = cap_strip.sum(axis=0) / (255.0 * ch)

    left_bound, right_bound = 0, W
    for ox, oy, ow, oh in other_photos:
        if oy + oh < cy or oy > cy + ch:
            continue
        if ox + ow <= px and ox + ow > left_bound:
            left_bound = ox + ow
        if ox >= px + pw and ox < right_bound:
            right_bound = ox

    run_lim = p.caption_horizontal_white_run
    x_left = px
    white_run = 0
    for k in range(px - 1, left_bound - 1, -1):
        if col_density[k] < p.caption_text_min_density:
            white_run += 1
            if white_run >= run_lim:
                break
        else:
            x_left = k
            white_run = 0

    x_right = px + pw
    white_run = 0
    for k in range(px + pw, right_bound):
        if col_density[k] < p.caption_text_min_density:
            white_run += 1
            if white_run >= run_lim:
                break
        else:
            x_right = k + 1
            white_run = 0

    cx = max(0, x_left)
    cw = min(W, x_right) - cx
    return (cx, cw) if cw >= pw else (px, pw)


def _detect_caption(bw_full: np.ndarray, photo_bbox, p: Params,
                    notes: list, other_photos, claimed_regions=None):
    H, W = bw_full.shape
    px, py, pw, ph = photo_bbox
    bottom = py + ph
    claimed_regions = claimed_regions or []

    def _overlaps(x0, y0, x1, y1):
        return any(y0 < cy1 and y1 > cy0 and x0 < cx1 and x1 > cx0
                   for cx0, cy0, cx1, cy1 in claimed_regions)

    # --- below ---
    search_end = min(H, bottom + p.caption_search_px)
    for ox, oy, ow, oh in other_photos:
        if max(0, min(ox + ow, px + pw) - max(ox, px)) < 0.2 * min(ow, pw):
            continue
        if oy > bottom and oy < search_end:
            search_end = oy

    if search_end - bottom >= 20:
        band = bw_full[bottom:search_end, px:px + pw]
        if band.size:
            row_density = band.sum(axis=1) / (255.0 * band.shape[1])
            s, e = _scan_caption_band(row_density, p)
            if s is not None and (e - s) >= 10:
                cy, ch = bottom + s, e - s
                cx, cw = (px, pw)
                if p.caption_horizontal_expand:
                    cx, cw = _expand_caption_horizontally(
                        bw_full, cy, ch, px, pw, p, other_photos)
                if not _overlaps(cx, cy, cx + cw, cy + ch):
                    notes.append(f"  caption(below): y=[{cy}..{cy+ch}]")
                    return (cx, cy, cw, ch)

    # --- above (fallback) ---
    if not p.caption_try_above:
        notes.append(f"  caption: none for photo at y={py}")
        return None

    search_start = max(0, py - p.caption_search_px)
    for ox, oy, ow, oh in other_photos:
        if max(0, min(ox + ow, px + pw) - max(ox, px)) < 0.2 * min(ow, pw):
            continue
        bot = oy + oh
        if bot < py and bot > search_start:
            search_start = bot
    for cx0, cy0, cx1, cy1 in claimed_regions:
        if cy1 <= py and cy1 > search_start:
            if max(0, min(cx1, px + pw) - max(cx0, px)) > 0:
                search_start = cy1

    if py - search_start < 20:
        return None

    band = bw_full[search_start:py, px:px + pw]
    if not band.size:
        return None
    row_density = band.sum(axis=1) / (255.0 * band.shape[1])
    s, e = _scan_caption_band(row_density, p, reverse=True)
    if s is None or (e - s) < 10:
        notes.append(f"  caption: none for photo at y={py}")
        return None

    cy, ch = search_start + s, e - s
    cx, cw = (px, pw)
    if p.caption_horizontal_expand:
        cx, cw = _expand_caption_horizontally(
            bw_full, cy, ch, px, pw, p, other_photos)
    if _overlaps(cx, cy, cx + cw, cy + ch):
        return None
    notes.append(f"  caption(above): y=[{cy}..{cy+ch}]")
    return (cx, cy, cw, ch)


def _crop_with_pad(rgb: np.ndarray, bbox, pad: int) -> np.ndarray:
    H, W = rgb.shape[:2]
    x, y, w, h = bbox
    return rgb[max(0, y - pad):min(H, y + h + pad),
               max(0, x - pad):min(W, x + w + pad)]


def _save_image(rgb: np.ndarray, out_path: Path, dpi: int) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    cv2.imwrite(str(out_path), bgr)
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(out_path) as _im:
            _im.save(out_path, dpi=(dpi, dpi))
    except Exception:
        pass


def _body_clip(page: fitz.Page, p: Params, notes: list):
    """Return (x, y, w, h) in pixel coords using header_footer margin
    detection, or None to use the full page."""
    try:
        from header_footer import detect_margins, Params as HFParams
    except ImportError:
        notes.append("body-crop: header_footer not importable — using full page")
        return None
    m = detect_margins(page, HFParams(dpi=p.dpi))
    x0 = int(getattr(m, "keep_left", 0) or 0)
    x1 = int(getattr(m, "keep_right", 0) or 0)
    y0 = int(getattr(m, "keep_top",  0) or 0)
    y1 = int(getattr(m, "keep_bottom", 0) or 0)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


# ----------------------------------------------------------------- public API

def extract_images(
    src_pdf,
    out_dir,
    *,
    dpi: int = 400,
    with_captions: bool = True,
    use_body_crop: bool = True,
    params: Optional[Params] = None,
    zip_path=None,
    verbose: bool = False,
    on_page=None,
) -> list[ExtractResult]:
    """Extract photographic regions from a scanned PDF.

    Parameters
    ----------
    src_pdf        Path to source PDF (use the original, not the stripped copy).
    out_dir        Directory for output PNGs (created if missing).
    dpi            Render resolution; 400 matches the page_export default.
    with_captions  Also save the caption strip for each photo.
    use_body_crop  Limit segmentation to the body region detected by
                   header_footer.detect_margins (falls back to full page).
    params         Params instance for tuning; dpi= kwarg takes precedence.
    zip_path       If given, bundle all output files into a ZIP here.
    verbose        Populate per-page diagnostic notes.
    on_page        Callable(page_num: int, total: int) called after each page.

    Returns
    -------
    list[ExtractResult] — one record per page.
    """
    p = params or Params(dpi=dpi)
    if dpi != p.dpi:
        p = Params(**{**p.__dict__, "dpi": dpi})

    src_pdf = Path(src_pdf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExtractResult] = []
    saved_paths: list[Path] = []

    with fitz.open(src_pdf) as doc:
        total = doc.page_count
        for pi, page in enumerate(doc):
            res = ExtractResult(page_index=pi)
            notes = res.notes if verbose else []

            rgb = _render_page(page, p.dpi)
            H, W = rgb.shape[:2]

            body = _body_clip(page, p, notes) if use_body_crop else None
            if body is not None:
                bx, by, bw_, bh_ = body
                bx = max(0, min(W - 1, bx)); by = max(0, min(H - 1, by))
                bw_ = max(1, min(W - bx, bw_)); bh_ = max(1, min(H - by, bh_))
                work_rgb = rgb[by:by + bh_, bx:bx + bw_]
                origin = (bx, by)
                notes.append(f"body crop: x=[{bx}..{bx+bw_}] y=[{by}..{by+bh_}]")
            else:
                work_rgb = rgb
                origin = (0, 0)

            bw_dark, bw_full, otsu_t = _build_masks(work_rgb, p)
            notes.append(f"otsu_t={otsu_t:.0f} dark_t={otsu_t * p.dark_factor:.0f}")

            photos = _detect_photo_bboxes(bw_dark, p, notes)
            if p.expand_after_detection:
                photos = [_expand_photo_bbox(b, bw_dark, p, bw_full) for b in photos]

            captions: list = []
            claimed: list = []
            for i, ph in enumerate(photos):
                cap = None
                if with_captions:
                    others = [o for j, o in enumerate(photos) if j != i]
                    cap = _detect_caption(bw_full, ph, p, notes, others,
                                          claimed_regions=claimed)
                captions.append(cap)
                if cap is not None:
                    cx, cy, cw, ch = cap
                    claimed.append((cx, cy, cx + cw, cy + ch))

            ox, oy = origin
            page_label = f"page{pi + 1:03d}"
            for fig_i, (ph, cap) in enumerate(zip(photos, captions), start=1):
                px_, py_, pw_, phh_ = ph
                full_ph = (px_ + ox, py_ + oy, pw_, phh_)
                fig_stem = f"{page_label}_fig{fig_i:02d}"

                photo_img = _crop_with_pad(rgb, full_ph, p.photo_pad_px)
                photo_path = out_dir / f"{fig_stem}.png"
                _save_image(photo_img, photo_path, p.dpi)
                res.saved_files.append(photo_path)
                saved_paths.append(photo_path)
                res.photos.append(full_ph)

                if cap is not None:
                    cx, cy, cw, ch = cap
                    full_cap = (cx + ox, cy + oy, cw, ch)
                    cap_img = _crop_with_pad(rgb, full_cap, p.caption_pad_px)
                    cap_path = out_dir / f"{fig_stem}_caption.png"
                    _save_image(cap_img, cap_path, p.dpi)
                    res.saved_files.append(cap_path)
                    saved_paths.append(cap_path)
                    res.captions.append(full_cap)
                else:
                    res.captions.append(None)

            results.append(res)
            if on_page:
                on_page(pi + 1, total)

    if zip_path is not None and saved_paths:
        zp = Path(zip_path)
        zp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in saved_paths:
                zf.write(f, arcname=f.name)

    return results
