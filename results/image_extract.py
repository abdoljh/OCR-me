"""
image_extract.py - Phase 1b of the Arabic-OCR pipeline.

Extracts photographic regions (and optionally their captions) from
scanned Arabic book pages. Companion to header_footer_v3.py and
page_export.py.

Design
------
Scanned book pages contain N photographs surrounded by Arabic
captions. Each PDF page is one big embedded JPEG, so PyMuPDF's
get_images() returns the whole page-scan, useless for extracting
individual photos. We segment in pixel space.

Key insight: photos and captions live in different intensity ranges.
Captions are mid-gray ink on white; photographic regions contain
large swaths of very dark pixels (deep shadows, dark clothing,
black-and-white tones). Otsu binarization conflates them. So we use
two masks:

  bw_dark : gray < OTSU * dark_factor    (~0.85)
            -> photographic regions only; caption text largely
               disappears.
            -> small closing kernel; one component per photo.

  bw_full : Otsu binarization
            -> all ink including caption text.
            -> used for caption detection.

The dark-factor approach is auto-adaptive: Otsu's threshold sits
around 140 across pages of widely different median brightness, so
0.85 * otsu lands at ~117-124 on every test page, which is the
empirically right cutoff for separating photos from caption text.

Pipeline per page:
  1. Render page at high DPI (default 400).
  2. Optionally crop to body region using header_footer_v3.
  3. Build bw_dark and bw_full.
  4. Connected components on bw_dark -> photo candidates.
  5. Filter by area, fill ratio, dimensions, aspect.
  6. Walk each photo's edges outward to recover faded boundaries.
  7. For each photo, scan downward in bw_full to find caption band.
     Allow caption to widen horizontally beyond the photo's column.
  8. Save photo and (optionally) caption with shared stem:
        page03_fig02.png
        page03_fig02_caption.png

API
---
extract_images(src_pdf, out_dir, *, dpi=400, with_captions=True,
               use_body_crop=True, params=Params(), zip_path=None,
               verbose=False) -> list[ExtractResult]

Always segments. Even when get_images() returns a "real" embedded
image, its colorspace, orientation, or clipping may differ from what
is visually displayed. Pixel-domain extraction is what the eye sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import zipfile

import cv2
import fitz  # PyMuPDF
import numpy as np


# ----------------------------------------------------------------- Params

@dataclass
class Params:
    """Tunable thresholds for photo / caption segmentation.

    Area / length values that should scale with the page are expressed
    as fractions of page dimensions, so the same defaults work at any
    DPI. Pixel-valued knobs are calibrated for 300-400 DPI scans.
    """

    # Render
    dpi: int = 400

    # Photo detection (operates on bw_dark)
    dark_factor: float = 0.85
    close_kernel_frac: float = 0.0015
    close_iters: int = 1
    min_area_frac: float = 0.02
    max_area_frac: float = 0.85
    min_fill_ratio: float = 0.35
    min_dim_frac: float = 0.10
    max_aspect_ratio: float = 4.0
    expand_after_detection: bool = True
    edge_walk_max_frac: float = 0.10    # max walk distance (frac of page)
    edge_walk_stop_white_run: int = 15  # >=15 truly-empty rows on bw_full
                                        # marks the photo's real edge.

    # Caption detection (operates on bw_full)
    caption_search_px: int = 600
    caption_min_white_above: int = 5
    caption_text_min_density: float = 0.005
    caption_text_max_density: float = 0.50
    caption_end_white_run: int = 35
    caption_horizontal_expand: bool = True
    caption_horizontal_white_run: int = 30  # px of whitespace that ends a horizontal walk
    caption_try_above: bool = True   # if no caption below, look above the photo
    caption_min_total_height: int = 15   # reject sub-15px "captions" — likely
                                         # photo-edge artifacts.
    caption_min_peak_density: float = 0.10  # at least one row of the band must
                                            # exceed this, or it's not text.

    # Output framing
    photo_pad_px: int = 10
    caption_pad_px: int = 8


# ----------------------------------------------------------------- Result

@dataclass
class ExtractResult:
    """Per-page extraction record."""
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

def _render_page(page, dpi: int) -> np.ndarray:
    """Render page to RGB ndarray. Cropbox-aware (the PyMuPDF default
    render respects whatever cropbox is set; passing clip= would
    double-clip, the lesson from the page_export bug)."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.height, pix.width, pix.n)


def _build_masks(rgb: np.ndarray, p: Params):
    """Return (bw_dark, bw_full, otsu_threshold).

    bw_dark : strict; gray < otsu * dark_factor. Photos survive
              cleanly, caption text mostly disappears.
    bw_full : Otsu. All ink. Used for caption detection.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    otsu_t, bw_full = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_t = otsu_t * p.dark_factor
    bw_dark = (gray < dark_t).astype(np.uint8) * 255
    return bw_dark, bw_full, otsu_t


def _detect_photo_bboxes(bw_dark: np.ndarray, p: Params, notes: list):
    """Connected components on the dark mask. Returns photo bboxes
    sorted top-to-bottom, then left-to-right within row band."""
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

        bbox_area = w * h
        fill = area / bbox_area if bbox_area else 0.0
        if fill < p.min_fill_ratio:
            notes.append(f"reject c{i}: fill={fill:.2f} (sparse - text)")
            continue

        if min(w, h) / min(W, H) < p.min_dim_frac:
            notes.append(f"reject c{i}: thin (min_dim_frac)")
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > p.max_aspect_ratio:
            notes.append(f"reject c{i}: aspect={aspect:.1f}")
            continue

        photos.append((int(x), int(y), int(w), int(h)))
        notes.append(f"keep   c{i}: bbox=({x},{y},{w},{h}) "
                     f"area_frac={area_frac:.3f} fill={fill:.2f}")

    # Sort: bucket by row band, then x within each band.
    def sort_key(b):
        x, y, w, h = b
        row_bucket = y // max(1, H // 20)
        return (row_bucket, x)
    photos.sort(key=sort_key)
    return photos


def _expand_photo_bbox(bbox, bw_dark: np.ndarray, p: Params,
                       bw_full: Optional[np.ndarray] = None):
    """Walk outward from each edge to recover gradually-fading photo
    boundaries that the strict dark mask cuts off (white robes, sky,
    pale backgrounds). The walker uses ``bw_full`` (Otsu — captures
    everything, including faint photo midtones) and stops at a real
    whitespace gap: a contiguous run of truly-empty rows/columns. Real
    inter-element whitespace on a scanned page is sharply empty on
    ``bw_full``; photo content — even faint — is not. So the natural
    stopping signal is "saw 15+ empty rows in a row". Caption text
    rows are not empty, but the gap between photo and caption is.

    Falls back to ``bw_dark`` if ``bw_full`` is not provided.
    """
    H, W = bw_dark.shape
    x, y, w, h = bbox
    max_v = int(p.edge_walk_max_frac * H)
    max_h = int(p.edge_walk_max_frac * W)
    stop_run = p.edge_walk_stop_white_run
    mask = bw_full if bw_full is not None else bw_dark

    def empty_row(arr):
        return (arr > 0).sum() / max(1, arr.size) < 0.005

    # Walk top edge upward
    walked = 0
    empty_run = 0
    while walked < max_v and y - 1 >= 0:
        row = mask[y - 1, x:x + w]
        if empty_row(row):
            empty_run += 1
            if empty_run >= stop_run:
                break
        else:
            empty_run = 0
        y -= 1
        h += 1
        walked += 1
    # Trim back the empty rows we accumulated at the very edge
    if empty_run > 0:
        y += empty_run
        h -= empty_run

    # Bottom
    walked = 0
    empty_run = 0
    while walked < max_v and y + h < H:
        row = mask[y + h, x:x + w]
        if empty_row(row):
            empty_run += 1
            if empty_run >= stop_run:
                break
        else:
            empty_run = 0
        h += 1
        walked += 1
    if empty_run > 0:
        h -= empty_run

    # Left
    walked = 0
    empty_run = 0
    while walked < max_h and x - 1 >= 0:
        col = mask[y:y + h, x - 1]
        if empty_row(col):
            empty_run += 1
            if empty_run >= stop_run:
                break
        else:
            empty_run = 0
        x -= 1
        w += 1
        walked += 1
    if empty_run > 0:
        x += empty_run
        w -= empty_run

    # Right
    walked = 0
    empty_run = 0
    while walked < max_h and x + w < W:
        col = mask[y:y + h, x + w]
        if empty_row(col):
            empty_run += 1
            if empty_run >= stop_run:
                break
        else:
            empty_run = 0
        w += 1
        walked += 1
    if empty_run > 0:
        w -= empty_run

    return (x, y, w, h)


def _scan_caption_band(row_density: np.ndarray, p: Params,
                       reverse: bool = False):
    """Walk through row_density looking for a caption band: a region
    of mid-density rows preceded by whitespace, followed by a long
    whitespace run.

    Returns (start, end) indices into row_density, or (None, None).
    Rejects spurious short blips by requiring that any reported band
    has at least ``caption_min_total_height`` rows whose max density
    is at least ``caption_min_peak_density``.
    """
    seq = list(enumerate(row_density))
    if reverse:
        seq = list(reversed(seq))

    cap_start = None
    cap_end = None
    consec_white = 0
    saw_white = False

    def _band_passes(s, e):
        if s is None or e is None:
            return False
        if reverse:
            lo, hi = (e, s) if s > e else (s, e)
        else:
            lo, hi = (s, e) if s <= e else (e, s)
        if hi - lo < p.caption_min_total_height:
            return False
        sub = row_density[lo:hi + 1]
        if sub.size == 0:
            return False
        if sub.max() < p.caption_min_peak_density:
            return False
        return True

    for k, (i, d) in enumerate(seq):
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
                    if _band_passes(cap_start, cap_end):
                        break
                    # Reject this band; reset and keep scanning
                    cap_start = None
                    cap_end = None
                    saw_white = True
            else:
                consec_white += 1
                if consec_white >= p.caption_min_white_above:
                    saw_white = True

    if cap_start is None or cap_end is None:
        return None, None
    if not _band_passes(cap_start, cap_end):
        return None, None

    if cap_start > cap_end:
        cap_start, cap_end = cap_end, cap_start

    return cap_start, cap_end + 1


def _expand_caption_horizontally(bw_full: np.ndarray, cy: int, ch: int,
                                 px: int, pw: int, p: Params,
                                 other_photos):
    """Walk left/right from the photo's column at the caption's
    vertical band, until a sustained whitespace run or another photo.
    Returns (cx, cw)."""
    H, W = bw_full.shape
    cap_strip = bw_full[cy:cy + ch, :]
    if cap_strip.size == 0:
        return px, pw
    col_density = cap_strip.sum(axis=0) / (255.0 * ch)

    left_bound = 0
    right_bound = W
    for ox, oy, ow, oh in other_photos:
        if oy + oh < cy or oy > cy + ch:
            continue  # photo doesn't vertically overlap caption
        if ox + ow <= px and ox + ow > left_bound:
            left_bound = ox + ow
        if ox >= px + pw and ox < right_bound:
            right_bound = ox

    run_lim = p.caption_horizontal_white_run

    x_left = px
    white_run = 0
    k = px - 1
    while k >= left_bound:
        if col_density[k] < p.caption_text_min_density:
            white_run += 1
            if white_run >= run_lim:
                break
        else:
            x_left = k
            white_run = 0
        k -= 1

    x_right = px + pw
    white_run = 0
    k = px + pw
    while k < right_bound:
        if col_density[k] < p.caption_text_min_density:
            white_run += 1
            if white_run >= run_lim:
                break
        else:
            x_right = k + 1
            white_run = 0
        k += 1

    cx = max(0, x_left)
    cw = min(W, x_right) - cx
    if cw < pw:
        cx, cw = px, pw
    return cx, cw


def _detect_caption(bw_full: np.ndarray, photo_bbox, p: Params,
                    notes: list, other_photos,
                    claimed_regions=None):
    """For a given photo bbox, find the caption band.

    First try directly below the photo (the common case). If nothing
    plausible is found and ``caption_try_above`` is set, try directly
    above. The above-pass is bounded by the next photo upward (or the
    top of the page region) so it cannot claim text from elsewhere.

    ``claimed_regions`` is a list of (x0, y0, x1, y1) rectangles
    already assigned to other photos as captions. Candidate captions
    overlapping these (in both x and y) are skipped.
    """
    H, W = bw_full.shape
    px, py, pw, ph = photo_bbox
    bottom = py + ph

    if claimed_regions is None:
        claimed_regions = []

    def _overlaps_claimed(x0, y0, x1, y1):
        for cx0, cy0, cx1, cy1 in claimed_regions:
            if y0 < cy1 and y1 > cy0 and x0 < cx1 and x1 > cx0:
                return True
        return False

    # ---- below pass ---------------------------------------------------
    search_end = min(H, bottom + p.caption_search_px)
    for ox, oy, ow, oh in other_photos:
        col_overlap = max(0, min(ox + ow, px + pw) - max(ox, px))
        if col_overlap < 0.2 * min(ow, pw):
            continue  # different column
        if oy > bottom and oy < search_end:
            search_end = oy

    if search_end - bottom >= 20:
        band = bw_full[bottom:search_end, px:px + pw]
        if band.size > 0:
            row_density = band.sum(axis=1) / (255.0 * band.shape[1])
            s, e = _scan_caption_band(row_density, p, reverse=False)
            if s is not None and (e - s) >= 10:
                cy = bottom + s
                ch = e - s
                cx, cw = (px, pw)
                if p.caption_horizontal_expand:
                    cx, cw = _expand_caption_horizontally(
                        bw_full, cy, ch, px, pw, p, other_photos)
                if not _overlaps_claimed(cx, cy, cx + cw, cy + ch):
                    notes.append(f"  caption(below): x=[{cx}..{cx+cw}] y=[{cy}..{cy+ch}] h={ch}")
                    return (cx, cy, cw, ch)
                else:
                    notes.append(f"  caption(below): claimed elsewhere y=[{cy}..{cy+ch}]")

    # ---- above pass (fallback) ---------------------------------------
    if not p.caption_try_above:
        notes.append(f"  caption: none under photo at y={py}")
        return None

    search_start = max(0, py - p.caption_search_px)
    for ox, oy, ow, oh in other_photos:
        col_overlap = max(0, min(ox + ow, px + pw) - max(ox, px))
        if col_overlap < 0.2 * min(ow, pw):
            continue
        bottom_of_other = oy + oh
        if bottom_of_other < py and bottom_of_other > search_start:
            search_start = bottom_of_other

    # Bound search_start by claimed caption regions above this photo
    # (only those that share x-extent)
    for cx0, cy0, cx1, cy1 in claimed_regions:
        if cy1 <= py and cy1 > search_start:
            x_overlap = max(0, min(cx1, px + pw) - max(cx0, px))
            if x_overlap > 0:
                search_start = cy1

    if py - search_start < 20:
        notes.append(f"  caption: none for photo at y={py} (no space above either)")
        return None

    band = bw_full[search_start:py, px:px + pw]
    if band.size == 0:
        notes.append(f"  caption: none for photo at y={py}")
        return None
    row_density = band.sum(axis=1) / (255.0 * band.shape[1])
    s, e = _scan_caption_band(row_density, p, reverse=True)
    if s is None or (e - s) < 10:
        notes.append(f"  caption: none for photo at y={py}")
        return None

    cy = search_start + s
    ch = e - s
    cx, cw = (px, pw)
    if p.caption_horizontal_expand:
        cx, cw = _expand_caption_horizontally(
            bw_full, cy, ch, px, pw, p, other_photos)
    if _overlaps_claimed(cx, cy, cx + cw, cy + ch):
        notes.append(f"  caption(above): claimed elsewhere y=[{cy}..{cy+ch}]")
        return None
    notes.append(f"  caption(above): x=[{cx}..{cx+cw}] y=[{cy}..{cy+ch}] h={ch}")
    return (cx, cy, cw, ch)


def _crop_with_pad(rgb: np.ndarray, bbox, pad: int) -> np.ndarray:
    H, W = rgb.shape[:2]
    x, y, w, h = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return rgb[y0:y1, x0:x1]


def _save_image(rgb: np.ndarray, out_path: Path, dpi: int) -> None:
    """Save RGB ndarray as PNG. Stamp DPI metadata via PIL because cv2
    doesn't write that into PNG headers."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    cv2.imwrite(str(out_path), bgr)
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(out_path) as _im:
            _im.save(out_path, dpi=(dpi, dpi))
    except Exception:
        pass  # PIL is a soft dependency for metadata only


def _maybe_get_body_clip(page, p: Params, use_body_crop: bool, notes: list):
    """If header_footer_v3 is importable, return (x, y, w, h) of body
    region in pixel coords at our render DPI. The detector renders at
    its own DPI internally; we point it at our DPI so coords align."""
    if not use_body_crop:
        return None
    try:
        from header_footer_v3 import detect_margins, Params as HFParams
    except ImportError:
        notes.append("body-crop: header_footer_v3 not importable - using full page")
        return None

    hf_params = HFParams(dpi=p.dpi)
    m = detect_margins(page, hf_params)
    if getattr(m, "keep_y0", None) is None:
        return None
    x0 = getattr(m, "keep_x0", 0) or 0
    x1 = getattr(m, "keep_x1", None) or 0
    y0 = m.keep_y0
    y1 = m.keep_y1
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


# ----------------------------------------------------------------- public API

def extract_images(src_pdf, out_dir, *, dpi: int = 400,
                   with_captions: bool = True,
                   use_body_crop: bool = True,
                   params: Optional[Params] = None,
                   zip_path=None,
                   verbose: bool = False):
    """Extract photographic regions from a scanned PDF.

    Parameters
    ----------
    src_pdf
        Path to source PDF (typically the original - we always do
        pixel-domain segmentation).
    out_dir
        Directory for output PNGs. Created if missing.
    dpi
        Render resolution. 400 matches the page_export default.
    with_captions
        If True, also save the caption strip below each photo as
        ``<stem>_caption.png``. Photo and caption share a common stem
        for easy pairing (``glob('*_caption.png')``).
    use_body_crop
        If True, run header_footer_v3 first and limit segmentation to
        the detected body region. Falls back gracefully if v3 is not
        importable.
    params
        Params instance for tuning.
    zip_path
        If given, also bundle all output files into a zip.
    verbose
        Populate per-page diagnostic notes.

    Returns
    -------
    list[ExtractResult] : one record per page.
    """
    p = params or Params(dpi=dpi)
    if dpi != p.dpi:
        # caller passed dpi explicitly; honor it over Params.dpi
        p = Params(**{**p.__dict__, "dpi": dpi})

    src_pdf = Path(src_pdf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    saved_paths = []

    with fitz.open(src_pdf) as doc:
        for pi, page in enumerate(doc):
            res = ExtractResult(page_index=pi)
            notes = res.notes if verbose else []

            rgb = _render_page(page, p.dpi)
            H, W = rgb.shape[:2]

            body = _maybe_get_body_clip(page, p, use_body_crop, notes)
            if body is not None:
                bx, by, bw_, bh_ = body
                bx = max(0, min(W - 1, bx))
                by = max(0, min(H - 1, by))
                bw_ = max(1, min(W - bx, bw_))
                bh_ = max(1, min(H - by, bh_))
                work_rgb = rgb[by:by + bh_, bx:bx + bw_]
                origin = (bx, by)
                if verbose:
                    notes.append(f"body crop: x=[{bx}..{bx+bw_}] y=[{by}..{by+bh_}]")
            else:
                work_rgb = rgb
                origin = (0, 0)

            bw_dark, bw_full, otsu_t = _build_masks(work_rgb, p)
            if verbose:
                notes.append(f"otsu_t={otsu_t:.0f} dark_t={otsu_t * p.dark_factor:.0f}")

            photos = _detect_photo_bboxes(bw_dark, p, notes)
            if p.expand_after_detection:
                photos = [_expand_photo_bbox(b, bw_dark, p, bw_full)
                          for b in photos]

            captions = []
            claimed = []
            for i, ph in enumerate(photos):
                if with_captions:
                    others = [o for j, o in enumerate(photos) if j != i]
                    cap = _detect_caption(bw_full, ph, p, notes, others,
                                          claimed_regions=claimed)
                else:
                    cap = None
                captions.append(cap)
                if cap is not None:
                    cx, cy, cw, ch = cap
                    claimed.append((cx, cy, cx + cw, cy + ch))

            ox, oy = origin
            page_label = f"page{pi+1:03d}"
            for i, (ph, cap) in enumerate(zip(photos, captions), start=1):
                px_, py_, pw_, phh_ = ph
                full_ph = (px_ + ox, py_ + oy, pw_, phh_)

                stem = f"{page_label}_fig{i:02d}"

                photo_img = _crop_with_pad(rgb, full_ph, p.photo_pad_px)
                photo_path = out_dir / f"{stem}.png"
                _save_image(photo_img, photo_path, p.dpi)
                res.saved_files.append(photo_path)
                saved_paths.append(photo_path)
                res.photos.append(full_ph)

                if cap is not None:
                    cx, cy, cw, ch = cap
                    full_cap = (cx + ox, cy + oy, cw, ch)
                    cap_img = _crop_with_pad(rgb, full_cap, p.caption_pad_px)
                    cap_path = out_dir / f"{stem}_caption.png"
                    _save_image(cap_img, cap_path, p.dpi)
                    res.saved_files.append(cap_path)
                    saved_paths.append(cap_path)
                    res.captions.append(full_cap)
                else:
                    res.captions.append(None)

            results.append(res)

    if zip_path is not None and saved_paths:
        zp = Path(zip_path)
        zp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in saved_paths:
                zf.write(f, arcname=f.name)

    return results
