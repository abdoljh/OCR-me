"""
Companion utilities for `header_footer_v3.py`.

Two operations, both built on top of the v3 detector:

1. `export_pages_as_images(in_pdf, out_dir, dpi=400, ...)` — render each page
   of a stripped PDF (i.e., with CropBox already set by `strip_pdf`) to a PNG
   at the requested DPI. Suited for downstream offline OCR.

2. `extract_footers_pdf(src_pdf, out_pdf, p=Params(), ...)` — re-detect each
   page's footer region in `src_pdf` (the ORIGINAL, uncropped PDF) and assemble
   a single PDF in which each output page contains exactly one source footer
   (rule + footnote text), labeled with the source page number. Pages without
   a detected footer are skipped.

Both helpers are independent of each other and call into header_footer_v3
without modifying it.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import fitz
import numpy as np

from header_footer_v3 import (
    Params,
    detect_margins,
    _render_gray,
)


# ---------------------------------------------------------------------------
# Task 1: export each (cropped) page as a high-resolution PNG, then zip.
# ---------------------------------------------------------------------------

def export_pages_as_images(
    in_pdf: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = 400,
    fmt: str = "png",
    zip_path: Optional[str | Path] = None,
) -> list[Path]:
    """
    Render every page of `in_pdf` to an image at `dpi`, save into `out_dir`,
    and (optionally) zip them.

    The rendering respects each page's CropBox, so when called on the output
    of `strip_pdf` the produced images are already header/footer-stripped.

    Parameters
    ----------
    in_pdf : path
        Source PDF (typically the output of `strip_pdf`).
    out_dir : path
        Directory to write `page_001.png` ... `page_NNN.png` into. Created if
        missing. Existing files with the same names are overwritten.
    dpi : int, default 400
        Render resolution. 400 DPI is a comfortable margin above the 300 DPI
        OCR floor; useful when a downstream engine wants extra detail.
    fmt : {"png", "tif", "jpg"}, default "png"
        Image format. PNG is lossless and is what Tesseract / Kraken / EasyOCR
        prefer. TIFF (".tif") is also lossless and what some scanners emit.
        JPEG is offered for completeness but is NOT recommended for OCR.
    zip_path : path or None
        If provided, all written images are bundled into a single zip at
        that path. The directory is left in place.

    Returns
    -------
    list[Path]
        The image paths written, in page order.
    """
    in_pdf = Path(in_pdf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().lstrip(".")
    if fmt not in {"png", "tif", "tiff", "jpg", "jpeg"}:
        raise ValueError(f"unsupported image format: {fmt!r}")
    ext = ".tif" if fmt in {"tif", "tiff"} else (".jpg" if fmt in {"jpg", "jpeg"} else ".png")

    doc = fitz.open(in_pdf)
    n = doc.page_count
    width = max(3, len(str(n)))  # zero-pad page numbers to allow lex sort
    written: list[Path] = []
    skipped: list[int] = []

    for i in range(n):
        page = doc.load_page(i)
        # Default rendering already respects the page's CropBox, so we do NOT
        # pass clip=page.cropbox here. Passing clip=cropbox on a page whose
        # CropBox is smaller than its MediaBox causes a double-clip — PyMuPDF
        # interprets the clip rect relative to the cropbox-translated origin,
        # which crops away ~3 lines at the top. (Verified empirically against
        # both A4 sample books at 300 and 400 DPI.)
        pix = page.get_pixmap(dpi=dpi)
        if pix.width == 0 or pix.height == 0:
            # Degenerate cropbox (can happen on a near-empty page where the
            # detector's keep-band collapsed). Skip rather than crash.
            skipped.append(i + 1)
            continue
        out_path = out_dir / f"page_{i + 1:0{width}d}{ext}"
        # Use cv2 for jpg/tif quality control, pixmap.save for png simplicity.
        if ext == ".png":
            pix.save(str(out_path))
        else:
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            params: list[int] = []
            if ext == ".jpg":
                params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            cv2.imwrite(str(out_path), arr, params)
        written.append(out_path)

    doc.close()

    if skipped:
        # Make the skip list discoverable without breaking the simple return type.
        print(f"  warning: skipped degenerate page(s) {skipped} in {in_pdf.name}")

    if zip_path is not None:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img in written:
                zf.write(img, arcname=img.name)

    return written


# ---------------------------------------------------------------------------
# Task 2: collect the detected footers into a single PDF.
# ---------------------------------------------------------------------------

def _footer_band_y(m, body_pad_px: int = 8) -> Optional[tuple[int, int]]:
    """
    Compute the inclusive (y0, y1) pixel band that the footer occupies on a
    page given a `PageMargins` from the v3 detector.

    Strategy
    --------
    - If `m.rule_y` is set: footer = [rule_y - small_pad, body_last_y]. We
      include a few pixels above the rule itself so the rule line is visible
      in the output and serves as a visual anchor.
    - Else if `m.footer_strip` is set: footer = the strip itself (a tight
      band; we add a small pad so descenders / dots aren't clipped).
    - Else: no footer detected → return None.

    `body_last_y` is the bottom of the source page (excluding the small
    safety margin used by detect_margins). To find a sensible bottom edge
    that excludes scanner artifacts, we use page_h - 2% of page height, which
    matches the artifact filter the detector itself uses internally.
    """
    if m.rule_y is not None:
        y0 = max(0, m.rule_y - body_pad_px)
        y1 = m.page_h - max(3, int(0.012 * m.page_h)) - 1
        if y1 <= y0:
            return None
        return (y0, y1)
    if m.footer_strip is not None:
        a, b = m.footer_strip
        y0 = max(0, a - body_pad_px)
        y1 = min(m.page_h - 1, b + body_pad_px)
        if y1 <= y0:
            return None
        return (y0, y1)
    return None


def extract_footers_pdf(
    src_pdf: str | Path,
    out_pdf: str | Path,
    *,
    p: Params = Params(),
    label_dpi_px: int = 48,
    label_margin_px: int = 24,
    img_dir: Optional[str | Path] = None,
    images_dpi: int = 400,
    images_fmt: str = "png",
    zip_path: Optional[str | Path] = None,
) -> list[int]:
    """
    Build a one-PDF compendium of the footers found in `src_pdf`.

    For each page that has a detectable footer (rule + footnote, or a footer
    strip), the corresponding pixel band of the ORIGINAL page is rasterized
    at `p.dpi`, prefixed with a small label band like "Page 3 — Sample.pdf",
    and inserted as a single page into the output PDF.

    Pages without a detected footer are skipped silently.

    Parameters
    ----------
    src_pdf, out_pdf : path
        Source PDF (the original, uncropped one) and the destination PDF for
        the bound-together footers.
    p : Params
        Detector params; reuses the same Params used for `strip_pdf`.
    label_dpi_px, label_margin_px : int
        Geometry of the small "Page N (filename)" header band rendered above
        each footer in the bound PDF. These do not affect the standalone
        images saved via `img_dir` (those are unlabeled — page numbers go in
        the filenames instead).
    img_dir : path or None, default None
        If provided, also save each detected footer as a standalone image
        (`footer_p001.png` etc.) into this directory at `images_dpi`. Useful
        when the footers themselves need OCR or post-processing.
    images_dpi : int, default 400
        Render resolution for the standalone images (matches the page-export
        default; the bound PDF uses `p.dpi` to keep itself lightweight).
    images_fmt : {"png", "tif", "jpg"}, default "png"
        Image format for standalone files. Same conventions as
        `export_pages_as_images`.
    zip_path : path or None, default None
        If provided, the standalone images are bundled into a zip at this
        path. Requires `img_dir` to be set as well.

    Returns
    -------
    list[int]
        1-based page numbers of source pages that contributed a footer.
    """
    src_pdf = Path(src_pdf)
    out_pdf = Path(out_pdf)
    src = fitz.open(src_pdf)
    out = fitz.open()
    contributed: list[int] = []

    # Validate / set up image-export side once before the page loop.
    saving_images = img_dir is not None
    if saving_images:
        img_dir = Path(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        fmt = images_fmt.lower().lstrip(".")
        if fmt not in {"png", "tif", "tiff", "jpg", "jpeg"}:
            raise ValueError(f"unsupported image format: {images_fmt!r}")
        ext = ".tif" if fmt in {"tif", "tiff"} else (".jpg" if fmt in {"jpg", "jpeg"} else ".png")
        # Pre-compute filename width once we know how many source pages exist.
        fname_width = max(3, len(str(src.page_count)))
    if zip_path is not None and not saving_images:
        raise ValueError("zip_path requires img_dir to also be set")
    saved_images: list[Path] = []

    for i, page in enumerate(src):
        m = detect_margins(page, p, verbose=False)
        m.page_index = i
        band = _footer_band_y(m)
        if band is None:
            continue
        y0, y1 = band

        # Render the page once at p.dpi for the bound PDF and slice the
        # footer band in pixel space — cheaper than a clip-rect re-render.
        gray = _render_gray(page, p.dpi)
        footer_img = gray[y0 : y1 + 1, :]
        h_strip, w_strip = footer_img.shape[:2]

        # Build a label strip with the source page number.
        label_h = label_dpi_px + 2 * label_margin_px
        label = np.full((label_h, w_strip), 255, dtype=np.uint8)
        label_text = f"Page {i + 1}  ({src_pdf.name})"
        cv2.putText(
            label,
            label_text,
            (label_margin_px, label_margin_px + label_dpi_px - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            0,
            2,
            cv2.LINE_AA,
        )
        # A thin separator line between label and footer.
        cv2.line(label, (0, label_h - 2), (w_strip - 1, label_h - 2), 0, 1)

        composite = np.vstack([label, footer_img])

        # Encode and insert as a new PDF page sized to match.
        ok, buf = cv2.imencode(".png", composite)
        if not ok:
            raise RuntimeError(f"PNG encode failed for footer of page {i + 1}")
        png_bytes = buf.tobytes()
        ch, cw = composite.shape[:2]
        pt_w = cw * 72.0 / p.dpi
        pt_h = ch * 72.0 / p.dpi
        new_page = out.new_page(width=pt_w, height=pt_h)
        new_page.insert_image(new_page.rect, stream=png_bytes)
        contributed.append(i + 1)

        # Standalone image: re-render the same band at images_dpi if it
        # differs from p.dpi, so the saved file matches the page-export
        # resolution. This second render is the price of decoupling image
        # quality from the (lighter) bound PDF; the alternative would be
        # upscaling the already-rendered footer, which would lose detail.
        if saving_images:
            if images_dpi == p.dpi:
                hi_footer = footer_img
            else:
                hi_gray = _render_gray(page, images_dpi)
                # Recompute pixel band for the new resolution by scaling.
                scale = images_dpi / p.dpi
                hy0 = int(round(y0 * scale))
                hy1 = int(round(y1 * scale))
                hi_footer = hi_gray[hy0 : hy1 + 1, :]
            out_path = img_dir / f"footer_p{i + 1:0{fname_width}d}{ext}"
            params: list[int] = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext == ".jpg" else []
            cv2.imwrite(str(out_path), hi_footer, params)
            # cv2.imwrite drops DPI metadata; re-stamp it via PIL so OCR tools
            # see the resolution baked into the file header.
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(out_path) as _im:
                    _im.save(out_path, dpi=(images_dpi, images_dpi))
            except Exception:  # pragma: no cover - PIL is a soft dep here
                pass
            saved_images.append(out_path)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if out.page_count == 0:
        # Avoid writing an empty PDF; fitz won't save zero-page docs cleanly.
        out.close()
        src.close()
        return []
    out.save(out_pdf, garbage=4, deflate=True)
    out.close()
    src.close()

    if zip_path is not None and saved_images:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img in saved_images:
                zf.write(img, arcname=img.name)

    return contributed
