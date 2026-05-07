"""
Companion utilities for header_footer.py.

1. export_pages_as_images(in_pdf, out_dir, dpi=400, fmt="png", zip_path=None)
   Render each page of a stripped PDF (CropBox already set) to PNG/TIF/JPG,
   optionally zipping the results.

2. extract_footers_pdf(src_pdf, out_pdf, p=Params(), ...)
   Re-detect each page's footer in the ORIGINAL uncropped PDF and assemble a
   single PDF where each output page shows one source footer with a page label.
   Pages without a detected footer are skipped.

   v2 additions: img_dir, images_dpi, images_fmt, zip_path — save standalone
   footer images at a configurable DPI and optionally zip them.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import cv2
import fitz
import numpy as np

from header_footer import Params, detect_margins, _render_gray


# ---------------------------------------------------------------------------
# Task 1: export (cropped) pages as images, optionally zipped.
# ---------------------------------------------------------------------------

def export_pages_as_images(
    in_pdf: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = 400,
    fmt: str = "png",
    zip_path: Optional[str | Path] = None,
) -> list[Path]:
    """Render every page of `in_pdf` to an image file at `dpi`.

    Respects each page's CropBox, so when called on the output of strip_pdf()
    the images are already header/footer-stripped.

    Returns the list of written image paths.
    """
    in_pdf  = Path(in_pdf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().lstrip(".")
    if fmt not in {"png", "tif", "tiff", "jpg", "jpeg"}:
        raise ValueError(f"unsupported format: {fmt!r}")
    ext = ".tif" if fmt in {"tif", "tiff"} else (".jpg" if fmt in {"jpg", "jpeg"} else ".png")

    doc = fitz.open(in_pdf)
    n = doc.page_count
    width = max(3, len(str(n)))
    written: list[Path] = []
    skipped: list[int] = []

    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        if pix.width == 0 or pix.height == 0:
            skipped.append(i + 1)
            continue
        out_path = out_dir / f"page_{i + 1:0{width}d}{ext}"
        if ext == ".png":
            pix.save(str(out_path))
        else:
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            params: list[int] = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext == ".jpg" else []
            cv2.imwrite(str(out_path), arr, params)
        written.append(out_path)

    doc.close()
    if skipped:
        print(f"  warning: skipped degenerate page(s) {skipped} in {in_pdf.name}")

    if zip_path is not None:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img in written:
                zf.write(img, arcname=img.name)

    return written


# ---------------------------------------------------------------------------
# Task 2: collect footer regions into a single labeled PDF.
# ---------------------------------------------------------------------------

def _footer_band_y(m, body_pad_px: int = 8) -> Optional[tuple[int, int]]:
    """Return inclusive (y0, y1) pixel range of the footer, or None."""
    if m.rule_y is not None:
        y0 = max(0, m.rule_y - body_pad_px)
        y1 = m.page_h - max(3, int(0.012 * m.page_h)) - 1
        return (y0, y1) if y1 > y0 else None
    if m.footer_strip is not None:
        a, b = m.footer_strip
        y0 = max(0, a - body_pad_px)
        y1 = min(m.page_h - 1, b + body_pad_px)
        return (y0, y1) if y1 > y0 else None
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
    on_page=None,
) -> list[int]:
    """Assemble all detected footer regions from `src_pdf` into `out_pdf`.

    Each output page shows one footer prefixed by a label ("Page N — file.pdf").
    Returns the 1-based page numbers of source pages that contributed a footer.

    Optional standalone image export:
      img_dir    — directory to write per-footer images (created if needed).
      images_dpi — DPI for standalone images (re-rendered if != p.dpi).
      images_fmt — "png", "jpg", or "tif".
      zip_path   — if given, zip all standalone images here.
    """
    src_pdf = Path(src_pdf)
    out_pdf = Path(out_pdf)
    src = fitz.open(src_pdf)
    total_pages = src.page_count
    out = fitz.open()
    contributed: list[int] = []

    _save_imgs = img_dir is not None
    img_paths: list[Path] = []
    fmt = images_fmt.lower().lstrip(".")
    ext = ".tif" if fmt in {"tif", "tiff"} else (".jpg" if fmt in {"jpg", "jpeg"} else ".png")
    if _save_imgs:
        img_dir = Path(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(src):
        m = detect_margins(page, p, verbose=False)
        m.page_index = i
        band = _footer_band_y(m)
        if band is None:
            continue
        y0, y1 = band

        gray = _render_gray(page, p.dpi)
        footer_img = gray[y0 : y1 + 1, :]
        h_strip, w_strip = footer_img.shape[:2]

        # Label band for the PDF.
        label_h = label_dpi_px + 2 * label_margin_px
        label = np.full((label_h, w_strip), 255, dtype=np.uint8)
        cv2.putText(
            label,
            f"Page {i + 1}  ({src_pdf.name})",
            (label_margin_px, label_margin_px + label_dpi_px - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA,
        )
        cv2.line(label, (0, label_h - 2), (w_strip - 1, label_h - 2), 0, 1)

        composite = np.vstack([label, footer_img])
        ok, buf = cv2.imencode(".png", composite)
        if not ok:
            raise RuntimeError(f"PNG encode failed for footer of page {i + 1}")

        ch, cw = composite.shape[:2]
        new_page = out.new_page(width=cw * 72.0 / p.dpi, height=ch * 72.0 / p.dpi)
        new_page.insert_image(new_page.rect, stream=buf.tobytes())
        contributed.append(i + 1)
        if on_page:
            on_page(i + 1, total_pages)

        # Save standalone footer image.
        if _save_imgs:
            n = len(contributed)
            img_path = img_dir / f"footer_{n:04d}_pg{i + 1}{ext}"
            if images_dpi != p.dpi:
                hi_gray = _render_gray(page, images_dpi)
                scale = images_dpi / p.dpi
                hi_y0 = max(0, int(round(y0 * scale)))
                hi_y1 = min(hi_gray.shape[0] - 1, int(round(y1 * scale)))
                save_img = hi_gray[hi_y0 : hi_y1 + 1, :]
            else:
                save_img = footer_img
            if ext == ".png":
                cv2.imwrite(str(img_path), save_img)
            else:
                save_bgr = (
                    cv2.cvtColor(save_img, cv2.COLOR_GRAY2BGR)
                    if save_img.ndim == 2 else save_img
                )
                jparams: list[int] = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext == ".jpg" else []
                cv2.imwrite(str(img_path), save_bgr, jparams)
            img_paths.append(img_path)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if out.page_count > 0:
        out.save(out_pdf, garbage=4, deflate=True)
    out.close()
    src.close()

    if zip_path is not None and img_paths:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img in img_paths:
                zf.write(img, arcname=img.name)

    return contributed
