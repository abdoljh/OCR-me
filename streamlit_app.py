import io
import hashlib
import os
import zipfile

import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from scipy.ndimage import (zoom as _zoom, percentile_filter,
                            affine_transform, gaussian_filter,
                            binary_dilation)

DEFAULT_DPI = 300
MIN_DPI = 150
MAX_DPI = 600
TESSDATA_CACHE = os.path.expanduser("~/.tessdata_custom")


def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@st.cache_data(show_spinner=False)
def _get_page_count(pdf_bytes: bytes) -> int:
    try:
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp = f.name
        out = subprocess.check_output(
            ["pdfinfo", tmp], stderr=subprocess.DEVNULL, timeout=10, text=True
        )
        os.unlink(tmp)
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return len(convert_from_bytes(pdf_bytes, dpi=36))


@st.cache_data(show_spinner=False)
def _render_page(pdf_bytes: bytes, page_num: int, dpi: int) -> Image.Image:
    return convert_from_bytes(
        pdf_bytes, dpi=dpi, first_page=page_num, last_page=page_num
    )[0]


@st.cache_data(show_spinner=False)
def _ocr_page(bw_bytes: bytes) -> str:
    """Run Tesseract on a binarized page image. Cached."""
    img = Image.open(io.BytesIO(bw_bytes))
    if os.path.exists(os.path.join(TESSDATA_CACHE, "ara.traineddata")):
        cfg = f'--oem 1 --psm 4 --tessdata-dir "{TESSDATA_CACHE}"'
    else:
        cfg = "--oem 1 --psm 4"
    try:
        return pytesseract.image_to_string(img, lang="ara", config=cfg)
    except Exception as exc:
        return f"[OCR error: {exc}]"


@st.cache_data(show_spinner=False)
def _binarize_page(pdf_bytes: bytes, page_num: int, dpi: int, threshold_pct: int) -> bytes:
    """Render and binarize one page; return PNG bytes. Cached per (file, page, dpi, threshold)."""
    img = _render_page(pdf_bytes, page_num, dpi)
    bw = _nlbin(img, threshold=threshold_pct / 100.0)
    buf = io.BytesIO()
    bw.save(buf, format="PNG")
    return buf.getvalue()


def _nlbin(img: Image.Image, threshold: float = 0.5) -> Image.Image:
    """Non-linear binarization — port of kraken's nlbin algorithm."""
    img = img.convert("L")
    raw = np.array(img, dtype=float) / 255.0
    image = raw - raw.min()
    if image.max() == 0:
        return img
    image /= image.max()

    m = _zoom(image, 0.5)
    m = percentile_filter(m, 80, size=(20, 2))
    m = percentile_filter(m, 80, size=(2, 20))
    mh, mw = m.shape
    oh, ow = image.shape
    m = affine_transform(m, np.diag([mh / oh, mw / ow]), output_shape=image.shape)
    w = min(image.shape[0], m.shape[0])
    h = min(image.shape[1], m.shape[1])
    flat = np.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)

    d0, d1 = flat.shape
    o0, o1 = int(0.1 * d0), int(0.1 * d1)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    v = est - gaussian_filter(est, 20.0)
    v = gaussian_filter(v ** 2, 20.0) ** 0.5
    v = v > 0.3 * v.max()
    v = binary_dilation(v, structure=np.ones((50, 1)))
    v = binary_dilation(v, structure=np.ones((1, 50)))
    est = est[v]
    lo = np.percentile(est, 5) if est.size else 0.0
    hi = np.percentile(est, 90) if est.size else 1.0
    flat -= lo
    if hi > lo:
        flat /= (hi - lo)
    flat = np.clip(flat, 0, 1)
    return Image.fromarray(np.uint8(255 * (flat > threshold)), mode="L")


def _build_txt(texts: list[str], stem: str) -> bytes:
    parts = [f"=== {stem} — Page {i} ===\n{t.strip()}" for i, t in enumerate(texts, 1)]
    return "\n\n".join(parts).encode("utf-8")


def _build_pdf(png_bytes_list: list[bytes], dpi: int) -> bytes:
    images = [Image.open(io.BytesIO(b)) for b in png_bytes_list]
    buf = io.BytesIO()
    images[0].save(
        buf, format="PDF", save_all=True, append_images=images[1:], resolution=dpi
    )
    return buf.getvalue()


def _build_zip(png_bytes_list: list[bytes], stem: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, png_bytes in enumerate(png_bytes_list, 1):
            zf.writestr(f"{stem}_page{i:03d}.png", png_bytes)
    return buf.getvalue()


def _build_tiff(png_bytes_list: list[bytes]) -> bytes:
    images = [Image.open(io.BytesIO(b)) for b in png_bytes_list]
    buf = io.BytesIO()
    images[0].save(
        buf, format="TIFF", save_all=True, append_images=images[1:],
        compression="tiff_deflate",
    )
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title="Arabic PDF Binarizer", page_icon="📄", layout="wide")
    st.title("Arabic PDF Binarizer")

    with st.sidebar:
        st.header("Settings")
        dpi = st.slider(
            "Rendering DPI",
            min_value=MIN_DPI,
            max_value=MAX_DPI,
            value=DEFAULT_DPI,
            step=50,
            help="Higher DPI = sharper image, slower rendering.",
        )
        threshold_pct = st.slider(
            "Binarization threshold",
            min_value=10,
            max_value=90,
            value=50,
            step=5,
            help=(
                "nlbin threshold (default 50 = 0.50). "
                "Increase if faint strokes disappear; "
                "decrease if background noise bleeds in."
            ),
        )

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload a PDF file to begin.")
        return

    for file_obj in uploaded_files:
        pdf_bytes = file_obj.read()
        file_hash = get_file_hash(pdf_bytes)
        stem = file_obj.name.removesuffix(".pdf")
        st.header(file_obj.name)

        try:
            total = _get_page_count(pdf_bytes)
        except Exception as exc:
            st.error(f"Could not read '{file_obj.name}': {exc}")
            continue

        # ── Render, binarize and OCR all pages (each step cached) ────────
        all_bw_bytes: list[bytes] = []
        all_texts: list[str] = []
        progress = st.progress(0, text=f"Page 1 of {total}…")
        for page_num in range(1, total + 1):
            progress.progress(
                page_num / total, text=f"Page {page_num} of {total}…"
            )
            try:
                bw_bytes = _binarize_page(pdf_bytes, page_num, dpi, threshold_pct)
            except Exception as exc:
                st.error(f"Page {page_num}: failed — {exc}")
                continue
            all_bw_bytes.append(bw_bytes)
            all_texts.append(_ocr_page(bw_bytes))

            orig = _render_page(pdf_bytes, page_num, dpi)
            bw   = Image.open(io.BytesIO(bw_bytes))
            col_orig, col_bin = st.columns(2)
            col_orig.image(orig, use_container_width=True,
                           caption=f"Page {page_num} — original")
            col_bin.image(bw,   use_container_width=True,
                          caption=f"Page {page_num} — binarized")
            col_bin.download_button(
                label=f"↓ Page {page_num} (PNG)",
                data=bw_bytes,
                file_name=f"{stem}_page{page_num:03d}.png",
                mime="image/png",
                key=f"png_{file_hash}_{page_num}",
            )

        progress.empty()

        # ── Bulk downloads ────────────────────────────────────────────────
        if all_bw_bytes:
            st.markdown("**Download all pages as:**")
            col_txt, col_pdf, col_tiff, col_zip = st.columns(4)

            col_txt.download_button(
                label="TXT",
                data=_build_txt(all_texts, stem),
                file_name=f"{stem}.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key=f"txt_{file_hash}",
            )
            col_pdf.download_button(
                label="PDF",
                data=_build_pdf(all_bw_bytes, dpi),
                file_name=f"{stem}_binarized.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_{file_hash}",
            )
            col_tiff.download_button(
                label="Multi-page TIFF",
                data=_build_tiff(all_bw_bytes),
                file_name=f"{stem}_binarized.tiff",
                mime="image/tiff",
                use_container_width=True,
                key=f"tiff_{file_hash}",
            )
            col_zip.download_button(
                label="ZIP (PNG per page)",
                data=_build_zip(all_bw_bytes, stem),
                file_name=f"{stem}_binarized.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_{file_hash}",
            )

        st.divider()


if __name__ == "__main__":
    main()
