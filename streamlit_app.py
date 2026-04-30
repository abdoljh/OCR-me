import io
import hashlib

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes
from scipy.ndimage import (zoom as _zoom, percentile_filter,
                            affine_transform, gaussian_filter,
                            binary_dilation)

DEFAULT_DPI = 300
MIN_DPI = 150
MAX_DPI = 600


def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@st.cache_data(show_spinner=False)
def _render_page(pdf_bytes: bytes, page_num: int, dpi: int) -> Image.Image:
    return convert_from_bytes(
        pdf_bytes, dpi=dpi, first_page=page_num, last_page=page_num
    )[0]


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


def nlbin(img: Image.Image, threshold: float = 0.5) -> Image.Image:
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
                "Increase if faint strokes disappear; decrease if background bleeds in."
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

    threshold = threshold_pct / 100.0

    for file_obj in uploaded_files:
        pdf_bytes = file_obj.read()
        st.header(file_obj.name)

        try:
            total = _get_page_count(pdf_bytes)
        except Exception as exc:
            st.error(f"Could not read '{file_obj.name}': {exc}")
            continue

        progress = st.progress(0, text=f"Rendering page 1 of {total}…")
        for page_num in range(1, total + 1):
            progress.progress(page_num / total, text=f"Rendering page {page_num} of {total}…")
            try:
                orig = _render_page(pdf_bytes, page_num, dpi)
            except Exception as exc:
                st.error(f"Page {page_num}: render failed — {exc}")
                continue

            bw = nlbin(orig, threshold=threshold)

            col_orig, col_bin = st.columns(2)
            col_orig.image(orig, use_container_width=True, caption=f"Page {page_num} — original")
            col_bin.image(bw, use_container_width=True, caption=f"Page {page_num} — binarized")

            buf = io.BytesIO()
            bw.save(buf, format="PNG")
            col_bin.download_button(
                label=f"Download page {page_num} (PNG)",
                data=buf.getvalue(),
                file_name=f"{file_obj.name}_page{page_num:03d}.png",
                mime="image/png",
                key=f"dl_{get_file_hash(pdf_bytes)}_{page_num}",
            )

        progress.empty()
        st.divider()


if __name__ == "__main__":
    main()
