import hashlib
import os
import urllib.request

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from pdf2image import convert_from_bytes

_NUMERAL_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

TESSERACT_OEM = "--oem 1"
DEFAULT_DPI = 400
MIN_DPI = 150
MAX_DPI = 600
DEFAULT_PSM = 4
PSM_OPTIONS = {
    4: "4 — Single column, variable sizes  (default)",
    6: "6 — Uniform text block",
    3: "3 — Auto layout detection  (multi-column documents)",
}

# Two-pass OCR constants
BORDER_PX = 100        # white padding added on all sides before OCR
HEADER_CONTENT_MM = 20 # page content depth treated as header zone
HEADER_CONF_MIN = 65   # minimum Tesseract word confidence to keep from header strip

# Language model options — downloaded once and cached on disk
TESSDATA_CACHE_DIR = os.path.expanduser("~/.tessdata_custom")
LANG_MODELS: dict[str, dict] = {
    "best": {
        "label": "tessdata_best  (highest accuracy · ~14 MB · recommended)",
        "url": "https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata",
        "filename": "ara.traineddata",
        "lang": "ara",
    },
    "amiri": {
        "label": "ara-amiri-3000  (fine-tuned Arabic numerals · ~10 MB)",
        "url": "https://github.com/Shreeshrii/tessdata_shreetest/raw/master/ara-amiri-3000.traineddata",
        "filename": "ara-amiri-3000.traineddata",
        "lang": "ara-amiri-3000",
    },
    "standard": {
        "label": "Standard  (apt-installed · fastest · no download)",
        "url": None,
        "filename": None,
        "lang": "ara",
    },
}
DEFAULT_MODEL = "best"


@st.cache_data(show_spinner=False)
def pdf_to_images(pdf_bytes: bytes, dpi: int) -> list:
    return convert_from_bytes(pdf_bytes, dpi=dpi)


@st.cache_resource(show_spinner=False)
def _ensure_model(model_key: str) -> tuple[str, str]:
    """Download model traineddata if needed. Returns (tesseract_lang, tessdata_dir)."""
    info = LANG_MODELS[model_key]
    if info["url"] is None:
        return info["lang"], ""
    os.makedirs(TESSDATA_CACHE_DIR, exist_ok=True)
    dest = os.path.join(TESSDATA_CACHE_DIR, info["filename"])
    if not os.path.exists(dest):
        try:
            urllib.request.urlretrieve(info["url"], dest)
        except Exception:
            st.warning(f"Could not download {info['label']} — falling back to standard model.")
            return LANG_MODELS["standard"]["lang"], ""
    return info["lang"], TESSDATA_CACHE_DIR


def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img


def _header_strip_px(dpi: int) -> int:
    """Height in pixels of the header zone: our border + top 20 mm of page content."""
    return BORDER_PX + int(HEADER_CONTENT_MM / 25.4 * dpi)


def _build_config(psm: int, tessdata_dir: str) -> str:
    parts = [TESSERACT_OEM, f"--psm {psm}"]
    if tessdata_dir:
        parts.append(f"--tessdata-dir {tessdata_dir}")
    return " ".join(parts)


def _ocr_strip_filtered(strip: Image.Image, lang: str, tessdata_dir: str) -> str:
    """OCR a narrow strip, returning only words with Tesseract confidence >= HEADER_CONF_MIN.

    Filters decoration/noise artefacts (low confidence) while keeping real
    header text (high confidence), even when both appear in the same strip.
    """
    config = _build_config(psm=6, tessdata_dir=tessdata_dir)
    try:
        data = pytesseract.image_to_data(
            strip,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError:
        return ""
    line_words: dict = {}
    for word, conf, block, par, line in zip(
        data["text"], data["conf"],
        data["block_num"], data["par_num"], data["line_num"],
    ):
        if word.strip() and int(conf) >= HEADER_CONF_MIN:
            line_words.setdefault((block, par, line), []).append(word)
    return "\n".join(" ".join(words) for words in line_words.values())


def ocr_page(img: Image.Image, psm: int, dpi: int, lang: str, tessdata_dir: str) -> str:
    fill = 255 if img.mode == "L" else (255, 255, 255)
    padded = ImageOps.expand(img, border=BORDER_PX, fill=fill)
    strip_h = _header_strip_px(dpi)

    # Pass 1: header strip — confidence-filtered to remove decoration noise
    header_strip = padded.crop((0, 0, padded.width, strip_h))
    header_text = _ocr_strip_filtered(header_strip, lang, tessdata_dir).strip()

    # Pass 2: body — cropped below the header strip, processed with chosen PSM
    body_img = padded.crop((0, strip_h, padded.width, padded.height))
    config = _build_config(psm=psm, tessdata_dir=tessdata_dir)
    try:
        body_text = pytesseract.image_to_string(body_img, lang=lang, config=config)
        text = (header_text + "\n\n" + body_text) if header_text else body_text
        return text.translate(_NUMERAL_TABLE)
    except pytesseract.TesseractError as exc:
        return f"[OCR error on this page: {exc}]"


def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def render_rtl_text(text: str) -> None:
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div style="'
        f"direction:rtl;"
        f"text-align:right;"
        f"font-family:Arial,sans-serif;"
        f"font-size:16px;"
        f"line-height:2;"
        f"background-color:#f8f8f8;"
        f"padding:12px 16px;"
        f"border-radius:6px;"
        f"border:1px solid #e0e0e0;"
        f"white-space:pre-wrap;"
        f'word-wrap:break-word;">'
        f"{safe_text}</div>",
        unsafe_allow_html=True,
    )


def render_page_result(
    file_idx: int,
    page_num: int,
    total: int,
    text: str,
    show_image: bool,
    img: Image.Image,
) -> None:
    st.subheader(f"Page {page_num} of {total}")
    if show_image:
        st.image(img, use_container_width=True, caption=f"Page {page_num}")
    render_rtl_text(text)
    st.text_area(
        label=f"Copy text — Page {page_num}",
        value=text,
        height=150,
        key=f"textarea_{file_idx}_{page_num}",
    )


def render_sidebar() -> tuple:
    with st.sidebar:
        st.header("Settings")
        model_key = st.selectbox(
            "Language model",
            options=list(LANG_MODELS.keys()),
            format_func=lambda k: LANG_MODELS[k]["label"],
            index=list(LANG_MODELS.keys()).index(DEFAULT_MODEL),
            help=(
                "tessdata_best and ara-amiri-3000 are downloaded once (~10–14 MB) and "
                "cached for the session. Both improve number and punctuation accuracy "
                "over the standard apt-installed model."
            ),
        )
        dpi = st.slider(
            "Rendering DPI",
            min_value=MIN_DPI,
            max_value=MAX_DPI,
            value=DEFAULT_DPI,
            step=50,
            help="Higher DPI improves OCR accuracy at the cost of speed and memory.",
        )
        psm = st.selectbox(
            "Page segmentation mode",
            options=list(PSM_OPTIONS.keys()),
            format_func=lambda x: PSM_OPTIONS[x],
            index=list(PSM_OPTIONS.keys()).index(DEFAULT_PSM),
            help=(
                "Applies to the body region only. The page header is always extracted "
                "separately via a confidence-filtered strip OCR. "
                "PSM 4 gives the best body accuracy for single-column Arabic documents."
            ),
        )
        show_images = st.checkbox("Show page images", value=False)
        preprocess = st.checkbox(
            "Enhance contrast (grayscale)",
            value=True,
            help="Converts to grayscale and boosts contrast. Recommended for most Arabic documents.",
        )
    return model_key, dpi, psm, show_images, preprocess


def main() -> None:
    st.set_page_config(page_title="Arabic PDF OCR", page_icon="📄", layout="wide")
    st.title("Arabic PDF OCR")

    model_key, dpi, psm, show_images, preprocess = render_sidebar()

    # Resolve language model (downloads if needed)
    info = LANG_MODELS[model_key]
    needs_download = (
        info["url"] is not None
        and info["filename"] is not None
        and not os.path.exists(os.path.join(TESSDATA_CACHE_DIR, info["filename"]))
    )
    if needs_download:
        size_hint = info["label"].split("·")[1].strip()
        with st.spinner(f"Downloading language model ({size_hint})…"):
            lang, tessdata_dir = _ensure_model(model_key)
    else:
        lang, tessdata_dir = _ensure_model(model_key)

    st.caption(
        f"Tesseract OCR · OEM 1 LSTM · PSM {psm} · "
        f"lang: {lang} · model: {info['label'].split('(')[0].strip()}"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more Arabic PDF files for text extraction.",
    )

    if not uploaded_files:
        st.info("Upload a PDF file above to begin.")
        return

    if "results" not in st.session_state:
        st.session_state["results"] = {}

    all_text_parts: list[str] = []

    for file_idx, file_obj in enumerate(uploaded_files):
        pdf_bytes = file_obj.read()
        cache_key = f"{get_file_hash(pdf_bytes)}_{dpi}_{preprocess}_{psm}_{model_key}"

        if cache_key not in st.session_state["results"]:
            try:
                with st.spinner(f"Rendering pages for {file_obj.name}…"):
                    images = pdf_to_images(pdf_bytes, dpi)
            except Exception as exc:
                st.error(f"Failed to render '{file_obj.name}': {exc}")
                continue

            total_pages = len(images)
            page_texts: list[str] = []
            progress_bar = st.progress(0, text=f"OCR: page 1 of {total_pages}")

            for i, img in enumerate(images):
                work_img = preprocess_image(img) if preprocess else img
                page_texts.append(ocr_page(work_img, psm, dpi, lang, tessdata_dir))
                progress_bar.progress(
                    (i + 1) / total_pages,
                    text=f"OCR: page {i + 1} of {total_pages}",
                )

            progress_bar.empty()
            st.session_state["results"][cache_key] = {
                "pages": page_texts,
                "images": images,
                "filename": file_obj.name,
            }

        result = st.session_state["results"][cache_key]
        page_texts = result["pages"]
        images = result["images"]
        filename = result["filename"]
        total = len(page_texts)

        st.header(filename)
        for page_num, (text, img) in enumerate(zip(page_texts, images), start=1):
            render_page_result(
                file_idx=file_idx,
                page_num=page_num,
                total=total,
                text=text,
                show_image=show_images,
                img=img,
            )
            all_text_parts.append(f"=== {filename} — Page {page_num} of {total} ===\n{text}")

        st.divider()

    if all_text_parts:
        combined = "\n\n".join(all_text_parts)
        st.download_button(
            label="Download all extracted text (.txt)",
            data=combined.encode("utf-8"),
            file_name="ocr_output.txt",
            mime="text/plain; charset=utf-8",
        )


if __name__ == "__main__":
    main()
