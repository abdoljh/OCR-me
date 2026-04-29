import hashlib
import os
import subprocess
import tempfile
import urllib.request

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
from pdf2image import convert_from_bytes

from post_process import clean_pages

_NUMERAL_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

TESSERACT_OEM = "--oem 1"
DEFAULT_DPI = 400
MIN_DPI = 150
MAX_DPI = 600
DISPLAY_DPI = 150  # DPI used only for UI image previews — OCR uses the user-selected DPI
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
def _get_page_count(pdf_bytes: bytes) -> int:
    """Return PDF page count using pdfinfo (fast). Falls back to 36-DPI render."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp = f.name
    try:
        out = subprocess.check_output(
            ["pdfinfo", tmp], stderr=subprocess.DEVNULL, timeout=10, text=True
        )
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    finally:
        os.unlink(tmp)
    return len(convert_from_bytes(pdf_bytes, dpi=36))


@st.cache_data(show_spinner=False)
def _render_display_page(pdf_bytes: bytes, page_num: int) -> Image.Image:
    """Render a single page at DISPLAY_DPI for UI preview. Cached."""
    return convert_from_bytes(
        pdf_bytes, dpi=DISPLAY_DPI, first_page=page_num, last_page=page_num
    )[0]


def _render_ocr_page(pdf_bytes: bytes, page_num: int, dpi: int) -> Image.Image:
    """Render a single page at OCR DPI. Not cached — caller discards after use."""
    return convert_from_bytes(
        pdf_bytes, dpi=dpi, first_page=page_num, last_page=page_num
    )[0]


@st.cache_resource(show_spinner=False)
def _ensure_model(model_key: str) -> tuple[str, str, bool]:
    """Download model traineddata if needed. Returns (lang, tessdata_dir, used_fallback)."""
    info = LANG_MODELS[model_key]
    if info["url"] is None:
        return info["lang"], "", False
    os.makedirs(TESSDATA_CACHE_DIR, exist_ok=True)
    dest = os.path.join(TESSDATA_CACHE_DIR, info["filename"])
    if not os.path.exists(dest):
        try:
            with urllib.request.urlopen(info["url"], timeout=30) as resp:
                with open(dest, "wb") as fh:
                    fh.write(resp.read())
        except Exception:
            return LANG_MODELS["standard"]["lang"], "", True
    return info["lang"], TESSDATA_CACHE_DIR, False


def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img


def _header_strip_px(dpi: int) -> int:
    """Height in pixels of the header zone: our border + top 20 mm of page content."""
    return BORDER_PX + int(HEADER_CONTENT_MM / 25.4 * dpi)


def _build_config(psm: int, tessdata_dir: str, disable_dict: bool = False) -> str:
    parts = [TESSERACT_OEM, f"--psm {psm}"]
    if tessdata_dir:
        parts.append(f'--tessdata-dir "{tessdata_dir}"')
    if disable_dict:
        parts += ["-c load_system_dawg=0", "-c load_freq_dawg=0"]
    return " ".join(parts)


def _ocr_strip_filtered(
    strip: Image.Image, lang: str, tessdata_dir: str, disable_dict: bool = False
) -> str:
    """OCR a narrow strip, returning only words with Tesseract confidence >= HEADER_CONF_MIN.

    Filters decoration/noise artefacts (low confidence) while keeping real
    header text (high confidence), even when both appear in the same strip.
    """
    config = _build_config(psm=6, tessdata_dir=tessdata_dir, disable_dict=disable_dict)
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


def _arabic_ratio(text: str) -> float:
    """Fraction of non-space characters that fall in the Arabic Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if "؀" <= c <= "ۿ") / len(chars)


def ocr_page(
    img: Image.Image,
    psm: int,
    dpi: int,
    lang: str,
    tessdata_dir: str,
    disable_dict: bool = False,
) -> str:
    fill = 255 if img.mode == "L" else (255, 255, 255)
    padded = ImageOps.expand(img, border=BORDER_PX, fill=fill)
    strip_h = _header_strip_px(dpi)

    # Pass 1: header strip — confidence-filtered to remove decoration noise
    header_strip = padded.crop((0, 0, padded.width, strip_h))
    header_text = _ocr_strip_filtered(
        header_strip, lang, tessdata_dir, disable_dict
    ).strip()

    # Pass 2: body — cropped below the header strip, processed with chosen PSM
    body_img = padded.crop((0, strip_h, padded.width, padded.height))
    config = _build_config(psm=psm, tessdata_dir=tessdata_dir, disable_dict=disable_dict)
    try:
        body_text = pytesseract.image_to_string(body_img, lang=lang, config=config)
        # Auto-retry with PSM 11 (sparse text) when the result is mostly non-Arabic.
        # TOC/front-matter pages that block-PSMs mangle often respond better to PSM 11.
        if psm not in (11, 12) and _arabic_ratio(body_text) < 0.35:
            sparse_cfg = _build_config(psm=11, tessdata_dir=tessdata_dir, disable_dict=disable_dict)
            try:
                sparse_text = pytesseract.image_to_string(body_img, lang=lang, config=sparse_cfg)
                if _arabic_ratio(sparse_text) > _arabic_ratio(body_text):
                    body_text = sparse_text
            except pytesseract.TesseractError:
                pass
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
        f"color:#1a1a1a;"
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
    pdf_bytes: bytes,
) -> None:
    st.subheader(f"Page {page_num} of {total}")
    if show_image:
        img = _render_display_page(pdf_bytes, page_num)
        st.image(img, use_container_width=True, caption=f"Page {page_num}")
    render_rtl_text(text)
    st.text_area(
        label=f"Copy text — Page {page_num}",
        value=text,
        height=150,
        key=f"textarea_{file_idx}_{page_num}",
    )


def render_sidebar() -> tuple[str, int, int, bool, bool, bool, bool, bool]:
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
        disable_dict = st.checkbox(
            "Disable Tesseract dictionary",
            value=False,
            help=(
                "Turns off the built-in word-frequency correction "
                "(load_system_dawg=0, load_freq_dawg=0). "
                "Helps when the dictionary incorrectly 'corrects' rare words, "
                "names, or specialised vocabulary."
            ),
        )
        st.subheader("Post-processing")
        move_footnotes = st.checkbox(
            "Extract footnotes to end",
            value=True,
            help="Detects footnote blocks (e.g. )1( or (7)) and moves them to the end of the cleaned document.",
        )
        arabic_indic = st.checkbox(
            "Convert to Arabic-Indic numerals",
            value=False,
            help="Converts Western digits (0–9) → Arabic-Indic (٠–٩) in the cleaned output.",
        )
    return model_key, dpi, psm, show_images, preprocess, move_footnotes, arabic_indic, disable_dict


def main() -> None:
    st.set_page_config(page_title="Arabic PDF OCR", page_icon="📄", layout="wide")
    st.title("Arabic PDF OCR")

    model_key, dpi, psm, show_images, preprocess, move_footnotes, arabic_indic, disable_dict = render_sidebar()

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
            lang, tessdata_dir, used_fallback = _ensure_model(model_key)
    else:
        lang, tessdata_dir, used_fallback = _ensure_model(model_key)

    if used_fallback:
        st.warning(f"Could not download {info['label']} — using standard model.")

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

    # Collect OCR results for all files first (spinners/progress bars render here)
    file_results: list[dict] = []
    for file_obj in uploaded_files:
        pdf_bytes = file_obj.read()
        cache_key = f"{get_file_hash(pdf_bytes)}_{dpi}_{preprocess}_{psm}_{model_key}_{disable_dict}"

        if cache_key not in st.session_state["results"]:
            total_pages = _get_page_count(pdf_bytes)
            if total_pages == 0:
                st.error(f"Could not read '{file_obj.name}' — is it a valid PDF?")
                continue

            page_texts: list[str] = []
            progress_bar = st.progress(0, text=f"OCR: page 1 of {total_pages}")

            for page_num in range(1, total_pages + 1):
                try:
                    img = _render_ocr_page(pdf_bytes, page_num, dpi)
                except Exception as exc:
                    page_texts.append(f"[Failed to render page {page_num}: {exc}]")
                    continue
                work_img = preprocess_image(img) if preprocess else img
                page_texts.append(ocr_page(work_img, psm, dpi, lang, tessdata_dir, disable_dict))
                del img, work_img  # free the large raster before next page
                progress_bar.progress(
                    page_num / total_pages,
                    text=f"OCR: page {page_num} of {total_pages}",
                )

            progress_bar.empty()
            st.session_state["results"][cache_key] = {
                "pages": page_texts,
                "filename": file_obj.name,
            }

        # pdf_bytes held only for this rerun (not stored in session state)
        file_results.append({
            **st.session_state["results"][cache_key],
            "pdf_bytes": pdf_bytes,
        })

    if not file_results:
        return

    tab_raw, tab_clean = st.tabs(["Raw OCR output", "Cleaned text"])

    with tab_raw:
        raw_parts: list[str] = []
        for file_idx, fr in enumerate(file_results):
            filename = fr["filename"]
            page_texts = fr["pages"]
            pdf_bytes = fr["pdf_bytes"]
            total = len(page_texts)
            st.header(filename)
            for page_num, text in enumerate(page_texts, start=1):
                render_page_result(
                    file_idx=file_idx,
                    page_num=page_num,
                    total=total,
                    text=text,
                    show_image=show_images,
                    pdf_bytes=pdf_bytes,
                )
                raw_parts.append(f"=== {filename} — Page {page_num} of {total} ===\n{text}")
            st.divider()
        st.download_button(
            label="Download raw text (.txt)",
            data="\n\n".join(raw_parts).encode("utf-8"),
            file_name="ocr_output_raw.txt",
            mime="text/plain; charset=utf-8",
            key="dl_raw",
        )

    with tab_clean:
        all_clean_bodies: list[str] = []
        all_footnotes: list[str] = []
        for fr in file_results:
            # Each file's pages are cleaned independently — no cross-file merging
            clean_result = clean_pages(
                fr["pages"],
                move_footnotes=move_footnotes,
                arabic_indic_numerals=arabic_indic,
            )
            if clean_result.body:
                if len(file_results) > 1:
                    st.subheader(fr["filename"])
                render_rtl_text(clean_result.body)
                all_clean_bodies.append(clean_result.body)
            all_footnotes.extend(clean_result.footnotes)

        if all_footnotes:
            st.markdown("---")
            st.markdown("**Footnotes**")
            render_rtl_text("\n\n".join(all_footnotes))

        download_parts = list(all_clean_bodies)
        if all_footnotes:
            download_parts.append("=== Footnotes ===\n" + "\n\n".join(all_footnotes))
        st.download_button(
            label="Download cleaned text (.txt)",
            data="\n\n".join(download_parts).encode("utf-8"),
            file_name="ocr_output_clean.txt",
            mime="text/plain; charset=utf-8",
            key="dl_clean",
        )


if __name__ == "__main__":
    main()
