import io
import hashlib
import os
import urllib.request
import warnings
import zipfile

import numpy as np
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

DEFAULT_DPI = 300
MIN_DPI = 150
MAX_DPI = 600

_MODEL_URL = (
    "https://raw.githubusercontent.com/OpenITI/AOCP_print_models"
    "/refs/heads/main/transcription/apt-20221130.mlmodel"
)
_MODEL_PATH = os.path.expanduser("~/.kraken_models/apt-20221130.mlmodel")

# Maps the UI bidi selection string to the value rpred expects.
_BIDI_OPTIONS = {
    "Auto — let kraken decide (True)": "auto",
    "Force RTL — override to right-to-left ('R')": "R",
    "Force LTR — override to left-to-right ('L')": "L",
    "Off — raw display order (False)": "off",
}
_BIDI_TO_RPRED = {"auto": True, "R": "R", "L": "L", "off": False}
# Reverse map: internal key → short display name for the config JSON snapshot.
_BIDI_SHORT = {v: k.split(" —")[0] for k, v in _BIDI_OPTIONS.items()}


def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@st.cache_resource(show_spinner=False)
def _load_model():
    """Download the Arabic model once and keep it in memory."""
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    if not os.path.exists(_MODEL_PATH):
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    from kraken.lib import models as kraken_models
    return kraken_models.load_any(_MODEL_PATH)


@st.cache_data(show_spinner=False)
def _get_page_count(pdf_bytes: bytes) -> int:
    try:
        import subprocess, tempfile
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
def _render_page_bytes(pdf_bytes: bytes, page_num: int, dpi: int) -> bytes:
    """Return the original rendered page as PNG bytes (colour, for rpred)."""
    buf = io.BytesIO()
    _render_page(pdf_bytes, page_num, dpi).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _binarize_page(pdf_bytes: bytes, page_num: int, dpi: int, threshold_pct: int) -> bytes:
    from kraken import binarization as kraken_bin
    img = _render_page(pdf_bytes, page_num, dpi)
    bw = kraken_bin.nlbin(img, threshold=threshold_pct / 100.0)
    buf = io.BytesIO()
    bw.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _ocr_page(
    orig_bytes: bytes,
    threshold_pct: int = 50,
    text_direction: str = "horizontal-rl",
    autocast: bool = False,
    pad: int = 16,
    bidi_key: str = "auto",
    no_legacy_polygons: bool = False,
    temperature: float = 1.0,
) -> tuple[str, list[float]]:
    """Full kraken pipeline: binarize → segment → ocr.

    Prefers the kraken CLI (the exact pipeline from the documentation):
        kraken -i page.png out.txt binarize segment ocr -m model.mlmodel
    Falls back to the Python API when the CLI is not on PATH (dev environments).

    orig_bytes: original colour/grey page render — the CLI binarises this itself;
                the Python-API fallback also binarises internally before segmenting.
    """
    import shutil, subprocess, sys, tempfile

    # ── Locate the kraken CLI ─────────────────────────────────────────────
    # When installed via pip the script lives next to the Python executable.
    kraken_bin = shutil.which("kraken") or os.path.join(
        os.path.dirname(sys.executable), "kraken"
    )
    use_cli = bool(kraken_bin) and os.path.isfile(kraken_bin)

    if use_cli:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            txt_path = os.path.join(tmpdir, "page.txt")
            Image.open(io.BytesIO(orig_bytes)).save(img_path, format="PNG")

            cmd = [
                kraken_bin,
                "-i", img_path, txt_path,
                "binarize", "-t", f"{threshold_pct / 100:.2f}",
                "segment", "-d", text_direction,
                "ocr", "-m", _MODEL_PATH, "-p", str(pad),
            ]
            # Bidi flags
            if bidi_key == "off":
                cmd.append("--no-bidi")
            elif bidi_key in ("R", "L"):
                cmd += ["--bidi-override", bidi_key]
            # Temperature (skip when default to avoid unsupported-flag errors on older builds)
            if temperature != 1.0:
                cmd += ["-T", str(temperature)]

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                if proc.returncode == 0 and os.path.exists(txt_path):
                    with open(txt_path, encoding="utf-8") as f:
                        return f.read().strip(), []
                # CLI ran but failed — surface the error so we can debug
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"kraken CLI exited {proc.returncode}: {proc.stderr[:400]}"
                    )
            except subprocess.TimeoutExpired:
                raise RuntimeError("kraken CLI timed out (>300 s)")

    # ── Python API fallback (dev / CLI-not-available) ─────────────────────
    from kraken import blla, binarization as _kbin, rpred as krpred
    model = _load_model()
    model.temperature = temperature

    orig_img = Image.open(io.BytesIO(orig_bytes))
    bw_img   = _kbin.nlbin(orig_img, threshold=threshold_pct / 100.0)

    bidi = _BIDI_TO_RPRED[bidi_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = blla.segment(bw_img, text_direction=text_direction, autocast=autocast)
        records = list(krpred.rpred(
            model, orig_img, seg,
            pad=pad,
            bidi_reordering=bidi,
            no_legacy_polygons=no_legacy_polygons,
        ))
    lines, confs = [], []
    for r in records:
        if r.prediction.strip():
            lines.append(r.prediction)
            avg_conf = float(np.mean(r.confidences)) if r.confidences else 0.0
            confs.append(avg_conf)
    return "\n".join(lines), confs


@st.cache_data(show_spinner=False)
def _build_txt(texts: tuple[str, ...], stem: str) -> bytes:
    parts = [f"=== {stem} — Page {i} ===\n{t.strip()}" for i, t in enumerate(texts, 1)]
    return "\n\n".join(parts).encode("utf-8")


@st.cache_data(show_spinner=False)
def _build_pdf(png_bytes_list: tuple[bytes, ...], dpi: int) -> bytes:
    images = [Image.open(io.BytesIO(b)) for b in png_bytes_list]
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:], resolution=dpi)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _build_tiff(png_bytes_list: tuple[bytes, ...]) -> bytes:
    images = [Image.open(io.BytesIO(b)) for b in png_bytes_list]
    buf = io.BytesIO()
    images[0].save(buf, format="TIFF", save_all=True, append_images=images[1:],
                   compression="tiff_deflate")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _build_zip(png_bytes_list: tuple[bytes, ...], stem: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, png_bytes in enumerate(png_bytes_list, 1):
            zf.writestr(f"{stem}_page{i:03d}.png", png_bytes)
    return buf.getvalue()


def _sidebar_settings() -> dict:
    """Render all sidebar controls and return a dict of current values."""
    with st.sidebar:
        st.header("Settings")

        # ── Image quality ────────────────────────────────────────────────
        st.subheader("Image quality")
        dpi = st.slider(
            "Rendering DPI", MIN_DPI, MAX_DPI, DEFAULT_DPI, step=50,
            help="Higher DPI = sharper image but slower rendering and more memory.",
        )
        threshold_pct = st.slider(
            "Binarization threshold (nlbin)", 10, 90, 50, step=5,
            help=(
                "nlbin ink/background threshold (0.10–0.90). "
                "Raise if faint strokes vanish; lower if background noise bleeds in."
            ),
        )

        # ── Segmentation ─────────────────────────────────────────────────
        st.subheader("Segmentation (blla)")
        text_direction = st.selectbox(
            "Text direction",
            ["horizontal-rl", "horizontal-lr", "vertical-rl", "vertical-lr"],
            index=0,
            help=(
                "Primary reading direction passed to blla.segment(). "
                "Arabic is right-to-left (horizontal-rl). "
                "Affects the reading-order heuristic that sorts detected lines."
            ),
        )
        autocast = st.checkbox(
            "Autocast (mixed precision)",
            value=False,
            help=(
                "Enable torch.autocast during segmentation inference. "
                "May speed up GPU inference; usually no effect on CPU."
            ),
        )

        # ── Recognition ──────────────────────────────────────────────────
        st.subheader("Recognition (rpred)")
        pad = st.slider(
            "Line padding (px)", 0, 64, 16, step=4,
            help=(
                "Blank white pixels added to the left and right of each "
                "extracted line image before it is fed to the model. "
                "More padding gives the LSTM context at line edges; "
                "too much pads with noise."
            ),
        )
        bidi_label = st.selectbox(
            "BiDi reordering",
            list(_BIDI_OPTIONS.keys()),
            index=0,
            help=(
                "Unicode bidirectional reordering applied to each output line. "
                "'Auto' lets kraken detect direction per line. "
                "'Force RTL/LTR' overrides. "
                "'Off' returns raw display order (may reverse Arabic words)."
            ),
        )
        bidi_key = _BIDI_OPTIONS[bidi_label]

        no_legacy_polygons = st.checkbox(
            "Force new polygon extractor",
            value=False,
            help=(
                "If unchecked, kraken uses the polygon extractor the model was "
                "trained with (legacy for older models). "
                "Forcing the new extractor on a legacy-trained model may hurt accuracy "
                "but can be useful for comparison."
            ),
        )
        temperature = st.slider(
            "Softmax temperature", 0.1, 3.0, 1.0, step=0.1,
            help=(
                "Scales logits before softmax: T<1 sharpens the distribution "
                "(higher peak confidence), T>1 flattens it. "
                "With greedy decoding the recognised characters do not change, "
                "but per-character confidence scores do — useful for spotting "
                "uncertain regions."
            ),
        )

        # ── Post-processing ───────────────────────────────────────────────
        st.subheader("Post-processing")
        apply_corrections = st.checkbox(
            "Apply word corrections",
            value=False,
            help=(
                "Run confusables.py word substitutions on every recognised line "
                "to fix systematic Arabic OCR errors (e.g. العكري→العسكري). "
                "Only safe, high-precision corrections are applied."
            ),
        )

        # ── Active config summary ─────────────────────────────────────────
        with st.expander("Active configuration", expanded=False):
            st.json({
                "dpi": dpi,
                "nlbin_threshold": threshold_pct / 100,
                "text_direction": text_direction,
                "autocast": autocast,
                "pad": pad,
                "bidi_reordering": _BIDI_SHORT[bidi_key],
                "no_legacy_polygons": no_legacy_polygons,
                "temperature": temperature,
                "apply_corrections": apply_corrections,
            })

    return dict(
        dpi=dpi,
        threshold_pct=threshold_pct,
        text_direction=text_direction,
        autocast=autocast,
        pad=pad,
        bidi_key=bidi_key,
        no_legacy_polygons=no_legacy_polygons,
        temperature=temperature,
        apply_corrections=apply_corrections,
    )


def main() -> None:
    st.set_page_config(page_title="Arabic PDF OCR", page_icon="📄", layout="wide")
    st.title("Arabic PDF OCR")

    cfg = _sidebar_settings()

    with st.spinner("Loading Arabic OCR model…"):
        try:
            _load_model()
        except Exception as exc:
            st.error(f"Could not load OCR model: {exc}")
            return

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)", type=["pdf"], accept_multiple_files=True,
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

        all_bw_bytes: list[bytes] = []
        all_texts: list[str] = []
        progress = st.progress(0, text=f"Page 1 of {total}…")

        for page_num in range(1, total + 1):
            progress.progress(page_num / total, text=f"Page {page_num} of {total}…")
            orig_bytes = _render_page_bytes(pdf_bytes, page_num, cfg["dpi"])
            bw_bytes   = _binarize_page(pdf_bytes, page_num, cfg["dpi"], cfg["threshold_pct"])

            try:
                text, confs = _ocr_page(
                    orig_bytes,
                    threshold_pct=cfg["threshold_pct"],
                    text_direction=cfg["text_direction"],
                    autocast=cfg["autocast"],
                    pad=cfg["pad"],
                    bidi_key=cfg["bidi_key"],
                    no_legacy_polygons=cfg["no_legacy_polygons"],
                    temperature=cfg["temperature"],
                )
            except Exception as exc:
                st.error(f"Page {page_num}: OCR failed — {exc}")
                text, confs = "", []

            if cfg["apply_corrections"] and text:
                from confusables import apply_word_corrections
                text = apply_word_corrections(text, include_gt_derived=True)

            all_bw_bytes.append(bw_bytes)
            all_texts.append(text)

            orig = Image.open(io.BytesIO(orig_bytes))
            bw   = Image.open(io.BytesIO(bw_bytes))

            col_orig, col_bin = st.columns(2)
            col_orig.image(orig, use_container_width=True,
                           caption=f"Page {page_num} — original")
            col_bin.image(bw, use_container_width=True,
                          caption=f"Page {page_num} — binarized")
            col_bin.download_button(
                label=f"↓ Page {page_num} (PNG)",
                data=bw_bytes,
                file_name=f"{stem}_page{page_num:03d}.png",
                mime="image/png",
                key=f"png_{file_hash}_{page_num}",
            )

            # OCR text + confidence for this page
            with st.expander(
                f"OCR text — Page {page_num}"
                + (f"  |  avg confidence {np.mean(confs):.2%}" if confs else ""),
                expanded=True,
            ):
                avg_conf = np.mean(confs) if confs else None
                if avg_conf is not None and avg_conf < 0.60:
                    st.warning(
                        f"Low average confidence ({avg_conf:.1%}). "
                        "Try adjusting DPI, threshold, or padding."
                    )
                st.text_area(
                    label="",
                    value=text,
                    height=220,
                    key=f"ocr_text_{file_hash}_{page_num}",
                    help="Arabic text extracted by kraken (right-to-left).",
                )
                if confs:
                    st.caption(
                        f"{len(confs)} lines recognised — "
                        f"min {min(confs):.1%} / mean {avg_conf:.1%} / max {max(confs):.1%}"
                    )

        progress.empty()

        if all_bw_bytes:
            st.markdown("**Download all pages as:**")
            col_txt, col_pdf, col_tiff, col_zip = st.columns(4)
            col_txt.download_button(
                label="TXT",
                data=_build_txt(tuple(all_texts), stem),
                file_name=f"{stem}.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key=f"txt_{file_hash}",
            )
            col_pdf.download_button(
                label="PDF",
                data=_build_pdf(tuple(all_bw_bytes), cfg["dpi"]),
                file_name=f"{stem}_binarized.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_{file_hash}",
            )
            col_tiff.download_button(
                label="Multi-page TIFF",
                data=_build_tiff(tuple(all_bw_bytes)),
                file_name=f"{stem}_binarized.tiff",
                mime="image/tiff",
                use_container_width=True,
                key=f"tiff_{file_hash}",
            )
            col_zip.download_button(
                label="ZIP (PNG per page)",
                data=_build_zip(tuple(all_bw_bytes), stem),
                file_name=f"{stem}_binarized.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_{file_hash}",
            )

        st.divider()


if __name__ == "__main__":
    main()
