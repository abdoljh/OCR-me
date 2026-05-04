import io
import hashlib
import os
import urllib.request
import warnings
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from pdf2image import convert_from_bytes

DEFAULT_DPI = 400
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
def _detect_margins(bw_bytes: bytes) -> tuple[int, int]:
    """Scan horizontal ink density to locate header/footer boundaries.

    Returns (top_crop_px, bot_crop_px): pixels to remove from each edge.
    Works by finding the first significant gap between the header ink-block
    and the body, and the last such gap between the body and the footer.
    Returns (0, 0) when no clear header/footer gap is detected.
    """
    img = Image.open(io.BytesIO(bw_bytes)).convert("L")
    arr = np.array(img)
    h, w = arr.shape

    ink = (arr < 128).mean(axis=1)                    # ink fraction per row
    smooth = np.convolve(ink, np.ones(5) / 5, mode="same")  # 5-row smoother

    INK_ROW  = 0.004   # row is "inky" if > 0.4 % of width is dark
    GAP_ROWS = 20      # gap must span ≥ 20 blank rows
    ZONE     = h // 5  # only search in the top/bottom 20 %

    def _gap_from_top(profile, end):
        in_ink = blank = 0
        for i in range(end):
            if profile[i] > INK_ROW:
                if in_ink and blank >= GAP_ROWS:
                    return i        # first body row after the header gap
                blank = 0
                in_ink = 1
            elif in_ink:
                blank += 1
        return 0

    def _gap_from_bottom(profile, start):
        in_ink = blank = 0
        for i in range(h - 1, start - 1, -1):
            if profile[i] > INK_ROW:
                if in_ink and blank >= GAP_ROWS:
                    return h - i - 1    # rows to remove from bottom
                blank = 0
                in_ink = 1
            elif in_ink:
                blank += 1
        return 0

    return int(_gap_from_top(smooth, ZONE)), int(_gap_from_bottom(smooth, h - ZONE))


@st.cache_data(show_spinner=False)
def _apply_crop(orig_bytes: bytes, top_px: int, bot_px: int, pad_px: int) -> bytes:
    """Crop top_px / bot_px rows then add pad_px white border. Returns PNG bytes."""
    img = Image.open(io.BytesIO(orig_bytes))
    w, h = img.size
    y0 = max(0, top_px)
    y1 = max(y0 + 1, h - bot_px)
    cropped = img.crop((0, y0, w, y1))
    if pad_px > 0:
        cropped = ImageOps.expand(cropped, border=pad_px, fill=(255, 255, 255))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _draw_crop_overlay(orig_bytes: bytes, top_px: int, bot_px: int) -> bytes:
    """Draw two red crop-boundary lines on the image for the preview column."""
    img = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    lw    = max(4, h // 300)          # line width scales with image height
    top_y = max(0, min(top_px, h - 1))
    bot_y = max(top_y + 1, h - max(0, bot_px))
    red   = (220, 30, 30)
    for dy in range(lw):
        draw.line([(0, top_y + dy), (w - 1, top_y + dy)], fill=red)
        draw.line([(0, bot_y - dy), (w - 1, bot_y - dy)], fill=red)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
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

    cli_stderr = ""   # captured for the fallback warning

    if use_cli:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            txt_path = os.path.join(tmpdir, "page.txt")
            Image.open(io.BytesIO(orig_bytes)).save(img_path, format="PNG")

            # Match the documented pipeline:
            #   kraken -i img.png out.txt segment -d DIR ocr -m MODEL ...
            # Note: binarize is intentionally omitted — BLLA works directly on
            # the colour render, and skipping it matches the Colab-verified flow.
            cmd = [
                kraken_bin,
                "-i", img_path, txt_path,
                "segment", "-d", text_direction,
                "ocr", "-m", _MODEL_PATH, "-p", str(pad),
            ]
            # Bidi reordering.
            # For Arabic (RTL), --base-dir R gives better word ordering than
            # Unicode auto-detect.  "auto" and "R" both use R as base; only
            # explicit LTR or off deviates from this.
            if bidi_key == "off":
                cmd.append("--no-reorder")
            elif bidi_key == "L":
                cmd += ["--base-dir", "L", "--reorder"]
            else:  # "R" or "auto"
                cmd += ["--base-dir", "R", "--reorder"]
            # Optional ocr flags
            if no_legacy_polygons:
                cmd.append("--no-legacy-polygons")
            if temperature != 1.0:
                cmd += ["-t", str(temperature)]

            # Completely clear PYTHONPATH so the subprocess uses only the venv's
            # site-packages and cannot pick up any local directory that shadows
            # the installed kraken package (Streamlit may inject the project root
            # via sitecustomize.py in addition to PYTHONPATH).
            env = os.environ.copy()
            env["PYTHONPATH"] = ""

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                    env=env, cwd=tmpdir,
                )
                if proc.returncode == 0 and os.path.exists(txt_path):
                    with open(txt_path, encoding="utf-8") as f:
                        return f.read().strip(), []
                cli_stderr = proc.stderr  # save for warning; fall through below
            except subprocess.TimeoutExpired:
                cli_stderr = "CLI timed out (>300 s)"

    # ── Python API fallback (dev / CLI-not-available / CLI-failed) ───────
    # cli_stderr is non-empty when the CLI was found but failed.
    if cli_stderr:
        # Surface the CLI failure as a Streamlit warning (not an exception)
        # so the user knows the fallback path was taken.
        import streamlit as _st
        _st.warning(f"kraken CLI failed — using Python API fallback.\n\n```\n{cli_stderr}\n```")

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


def _anthropic_key() -> str:
    """Return Anthropic API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("ANTHROPIC_API_KEY", "")


@st.cache_data(show_spinner=False)
def _ocr_page_claude(orig_bytes: bytes) -> tuple[str, list[float]]:
    """Extract Arabic text from a page image using Claude Haiku vision.

    Uses claude-haiku-4-5-20251001.  Handles both modern printed Arabic and
    Quranic Uthmanic script.  Requires ANTHROPIC_API_KEY in Streamlit secrets
    or the environment.  Cost: ~$0.004 per page at 400 DPI.
    """
    import anthropic
    import base64

    api_key = _anthropic_key()
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Add it to Streamlit secrets (Settings → Secrets) or set the environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = base64.standard_b64encode(orig_bytes).decode()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all Arabic text from this page image exactly as written.\n"
                        "• Preserve all diacritical marks (tashkeel) precisely.\n"
                        "• For Quranic pages: keep verse numbers in Arabic-Indic numerals "
                        "  e.g. ﴿١٢٣﴾ or (١٢٣).\n"
                        "• Separate paragraphs with a blank line.\n"
                        "• Output ONLY the extracted text — no labels, commentary, "
                        "  or translation."
                    ),
                },
            ],
        }],
    )
    return response.content[0].text.strip(), []


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
            help="Higher DPI = sharper image. 400 DPI is optimal for Apple Live Text.",
        )

        # ── Crop margins ──────────────────────────────────────────────────
        # Slider keys are pre-seeded in main() before this function runs so
        # there is never a value= / session_state conflict (Streamlit 1.57+).
        st.subheader("Crop margins")
        auto_detect = st.checkbox(
            "Auto-detect from first page",
            value=True,
            key="auto_detect_crop",
            help=(
                "Scan ink density to locate header/footer gaps on page 1. "
                "Sliders update automatically — adjust to fine-tune."
            ),
        )
        top_crop_pct = st.slider(
            "Top crop (%)", 0, 50, step=1, key="top_crop_pct",
            help="Percentage of page height removed from the top edge (page-number row, etc.).",
        )
        bot_crop_pct = st.slider(
            "Bottom crop (%)", 0, 50, step=1, key="bot_crop_pct",
            help="Percentage of page height removed from the bottom edge (footer, running title).",
        )
        pad_px = st.slider(
            "White padding (px)", 0, 50, step=2, key="crop_pad_px",
            help="White border added around the cropped image.",
        )

        # ── OCR (optional) ────────────────────────────────────────────────
        # The primary purpose of this app is to produce high-quality cropped
        # colour images for Apple Live Text (offline, free, near-perfect Arabic).
        # OCR via Claude Haiku or kraken is an optional extra for cases where
        # a digital text file is also needed.
        st.subheader("OCR (optional)")
        run_ocr = st.checkbox(
            "Extract text as well",
            value=False,
            help=(
                "Run an OCR engine to also produce a .txt file.\n\n"
                "The primary output is always the cropped colour images. "
                "OCR adds processing time (and API cost for Claude)."
            ),
        )

        # Set defaults for OCR params (used when run_ocr is False).
        engine = "claude"
        threshold_pct, text_direction = 50, "horizontal-rl"
        autocast, pad = False, 16
        bidi_key = "auto"
        no_legacy_polygons, temperature, apply_corrections = False, 1.0, False

        if run_ocr:
            engine_label = st.selectbox(
                "Engine",
                ["Claude Haiku (API, ~$0.004/page)", "kraken (offline, free)"],
                index=0,
                help=(
                    "Claude Haiku: near-perfect Arabic accuracy, handles Quranic "
                    "Uthmanic script with full tashkeel. Requires ANTHROPIC_API_KEY "
                    "in Streamlit secrets.\n\n"
                    "kraken: fully offline, no API key. Good for modern printed "
                    "Arabic; less reliable on Quranic script and decorative fonts."
                ),
            )
            engine = "claude" if engine_label.startswith("Claude") else "kraken"

            if engine == "claude" and not _anthropic_key():
                st.warning(
                    "ANTHROPIC_API_KEY not set — kraken will be used as fallback. "
                    "Add the key in **Settings → Secrets** to enable Claude.",
                    icon="⚠️",
                )

            kraken_label = (
                "kraken settings (fallback)" if engine == "claude" else "kraken settings"
            )
            with st.expander(kraken_label, expanded=(engine == "kraken")):
                threshold_pct = st.slider(
                    "Binarization threshold (nlbin)", 10, 90, 50, step=5,
                    help=(
                        "nlbin threshold (0.10–0.90). "
                        "Raise if faint strokes vanish; lower if background noise bleeds in."
                    ),
                )
                text_direction = st.selectbox(
                    "Text direction",
                    ["horizontal-rl", "horizontal-lr", "vertical-rl", "vertical-lr"],
                    index=0,
                    help="Arabic is right-to-left (horizontal-rl).",
                )
                autocast = st.checkbox(
                    "Autocast (mixed precision)", value=False,
                    help="Enable torch.autocast during segmentation. Usually no effect on CPU.",
                )
                pad = st.slider(
                    "Line padding (px)", 0, 64, 16, step=4,
                    help="Blank pixels added to each line edge before recognition.",
                )
                bidi_label = st.selectbox(
                    "BiDi reordering", list(_BIDI_OPTIONS.keys()), index=0,
                    help="Unicode bidi reordering. 'Auto' lets kraken detect per line.",
                )
                bidi_key = _BIDI_OPTIONS[bidi_label]
                no_legacy_polygons = st.checkbox(
                    "Force new polygon extractor", value=False,
                    help="May hurt accuracy on older models.",
                )
                temperature = st.slider(
                    "Softmax temperature", 0.1, 3.0, 1.0, step=0.1,
                    help="Affects confidence scores only, not character predictions.",
                )

            apply_corrections = st.checkbox(
                "Apply word corrections",
                value=False,
                help=(
                    "Run confusables.py substitutions to fix systematic kraken errors "
                    "(tatweel stripping, فى→في, بنفه→بنفسه …). "
                    "Claude usually produces these correctly already."
                ),
            )

        # ── Active config summary ─────────────────────────────────────────
        with st.expander("Active configuration", expanded=False):
            cfg_json: dict = {
                "dpi": dpi,
                "auto_detect_crop": auto_detect,
                "top_crop_pct": top_crop_pct,
                "bot_crop_pct": bot_crop_pct,
                "pad_px": pad_px,
                "run_ocr": run_ocr,
            }
            if run_ocr:
                cfg_json["engine"] = engine
                if engine == "claude":
                    cfg_json["model"] = "claude-haiku-4-5-20251001"
                else:
                    cfg_json.update({
                        "nlbin_threshold": threshold_pct / 100,
                        "text_direction": text_direction,
                        "autocast": autocast,
                        "pad": pad,
                        "bidi_reordering": _BIDI_SHORT[bidi_key],
                        "no_legacy_polygons": no_legacy_polygons,
                        "temperature": temperature,
                    })
                cfg_json["apply_corrections"] = apply_corrections
            st.json(cfg_json)

    return dict(
        dpi=dpi,
        auto_detect=auto_detect,
        top_crop_pct=top_crop_pct,
        bot_crop_pct=bot_crop_pct,
        pad_px=pad_px,
        run_ocr=run_ocr,
        engine=engine,
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

    # Seed crop slider keys with defaults BEFORE _sidebar_settings() renders widgets.
    for _k, _d in [("top_crop_pct", 0), ("bot_crop_pct", 0), ("crop_pad_px", 20)]:
        if _k not in st.session_state:
            st.session_state[_k] = _d
    # Flush pending auto-detect values from a previous st.rerun().
    for _k in ("top_crop_pct", "bot_crop_pct"):
        _pk = f"_pending_{_k}"
        if _pk in st.session_state:
            st.session_state[_k] = st.session_state.pop(_pk)

    cfg = _sidebar_settings()

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)", type=["pdf"], accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info(
            "Upload a PDF to begin.  "
            "Adjust the **Crop margins** sliders in the sidebar, then step through "
            "each page to fine-tune the crop before downloading."
        )
        return

    for file_obj in uploaded_files:
        # Cache raw bytes — st.rerun() moves the UploadedFile cursor to EOF.
        pdf_cache_key = f"pdf_{file_obj.name}_{file_obj.size}"
        if pdf_cache_key not in st.session_state:
            raw = file_obj.read()
            if raw:
                st.session_state[pdf_cache_key] = raw
        pdf_bytes = st.session_state.get(pdf_cache_key)
        if not pdf_bytes:
            st.error(f"Could not read '{file_obj.name}' — please re-upload.")
            continue

        file_hash = get_file_hash(pdf_bytes)
        stem = file_obj.name.removesuffix(".pdf")
        st.header(file_obj.name)

        try:
            total = _get_page_count(pdf_bytes)
        except Exception as exc:
            st.error(f"Could not read '{file_obj.name}': {exc}")
            continue

        # ── Auto-detect margins (once per file) ───────────────────────────
        det_key = f"crop_det_{file_hash}"
        if cfg["auto_detect"] and det_key not in st.session_state:
            with st.spinner("Detecting header/footer margins from page 1…"):
                bw_p1 = _binarize_page(pdf_bytes, 1, cfg["dpi"], 50)
                top_px_det, bot_px_det = _detect_margins(bw_p1)
                h_p1 = Image.open(io.BytesIO(bw_p1)).height
            st.session_state["_pending_top_crop_pct"] = min(50, round(top_px_det / h_p1 * 100))
            st.session_state["_pending_bot_crop_pct"] = min(50, round(bot_px_det / h_p1 * 100))
            st.session_state[det_key] = True
            st.rerun()

        # Wizard state ─────────────────────────────────────────────────────
        wiz_page_key = f"wiz_page_{file_hash}"
        wiz_done_key = f"wiz_done_{file_hash}"
        if wiz_page_key not in st.session_state:
            st.session_state[wiz_page_key] = 1

        # ══════════════════════════════════════════════════════════════════
        # DOWNLOAD VIEW — all pages cropped, optional OCR, download buttons.
        # ══════════════════════════════════════════════════════════════════
        if st.session_state.get(wiz_done_key, False):
            all_cropped_bytes: list[bytes] = []
            all_texts: list[str] = []

            prog = st.progress(0, text="Building images…")
            for pn in range(1, total + 1):
                prog.progress(pn / total, text=f"Page {pn} of {total}…")
                ob = _render_page_bytes(pdf_bytes, pn, cfg["dpi"])
                h  = Image.open(io.BytesIO(ob)).height
                pt = st.session_state.get(f"p_top_{file_hash}_{pn}", cfg["top_crop_pct"])
                pb = st.session_state.get(f"p_bot_{file_hash}_{pn}", cfg["bot_crop_pct"])
                cb = _apply_crop(ob, round(pt / 100 * h), round(pb / 100 * h), cfg["pad_px"])
                all_cropped_bytes.append(cb)

                text, confs = "", []
                if cfg["run_ocr"]:
                    if cfg["engine"] == "claude":
                        try:
                            text, confs = _ocr_page_claude(cb)
                        except Exception:
                            cfg["engine"] = "kraken"
                    if cfg["engine"] == "kraken":
                        try:
                            _load_model()
                            text, confs = _ocr_page(
                                cb,
                                threshold_pct=cfg["threshold_pct"],
                                text_direction=cfg["text_direction"],
                                autocast=cfg["autocast"],
                                pad=cfg["pad"],
                                bidi_key=cfg["bidi_key"],
                                no_legacy_polygons=cfg["no_legacy_polygons"],
                                temperature=cfg["temperature"],
                            )
                        except Exception:
                            pass
                    if cfg["apply_corrections"] and text:
                        from confusables import apply_word_corrections
                        text = apply_word_corrections(text, include_gt_derived=True)
                all_texts.append(text)
            prog.empty()

            # Thumbnail grid (up to 5 per row).
            THUMB_COLS = 5
            for row_start in range(0, total, THUMB_COLS):
                row_pages = range(row_start + 1, min(row_start + THUMB_COLS + 1, total + 1))
                thumb_row = st.columns(len(row_pages))
                for i, pn in enumerate(row_pages):
                    pt = st.session_state.get(f"p_top_{file_hash}_{pn}", cfg["top_crop_pct"])
                    pb = st.session_state.get(f"p_bot_{file_hash}_{pn}", cfg["bot_crop_pct"])
                    thumb_row[i].image(
                        Image.open(io.BytesIO(all_cropped_bytes[pn - 1])),
                        use_container_width=True,
                        caption=f"p.{pn}  ↑{pt}% ↓{pb}%",
                    )

            # Download buttons.
            st.markdown("**Download all pages as:**")
            n_cols = 4 if cfg["run_ocr"] else 3
            dl_cols = st.columns(n_cols)
            dl_cols[0].download_button(
                "PDF",
                data=_build_pdf(tuple(all_cropped_bytes), cfg["dpi"]),
                file_name=f"{stem}_cropped.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_{file_hash}",
            )
            dl_cols[1].download_button(
                "Multi-page TIFF",
                data=_build_tiff(tuple(all_cropped_bytes)),
                file_name=f"{stem}_cropped.tiff",
                mime="image/tiff",
                use_container_width=True,
                key=f"tiff_{file_hash}",
            )
            dl_cols[2].download_button(
                "ZIP (PNG per page)",
                data=_build_zip(tuple(all_cropped_bytes), stem),
                file_name=f"{stem}_cropped.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_{file_hash}",
            )
            if cfg["run_ocr"]:
                dl_cols[3].download_button(
                    "TXT",
                    data=_build_txt(tuple(all_texts), stem),
                    file_name=f"{stem}.txt",
                    mime="text/plain; charset=utf-8",
                    use_container_width=True,
                    key=f"txt_{file_hash}",
                )

            st.divider()
            if st.button("← Back to page editor", key=f"back_{file_hash}"):
                st.session_state[wiz_done_key] = False
                st.rerun()

        # ══════════════════════════════════════════════════════════════════
        # WIZARD VIEW — one page at a time with crop sliders and navigation.
        # ══════════════════════════════════════════════════════════════════
        else:
            current = st.session_state[wiz_page_key]

            # Per-page crop keys — initialise from global setting on first visit.
            p_top_key = f"p_top_{file_hash}_{current}"
            p_bot_key = f"p_bot_{file_hash}_{current}"
            if p_top_key not in st.session_state:
                st.session_state[p_top_key] = cfg["top_crop_pct"]
            if p_bot_key not in st.session_state:
                st.session_state[p_bot_key] = cfg["bot_crop_pct"]

            # Crop sliders — placed ABOVE the preview so the user sets values
            # before reading the result.
            st.caption(f"**Page {current} of {total}**")
            sl_left, sl_right = st.columns(2)
            sl_left.slider(
                f"Top crop (%) — page {current}", 0, 50, step=1,
                key=p_top_key,
                help="Rows to remove from the top of this page.",
            )
            sl_right.slider(
                f"Bottom crop (%) — page {current}", 0, 50, step=1,
                key=p_bot_key,
                help="Rows to remove from the bottom of this page.",
            )

            # Compute crop from the (possibly just-updated) slider values.
            orig_bytes = _render_page_bytes(pdf_bytes, current, cfg["dpi"])
            h_page   = Image.open(io.BytesIO(orig_bytes)).height
            page_top = st.session_state[p_top_key]
            page_bot = st.session_state[p_bot_key]
            top_px   = round(page_top / 100 * h_page)
            bot_px   = round(page_bot / 100 * h_page)
            cropped_bytes = _apply_crop(orig_bytes, top_px, bot_px, cfg["pad_px"])

            # Preview: overlay left, cropped result right.
            overlay_bytes = _draw_crop_overlay(orig_bytes, top_px, bot_px)
            col_left, col_right = st.columns(2)
            col_left.image(overlay_bytes, use_container_width=True,
                           caption=f"Page {current} — crop lines")
            col_right.image(Image.open(io.BytesIO(cropped_bytes)),
                            use_container_width=True,
                            caption=f"Page {current} — result")
            col_right.download_button(
                f"↓ Page {current} (PNG)",
                data=cropped_bytes,
                file_name=f"{stem}_page{current:03d}.png",
                mime="image/png",
                key=f"png_{file_hash}_{current}",
            )

            # Navigation row.
            st.write("")
            nav_prev, nav_mid, nav_next = st.columns([1, 2, 1])
            with nav_prev:
                if st.button("← Prev", disabled=(current == 1),
                             key=f"prev_{file_hash}", use_container_width=True):
                    st.session_state[wiz_page_key] = current - 1
                    st.rerun()
            with nav_mid:
                if st.button(
                    "⬇ Proceed to downloads",
                    key=f"skip_{file_hash}",
                    use_container_width=True,
                    help="Use current crop settings for any un-visited pages and go to downloads.",
                ):
                    st.session_state[wiz_done_key] = True
                    st.rerun()
            with nav_next:
                label = "Next →" if current < total else "✓ Done"
                btn_type = "secondary" if current < total else "primary"
                if st.button(label, key=f"next_{file_hash}",
                             use_container_width=True, type=btn_type):
                    if current < total:
                        st.session_state[wiz_page_key] = current + 1
                        st.rerun()
                    else:
                        st.session_state[wiz_done_key] = True
                        st.rerun()

            st.divider()


if __name__ == "__main__":
    main()
