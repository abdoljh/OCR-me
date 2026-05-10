import io
import hashlib
import math
import os
import shutil
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from pdf2image import convert_from_bytes

DEFAULT_DPI = 400
MIN_DPI = 150
MAX_DPI = 600
ZIP_SPLIT_DEFAULT_MB = 250

_MODEL_URL = (
    "https://raw.githubusercontent.com/OpenITI/AOCP_print_models"
    "/refs/heads/main/transcription/apt-20221130.mlmodel"
)
_MODEL_PATH = os.path.expanduser("~/.kraken_models/apt-20221130.mlmodel")

_BIDI_OPTIONS = {
    "Auto — let kraken decide (True)": "auto",
    "Force RTL — override to right-to-left ('R')": "R",
    "Force LTR — override to left-to-right ('L')": "L",
    "Off — raw display order (False)": "off",
}
_BIDI_TO_RPRED = {"auto": True, "R": "R", "L": "L", "off": False}
_BIDI_SHORT    = {v: k.split(" —")[0] for k, v in _BIDI_OPTIONS.items()}


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
    """
    img = Image.open(io.BytesIO(bw_bytes)).convert("L")
    arr = np.array(img)
    h, w = arr.shape

    ink = (arr < 128).mean(axis=1)
    smooth = np.convolve(ink, np.ones(5) / 5, mode="same")

    INK_ROW  = 0.004
    GAP_ROWS = 20
    ZONE     = h // 3

    def _gap_from_top(profile, end):
        in_ink = blank = 0
        for i in range(end):
            if profile[i] > INK_ROW:
                if in_ink and blank >= GAP_ROWS:
                    return i
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
                    return h - i - 1
                blank = 0
                in_ink = 1
            elif in_ink:
                blank += 1
        return 0

    return int(_gap_from_top(smooth, ZONE)), int(_gap_from_bottom(smooth, h - ZONE))


@st.cache_data(show_spinner=False)
def _detect_margins_v2(pdf_bytes: bytes, page_num: int, dpi: int) -> tuple[int, int, int, int]:
    """Smart per-page header/footer detection using header_footer.py.

    Returns (top_crop_px, bot_crop_px, page_h, page_w) at `dpi`.
    """
    import fitz
    from header_footer import detect_margins, Params

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num - 1]
    m = detect_margins(page, Params(dpi=dpi))
    doc.close()
    top_crop_px = m.keep_top
    bot_crop_px = max(0, m.page_h - 1 - m.keep_bottom)
    return top_crop_px, bot_crop_px, m.page_h, m.page_w


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
    lw    = max(4, h // 300)
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
    """Full kraken pipeline: binarize -> segment -> ocr."""
    import shutil, subprocess, sys, tempfile

    kraken_bin = shutil.which("kraken") or os.path.join(
        os.path.dirname(sys.executable), "kraken"
    )
    use_cli = bool(kraken_bin) and os.path.isfile(kraken_bin)

    cli_stderr = ""

    if use_cli:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            txt_path = os.path.join(tmpdir, "page.txt")
            Image.open(io.BytesIO(orig_bytes)).save(img_path, format="PNG")

            cmd = [
                kraken_bin,
                "-i", img_path, txt_path,
                "segment", "-d", text_direction,
                "ocr", "-m", _MODEL_PATH, "-p", str(pad),
            ]
            if bidi_key == "off":
                cmd.append("--no-reorder")
            elif bidi_key == "L":
                cmd += ["--base-dir", "L", "--reorder"]
            else:
                cmd += ["--base-dir", "R", "--reorder"]
            if no_legacy_polygons:
                cmd.append("--no-legacy-polygons")
            if temperature != 1.0:
                cmd += ["-t", str(temperature)]

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
                cli_stderr = proc.stderr
            except subprocess.TimeoutExpired:
                cli_stderr = "CLI timed out (>300 s)"

    if cli_stderr:
        st.warning(f"kraken CLI failed — using Python API fallback.\n\n```\n{cli_stderr}\n```")

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
    """Extract Arabic text from a page image using Claude Haiku vision."""
    import anthropic
    import base64

    api_key = _anthropic_key()
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Add it to Streamlit secrets (Settings -> Secrets) or set the environment variable."
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
                        "• Output ONLY the extracted text -- no labels, commentary, "
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
def _build_pdf(png_paths: tuple[str, ...], dpi: int) -> bytes:
    images = []
    for path in png_paths:
        img = Image.open(path).convert("L")
        w, h = img.size
        images.append(img.resize((w // 2, h // 2), Image.LANCZOS))
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:],
                   resolution=dpi // 2)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _build_tiff(png_paths: tuple[str, ...], dpi: int) -> bytes:
    images = []
    for path in png_paths:
        img = Image.open(path).convert("L")
        w, h = img.size
        images.append(img.resize((w // 2, h // 2), Image.LANCZOS))
    buf = io.BytesIO()
    half_dpi = dpi // 2
    images[0].save(buf, format="TIFF", save_all=True, append_images=images[1:],
                   compression="tiff_deflate", dpi=(half_dpi, half_dpi))
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _build_zip(png_paths: tuple[str, ...], stem: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, path in enumerate(png_paths, 1):
            with open(path, "rb") as f:
                zf.writestr(f"{stem}_page{i:03d}.png", f.read())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Disk-based pipeline helpers (no @st.cache_data — side effects + large output)
# ---------------------------------------------------------------------------

def _estimate_pages_mb(n_pages: int, dpi: int) -> float:
    """Estimate uncompressed PNG size: ~4 MB/page at 400 DPI, scales with DPI²."""
    return n_pages * 4.0 * (dpi / 400) ** 2


def _stream_pages_to_zips(
    src_pdf: str,
    split_dir: Path,
    stem: str,
    max_mb: float,
    dpi: int,
    on_page=None,
) -> list[str]:
    """Render PDF pages one at a time directly into split ZIP(s) — no intermediate PNGs on disk.

    Each page's PNG bytes are written straight into the ZIP, then discarded.
    Peak additional disk: only the growing current ZIP part (≤ max_mb).
    on_page(page_num, total) is called after each page is added.
    Single-chunk: `{stem}_pages.zip`; multi-chunk: `{stem}_pages_part_001.zip`, …
    Returns list of final ZIP paths.
    """
    import fitz as _fitz

    split_dir.mkdir(parents=True, exist_ok=True)
    doc = _fitz.open(src_pdf)
    n = doc.page_count
    width = max(3, len(str(n)))
    max_bytes = int(max_mb * 1024 * 1024)

    tmp_paths: list[Path] = []
    cur_size = 0
    cur_zf: zipfile.ZipFile | None = None

    def _next_zip() -> None:
        nonlocal cur_zf, cur_size
        if cur_zf is not None:
            cur_zf.close()
        p = split_dir / f"__part_{len(tmp_paths) + 1:03d}.zip"
        tmp_paths.append(p)
        cur_zf = zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED)
        cur_size = 0

    _next_zip()
    try:
        for i in range(n):
            page = doc.load_page(i)
            pix  = page.get_pixmap(dpi=dpi)
            png  = pix.tobytes("png")
            pix  = None          # release pixmap memory immediately
            fname = f"page_{i + 1:0{width}d}.png"
            if cur_size > 0 and cur_size + len(png) > max_bytes:
                _next_zip()
            cur_zf.writestr(fname, png)  # type: ignore[union-attr]
            cur_size += len(png)
            if on_page:
                on_page(i + 1, n)
    finally:
        if cur_zf is not None:
            cur_zf.close()
        doc.close()

    # Rename temp files to final names.
    if len(tmp_paths) == 1:
        final = split_dir / f"{stem}_pages.zip"
        tmp_paths[0].rename(final)
        return [str(final)]
    final_paths: list[str] = []
    for j, p in enumerate(tmp_paths, 1):
        final = split_dir / f"{stem}_pages_part_{j:03d}.zip"
        p.rename(final)
        final_paths.append(str(final))
    return final_paths


# ---------------------------------------------------------------------------
# Per-mode UI renderers
# ---------------------------------------------------------------------------

def _render_download_selector(
    files: list[dict],
    file_hash: str,
    key_prefix: str,
) -> None:
    """One download button at a time, gated by a selectbox.

    Streamlit stores each download_button's data in session memory for the
    lifetime of the render.  Loading all files simultaneously (e.g. 5 × 250 MB)
    exceeds Streamlit Cloud's 1 GB RAM limit.  This function shows only the
    currently-selected file's button, bounding peak memory to ≤ one file.

    files: list of {"path": str, "mime": str}
    """
    valid = [f for f in files if Path(f["path"]).exists()]
    if not valid:
        st.warning("Download files are no longer available. Please reset and re-run.")
        return

    labels = [
        f"{Path(f['path']).name}  ({Path(f['path']).stat().st_size / 1_048_576:.0f} MB)"
        for f in valid
    ]

    sel = st.selectbox(
        "Select file:",
        range(len(valid)),
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_sel_{file_hash}",
    )

    f = valid[sel]
    fname = Path(f["path"]).name
    with open(f["path"], "rb") as fh:
        st.download_button(
            f"⬇ {fname}",
            data=fh.read(),
            file_name=fname,
            mime=f["mime"],
            key=f"{key_prefix}_dl_{file_hash}_{sel}",
            use_container_width=True,
        )
    if len(valid) > 1:
        st.caption(f"File {sel + 1} of {len(valid)} — use the selector above to switch.")


def _do_single_book(
    pdf_bytes: bytes,
    dpi: int,
    include_footers: bool,
    include_photos: bool,
    zip_split_mb: float,
    tmpdir: Path,
    stem: str,
    total: int,
) -> dict:
    """Full pipeline with staged st.status progress updates.

    Stages shown to the user:
      1. Detect & strip margins (one low-DPI render per page via strip_pdf)
      2. Stream render → ZIP(s) at export DPI with per-page progress bar
      3. Extract footnote regions (optional)
      4. Extract photographs (optional)
    """
    from header_footer import strip_pdf, Params
    from page_export import extract_footers_pdf
    from image_extract import extract_images

    p = Params(dpi=dpi)
    src_path   = tmpdir / "input.pdf"
    strip_path = tmpdir / "stripped.pdf"
    split_dir  = tmpdir / "zips"
    foot_path  = tmpdir / "footers.pdf"

    src_path.write_bytes(pdf_bytes)

    with st.status("Processing…", expanded=True) as status:
        st.write(f"⚙️ Detecting and stripping margins on {total} page(s)…")
        prog1 = st.progress(0)

        def _on_strip(pg: int, n: int) -> None:
            prog1.progress(pg / n, text=f"Page {pg} of {n}")

        strip_pdf(str(src_path), str(strip_path), p, on_page=_on_strip)
        prog1.empty()

        st.write(f"🖼 Rendering {total} page(s) at {dpi} DPI…")
        prog2 = st.progress(0)

        def _on_page(pg: int, n: int) -> None:
            prog2.progress(pg / n, text=f"Page {pg} of {n}")

        page_zips = _stream_pages_to_zips(
            str(strip_path), split_dir, stem, zip_split_mb, dpi, _on_page
        )
        prog2.empty()

        foot_pages: list[int] = []
        footers_zip_path: str | None = None
        if include_footers:
            st.write("📑 Extracting footnote regions…")
            prog3 = st.progress(0)

            def _on_foot(pg: int, n: int) -> None:
                prog3.progress(pg / n, text=f"Page {pg} of {n}")

            foot_img_dir = tmpdir / "footer_imgs"
            fzip = tmpdir / f"{stem}_footers_imgs.zip"
            foot_pages = extract_footers_pdf(
                str(src_path), str(foot_path), p=p,
                img_dir=str(foot_img_dir),
                images_dpi=dpi,
                zip_path=str(fzip),
                on_page=_on_foot,
            )
            prog3.empty()
            if foot_pages and fzip.exists():
                footers_zip_path = str(fzip)

        photos_zip_path: str | None = None
        n_photos = 0
        if include_photos:
            st.write("📷 Extracting photographs…")
            prog4 = st.progress(0)

            def _on_photo(pg: int, n: int) -> None:
                prog4.progress(pg / n, text=f"Page {pg} of {n}")

            photo_dir = tmpdir / "photos"
            pzip = tmpdir / f"{stem}_photos.zip"
            photo_results = extract_images(
                str(src_path), str(photo_dir),
                dpi=dpi,
                with_captions=True,
                use_body_crop=True,
                zip_path=str(pzip),
                on_page=_on_photo,
            )
            prog4.empty()
            n_photos = sum(len(r.photos) for r in photo_results)
            if n_photos and pzip.exists():
                photos_zip_path = str(pzip)

        n_zips = len(page_zips)
        n_foot = len(foot_pages)
        status.update(
            label=(
                f"Done — {total} page(s) rendered"
                + (f" into {n_zips} ZIP part{'s' if n_zips > 1 else ''}" if n_zips else "")
                + (f", {n_foot} page(s) with footnotes" if include_footers else "")
                + (f", {n_photos} photograph{'s' if n_photos != 1 else ''} extracted" if include_photos else "")
                + "."
            ),
            state="complete",
        )

    return {
        "page_zips": page_zips,
        "footers_pdf": str(foot_path) if include_footers and foot_pages and foot_path.exists() else None,
        "footers_zip": footers_zip_path,
        "n_pages_with_footers": n_foot,
        "photos_zip": photos_zip_path,
        "n_photos": n_photos,
    }


def _render_single_book_ui(
    pdf_bytes: bytes, file_hash: str, stem: str, total: int, cfg: dict,
) -> None:
    result_key = f"sb_result_{file_hash}"
    tmpdir_key = f"sb_tmpdir_{file_hash}"

    if result_key not in st.session_state:
        est_mb = _estimate_pages_mb(total, cfg["dpi"])
        n_parts = max(1, math.ceil(est_mb / cfg["zip_split_mb"]))
        parts_note = f" → {n_parts} ZIP part{'s' if n_parts > 1 else ''}" if n_parts > 1 else ""
        st.caption(
            f"{total} page(s) · ~{est_mb:.0f} MB estimated at {cfg['dpi']} DPI{parts_note}"
        )

        col_btn, col_info = st.columns([1, 4])
        run_clicked = col_btn.button(
            "▶ Run",
            key=f"sb_run_{file_hash}",
            type="primary",
            use_container_width=True,
        )
        extras = []
        if cfg["include_footers"]:
            extras.append("footnotes")
        if cfg["include_photos"]:
            extras.append("photographs")
        col_info.caption(
            "Strips headers/footers automatically and exports all pages as colour PNGs."
            + (f" Also extracts {' and '.join(extras)}." if extras else "")
        )

        if run_clicked:
            tmpdir = Path(tempfile.mkdtemp(prefix="sb_"))
            st.session_state[tmpdir_key] = str(tmpdir)
            try:
                result = _do_single_book(
                    pdf_bytes, cfg["dpi"], cfg["include_footers"],
                    cfg["include_photos"], cfg["zip_split_mb"], tmpdir, stem, total,
                )
            except Exception as exc:
                shutil.rmtree(str(tmpdir), ignore_errors=True)
                st.session_state.pop(tmpdir_key, None)
                st.error(f"Processing failed: {exc}")
                return
            st.session_state[result_key] = result
            st.rerun()
    else:
        result = st.session_state[result_key]
        n_zips = len(result["page_zips"])
        n_foot = result["n_pages_with_footers"]
        n_photos = result.get("n_photos", 0)
        st.success(
            f"Done — {total} page(s) processed"
            + (f", {n_zips} ZIP part{'s' if n_zips > 1 else ''}" if n_zips else "")
            + (f", {n_foot} page(s) with footnotes" if cfg["include_footers"] else "")
            + (f", {n_photos} photograph{'s' if n_photos != 1 else ''} extracted"
               if cfg["include_photos"] else "")
            + ".",
            icon="✅",
        )

        dl_files = [{"path": zp, "mime": "application/zip"} for zp in result["page_zips"]]
        if result.get("footers_pdf"):
            dl_files.append({"path": result["footers_pdf"], "mime": "application/pdf"})
        if result.get("footers_zip"):
            dl_files.append({"path": result["footers_zip"], "mime": "application/zip"})
        if result.get("photos_zip"):
            dl_files.append({"path": result["photos_zip"], "mime": "application/zip"})
        _render_download_selector(dl_files, file_hash, "sb")
        if cfg["include_footers"] and not n_foot:
            st.info("No footnote sections were detected in this document.")
        if cfg["include_photos"] and not n_photos:
            st.info("No photographs were detected in this document.")

        if st.button("↺ Reset", key=f"sb_reset_{file_hash}"):
            tmpdir = st.session_state.pop(tmpdir_key, None)
            if tmpdir and os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
            st.session_state.pop(result_key, None)
            st.rerun()


def _do_raw_export(
    pdf_bytes: bytes,
    dpi: int,
    zip_split_mb: float,
    tmpdir: Path,
    stem: str,
    total: int,
) -> list[str]:
    """Export original pages (un-stripped) to split ZIP(s) with per-page progress."""
    src_path  = tmpdir / "input.pdf"
    split_dir = tmpdir / "zips"
    src_path.write_bytes(pdf_bytes)

    with st.status("Exporting…", expanded=True) as status:
        st.write(f"🖼 Rendering {total} page(s) at {dpi} DPI…")
        prog = st.progress(0)

        def _on_page(pg: int, n: int) -> None:
            prog.progress(pg / n, text=f"Page {pg} of {n}")

        zip_paths = _stream_pages_to_zips(
            str(src_path), split_dir, stem, zip_split_mb, dpi, _on_page
        )
        prog.empty()
        n_zips = len(zip_paths)
        status.update(
            label=f"Done — {total} page(s) exported into {n_zips} ZIP{'s' if n_zips > 1 else ''}.",
            state="complete",
        )

    return zip_paths


def _render_raw_export_ui(
    pdf_bytes: bytes, file_hash: str, stem: str, total: int, cfg: dict,
) -> None:
    result_key = f"re_result_{file_hash}"
    tmpdir_key = f"re_tmpdir_{file_hash}"

    if result_key not in st.session_state:
        est_mb = _estimate_pages_mb(total, cfg["dpi"])
        n_parts = max(1, math.ceil(est_mb / cfg["zip_split_mb"]))
        parts_note = f" → {n_parts} ZIP part{'s' if n_parts > 1 else ''}" if n_parts > 1 else ""
        st.caption(
            f"{total} page(s) · ~{est_mb:.0f} MB estimated at {cfg['dpi']} DPI{parts_note}"
        )

        col_btn, col_info = st.columns([1, 4])
        run_clicked = col_btn.button(
            "▶ Run",
            key=f"re_run_{file_hash}",
            type="primary",
            use_container_width=True,
        )
        col_info.caption(
            "Exports all original pages as colour PNG images (no header/footer removal)."
        )

        if run_clicked:
            tmpdir = Path(tempfile.mkdtemp(prefix="re_"))
            st.session_state[tmpdir_key] = str(tmpdir)
            try:
                zip_paths = _do_raw_export(
                    pdf_bytes, cfg["dpi"], cfg["zip_split_mb"], tmpdir, stem, total
                )
            except Exception as exc:
                shutil.rmtree(str(tmpdir), ignore_errors=True)
                st.session_state.pop(tmpdir_key, None)
                st.error(f"Export failed: {exc}")
                return
            st.session_state[result_key] = {"zip_paths": zip_paths}
            st.rerun()
    else:
        result = st.session_state[result_key]
        n_zips = len(result["zip_paths"])
        st.success(
            f"Done — {total} page(s) exported, {n_zips} ZIP part{'s' if n_zips > 1 else ''}.",
            icon="✅",
        )

        _render_download_selector(
            [{"path": zp, "mime": "application/zip"} for zp in result["zip_paths"]],
            file_hash, "re",
        )

        if st.button("↺ Reset", key=f"re_reset_{file_hash}"):
            tmpdir = st.session_state.pop(tmpdir_key, None)
            if tmpdir and os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
            st.session_state.pop(result_key, None)
            st.rerun()


def _render_visual_ui(
    pdf_bytes: bytes, file_hash: str, stem: str, total: int, cfg: dict,
) -> None:
    # Fast mode: detect from page 1 once per file.
    det_key = f"crop_det_{file_hash}"
    if cfg["detect_mode"] == "Fast (page 1)" and det_key not in st.session_state:
        with st.spinner("Detecting header/footer margins from page 1…"):
            bw_p1 = _binarize_page(pdf_bytes, 1, cfg["dpi"], 50)
            top_px_det, bot_px_det = _detect_margins(bw_p1)
            h_p1 = Image.open(io.BytesIO(bw_p1)).height
        st.session_state["_pending_top_crop_pct"] = min(75, round(top_px_det / h_p1 * 100))
        st.session_state["_pending_bot_crop_pct"] = min(75, round(bot_px_det / h_p1 * 100))
        st.session_state[det_key] = True
        st.rerun()

    wiz_page_key = f"wiz_page_{file_hash}"
    wiz_done_key = f"wiz_done_{file_hash}"
    if wiz_page_key not in st.session_state:
        st.session_state[wiz_page_key] = 1

    # =========================================================================
    # DOWNLOAD VIEW
    # =========================================================================
    if st.session_state.get(wiz_done_key, False):
        tmpdir_key = f"tmpdir_{file_hash}"
        if tmpdir_key not in st.session_state:
            st.session_state[tmpdir_key] = tempfile.mkdtemp(prefix="ocr_me_")
        tmpdir = st.session_state[tmpdir_key]

        all_texts: list[str] = []

        THUMB_COLS = 5
        thumb_cols: list[list] = []
        for row_start in range(0, total, THUMB_COLS):
            n_in_row = min(THUMB_COLS, total - row_start)
            thumb_cols.append(list(st.columns(n_in_row)))

        prog = st.progress(0, text="Building images…")
        png_paths: list[str] = []

        for pn in range(1, total + 1):
            prog.progress(pn / total, text=f"Page {pn} of {total}…")
            png_path = os.path.join(tmpdir, f"page{pn:03d}.png")
            png_paths.append(png_path)

            pt = st.session_state.get(f"p_top_{file_hash}_{pn}", cfg["top_crop_pct"])
            pb = st.session_state.get(f"p_bot_{file_hash}_{pn}", cfg["bot_crop_pct"])

            if not os.path.exists(png_path):
                ob = _render_page_bytes(pdf_bytes, pn, cfg["dpi"])
                h  = Image.open(io.BytesIO(ob)).height
                cb = _apply_crop(ob, round(pt / 100 * h), round(pb / 100 * h), cfg["pad_px"])
                with open(png_path, "wb") as fh:
                    fh.write(cb)

            with open(png_path, "rb") as fh:
                cb = fh.read()

            col = thumb_cols[(pn - 1) // THUMB_COLS][(pn - 1) % THUMB_COLS]
            col.image(Image.open(io.BytesIO(cb)), width="stretch",
                      caption=f"p.{pn}  ↑{pt}% ↓{pb}%")
            col.download_button(
                f"↓ p.{pn}",
                data=cb,
                file_name=f"{stem}_page{pn:03d}.png",
                mime="image/png",
                key=f"dl_thumb_{file_hash}_{pn}",
            )

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

        st.markdown("**Download all pages as:**")
        n_cols = 4 if cfg["run_ocr"] else 3
        dl_cols = st.columns(n_cols)
        dl_cols[0].download_button(
            "PDF",
            data=_build_pdf(tuple(png_paths), cfg["dpi"]),
            file_name=f"{stem}_cropped.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_{file_hash}",
        )
        dl_cols[1].download_button(
            "Multi-page TIFF",
            data=_build_tiff(tuple(png_paths), cfg["dpi"]),
            file_name=f"{stem}_cropped.tiff",
            mime="image/tiff",
            use_container_width=True,
            key=f"tiff_{file_hash}",
        )
        dl_cols[2].download_button(
            "ZIP (PNG per page)",
            data=_build_zip(tuple(png_paths), stem),
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
            tmpdir_key2 = f"tmpdir_{file_hash}"
            if tmpdir_key2 in st.session_state:
                shutil.rmtree(st.session_state.pop(tmpdir_key2), ignore_errors=True)
            st.session_state[wiz_done_key] = False
            st.rerun()

    # =========================================================================
    # WIZARD VIEW
    # =========================================================================
    else:
        current = st.session_state[wiz_page_key]

        p_top_key = f"p_top_{file_hash}_{current}"
        p_bot_key = f"p_bot_{file_hash}_{current}"
        if p_top_key not in st.session_state:
            if cfg["detect_mode"] == "Smart (per page)":
                with st.spinner(f"Smart-detecting margins on page {current}…"):
                    try:
                        top_px, bot_px, h_det, _ = _detect_margins_v2(
                            pdf_bytes, current, cfg["dpi"]
                        )
                        st.session_state[p_top_key] = min(75, round(top_px / h_det * 100))
                        st.session_state[p_bot_key] = min(75, round(bot_px / h_det * 100))
                    except Exception:
                        st.session_state[p_top_key] = cfg["top_crop_pct"]
                        st.session_state[p_bot_key] = cfg["bot_crop_pct"]
            else:
                st.session_state[p_top_key] = cfg["top_crop_pct"]
                st.session_state[p_bot_key] = cfg["bot_crop_pct"]

        st.caption(f"**Page {current} of {total}**")
        sl_left, sl_right = st.columns(2)
        sl_left.slider(
            f"Top crop (%) -- page {current}", 0, 75, step=1,
            key=p_top_key,
            help="Rows to remove from the top of this page.",
        )
        sl_right.slider(
            f"Bottom crop (%) -- page {current}", 0, 75, step=1,
            key=p_bot_key,
            help="Rows to remove from the bottom of this page.",
        )

        orig_bytes = _render_page_bytes(pdf_bytes, current, cfg["dpi"])
        h_page   = Image.open(io.BytesIO(orig_bytes)).height
        page_top = st.session_state[p_top_key]
        page_bot = st.session_state[p_bot_key]
        top_px   = round(page_top / 100 * h_page)
        bot_px   = round(page_bot / 100 * h_page)
        cropped_bytes = _apply_crop(orig_bytes, top_px, bot_px, cfg["pad_px"])

        tmpdir_key = f"tmpdir_{file_hash}"
        if tmpdir_key not in st.session_state:
            st.session_state[tmpdir_key] = tempfile.mkdtemp(prefix="ocr_me_")
        _wiz_png = os.path.join(st.session_state[tmpdir_key], f"page{current:03d}.png")
        with open(_wiz_png, "wb") as _fh:
            _fh.write(cropped_bytes)

        overlay_bytes = _draw_crop_overlay(orig_bytes, top_px, bot_px)
        col_left, col_right = st.columns(2)
        col_left.image(overlay_bytes, width="stretch",
                       caption=f"Page {current} -- crop lines")
        col_right.image(Image.open(io.BytesIO(cropped_bytes)),
                        width="stretch",
                        caption=f"Page {current} -- result")
        col_right.download_button(
            f"↓ Page {current} (PNG)",
            data=cropped_bytes,
            file_name=f"{stem}_page{current:03d}.png",
            mime="image/png",
            key=f"png_{file_hash}_{current}",
        )

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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar_settings() -> dict:
    """Render all sidebar controls and return a dict of current values."""
    with st.sidebar:
        st.header("Settings")

        mode = st.radio(
            "Mode",
            ["Single Book", "Batch", "Raw Export", "Visual"],
            index=0,
            key="app_mode",
            help=(
                "**Single Book**: Auto-detect & strip headers/footers, export all pages "
                "as colour images in a ZIP (+ optional footers PDF/ZIP).\n\n"
                "**Batch**: Same as Single Book — processes every uploaded file.\n\n"
                "**Raw Export**: Export the original pages as colour images with no "
                "header/footer removal.\n\n"
                "**Visual**: Step through pages one-by-one with manual crop sliders "
                "and optional OCR."
            ),
        )

        st.subheader("Image quality")
        dpi = st.slider(
            "Rendering DPI", MIN_DPI, MAX_DPI, DEFAULT_DPI, step=50,
            help="Higher DPI = sharper image. 400 DPI is optimal for Apple Live Text.",
        )

        # Defaults for options not rendered in the current mode.
        include_footers = False
        include_photos  = False
        zip_split_mb    = float(ZIP_SPLIT_DEFAULT_MB)
        detect_mode     = "Off"
        top_crop_pct    = 0
        bot_crop_pct    = 0
        pad_px          = 20
        run_ocr         = False
        engine          = "claude"
        threshold_pct   = 50
        text_direction  = "horizontal-rl"
        autocast        = False
        pad             = 16
        bidi_key        = "auto"
        no_legacy_polygons = False
        temperature     = 1.0
        apply_corrections = False

        if mode in ("Single Book", "Batch"):
            st.subheader("Export options")
            include_footers = st.checkbox(
                "Include footers PDF & ZIP",
                value=True,
                key="include_footers",
                help=(
                    "Assemble all detected footnote sections into a labeled PDF "
                    "and a separate image ZIP."
                ),
            )
            include_photos = st.checkbox(
                "Extract photographs",
                value=False,
                key="include_photos",
                help=(
                    "Detect photographic regions and their captions in each page "
                    "and save them as individual PNGs in a ZIP. Uses pixel-domain "
                    "dark-region segmentation — works even when the PDF has no "
                    "embedded image objects."
                ),
            )
            zip_split_mb = float(st.number_input(
                "ZIP split size (MB)",
                min_value=50, max_value=1000, value=ZIP_SPLIT_DEFAULT_MB, step=50,
                help="Split the pages ZIP into parts no larger than this.",
            ))

        elif mode == "Raw Export":
            st.subheader("Export options")
            zip_split_mb = float(st.number_input(
                "ZIP split size (MB)",
                min_value=50, max_value=1000, value=ZIP_SPLIT_DEFAULT_MB, step=50,
                help="Split the pages ZIP into parts no larger than this.",
            ))

        elif mode == "Visual":
            st.subheader("Crop margins")
            detect_mode = st.radio(
                "Auto-detect margins",
                ["Off", "Fast (page 1)", "Smart (per page)"],
                index=0,
                key="detect_mode",
                help=(
                    "**Off**: use the sliders below as-is for all pages.\n\n"
                    "**Fast (page 1)**: scans ink density on page 1 and pre-fills "
                    "the sliders — same crop applied to every page.\n\n"
                    "**Smart (per page)**: uses morphological analysis (OpenCV) to "
                    "detect header, running title, footnote rule, and page-number "
                    "footer independently for each page. "
                    "Handles pages where a footnote takes most of the space."
                ),
            )
            top_crop_pct = st.slider(
                "Top crop (%)", 0, 75, step=1, key="top_crop_pct",
                help="Percentage of page height removed from the top edge.",
            )
            bot_crop_pct = st.slider(
                "Bottom crop (%)", 0, 75, step=1, key="bot_crop_pct",
                help="Percentage of page height removed from the bottom edge.",
            )
            pad_px = st.slider(
                "White padding (px)", 0, 50, step=2, key="crop_pad_px",
                help="White border added around the cropped image.",
            )

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

            if run_ocr:
                engine_label = st.selectbox(
                    "Engine",
                    ["Claude Haiku (API, ~$0.004/page)", "kraken (offline, free)"],
                    index=0,
                    help=(
                        "Claude Haiku: near-perfect Arabic accuracy, handles Quranic "
                        "Uthmanic script with full tashkeel. Requires ANTHROPIC_API_KEY.\n\n"
                        "kraken: fully offline, no API key needed."
                    ),
                )
                engine = "claude" if engine_label.startswith("Claude") else "kraken"

                if engine == "claude" and not _anthropic_key():
                    st.warning(
                        "ANTHROPIC_API_KEY not set -- kraken will be used as fallback. "
                        "Add the key in **Settings -> Secrets** to enable Claude.",
                        icon="⚠️",
                    )

                kraken_label = (
                    "kraken settings (fallback)" if engine == "claude" else "kraken settings"
                )
                with st.expander(kraken_label, expanded=(engine == "kraken")):
                    threshold_pct = st.slider(
                        "Binarization threshold (nlbin)", 10, 90, 50, step=5,
                        help="Raise if faint strokes vanish; lower if noise bleeds in.",
                    )
                    text_direction = st.selectbox(
                        "Text direction",
                        ["horizontal-rl", "horizontal-lr", "vertical-rl", "vertical-lr"],
                        index=0,
                        help="Arabic is right-to-left (horizontal-rl).",
                    )
                    autocast = st.checkbox(
                        "Autocast (mixed precision)", value=False,
                        help="Enable torch.autocast during segmentation.",
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
                        "Run confusables.py substitutions to fix systematic kraken errors. "
                        "Claude usually produces these correctly already."
                    ),
                )

        with st.expander("Active configuration", expanded=False):
            cfg_json: dict = {"mode": mode, "dpi": dpi}
            if mode in ("Single Book", "Batch"):
                cfg_json.update({"include_footers": include_footers,
                                 "include_photos": include_photos,
                                 "zip_split_mb": zip_split_mb})
            elif mode == "Raw Export":
                cfg_json["zip_split_mb"] = zip_split_mb
            elif mode == "Visual":
                cfg_json.update({
                    "detect_mode": detect_mode,
                    "top_crop_pct": top_crop_pct,
                    "bot_crop_pct": bot_crop_pct,
                    "pad_px": pad_px,
                    "run_ocr": run_ocr,
                })
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
        mode=mode,
        dpi=dpi,
        include_footers=include_footers,
        include_photos=include_photos,
        zip_split_mb=zip_split_mb,
        detect_mode=detect_mode,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Arabic PDF2OCR", page_icon="\U0001f4c4", layout="wide")
    st.title("Arabic PDF2OCR")

    # Seed Visual-mode crop slider keys before sidebar renders widgets.
    for _k, _d in [("top_crop_pct", 0), ("bot_crop_pct", 0), ("crop_pad_px", 20)]:
        if _k not in st.session_state:
            st.session_state[_k] = _d
    # Flush pending auto-detect values from a previous st.rerun().
    for _k in ("top_crop_pct", "bot_crop_pct"):
        _pk = f"_pending_{_k}"
        if _pk in st.session_state:
            st.session_state[_k] = st.session_state.pop(_pk)

    cfg = _sidebar_settings()
    mode = cfg["mode"]

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)", type=["pdf"], accept_multiple_files=True,
    )
    if not uploaded_files:
        hints = {
            "Single Book": (
                "Upload a PDF to auto-detect and strip headers/footers, "
                "then download all cropped pages as a colour-image ZIP."
            ),
            "Batch": (
                "Upload one or more PDFs to process them all automatically."
            ),
            "Raw Export": (
                "Upload a PDF to export all original pages as colour images "
                "(no header/footer removal)."
            ),
            "Visual": (
                "Upload a PDF to step through pages one-by-one and set crop "
                "margins manually. Optional OCR is also available."
            ),
        }
        st.info(hints.get(mode, "Upload a PDF to begin."))
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

        if mode in ("Single Book", "Batch"):
            _render_single_book_ui(pdf_bytes, file_hash, stem, total, cfg)
        elif mode == "Raw Export":
            _render_raw_export_ui(pdf_bytes, file_hash, stem, total, cfg)
        elif mode == "Visual":
            _render_visual_ui(pdf_bytes, file_hash, stem, total, cfg)

        st.divider()


if __name__ == "__main__":
    main()
