# Deploying a kraken-based Streamlit App — Requirements & Pitfalls

This document captures every hard-won lesson from deploying this Arabic OCR
app to Streamlit Community Cloud.  Treat it as a living reference: update it
whenever a new deployment blocker is encountered.

---

## 1. Python Version — the Single Most Important Setting

### The constraint

kraken 7.x carries a hard metadata restriction:

```
Requires-Python: >=3.10,<3.14
```

Streamlit Community Cloud **ignores `runtime.txt` and `.python-version`**.
Both files are present in this repo but have no effect.  The only way to
choose a Python version is through the **Advanced settings** dialog that
appears when you first deploy the app.

### How to set it

1. Go to <https://share.streamlit.io> and click **Create app**.
2. Point it at the correct repo, branch, and `streamlit_app.py`.
3. Click **Advanced settings** (below the main form).
4. Set **Python version → 3.12**.
5. Deploy.

> **Critical:** Python version cannot be changed after deployment.
> If the wrong version was chosen, delete the app and redeploy.

### Why exactly 3.12?

| Python | kraken 7 | coremltools 9 | scikit-image 0.25 | scipy 1.15 | lightning/pytorch-lightning |
|--------|----------|--------------|-------------------|------------|----------------------------|
| 3.10   | ✓        | ✓            | ✓                 | ✓          | ✓ |
| 3.11   | ✓        | ✓            | ✓                 | ✓          | ✓ |
| **3.12** | **✓**  | **✓**        | **✓**             | **✓**      | **✓** |
| 3.13   | ✓        | ✓            | ✓                 | ✓          | ✓ |
| 3.14   | ✗        | ✗            | ✗                 | ✗          | ✗ |

Python 3.14 breaks *every single* compiled dependency in the chain.

---

## 2. The `lightning` Package — Quarantine Workaround

### What happened

The `lightning` package (required by kraken as `lightning~=2.6.0`) was
**quarantined on PyPI** on 30 April 2026 after a bad actor pushed versions
2.6.2 and 2.6.3.  Quarantine means the package is invisible to pip and uv;
no version can be downloaded or installed.

Error seen in deployment logs:

```
lightning~=2.6.0 (from versions: none)
```

### The solution — `lightning-compat/` shim

`pytorch-lightning 2.6.1` was **not** quarantined.  It ships two packages
inside its wheel:

- `pytorch_lightning` — the high-level training API
- `lightning_fabric` — the low-level `Fabric` class

The only import kraken needs at **inference time** (i.e. when running OCR,
not training) is:

```python
# kraken/lib/vgsl/model.py line 28
from lightning.fabric import Fabric
```

The `lightning-compat/` directory in this repo is a tiny Python package
that:

1. Declares itself to pip as `lightning==2.6.1` (satisfies kraken's
   `lightning~=2.6.0` dependency check).
2. Delegates the `lightning.fabric` namespace to `lightning_fabric`
   (provided by `pytorch-lightning`).

**File layout:**

```
lightning-compat/
├── pyproject.toml              # name="lightning", version="2.6.1",
│                               # depends on pytorch-lightning==2.6.1
└── lightning/
    ├── __init__.py             # re-exports seed_everything from lightning_fabric
    └── fabric/
        └── __init__.py         # exposes Fabric from lightning_fabric.fabric
```

**`requirements.txt` entry:**

```
lightning @ ./lightning-compat
```

pip/uv install the local directory as the `lightning` package, which:

- Satisfies kraken's dep resolver (sees `lightning==2.6.1` already installed)
- Pulls in `pytorch-lightning==2.6.1` as a transitive dependency
- Provides working `from lightning.fabric import Fabric` at runtime

> **Note:** If/when the PyPI quarantine on `lightning` is lifted and the
> package is restored, replace `lightning @ ./lightning-compat` with
> `lightning~=2.6.0` and delete the `lightning-compat/` directory.

### Why not install from GitHub?

```
lightning @ git+https://github.com/Lightning-AI/pytorch-lightning.git@2.6.1
```

This would also bypass PyPI quarantine, but requires building from source
(slow, ~5 min build on Streamlit Cloud), and pip still enforces
`Requires-Python` from the cloned `pyproject.toml`.  The local shim is
faster, smaller, and more robust.

---

## 3. Full Dependency Reference

### `requirements.txt`

```
streamlit
torch>=2.4.0,<=2.10.0
lightning @ ./lightning-compat   # local shim; see Section 2
kraken==7.0.1
pdf2image
Pillow
numpy
scipy
```

| Package | Version constraint | Why pinned |
|---------|-------------------|-----------|
| `torch` | `>=2.4.0,<=2.10.0` | kraken 7 hard requirement; torch 2.10 has cp312 AND cp314 wheels |
| `lightning @ ./lightning-compat` | local | PyPI quarantine bypass |
| `kraken` | `==7.0.1` | Latest stable; pin to avoid surprise upgrades |
| Others | unpinned | Streamlit/PIL/pdf2image track upstream; no known conflicts |

### `packages.txt` (apt dependencies)

```
poppler-utils
```

Required by `pdf2image` for the `pdfinfo` and `pdftoppm` CLI tools.

### `runtime.txt` / `.python-version`

These files are present but **ignored by Streamlit Cloud**.  They are kept
as documentation of intent and may work on other platforms (Hugging Face
Spaces, Render, Railway).

---

## 4. Model Download

The Arabic OCR model is **not bundled** in the repo (it is ~20 MB).  It is
downloaded at first startup from:

```
https://raw.githubusercontent.com/OpenITI/AOCP_print_models
    /refs/heads/main/transcription/apt-20221130.mlmodel
```

Cached at `~/.kraken_models/apt-20221130.mlmodel`.  Streamlit Cloud's
ephemeral filesystem means the model is re-downloaded on each cold start.
The `@st.cache_resource` decorator ensures it is only downloaded once per
running instance.

---

## 5. Alternative Platforms

If Streamlit Community Cloud continues to cause issues, the following
platforms all respect Python version selection via Dockerfile or settings:

| Platform | Python version control | Cost |
|----------|----------------------|------|
| Hugging Face Spaces | `python_version: "3.12"` in README YAML header | Free tier |
| Render | Dockerfile | Free tier |
| Railway | Dockerfile | Free tier |
| Google Cloud Run | Dockerfile | Pay per use |

**HF Spaces README header example:**

```yaml
---
title: Arabic PDF OCR
sdk: streamlit
sdk_version: "1.44"
python_version: "3.12"
app_file: streamlit_app.py
---
```

---

## 6. Deployment Checklist

Before every deploy:

- [ ] Python 3.12 selected in Streamlit Cloud Advanced settings
- [ ] `lightning-compat/` directory present in repo
- [ ] `requirements.txt` references `lightning @ ./lightning-compat`
- [ ] `packages.txt` contains `poppler-utils`
- [ ] `streamlit_app.py` uses `@st.cache_resource` for model (avoids re-download)
- [ ] Model URL is reachable (test: `curl -I <url>`)
- [ ] No secrets or API keys committed to the repo

---

## 7. Known Constraints of kraken 7.0.1

| Constraint | Impact |
|-----------|--------|
| `requires-python <3.14` | Cannot deploy on Python 3.14+ without patching the wheel |
| `lightning~=2.6.0` required | Blocked if lightning is quarantined on PyPI |
| `coremltools~=9.0` required | No cp314 wheels; blocks Python 3.14 |
| Greedy CTC decoding only | No beam search; character accuracy is model-limited |
| No streaming inference | Entire page binarized and segmented before first character |
| Model download at startup | Cold starts take 30–60 s on slow connections |
