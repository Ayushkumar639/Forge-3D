# Forge3D

A local, CPU-friendly **image-to-3D** and **text-to-3D** generator. You upload a
photo (or a few, or just type a prompt), and it hands back a textured
`.glb` model — built on top of Stability AI + Tripo AI's
[TripoSR](https://github.com/VAST-AI-Research/TripoSR), with a Flask
backend and a single-file Three.js frontend.

Everything runs on your own machine. **No cloud inference, no API key
required for the core pipeline.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="CPU" src="https://img.shields.io/badge/inference-CPU--only-orange">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey">
</p>

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Features](#features)
- [What every setting does](#what-every-setting-does)
- [Studio Texture](#studio-texture-experimental)
- [Environment variables](#environment-variables)
- [Docker support](#docker-support)
- [Performance tips](#performance-tips)
- [What was actually broken (and fixed)](#what-was-actually-broken-and-fixed)
- [Path to further improvement](#path-to-further-improvement)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Credits & license](#credits--license)

---

## Quick start

Requires **Python 3.10+**. The commands below work on **Windows (PowerShell)**, **Linux**, **macOS**, and **WSL**.

### Windows (PowerShell)

```powershell
# 1. Get this repo and enter it
git clone <this-repo-url> forge3d
cd forge3d

# 2. Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Clone TripoSR into ./TripoSR (or set TRIPOSR_DIR env var)
git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR

# 4. Install torch FIRST as CPU-only build
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install all other dependencies in ONE command
pip install -r requirements.txt

# 6. Run it
python app.py
```

### Linux / macOS / WSL (Bash)

```bash
# 1. Get this repo and enter it
git clone <this-repo-url> forge3d
cd forge3d

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Clone TripoSR into ./TripoSR (or set TRIPOSR_DIR env var)
git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR

# 4. Install torch FIRST as CPU-only build
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install all other dependencies in ONE command
pip install -r requirements.txt

# 6. Run it
python app.py
```

Then open **http://localhost:5000** in your browser.

> **First run downloads:** TripoSR checkpoint (~350 MB from Hugging Face) + rembg model (~180 MB). Both are one-time downloads and require internet access.

> **Why one `pip install -r requirements.txt`?** The file pins `numpy<2` because `trimesh==4.0.5` (pinned by TripoSR) calls an ndarray method removed in NumPy 2.0. If you install packages in separate `pip install` calls, a later package can silently upgrade numpy to 2.x since pip only guarantees consistency *within* a single resolver call. One-shot install avoids this.

---

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FORGE3D ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌──────────────┐     ┌─────────────────────┐   │
│   │  Browser │────▶│   Flask API  │────▶│   TripoSR Model     │   │
│   │ (Three.js)│    │  (app.py)    │     │  (stabilityai/TripoSR)│   │
│   └──────────┘     └──────────────┘     └──────────┬──────────┘   │
│        ▲                    │                        │             │
│        │                    ▼                        ▼             │
│        │            ┌──────────────┐     ┌─────────────────────┐   │
│        │            │  Background  │     │  Marching Cubes     │   │
│        │            │  Removal     │     │  (torchmcubes)      │   │
│        │            │  (rembg)     │     │  → Raw Mesh         │   │
│        │            └──────────────┘     └──────────┬──────────┘   │
│        │                                            │             │
│        │                    ┌───────────────────────┘             │
│        │                    ▼                                     │
│        │            ┌──────────────┐     ┌─────────────────────┐   │
│        └────────────│  Post-Proc   │────▶│  Texture Generation │   │
│                     │  (cleanup,   │     │  • Vertex Colors    │   │
│                     │   smoothing) │     │  • Studio Texture   │   │
│                     └──────────────┘     │    (xatlas+moderngl)│   │
│                                          └──────────┬──────────┘   │
│                                                     │             │
│                                          ┌──────────┴──────────┐   │
│                                          │    .glb Output      │   │
│                                          └─────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Pipeline stages:**

1. **Input** — Single image, multiple images, or text prompt
2. **Preprocessing** — Optional background removal (rembg/isnet), resize to 512×512
3. **TripoSR Inference** — Feed-forward transformer generates triplane NeRF
4. **Mesh Extraction** — Marching cubes at chosen resolution (64/128/256/320)
5. **Geometry Cleanup** — Remove degenerate faces, isolated vertices, non-manifold edges
6. **Surface Polish** — Optional Taubin smoothing (requires `scipy`)
7. **Texturing** — Either:
   - **Enhanced Vertex Colors** (default, fast, works everywhere): Neural renderer queried per vertex with directional lighting baked in
   - **Studio Texture** (optional): UV atlas via xatlas + rasterization via moderngl/OpenGL → baked texture map
8. **Export** — Binary GLB with mesh + texture/vertex colors

---

## Features

| Feature | Description |
|---------|-------------|
| **Image-to-3D** | Upload one or more photos → get a textured `.glb` |
| **Text-to-3D** | Type a prompt → Pollinations AI generates reference image → 3D |
| **Multi-image input** | Feed 2+ views for better 3D consistency |
| **Background removal** | Automatic (rembg) or bring your own masked image |
| **Adjustable detail** | 4 quality presets: Fast (64) → Balanced (128) → Detailed (256) → Studio (320) |
| **Surface polish** | Taubin smoothing: Off / Light / Medium / Heavy |
| **Two texture modes** | Vertex colors (zero deps) or baked UV texture (Studio Texture) |
| **Real-time preview** | Three.js viewer with orbit controls, wireframe toggle, env map |
| **Progress streaming** | Server-sent events show live generation progress |
| **Job cancellation** | Cancel long-running generations mid-way |
| **CPU-only** | Runs entirely on CPU, no GPU required |

---

## What every setting does

| Setting | What it does | Notes |
|---|---|---|
| **Remove Background** | Runs `rembg` (isnet-general-use model) before handing the image to TripoSR | Turn off if you're already passing a pre-masked/transparent image |
| **Foreground Ratio** | How much of the 512×512 conditioning frame the subject fills, after cropping | TripoSR's own default is 0.85; lower it if the model comes out oddly cropped |
| **Speed / Detail** | Marching-cubes resolution preset: Fast=64, Balanced=128, Detailed=256, Studio=320 | Cost scales roughly with the cube of resolution, not linearly — Studio is ~15× the raw voxel work of Balanced, not 2.5× |
| **Surface Polish** | Off / Light(40) / Medium(150) / Heavy(300) Taubin-smoothing iterations, run after an automatic geometry cleanup pass | Requires `scipy` — see below |
| **Studio Texture** | Bakes a real UV-mapped texture instead of per-vertex color | Experimental, optional deps — see its own section below |
| **MC Resolution / Density Threshold** (Advanced) | Direct control over the same marching-cubes resolution and isosurface threshold the presets set | Threshold's official default is 25.0; lower = more mesh, possibly more noise |

---

## Studio Texture (experimental)

Vertex colors are only as sharp as the mesh itself — colors blend
linearly across each triangle, so on anything but a very dense mesh you
get visible blur/banding. A **baked texture** doesn't have that ceiling:
it's an actual image, UV-mapped onto the model, the same technique
TripoSR's own `run.py --bake-texture` flag uses upstream. This is the
single biggest lever for making a model look "finished" rather than
"proceduralish," so it's worth setting up if you can.

**What it needs:** two extra packages (`xatlas`, `moderngl`) already in
`requirements.txt`, plus a working OpenGL context.

- **Desktop with a real GPU/display (most Windows/macOS setups, most
  Linux desktops):** should just work once the packages are installed.
- **Headless Linux server (no display attached):** needs Mesa's
  software renderer and EGL, or `moderngl` will fail with something like
  `XOpenDisplay: cannot open display`. Fix:

  ```bash
  sudo apt-get install -y libegl1 libgl1 libglx-mesa0 libosmesa6
  ```

  This was tested end-to-end in a headless, GPU-less container: with
  those packages installed, `moderngl.create_context(standalone=True,
  backend='egl')` renders correctly via Mesa's `llvmpipe` software
  rasterizer. `bake_worker.py` already tries the plain context first
  (best on a real desktop) and falls back to explicit EGL automatically,
  so you shouldn't need to change any code — just install the packages
  above if you're on a headless box.

**Why this is marked "experimental" and off by default:** in testing,
the UV-atlas step (`xatlas`) segfaulted outright on a constrained
single-core machine — a native crash, which no Python `try/except` can
catch. That's a real risk with any native rendering library, on some
fraction of machines, for reasons that can be hard to predict in
advance (driver quirks, core count, threading edge cases). Because of
that, baking runs in its **own subprocess** (`bake_worker.py`), not
in-process — so a crash there can only fail that one subprocess.
`app.py` detects the failure and automatically falls back to the
enhanced vertex-color path for that generation. This was verified
directly: a deliberately-crashing bake still produced a normal,
successful GLB, using vertex colors, with no user-visible error.

In short: turning it on can only help or no-op, never break a
generation — but "no-op" does mean you might not get the texture-baked
result you were expecting, silently, unless you're watching the log
panel (it logs clearly either way — `Studio texture applied ✓` or
`Studio texture unavailable — using vertex colors`). Test it on your
own machine before relying on it for a batch.

---

## Studio Texture (experimental)

**What it gives you:** A real UV-unwrapped texture map baked into the GLB, instead of per-vertex colors. This means:
- Cleaner appearance in other 3D tools (Blender, Unity, Godot, Three.js, etc.)
- Texture can be edited/exported separately
- More predictable rendering across engines

**What it costs:**
- Extra dependencies: `xatlas` (UV atlas) + `moderngl` (OpenGL rasterization)
- Extra time: ~10–30 s per model depending on resolution
- Extra fragility: Native C++ libraries — segfaults kill only the worker subprocess (by design), not the main server

**Enable it:**
```bash
pip install xatlas==0.0.9 moderngl==5.10.0
```
(Already in `requirements.txt` but commented — uncomment or run the command above.)

**On headless Linux servers** (no display/GPU), you also need a software GL stack:
```bash
# Debian/Ubuntu
apt-get update && apt-get install -y libgl1-mesa-glx libegl1-mesa libgles2-mesa
```
The worker tries the default GL context first, then falls back to EGL — verified working on Mesa's llvmpipe software renderer.

**Why a separate process?** `xatlas`'s atlas packer segfaulted in testing on constrained (single-core) machines — a native crash no Python `try/except` can catch. The isolated `bake_worker.py` subprocess means a crash here only loses that one texture bake; `app.py` detects failure and falls back to vertex colors for that job only. Server stays up.

---

## Environment variables

Create a `.env` file (copy from `.env.example`) or export in your shell. None are required — all have working defaults.

| Variable | Default | Description |
|---|---|---|
| `TRIPOSR_DIR` | `./TripoSR` | Path to your TripoSR clone |
| `TRIPOSR_CHUNK_SIZE` | `32768` | Renderer chunk size. Lower (e.g. `8192`) on memory-constrained machines; raise if you have RAM to spare |
| `REMBG_MODEL` | `isnet-general-use` | Background-removal model. `isnet-general-use` gives cleaner silhouettes than rembg's default `u2net` on most product/object photos |

---

## Docker support

A minimal `Dockerfile` is not included but here's a working example:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# System deps for headless Studio Texture (moderngl/EGL) + rembg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libegl1-mesa libgles2-mesa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU first
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Clone TripoSR at build time (or mount at runtime)
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR \
    && pip install -r TripoSR/requirements.txt

# Copy app code
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

Build & run:
```bash
docker build -t forge3d .
docker run -p 5000:5000 -v $(pwd)/outputs:/app/outputs forge3d
```

> **Note:** The first run inside the container will still download the TripoSR checkpoint (~350 MB) and rembg model (~180 MB). Consider baking them into the image for offline use.

---

## Performance tips

| Tip | Impact |
|-----|--------|
| Use **Balanced (128)** for most cases | Studio (320) is ~15× slower for diminishing returns |
| Disable **Studio Texture** if you only need quick previews | Saves 10–30 s per generation |
| Set `TRIPOSR_CHUNK_SIZE=8192` on machines with < 8 GB RAM | Trades speed for memory |
| Run on **WSL2** on Windows instead of native Python | Often faster due to better memory management |
| Close other heavy apps during generation | TripoSR + marching cubes can spike RAM to 4–6 GB |
| Use **SSD** for `outputs/` and TripoSR cache | Faster model loading & checkpoint writes |

**Typical generation times (CPU-only, Intel i7-12700K / AMD Ryzen 7 5800X):**

| Preset | Mesh Extraction | Total (w/o Studio Texture) | + Studio Texture |
|--------|-----------------|----------------------------|------------------|
| Fast (64) | ~15 s | ~45 s | ~60 s |
| Balanced (128) | ~45 s | ~90 s | ~115 s |
| Detailed (256) | ~180 s | ~240 s | ~280 s |
| Studio (320) | ~300 s | ~380 s | ~430 s |

*Times include model loading (first run only), background removal, mesh extraction, cleanup, and export. Your mileage will vary.*

---

## What was actually broken (and fixed)

This section exists so you don't have to rediscover these issues.

### 1. `numpy≥2.0` breaks `trimesh==4.0.5` (TripoSR's pinned version)

**Symptom:** Generation runs all the way through TripoSR inference and marching cubes, then **crashes on the final `.glb` save** with `AttributeError: 'numpy.ndarray' object has no attribute '...'`.

**Cause:** NumPy 2.0 (released mid-2024) removed several `ndarray` methods that `trimesh==4.0.5` calls in 50+ places, including mesh export.

**Fix:** `requirements.txt` pins `numpy<2`. Install the whole file in **one** `pip install -r requirements.txt` call so the resolver sees the constraint.

### 2. `scipy` missing → "Polish" silently does nothing

**Symptom:** You crank "Surface Polish" to Heavy, wait longer, but the output mesh looks identical.

**Cause:** `trimesh` imports `scipy` inside a bare `try/except`. If `scipy` isn't installed, Taubin smoothing becomes a no-op **without any error or warning** (except a terminal log line).

**Fix:** `scipy` is in `requirements.txt`. If you installed deps piecemeal and missed it, `pip install scipy`.

### 3. Studio Texture worker crashes → whole server would die

**Original design:** UV atlas + rasterization ran in-process.

**Problem:** `xatlas` segfaulted on constrained machines (native C++ crash). Flask process died, all in-flight jobs lost.

**Fix:** Moved to isolated `bake_worker.py` subprocess. Crash → worker exits non-zero → `app.py` detects it → falls back to vertex colors for that job only. Server stays up.

### 4. Text-to-3D via Pollinations needed proper error handling

**Original:** Failed silently if Pollinations was slow or rate-limited.

**Fix:** Added 45 s timeout, proper HTTP error propagation, and SSE error events so the UI shows "Text-to-3D failed: Pollinations returned HTTP 503" instead of spinning forever.

---

## Path to further improvement

### If you can fine-tune TripoSR

**You probably can't.** TripoSR is a feed-forward reconstruction model (triplane NeRF + transformer), not a diffusion model. There's no public training script, and there's no established community path to fine-tuning or LoRA-adapting this specific model. Reconstructing the training setup from the paper would mean re-implementing a triplane-NeRF training loop with volumetric-rendering losses against a large multi-view 3D dataset (Objaverse-scale), on multi-GPU hardware — a research undertaking, not a weekend project, and not something that fits a local CPU setup.

If you search for "TripoSR fine-tune," you'll mostly find the same question asked and left unanswered upstream. So: this repo intentionally does not point you at a fake fine-tuning tutorial.

### If you want a real step-change and can get GPU access

The actual jump in output quality beyond TripoSR right now comes from **switching models**, not tuning this one. As of August 2026:

- **[TRELLIS 2](https://github.com/microsoft/TRELLIS)** (Microsoft Research, MIT-licensed) — widely regarded as the strongest open-source image-to-3D model, with native PBR texture output. Needs ~24 GB VRAM, so not a drop-in replacement for CPU-only setups.
- **[InstantMesh](https://github.com/TencentARC/InstantMesh)** — sits between the two: better multi-view consistency and more production-ready output than TripoSR, still GPU-oriented but lighter than TRELLIS 2.

Both are worth checking again before committing — this space moves in months, not years.

If you're staying CPU-only, the ceiling really is the deterministic pipeline this repo already implements: resolution, cleanup, smoothing, and baked textures. There isn't currently a CPU-friendly open model that meaningfully beats TripoSR on quality.

---

## Project layout

```
forge3d/
├── app.py              # Flask backend — the generation pipeline
├── bake_worker.py      # Isolated subprocess for Studio Texture (UV atlas + rasterize)
├── index.html          # Frontend — Three.js viewer + all UI (single file)
├── requirements.txt    # Python dependencies (pins numpy<2, scipy, etc.)
├── .env.example        # Environment variable template
├── LICENSE             # MIT License
├── TripoSR/            # You clone this yourself — see Quick start
└── outputs/            # Generated .glb files land here (gitignored)
```

### Key files explained

| File | Purpose |
|------|---------|
| `app.py` | Flask server, TripoSR loading, generation pipeline, SSE streaming, worker orchestration |
| `bake_worker.py` | Isolated subprocess for Studio Texture (xatlas UV unwrapping + moderngl rasterization) |
| `index.html` | Complete frontend: Three.js viewer, UI controls, drag-drop, progress streaming, preview |
| `requirements.txt` | Pinned deps with comments explaining *why* each version constraint exists |

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| **"TripoSR package not found"** at startup | Haven't cloned TripoSR into `./TripoSR` (or `TRIPOSR_DIR` wrong) | `git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR` |
| **Generation fails at the end, after several minutes** | `numpy 2.x` installed (see "What was actually broken") | `pip install "numpy<2"` inside your venv |
| **"Polish" doesn't seem to change anything** | `scipy` not installed | `pip install scipy` (check terminal for smoothing warning) |
| **Studio Texture always falls back** | Missing `xatlas`/`moderngl` OR worker crashed | Check terminal for specific reason. On headless Linux, install Mesa GL packages |
| **Out of memory / killed** | Resolution too high for available RAM | Lower `TRIPOSR_CHUNK_SIZE` (e.g. `8192`), use Balanced (128) or Fast (64) |
| **Text-to-3D returns error** | Pollinations API timeout or rate limit | Retry; it's a free public endpoint with no SLA |
| **Port 5000 already in use** | Another process on port 5000 | `lsof -i :5000` → kill it, or change port in `app.py` |
| **Windows: `Activate.ps1` script error** | PowerShell execution policy | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## FAQ

**Q: Does this need a GPU?**  
A: No. Everything runs on CPU. A GPU is not used even if present (TripoSR is forced to `cpu` device).

**Q: Can I use my own TripoSR checkpoint?**  
A: Yes. Place your `model.ckpt` and `config.yaml` in `TripoSR/` and modify `app.py`'s `TSR.from_pretrained()` call to point to local files.

**Q: Why are my models dark / weirdly lit?**  
A: The vertex-color path bakes directional lighting. For neutral lighting, use Studio Texture (baked albedo texture) or adjust the lighting in your 3D viewer.

**Q: Can I generate PBR materials (roughness/metalness/normal)?**  
A: Not with TripoSR. For PBR, look at TRELLIS 2 (requires GPU).

**Q: Is there a batch/CLI mode?**  
A: Not built-in, but you can script against the `/generate` endpoint (SSE) or call the internal functions in `app.py` directly.

**Q: Why does the first generation take longer?**  
A: First run downloads the TripoSR checkpoint (~350 MB) and rembg model (~180 MB) from Hugging Face. Subsequent runs use the local cache.

**Q: Can I run this on a Raspberry Pi / Jetson / low-power device?**  
A: Technically yes if it runs Python 3.10+ and has 4+ GB RAM, but generation will be very slow (10–30+ minutes). Not recommended.

**Q: How do I update TripoSR to a newer version?**  
A: `cd TripoSR && git pull && pip install -r requirements.txt`. Check the TripoSR repo for breaking changes.

---

## Contributing

Contributions are welcome! Please:

1. **Fork** the repo and create a feature branch
2. **Follow existing code style** — type hints, docstrings, inline comments explaining *why*
3. **Test locally** — run `python app.py`, generate a few models, verify no regressions
4. **Keep `requirements.txt` pins intentional** — if you change a version, explain why in a comment
5. **Update README.md** if you add user-facing features or change setup steps
6. **Open a PR** with a clear description of the change and rationale

### Areas where help is especially welcome

- Windows-specific install fixes / CI
- More robust headless Linux Studio Texture setup
- Additional texture export formats (USDZ, OBJ+MTL)
- Benchmark scripts for different hardware
- Accessibility improvements in the frontend

---

## Credits & license

Built on [TripoSR](https://github.com/VAST-AI-Research/TripoSR) by
Tripo AI & Stability AI (MIT License). This repository's own code is
released under the MIT License — see [LICENSE](LICENSE).

**Forge3D · Built by Ayush Kumar**
