# Forge3D

A local, CPU-friendly image-to-3D and text-to-3D generator. You upload a
photo (or a few, or just type a prompt), and it hands back a textured
`.glb` model — built on top of Stability AI + Tripo AI's
[TripoSR](https://github.com/VAST-AI-Research/TripoSR), with a Flask
backend and a single-file Three.js frontend.

Everything runs on your own machine. No cloud inference, no API key
required for the core pipeline.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="CPU" src="https://img.shields.io/badge/inference-CPU--only-orange">
</p>

---

## Contents

- [Quick start](#quick-start)
- [What every setting does](#what-every-setting-does)
- [Studio Texture](#studio-texture-experimental)
- [What was actually broken (and fixed)](#what-was-actually-broken-and-fixed)
- [Path to further improvement](#path-to-further-improvement)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Credits & license](#credits--license)

---

## Quick start

Requires Python 3.10+. These commands are written for a Unix-like shell
(Linux/macOS/WSL); on native Windows use PowerShell equivalents (`venv\Scripts\activate`, etc).

```bash
# 1. Get this repo and enter it
git clone <this-repo-url> forge3d
cd forge3d

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Clone TripoSR itself into ./TripoSR (app.py expects it exactly here,
#    or set TRIPOSR_DIR to point somewhere else)
git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR

# 4. Install torch FIRST, as a CPU-only build, so step 5 doesn't pull a
#    multi-GB CUDA build you don't need on a CPU-only setup
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install everything else in ONE command, so pip's resolver sees every
#    constraint at once (see "numpy" note below for why this matters)
pip install -r requirements.txt

# 6. Run it
python app.py
```

Then open **http://localhost:5000**.

First generation will download the TripoSR checkpoint (~350MB) from
Hugging Face, and rembg will download its background-removal model
(~180MB) — both one-time, both need network access.

> **Why one `pip install -r requirements.txt` and not several smaller
> calls?** `requirements.txt` pins `numpy<2` for a real reason (see
> below). If you install packages one at a time across several `pip
> install` calls, a later package can silently pull numpy back up to 2.x
> because pip only guarantees consistency *within* a single resolver
> call. Installing the whole file in one shot avoids that.

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

## What was actually broken (and fixed)

This section exists because several of these were **silent** failures —
things that don't throw an error message pointing at the cause, they
just quietly produce a worse result (or, in two cases, no result at
all). If you were wondering why quality seemed capped or a setting
didn't seem to do anything, one of these is probably why.

1. **The web page 404'd on a fresh checkout.** `app.py` served
   `index.html` from a `static/` subfolder that never existed in this
   project's actual file layout — flat, `app.py` and `index.html` side
   by side. Fixed by serving directly from the project folder.

2. **GLB export crashed on the last step, after the entire (multi-minute)
   generation had already run.** `trimesh==4.0.5` — the exact version
   TripoSR's own `requirements.txt` pins — calls an ndarray method
   (`.ptp()`) that NumPy 2.0 removed in mid-2024, in 50+ places across
   the library, including mesh export. Any modern `pip install numpy`
   (2.x is the default today) made every generation fail right as the
   finished model was about to be written to disk. This was found by
   actually running the export path, not just reading the code — the
   traceback pointed at `trimesh/util.py`'s `allclose()`. Fixed by
   pinning `numpy<2` in `requirements.txt`, which is genuinely what this
   pinned trimesh version was built against. `app.py` also prints a
   clear warning at startup if it detects numpy 2.x anyway, so this
   doesn't fail silently even if your environment ends up mismatched
   some other way.

3. **"Polish" (Taubin smoothing) could silently do nothing.**
   `trimesh.smoothing` imports `scipy` inside a bare
   `try/except ImportError: pass`. TripoSR's own `requirements.txt`
   never lists `scipy` (vanilla TripoSR doesn't smooth anything), so a
   clean install can easily end up without it — at which point Polish
   (Medium, the default) silently no-ops instead of raising an error you'd
   notice. Confirmed directly: with scipy absent, the exact call this app
   makes throws `NameError: name 'coo_matrix' is not defined`. Fixed by
   adding `scipy` to `requirements.txt`, plus a startup check and a
   `try/except` around the actual call so it degrades to "skip
   smoothing, log why" instead of taking the whole generation down if
   scipy is ever missing anyway.

4. **A live-looking Hugging Face API token was hardcoded in the shipped
   frontend JavaScript.** Anyone opening dev tools — or anyone who forks
   this repo — gets that credential. Removed outright rather than
   patched. Separately, Hugging Face's free serverless inference API has
   moved away from directly reliable text-to-image support since that
   code was written (their `hf-inference` provider now focuses on
   smaller CPU-friendly tasks; image generation goes through a
   provider-routed, largely credit-metered system), so that fallback
   tier wasn't reliably working anyway. The text-to-image chain is now:
   current Pollinations.ai endpoint → legacy Pollinations.ai endpoint →
   local procedural placeholder (all client-side, no key required for
   the two Pollinations tiers, all three needing zero setup).

5. **Multi-image mode never actually told the frontend which file
   finished, even though the frontend already had correct code to
   display it.** The backend intercepted each real `result_ready`
   event and substituted a generic progress message instead of
   forwarding it, so the per-file log lines (`✓ Model ready: photo2.jpg`)
   could never fire. The final gallery still worked, via a separate
   server-side results list attached to the closing `done` event, so
   this wasn't visible unless you were watching the log closely. Fixed
   by forwarding the real event.

6. **Dead code left over from earlier iterations:** a
   `/genimg/<id>` preview reference on the frontend that had no matching
   backend route (harmless — a separate, working code path already set
   the preview image directly from the generated blob — but confusing to
   read), and an unused `canvas_input` form field the backend never
   read (it detects a canvas-generated placeholder via a pixel watermark
   instead). Both removed.

7. **No mesh cleanup pass.** Raw marching-cubes output can carry
   duplicate/degenerate faces and inconsistent winding; none of that was
   being cleaned up before smoothing ran on top of it. Added a cleanup
   step (`merge_vertices`, drop degenerate/duplicate faces, fix normals)
   that runs first.

8. **Marching-cubes resolution was capped well below what TripoSR
   itself considers its own default.** The old "Detailed" preset topped
   out at 192, and the advanced slider couldn't go higher. TripoSR's own
   reference Gradio app defaults to 256 with a max of 320. Presets and
   the slider ceiling now go up to 320 to match, while leaving the
   default (Balanced/128) unchanged so nothing gets slower without you
   asking for it.

9. **Vertex-color-only output**, which blurs across large triangles at
   low-to-medium mesh density. Added optional UV-texture baking (Studio
   Texture, above) as a real alternative.

10. **rembg used its implicit default (u2net) with no session reuse.**
    Switched to `isnet-general-use`, which gives cleaner silhouettes on
    general objects — every downstream step (crop, resize, orientation)
    inherits whatever the mask gets wrong, so this has an outsized effect
    on final quality. Falls back to the default model automatically if
    it can't be fetched.

---

## Path to further improvement

Everything above gets you a correctly-working, better-tuned version of
*this same pipeline*. If you've turned on Studio Texture, pushed
resolution to Studio, and it's still not where you want it, here's
where the ceiling actually is and what's realistic beyond it.

### What's still free to tune (do this first)

In priority order, roughly by how much it tends to move output quality
for the effort involved:

1. **Input photo quality.** This has more headroom than any pipeline
   setting. TripoSR is a single-image reconstruction model — it can't
   recover detail that isn't visible in the photo. Even, diffuse
   lighting (no hard shadows), a plain background, the object filling
   most of the frame, and a three-quarter angle that shows depth all
   measurably help.
2. **`fg_ratio`** — if the subject comes out oddly cropped or the
   proportions look off, this is usually why.
3. **MC Resolution / threshold**, via the presets or the advanced
   sliders.
4. **Studio Texture**, if your machine supports it (see above).
5. **Polish level** — Heavy can over-smooth thin/sharp features; if
   edges look melted, try Light or Medium instead of Heavy.

### Why "fine-tune the model" isn't the right next step

It's worth being direct about this rather than gesturing at a training
recipe that wouldn't actually work: **TripoSR's own GitHub repository
ships inference code only.** Its README points to the technical report
for architecture and training details but doesn't release a training
script, and there's no official or established community path to
fine-tuning or LoRA-adapting this specific model the way you might with
a diffusion checkpoint. Reconstructing the training setup from the paper
would mean re-implementing a triplane-NeRF training loop with
volumetric-rendering losses against a large multi-view 3D dataset
(something Objaverse-scale), on multi-GPU hardware — a research
undertaking, not a weekend project, and not something that fits a local
CPU setup at all. If you search for "TripoSR fine-tune," you'll mostly
find the same question asked and left unanswered upstream. So: this
repo intentionally does not point you at a fake fine-tuning tutorial.

### If you want a real step-change and can get GPU access

The actual jump in output quality beyond TripoSR right now comes from
switching models, not tuning or fine-tuning this one. As of when this
was written (August 2026):

- **[TRELLIS 2](https://github.com/microsoft/TRELLIS)** (Microsoft
  Research, MIT-licensed) is widely regarded as the strongest
  open-source image-to-3D model available, with native PBR texture
  output — but wants roughly a 24GB-class GPU, so it's not a drop-in
  replacement for a CPU-only setup.
- **[InstantMesh](https://github.com/TencentARC/InstantMesh)** sits
  between the two: better multi-view consistency and more
  production-ready output than TripoSR, still GPU-oriented but lighter
  than TRELLIS 2.

Both are worth checking again before committing — this space moves in
months, not years, so search for their current state rather than taking
this list as final.

If you're staying CPU-only, the ceiling really is the deterministic
pipeline this repo already implements: resolution, cleanup, smoothing,
and baked textures. There isn't currently a CPU-friendly open model that
meaningfully beats TripoSR on quality.

---

## Project layout

```
forge3d/
├── app.py              # Flask backend — the generation pipeline
├── bake_worker.py       # Isolated subprocess for Studio Texture (UV atlas + rasterize)
├── index.html           # Frontend — Three.js viewer + all UI
├── requirements.txt
├── .env.example
├── TripoSR/              # you clone this yourself — see Quick start
└── outputs/              # generated .glb files land here (gitignored)
```

## Troubleshooting

- **"TripoSR package not found"** at startup → you haven't cloned
  TripoSR into `./TripoSR` yet (or `TRIPOSR_DIR` points somewhere
  wrong). See Quick start step 3.
- **Generation fails right at the end, after several minutes** → almost
  certainly the numpy 2.x issue above. Run
  `python -c "import numpy; print(numpy.__version__)"` inside your venv;
  if it says 2.x, `pip install "numpy<2"`.
- **"Polish" doesn't seem to change anything** → check the terminal log
  for a smoothing warning; you're likely missing `scipy`.
- **Studio Texture always falls back** → check the terminal for the
  specific reason (missing packages vs. a failed/crashed render). On a
  headless Linux box, see the apt packages above.
- Environment variables you can set instead of editing code:
  `TRIPOSR_DIR` (where TripoSR is cloned), `TRIPOSR_CHUNK_SIZE` (renderer
  chunk size, lower it on very memory-constrained machines),
  `REMBG_MODEL` (background-removal model name).

## Credits & license

Built on [TripoSR](https://github.com/VAST-AI-Research/TripoSR) by
Tripo AI & Stability AI (MIT License). This repository's own code is
released under the MIT License — see [LICENSE](LICENSE).

Forge3D · Built by Ayush Kumar
