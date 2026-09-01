import sys
sys.path.append("TripoSR")
import io
import json
import os
import queue
import random
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
TRIPOSR_DIR = Path(os.environ.get("TRIPOSR_DIR", HERE / "TripoSR")).resolve()
sys.path.insert(0, str(TRIPOSR_DIR))

app = Flask(__name__)
CORS(app)
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

model = None
rembg_session = None
active_jobs = {}

def _check_environment():
    np_major = int(np.__version__.split(".")[0])
    if np_major >= 2:
        print("\n" + "=" * 70)
        print(f"  Forge3D: numpy {np.__version__} is installed.")
        print("  trimesh==4.0.5 (the version TripoSR's own requirements.txt pins)")
        print("  calls an ndarray method that NumPy 2.0 removed, in 50+ places --")
        print("  including mesh export. Left as-is, generation runs all the way")
        print("  through TripoSR and then crashes on the FINAL save step, so you")
        print("  only find out after waiting through the whole thing.")
        print('  Fix:  pip install "numpy<2"')
        print("=" * 70 + "\n")

    triposr_ok = (TRIPOSR_DIR / "tsr" / "system.py").exists()
    if not triposr_ok:
        print("\n" + "=" * 70)
        print("  Forge3D: TripoSR package not found at:")
        print(f"    {TRIPOSR_DIR}")
        print("  Generation will fail until you clone it there:")
        print(f'    git clone https://github.com/VAST-AI-Research/TripoSR.git "{TRIPOSR_DIR}"')
        print(f'    pip install -r "{TRIPOSR_DIR / "requirements.txt"}"')
        print("  See README.md 'Setup' for the full sequence.")
        print("=" * 70 + "\n")

    try:
        import scipy
    except ImportError:
        print("\n" + "=" * 70)
        print("  Forge3D: scipy is not installed.")
        print("  'Polish' (Taubin smoothing) will silently do nothing without it.")
        print("  Fix:  pip install scipy")
        print("=" * 70 + "\n")

    try:
        import xatlas, moderngl 
        print("  Studio Texture: available (xatlas + moderngl found).")
    except ImportError:
        print("  Studio Texture: not available (optional -- pip install xatlas moderngl to enable).")

    return triposr_ok


TRIPOSR_READY = _check_environment()

def load_model():
    global model
    if model is not None:
        return model
    if not (TRIPOSR_DIR / "tsr" / "system.py").exists():
        raise RuntimeError(
            f"TripoSR is not installed at {TRIPOSR_DIR}. Run: git clone "
            f'https://github.com/VAST-AI-Research/TripoSR.git "{TRIPOSR_DIR}" '
            f"-- see README.md 'Setup'."
        )
    print("Loading TripoSR model (first run downloads the checkpoint from Hugging Face)…")
    from tsr.system import TSR

    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    chunk_size = int(os.environ.get("TRIPOSR_CHUNK_SIZE", "32768"))
    model.renderer.set_chunk_size(chunk_size)
    model.to("cpu")
    print("Model ready!")
    return model


def get_rembg_session():
    global rembg_session
    if rembg_session is not None:
        return rembg_session
    import rembg

    model_name = os.environ.get("REMBG_MODEL", "isnet-general-use")
    try:
        rembg_session = rembg.new_session(model_name)
        print(f"  rembg model: {model_name}")
    except Exception as e:
        print(f"  Could not load rembg model '{model_name}' ({e}); using rembg's default instead.")
        rembg_session = rembg.new_session()
    return rembg_session

def _fill_bg_gray(image_rgba):
    a = np.array(image_rgba).astype(np.float32) / 255.0
    rgb = a[:, :, :3] * a[:, :, 3:4] + (1 - a[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _resize_foreground(img_rgba, ratio):
    a = np.array(img_rgba)
    yw, xw = np.where(a[..., 3] > 0)
    if len(yw) == 0:
        return img_rgba
    fg = a[yw.min():yw.max(), xw.min():xw.max()]
    sz = max(fg.shape[0], fg.shape[1])
    p0h = (sz - fg.shape[0]) // 2
    p1h = sz - fg.shape[0] - p0h
    p0w = (sz - fg.shape[1]) // 2
    p1w = sz - fg.shape[1] - p0w
    sq = np.pad(fg, ((p0h, p1h), (p0w, p1w), (0, 0)), constant_values=0)
    nsz = int(sz / ratio)
    o = (nsz - sz) // 2
    return Image.fromarray(np.pad(sq, ((o, nsz - sz - o), (o, nsz - sz - o), (0, 0)), constant_values=0))


def sharpen_input(img_rgb):
    img = img_rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))
    return ImageEnhance.Contrast(img).enhance(1.12)


def preprocess(img_pil, remove_bg=True, fg_ratio=0.85):

    rgb0 = np.array(img_pil.convert("RGB"))
    r0, g0, b0 = int(rgb0[0, 0, 0]), int(rgb0[0, 0, 1]), int(rgb0[0, 0, 2])
    is_canvas = r0 == 1 and g0 == 2 and b0 == 3
    print(f"  Pixel(0,0)=({r0},{g0},{b0})  canvas={is_canvas}")

    img = img_pil.convert("RGBA")

    if is_canvas:
        img = img.resize((512, 512), Image.LANCZOS).convert("RGB")
        arr = np.array(img)
        arr[0, 0] = [127, 127, 127]
        img = Image.fromarray(arr)
        img = sharpen_input(img)
        return img, True

    if remove_bg:
        import rembg

        if np.array(img)[:, :, 3].min() == 255:
            print("  Removing background…")
            img = rembg.remove(img.convert("RGB"), session=get_rembg_session())
        else:
            print("  Already transparent — skipping rembg")
        img = _resize_foreground(img, fg_ratio)
        img = _fill_bg_gray(img)
    else:
        img = _fill_bg_gray(img)
    img = img.resize((512, 512), Image.LANCZOS).convert("RGB")
    img = sharpen_input(img)
    return img, False

def run_triposr(img_rgb, mc_resolution=192, threshold=25.0, is_canvas=False):

    import torch
    import trimesh as tm

    img_np = np.array(img_rgb).astype(np.float32) / 255.0
    m = load_model()
    print(f"  TripoSR mc_res={mc_resolution} threshold={threshold} canvas={is_canvas}")
    with torch.no_grad():
        scene_code = m([img_np], device="cpu")
    meshes = m.extract_mesh(scene_code, has_vertex_color=True, resolution=mc_resolution, threshold=threshold)
    mesh = meshes[0]
    if not is_canvas:
        mesh.apply_transform(tm.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        mesh.apply_transform(tm.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return mesh, scene_code

def cleanup_mesh(mesh):
    import trimesh as tm

    try:
        before = len(mesh.faces)
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        tm.repair.fix_normals(mesh)
        print(f"  Cleanup: {before} → {len(mesh.faces)} faces")
    except Exception as e:
        print(f"  Cleanup warning (continuing with uncleaned mesh): {e}")
    return mesh

def polish_mesh(mesh, level=2):
    import trimesh as tm

    iters = {0: 0, 1: 40, 2: 150, 3: 300}.get(int(level), 150)
    if iters == 0:
        return mesh
    try:
        print(f"  Taubin ×{iters}…")
        tm.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=iters)
    except Exception as e:
        print(f"  ! Smoothing skipped due to: {e}")
        print("    (if this says 'coo_matrix' or similar: pip install scipy)")
    return mesh


def subdivide_mesh(mesh, level=2):
    if int(level) < 3:
        return mesh
    print("  Subdividing…")
    import trimesh as tm

    mesh = mesh.subdivide()
    try:
        tm.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=60)
    except Exception as e:
        print(f"  ! Post-subdivide smoothing skipped due to: {e}")
    return mesh


def enhance_colors(mesh, is_canvas=False):
    if is_canvas:
        print("  Colors: light enhancement only (canvas)")
        vc = mesh.visual.vertex_colors[:, :3].astype(np.uint8)
        img = Image.fromarray(vc.reshape(1, -1, 3), "RGB")
        img = ImageEnhance.Color(img).enhance(1.3)
        arr = np.array(img).reshape(-1, 3).astype(np.uint8)
        mesh.visual.vertex_colors = np.hstack([arr, mesh.visual.vertex_colors[:, 3:4]])
        return mesh
    vc = mesh.visual.vertex_colors[:, :3].astype(np.uint8)
    img = Image.fromarray(vc.reshape(1, -1, 3), "RGB")
    img = ImageEnhance.Color(img).enhance(1.6)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    arr = np.clip(np.array(img).reshape(-1, 3).astype(np.float32) / 255.0, 0, 1)
    arr = (arr ** 0.78 * 255).astype(np.uint8)
    mesh.visual.vertex_colors = np.hstack([arr, mesh.visual.vertex_colors[:, 3:4]])
    print("  Colors enhanced")
    return mesh

def positions_to_colors(model, scene_code, positions_texture, texture_resolution):
    import torch

    positions = torch.tensor(positions_texture.reshape(-1, 4)[:, :-1])
    with torch.no_grad():
        queried_grid = model.renderer.query_triplane(model.decoder, positions, scene_code)
    rgb_f = queried_grid["color"].numpy().reshape(-1, 3)
    rgba_f = np.insert(rgb_f, 3, positions_texture.reshape(-1, 4)[:, -1], axis=1)
    rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba_f.reshape(texture_resolution, texture_resolution, 4)


def bake_texture_isolated(mesh, model, scene_code, resolution=1024, timeout=90):

    worker = HERE / "bake_worker.py"
    if not worker.exists():
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "mesh.npz"
            out_path = Path(td) / "atlas.npz"
            np.savez(in_path, vertices=mesh.vertices.astype(np.float32), faces=mesh.faces.astype(np.int32))

            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "1")

            proc = subprocess.run(
                [sys.executable, str(worker), str(in_path), str(out_path), str(resolution)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if proc.returncode != 0 or not out_path.exists():
                reason = f"exit code {proc.returncode}"
                if proc.stderr and proc.stderr.strip():
                    reason += " — " + proc.stderr.strip().splitlines()[-1]
                print(f"  Studio Texture unavailable ({reason}); using enhanced vertex colors instead.")
                return None

            atlas_data = np.load(out_path)
            vmapping = atlas_data["vmapping"]
            indices = atlas_data["indices"]
            uvs = atlas_data["uvs"]
            positions_texture = atlas_data["positions_texture"]

        colors_texture = positions_to_colors(model, scene_code, positions_texture, resolution)
        return {"vmapping": vmapping, "indices": indices, "uvs": uvs, "colors": colors_texture}
    except subprocess.TimeoutExpired:
        print(f"  Studio Texture timed out after {timeout}s; using enhanced vertex colors instead.")
        return None
    except Exception as e:
        print(f"  Studio Texture failed ({e}); using enhanced vertex colors instead.")
        return None


def build_textured_mesh(mesh, bake_result):
    import trimesh as tm

    new_verts = mesh.vertices[bake_result["vmapping"]]
    new_faces = bake_result["indices"]
    tex_arr = (np.clip(bake_result["colors"][..., :3], 0, 1) * 255).astype(np.uint8)
    tex_img = Image.fromarray(tex_arr).transpose(Image.FLIP_TOP_BOTTOM)
    tex_img = ImageEnhance.Color(tex_img).enhance(1.25)
    tex_img = ImageEnhance.Contrast(tex_img).enhance(1.08)

    new_mesh = tm.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    material = tm.visual.material.PBRMaterial(baseColorTexture=tex_img, metallicFactor=0.02, roughnessFactor=0.85)
    new_mesh.visual = tm.visual.TextureVisuals(uv=bake_result["uvs"], material=material)
    return new_mesh

def sanitize_mesh(mesh):
    verts = mesh.vertices
    bad = ~np.isfinite(verts).all(axis=1)
    if bad.any():
        print(f"  Sanitize: removing {bad.sum()} non-finite vertices before export")
        mesh.update_vertices(~bad)
    if len(mesh.vertices) and not np.isfinite(mesh.vertex_normals).all():
        bad_n = ~np.isfinite(mesh.vertex_normals).all(axis=1)
        print(f"  Sanitize: patching {bad_n.sum()} non-finite normals")
        mesh.vertex_normals[bad_n] = [0.0, 0.0, 1.0]
    return mesh

def save_mesh(mesh, job_id):
    import trimesh as tm

    mesh = sanitize_mesh(mesh)
    scene = tm.scene.Scene(geometry={"mesh": mesh})
    glb = scene.export(file_type="glb")
    p = OUTPUT_DIR / f"{job_id}.glb"
    p.write_bytes(glb)
    print(f"  Saved {len(glb) // 1024}KB → {p}")
    return job_id


def full_pipeline(img_bytes, fname, remove_bg, fg_ratio, mc_res, threshold, polish, studio_texture, q_emit):

    job_id = str(uuid.uuid4())[:8]
    img = Image.open(io.BytesIO(img_bytes))

    q_emit({"type": "progress", "pct": 10, "msg": "Preprocessing…", "phase": "particles"})
    processed, is_canvas = preprocess(img, remove_bg, fg_ratio)
    processed.save(OUTPUT_DIR / f"{job_id}_input.png")

    if is_canvas:
        q_emit({"type": "progress", "pct": 18, "msg": "Canvas input — skipping rembg & orient fix…", "phase": "converge"})
    else:
        q_emit({"type": "progress", "pct": 18, "msg": "Preprocessing done. Running TripoSR…", "phase": "converge"})

    q_emit({"type": "progress", "pct": 25, "msg": "Running TripoSR neural network…", "phase": "converge"})
    mesh, scene_code = run_triposr(processed, mc_res, threshold, is_canvas=is_canvas)

    q_emit({"type": "progress", "pct": 68, "msg": "Cleaning up raw geometry…", "phase": "wireframe"})
    mesh = cleanup_mesh(mesh)

    q_emit({"type": "progress", "pct": 75, "msg": "Polishing surface…", "phase": "wireframe"})
    mesh = polish_mesh(mesh, polish)

    q_emit({"type": "progress", "pct": 83, "msg": "Subdividing mesh…", "phase": "fill"})
    mesh = subdivide_mesh(mesh, polish)

    textured = False
    if studio_texture:
        q_emit({"type": "progress", "pct": 88, "msg": "Baking studio texture…", "phase": "fill"})
        bake_result = bake_texture_isolated(mesh, model, scene_code, resolution=1024)
        if bake_result is not None:
            mesh = build_textured_mesh(mesh, bake_result)
            textured = True
            q_emit({"type": "progress", "pct": 93, "msg": "Studio texture applied ✓", "phase": "fill"})
        else:
            q_emit({"type": "progress", "pct": 93, "msg": "Studio texture unavailable — using vertex colors", "phase": "fill"})

    if not textured:
        q_emit({"type": "progress", "pct": 93, "msg": "Enhancing colors…", "phase": "fill"})
        mesh = enhance_colors(mesh, is_canvas=is_canvas)

    q_emit({"type": "progress", "pct": 97, "msg": "Saving…", "phase": "materialise"})
    save_mesh(mesh, job_id)
    return job_id, textured

def _worker(form_data, files_data, job_id, q, stop):
    def emit(msg):
        q.put(msg)

    def stopped():
        if stop.is_set():
            emit({"type": "cancelled"})
            return True
        return False

    try:
        remove_bg = form_data.get("remove_bg", "true") == "true"
        fg_ratio = float(form_data.get("fg_ratio", 0.85))
        mc_res = int(form_data.get("mc_resolution", 128))
        threshold = float(form_data.get("threshold", 25.0))
        polish = int(form_data.get("polish", 2))
        studio_texture = form_data.get("studio_texture", "false") == "true"
        mode = form_data.get("mode", "single")

        if mode == "multiple":
            valid = [(n, d) for n, d in files_data if n]
            if not valid:
                emit({"type": "error", "message": "No images"})
                return
            for i, (fname, img_bytes) in enumerate(valid[:6]):
                if stopped():
                    return
                pct = int(i / len(valid) * 85)
                emit({"type": "progress", "pct": pct + 5, "msg": f"Processing {fname} ({i + 1}/{len(valid)})…", "phase": "converge"})
                fid, textured = full_pipeline(img_bytes, fname, remove_bg, fg_ratio, mc_res, threshold, polish, studio_texture, emit)
                if stopped():
                    return
                q.put({"type": "result_ready", "file_id": fid, "name": fname, "textured": textured})
            emit({"type": "done"})
        else:
            if not files_data:
                emit({"type": "error", "message": "No image"})
                return
            fname, img_bytes = files_data[0]
            fid, textured = full_pipeline(img_bytes, fname, remove_bg, fg_ratio, mc_res, threshold, polish, studio_texture, emit)
            if stopped():
                return
            emit({"type": "done", "results": [{"file_id": fid, "textured": textured}]})

    except Exception as e:
        traceback.print_exc()
        emit({"type": "error", "message": str(e)})
    finally:
        active_jobs.pop(job_id, None)


@app.route("/")
def index():
    return send_from_directory(str(HERE), "index.html")

@app.route("/text2img")
def text2img():
    prompt = request.args.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    key = os.environ.get("POLLINATIONS_API_KEY", "")
    if not key:
        return jsonify({"error": "POLLINATIONS_API_KEY not set in .env"}), 501

    import requests
    from urllib.parse import quote

    seed = random.randint(0, 99999)
    url = f"https://gen.pollinations.ai/image/{quote(prompt)}?width=512&height=512&model=flux&seed={seed}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=45)
        content_type = r.headers.get("content-type", "")
        if r.ok and content_type.startswith("image/"):
            return Response(r.content, mimetype=content_type)
        return jsonify({"error": f"Pollinations returned HTTP {r.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/generate", methods=["POST"])
def generate():
    job_id = str(uuid.uuid4())[:8]
    q = queue.Queue()
    stop = threading.Event()
    active_jobs[job_id] = {"queue": q, "stop": stop}

    form_data = request.form.to_dict(flat=True)
    mode = form_data.get("mode", "single")
    files_data = []
    if mode == "multiple":
        for f in request.files.getlist("images"):
            if f.filename:
                files_data.append((f.filename, f.read()))
    elif "image" in request.files:
        f = request.files["image"]
        if f.filename:
            files_data.append((f.filename, f.read()))

    t = threading.Thread(target=_worker, args=(form_data, files_data, job_id, q, stop), daemon=True)
    t.start()

    def stream():
        yield f"data: {json.dumps({'type': 'started', 'job_id': job_id})}\n\n"
        results = []
        while True:
            try:
                msg = q.get(timeout=240)
                if msg["type"] == "result_ready":
                    results.append({"file_id": msg["file_id"], "name": msg.get("name", ""), "textured": msg.get("textured", False)})
                    yield f"data: {json.dumps(msg)}\n\n"
                elif msg["type"] == "done" and not msg.get("results"):
                    msg["results"] = results
                    yield f"data: {json.dumps(msg)}\n\n"
                    break
                else:
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg["type"] in ("done", "error", "cancelled"):
                        break
            except queue.Empty:
                yield 'data: {"type": "heartbeat"}\n\n'

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    job = active_jobs.get(job_id)
    if job:
        job["stop"].set()
        return jsonify({"cancelled": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/download/<file_id>")
def download(file_id):
    p = OUTPUT_DIR / f"{file_id}.glb"
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(p), mimetype="model/gltf-binary", as_attachment=True, download_name=f"forge3d_{file_id}.glb")


@app.route("/preview/<file_id>")
def preview(file_id):
    p = OUTPUT_DIR / f"{file_id}.glb"
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(p), mimetype="model/gltf-binary")


if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║      🧊 Forge3D Local Server          ║")
    print("  ╠══════════════════════════════════════╣")
    print("  ║  Open: http://localhost:5000          ║")
    print("  ╚══════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
