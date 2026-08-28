#!/usr/bin/env python3
"""
bake_worker.py — isolated worker for the "Studio Texture" feature.

WHY THIS IS A SEPARATE PROCESS
-------------------------------
UV-atlas generation (xatlas) and its rasterization (moderngl/OpenGL) are both
native, C/C++-backed libraries. In testing, xatlas's atlas packer segfaulted
outright on a constrained (single-core) machine — a native crash, not a
Python exception, so nothing inside the Flask process could have caught it.
Running this step in its own subprocess means a crash here can only kill
this one subprocess; app.py detects the failure and falls back to enhanced
vertex colors for that generation, instead of losing the whole server.

This worker only needs mesh geometry (vertices + faces) — NOT the TripoSR
model — so it never has to reload model weights. Only the final step
(querying the neural network for a color at each rasterized position) needs
the model, and that step is plain, safe PyTorch inference, so app.py does
that part itself, back in the main process, using the model it already has
loaded in memory.

USAGE
-----
    python bake_worker.py <input.npz> <output.npz> <resolution>

input.npz  — arrays "vertices" (N,3 float32) and "faces" (M,3 int32)
output.npz — on success, arrays "vmapping" (K,), "indices" (M,3),
             "uvs" (K,2), "positions_texture" (R,R,4 float32)

Exit code 0 + output.npz written = success.
Any other outcome (nonzero exit, timeout, no file, crash) = caller falls
back to vertex colors. This script deliberately does not import anything
from app.py or the tsr package, and must have no side effect on them.
"""
import sys
import traceback

import numpy as np


def make_atlas(vertices, faces, texture_resolution, texture_padding):
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    options = xatlas.PackOptions()
    options.resolution = texture_resolution
    options.padding = texture_padding
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    return vmapping, indices, uvs


def make_gl_context():
    """Try the platform default first (works out of the box on most desktops
    with a real GPU/display). Fall back to an explicit EGL context, which is
    what makes this work on a headless Linux server with no display attached
    — verified against Mesa's llvmpipe software renderer. See README.md
    'Studio Texture on a headless server' for the exact system packages."""
    import moderngl

    try:
        return moderngl.create_context(standalone=True)
    except Exception:
        return moderngl.create_context(standalone=True, backend="egl")


def rasterize_position_atlas(vertices, vmapping, indices, uvs, texture_resolution, texture_padding):
    ctx = make_gl_context()
    try:
        basic_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_uv;
                in vec3 in_pos;
                out vec3 v_pos;
                void main() {
                    v_pos = in_pos;
                    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_pos;
                out vec4 o_col;
                void main() {
                    o_col = vec4(v_pos, 1.0);
                }
            """,
        )
        gs_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_uv;
                in vec3 in_pos;
                out vec3 vg_pos;
                void main() {
                    vg_pos = in_pos;
                    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
                }
            """,
            geometry_shader="""
                #version 330
                uniform float u_resolution;
                uniform float u_dilation;
                layout (triangles) in;
                layout (triangle_strip, max_vertices = 12) out;
                in vec3 vg_pos[];
                out vec3 vf_pos;
                void lineSegment(int aidx, int bidx) {
                    vec2 a = gl_in[aidx].gl_Position.xy;
                    vec2 b = gl_in[bidx].gl_Position.xy;
                    vec3 aCol = vg_pos[aidx];
                    vec3 bCol = vg_pos[bidx];

                    vec2 dir = normalize((b - a) * u_resolution);
                    vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;

                    gl_Position = vec4(a + offset, 0.0, 1.0);
                    vf_pos = aCol;
                    EmitVertex();
                    gl_Position = vec4(a - offset, 0.0, 1.0);
                    vf_pos = aCol;
                    EmitVertex();
                    gl_Position = vec4(b + offset, 0.0, 1.0);
                    vf_pos = bCol;
                    EmitVertex();
                    gl_Position = vec4(b - offset, 0.0, 1.0);
                    vf_pos = bCol;
                    EmitVertex();
                }
                void main() {
                    lineSegment(0, 1);
                    lineSegment(1, 2);
                    lineSegment(2, 0);
                    EndPrimitive();
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 vf_pos;
                out vec4 o_col;
                void main() {
                    o_col = vec4(vf_pos, 1.0);
                }
            """,
        )
        uvs_f = uvs.flatten().astype("f4")
        pos_f = vertices[vmapping].flatten().astype("f4")
        idx_f = indices.flatten().astype("i4")
        vbo_uvs = ctx.buffer(uvs_f)
        vbo_pos = ctx.buffer(pos_f)
        ibo = ctx.buffer(idx_f)
        vao_content = [
            vbo_uvs.bind("in_uv", layout="2f"),
            vbo_pos.bind("in_pos", layout="3f"),
        ]
        basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
        gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
        fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((texture_resolution, texture_resolution), 4, dtype="f4")]
        )
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        gs_prog["u_resolution"].value = texture_resolution
        gs_prog["u_dilation"].value = texture_padding
        gs_vao.render()
        basic_vao.render()

        fbo_bytes = fbo.color_attachments[0].read()
        fbo_np = np.frombuffer(fbo_bytes, dtype="f4").reshape(texture_resolution, texture_resolution, 4)
        return fbo_np.copy()
    finally:
        ctx.release()


def main():
    if len(sys.argv) != 4:
        print("usage: bake_worker.py <input.npz> <output.npz> <resolution>", file=sys.stderr)
        return 2

    in_path, out_path, resolution = sys.argv[1], sys.argv[2], int(sys.argv[3])
    data = np.load(in_path)
    vertices = data["vertices"].astype(np.float32)
    faces = data["faces"].astype(np.uint32)

    texture_padding = round(max(2, resolution / 256))
    vmapping, indices, uvs = make_atlas(vertices, faces, resolution, texture_padding)
    positions_texture = rasterize_position_atlas(vertices, vmapping, indices, uvs, resolution, texture_padding)

    np.savez(
        out_path,
        vmapping=vmapping,
        indices=indices,
        uvs=uvs,
        positions_texture=positions_texture,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
