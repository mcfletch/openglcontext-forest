#!/usr/bin/env python
"""Regenerate the demo's runtime tree assets from the source .glb models.

The scene loads baked NumPy meshes (``*.npz``) + textures + impostor billboards, not
the glTF models directly, so startup is a plain ``np.load`` -> VBO upload with no
glTF parsing. This script bakes those runtime assets from the CC-BY source models in
``assets-source/`` — it is the reproducible provenance chain behind
``ASSET-LICENSES.md``.

For each species it:
  * pulls the configured opaque (trunk/bark/branch) and foliage (leaves/cluster)
    geometries out of the glb in *world* space (so the glTF Y-up node transforms are
    applied), merging multi-primitive foliage;
  * normalises the whole tree to unit height, base at y=0, XZ centroid at the origin
    (the convention the renderer + per-tree scaling assume);
  * writes ``<species>.npz`` with the part key scheme the scene expects
    (fir/noel/real: o*/b*; maple: b*/c*), each part = P/N/U/I
    (positions, normals, uvs, triangle indices);
  * saves each part's base-colour texture as PNG;
  * (unless --no-impostors) renders a front-on orthographic billboard to
    ``<impostor>.png`` for the far LOD.

Usage (from the demo project root, with the package importable)::

    python tools/bake_assets.py                 # bake everything into the package
    python tools/bake_assets.py --out /tmp/bake # bake to a scratch dir
    python tools/bake_assets.py --only fir real3 --no-impostors

Requires: trimesh (glb parsing), numpy, pillow; glfw + PyOpenGL for --impostors.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SOURCE = os.path.join(PROJECT, "assets-source")
DEFAULT_OUT = os.path.join(PROJECT, "src", "openglcontext_forest_demo", "assets")

FIR = "fir_tree.glb"; NOEL = "noel_pine_tree.glb"
MAPLE = "maple_trees_pack_lowpoly_game_ready_lods.glb"
REAL = "realistic_trees_collection.glb"

# Per output species: source glb, opaque + foliage glTF node names, the npz key
# prefixes for each part, and the texture output stems. Node names were resolved by
# matching the shipped npz vertex counts back to the source geometries.
SPECIES = [
    {"out": "fir", "glb": FIR, "opaque": ["Fir01_LOD0_M_Bark.007_0"],
         "foliage": ["Fir01_LOD0_M_Branch.007_0"], "okey": "o", "fkey": "b",
         "otex": "fir_bark", "ftex": "fir_branch", "imp": "fir_imp"},
    {"out": "noel", "glb": NOEL, "opaque": ["noel_pine_tree_trunk_0"],
         "foliage": ["noel_pine_tree_leaves_0"], "okey": "o", "fkey": "b",
         "otex": "noel_bark", "ftex": "noel_branch", "imp": "noel_imp"},
]
_MAPLE = [("maple0", "medium_1"), ("maple1", "medium_2"),
          ("maple2", "small_1"), ("maple3", "large_3")]
for out, tag in _MAPLE:
    SPECIES.append({
        "out": out, "glb": MAPLE, "opaque": [f"Acer_{tag}_LOD2_Bark_Mat_0"],
        "foliage": [f"Acer_{tag}_LOD2_Cluster_Mat_0"], "okey": "b", "fkey": "c",
        "otex": "maple_bark", "ftex": "maple_leaves", "imp": "maple_imp" + out[-1]})
_REAL = [
    ("real0", "Tree EZTree0.Large_branches_0", ["Tree EZTree0.Large_leaves_0"]),
    ("real1", "Tree EZTree0.Medium010_branches.010_0", ["Tree EZTree0.Medium010_leaves.010_0"]),
    ("real2", "Tree EZTree0.Medium011_branches.011_0", ["Tree EZTree0.Medium011_leaves.011_0"]),
    ("real3", "Tree EZTree1.Bush006_branches.006_0", ["Tree EZTree1.Bush006_leaves.006_0"]),
    ("real4", "Tree EZTree1.Large001_branches.001_0",
     ["Tree EZTree1.Large001_leaves.001_0", "Tree EZTree1.Large001_leaves.001_1"]),
    ("real5", "Tree EZTree1.Large009_branches.009_0", ["Tree EZTree1.Large009_leaves.009_0"]),
    ("real6", "Tree EZTree1.Medium002_branches.002_0", ["Tree EZTree1.Medium002_leaves.002_0"]),
]
for i, (out, br, lf) in enumerate(_REAL):
    SPECIES.append({"out": out, "glb": REAL, "opaque": [br], "foliage": lf, "okey": "o", "fkey": "b",
                        "otex": f"real_br{i}", "ftex": f"real_lf{i}", "imp": f"imp{i}"})


def _world_part(scene, names):
    """Merge named glTF nodes into world-space (P, N, U, I) arrays."""
    import trimesh
    P = []; N = []; U = []; I = []; base = 0
    for name in names:
        T, gname = scene.graph[name]
        g = scene.geometry[gname]
        v = trimesh.transformations.transform_points(g.vertices, T)
        n = g.vertex_normals @ T[:3, :3].T
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        uv = np.asarray(g.visual.uv, np.float32) if g.visual.uv is not None else np.zeros((len(v), 2), np.float32)
        P.append(v); N.append(n); U.append(uv)
        I.append(g.faces.astype(np.uint32) + base); base += len(v)
    return (np.concatenate(P), np.concatenate(N), np.concatenate(U),
            np.concatenate(I).ravel().astype(np.uint32))


def _texture(scene, name, path):
    g = scene.geometry[scene.graph[name][1]]
    mat = getattr(g.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        raise RuntimeError(f"no base-colour texture on {name}")
    img.convert("RGBA").save(path)


def bake_geometry(scene, spec, out_dir):
    """Write <species>.npz (normalised) and the two part textures. Returns metrics."""
    oP, oN, oU, oI = _world_part(scene, spec["opaque"])
    fP, fN, fU, fI = _world_part(scene, spec["foliage"])
    allv = np.concatenate([oP, fP])
    h = float(np.ptp(allv[:, 1])) or 1.0
    ymin = float(allv[:, 1].min())
    cx = float(allv[:, 0].mean()); cz = float(allv[:, 2].mean())

    def norm(P):
        w = P.copy(); w[:, 0] -= cx; w[:, 2] -= cz; w[:, 1] -= ymin
        return (w / h).astype(np.float32)

    ok, fk = spec["okey"], spec["fkey"]
    data = {ok + "P": norm(oP), ok + "N": oN.astype(np.float32),
            ok + "U": oU.astype(np.float32), ok + "I": oI,
            fk + "P": norm(fP), fk + "N": fN.astype(np.float32),
            fk + "U": fU.astype(np.float32), fk + "I": fI}
    np.savez(os.path.join(out_dir, spec["out"] + ".npz"), **data)
    _texture(scene, spec["opaque"][0], os.path.join(out_dir, spec["otex"] + ".png"))
    _texture(scene, spec["foliage"][0], os.path.join(out_dir, spec["ftex"] + ".png"))
    return {"verts": len(oP) + len(fP), "h": h}


def bake_impostor(scene, spec, out_dir, size=512):
    """Render a front-on orthographic RGBA billboard of the tree to <impostor>.png."""
    import ctypes

    from OpenGL.GL import (
        GL_ARRAY_BUFFER,
        GL_BLEND,
        GL_CLAMP_TO_EDGE,
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_ATTACHMENT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_COMPONENT24,
        GL_DEPTH_TEST,
        GL_ELEMENT_ARRAY_BUFFER,
        GL_FALSE,
        GL_FLOAT,
        GL_FRAGMENT_SHADER,
        GL_FRAMEBUFFER,
        GL_LINEAR,
        GL_RENDERBUFFER,
        GL_RGBA,
        GL_RGBA8,
        GL_STATIC_DRAW,
        GL_TEXTURE0,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TRIANGLES,
        GL_TRUE,
        GL_UNSIGNED_BYTE,
        GL_UNSIGNED_INT,
        GL_VERTEX_SHADER,
        glActiveTexture,
        glBindBuffer,
        glBindFramebuffer,
        glBindRenderbuffer,
        glBindTexture,
        glBindVertexArray,
        glBufferData,
        glClear,
        glClearColor,
        glDisable,
        glDrawElements,
        glEnable,
        glEnableVertexAttribArray,
        glFramebufferRenderbuffer,
        glFramebufferTexture2D,
        glGenBuffers,
        glGenerateMipmap,
        glGenFramebuffers,
        glGenRenderbuffers,
        glGenTextures,
        glGenVertexArrays,
        glGetUniformLocation,
        glReadPixels,
        glRenderbufferStorage,
        glTexImage2D,
        glTexParameteri,
        glUniform1i,
        glUniformMatrix4fv,
        glUseProgram,
        glVertexAttribPointer,
        glViewport,
    )
    from OpenGL.GL.shaders import compileProgram, compileShader

    d = np.load(os.path.join(out_dir, spec["out"] + ".npz"))
    ok, fk = spec["okey"], spec["fkey"]
    parts = [(d[ok + "P"], d[ok + "U"], d[ok + "I"], spec["otex"]),
             (d[fk + "P"], d[fk + "U"], d[fk + "I"], spec["ftex"])]

    VS = """#version 330 core
    layout(location=0) in vec3 p; layout(location=1) in vec2 uv;
    uniform mat4 M; out vec2 vUV; void main(){ vUV=uv; gl_Position=M*vec4(p,1.0); }"""
    FS = """#version 330 core
    in vec2 vUV; uniform sampler2D tex; out vec4 c;
    void main(){ vec4 t=texture(tex,vUV); if(t.a<0.5) discard; c=vec4(t.rgb,1.0); }"""
    prog = compileProgram(compileShader(VS, GL_VERTEX_SHADER), compileShader(FS, GL_FRAGMENT_SHADER))

    # front orthographic: x in [-0.5,0.5], y in [0,1] -> clip. Column-vector matrix.
    M = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, -1.0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]], np.float32)

    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    ct = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, ct)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0)
    rb = glGenRenderbuffers(1); glBindRenderbuffer(GL_RENDERBUFFER, rb)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb)
    glViewport(0, 0, size, size); glClearColor(0, 0, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST); glDisable(GL_BLEND)
    glUseProgram(prog)
    glUniformMatrix4fv(glGetUniformLocation(prog, "M"), 1, GL_TRUE, M)  # row-major -> transpose
    glUniform1i(glGetUniformLocation(prog, "tex"), 0)

    def tex2d(path):
        im = Image.open(path).convert("RGBA")
        t = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, t)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, im.width, im.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, np.asarray(im))
        glGenerateMipmap(GL_TEXTURE_2D)
        for k, v in [(GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                     (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
            glTexParameteri(GL_TEXTURE_2D, k, v)
        return t

    for P, U, I, texname in parts:
        mesh = np.concatenate([P, U], 1).astype(np.float32)
        vao = glGenVertexArrays(1); glBindVertexArray(vao)
        vb = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vb)
        glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12)); glEnableVertexAttribArray(1)
        ib = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib)
        idx = I.astype(np.uint32); glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        glActiveTexture(GL_TEXTURE0); tex2d(os.path.join(out_dir, texname + ".png"))
        glDrawElements(GL_TRIANGLES, len(idx), GL_UNSIGNED_INT, None)

    buf = glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE)
    img = Image.frombytes("RGBA", (size, size), buf).transpose(Image.FLIP_TOP_BOTTOM)
    img.save(os.path.join(out_dir, spec["imp"] + ".png"))
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output asset directory")
    ap.add_argument("--source", default=SOURCE, help="directory of source .glb models")
    ap.add_argument("--only", nargs="*", help="only these species (default: all)")
    ap.add_argument("--no-impostors", action="store_true", help="skip the GL impostor bake")
    args = ap.parse_args(argv)

    import trimesh
    os.makedirs(args.out, exist_ok=True)
    specs = [s for s in SPECIES if not args.only or s["out"] in args.only]

    if not args.no_impostors:
        import glfw
        if not glfw.init():
            raise SystemExit("glfw init failed (needed for impostors; use --no-impostors)")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        win = glfw.create_window(64, 64, "bake", None, None); glfw.make_context_current(win)

    scenes = {}
    for spec in specs:
        if spec["glb"] not in scenes:
            scenes[spec["glb"]] = trimesh.load(os.path.join(args.source, spec["glb"]))
        m = bake_geometry(scenes[spec["glb"]], spec, args.out)
        msg = f"  {spec['out']:<8} verts={m['verts']:<7}"
        if not args.no_impostors:
            bake_impostor(scenes[spec["glb"]], spec, args.out); msg += " +impostor"
        print(msg); sys.stdout.flush()
    print(f"baked {len(specs)} species -> {args.out}")


if __name__ == "__main__":
    main()
