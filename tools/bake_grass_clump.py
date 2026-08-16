#! /usr/bin/env python
"""Bake a front-on orthographic impostor billboard of the grass clump.

The near LOD draws the real clump mesh (see ``vegetation/clumps.py``); the mid/far
LOD needs a flat card whose colours MATCH the clump (dark root -> pale tip), so the
cross-fade from geometry to billboard is invisible. This renders the clump front-on
to an RGBA PNG whose alpha is the blade silhouette (transparent background), and
prints the card aspect (full width / height) to use as the billboard ``width``.

    python tools/bake_grass_clump.py           # -> assets/grass_clump_imp.png
    python tools/bake_grass_clump.py --size 512

Requires glfw + PyOpenGL (offscreen GL), numpy, pillow.
"""
import argparse
import ctypes
import os
import sys

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "openglcontext"))
DEFAULT_GLB = os.path.join(HERE, "..", "src", "openglcontext_forest_demo", "assets", "basic-clump.glb")
DEFAULT_OUT = os.path.join(HERE, "..", "src", "openglcontext_forest_demo", "assets", "grass_clump_imp.png")


def bake(glb, out, size=512):
    import glfw
    from OpenGLContext.scenegraph.vegetation import load_clump_glb

    P, _N, UV, idx, texpath = load_clump_glb(glb)          # height-normalised to 1.0
    hx = float(np.max(np.abs(P[:, 0])))                    # front-view half-width
    hy = float(P[:, 1].max())

    if not glfw.init():
        raise SystemExit("glfw init failed")
    for h, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3),
                 (glfw.CONTEXT_VERSION_MINOR, 3), (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(h, v)
    win = glfw.create_window(64, 64, "bake", None, None); glfw.make_context_current(win)

    VS = """#version 330 core
    layout(location=0) in vec3 p; layout(location=1) in vec2 uv;
    uniform mat4 M; out vec2 vUV; void main(){ vUV=uv; gl_Position=M*vec4(p,1.0); }"""
    FS = """#version 330 core
    in vec2 vUV; uniform sampler2D tex; out vec4 c;
    void main(){ vec4 t=texture(tex,vUV); if(t.a<0.33) discard; c=vec4(t.rgb,1.0); }"""
    prog = compileProgram(compileShader(VS, GL_VERTEX_SHADER), compileShader(FS, GL_FRAGMENT_SHADER))

    # front orthographic: x in [-hx,hx] -> [-1,1], y in [0,hy] -> [-1,1], z -> depth.
    M = np.array([[1.0 / hx, 0, 0, 0], [0, 2.0 / hy, 0, -1.0],
                  [0, 0, -1.0, 0], [0, 0, 0, 1.0]], np.float32)

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
    glUniformMatrix4fv(glGetUniformLocation(prog, "M"), 1, GL_TRUE, M)   # row-major -> transpose
    glUniform1i(glGetUniformLocation(prog, "tex"), 0)

    im = Image.open(texpath).convert("RGBA")
    t = glGenTextures(1); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, t)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, im.width, im.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, np.asarray(im))
    glGenerateMipmap(GL_TEXTURE_2D)
    for k, v in [(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                 (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
        glTexParameteri(GL_TEXTURE_2D, k, v)

    mesh = np.concatenate([P, UV], 1).astype(np.float32)
    vao = glGenVertexArrays(1); glBindVertexArray(vao)
    vb = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vb)
    glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12)); glEnableVertexAttribArray(1)
    ib = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib)
    ii = idx.astype(np.uint32); glBufferData(GL_ELEMENT_ARRAY_BUFFER, ii.nbytes, ii, GL_STATIC_DRAW)
    glDrawElements(GL_TRIANGLES, len(ii), GL_UNSIGNED_INT, None)

    buf = glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE)
    img = Image.frombytes("RGBA", (size, size), buf).transpose(Image.FLIP_TOP_BOTTOM)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    cov = (np.asarray(img)[..., 3] > 10).mean()
    glfw.terminate()
    print(f"baked {out}  ({size}x{size}, width/height aspect={2 * hx / hy:.2f}, "
          f"silhouette coverage={100 * cov:.1f}%)")
    return 2 * hx / hy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)
    bake(args.glb, args.out, args.size)


if __name__ == "__main__":
    main()
