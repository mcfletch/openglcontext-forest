# openglcontext-forest-demo

A walkable, near-photorealistic forest demo for
[OpenGLContext](http://pyopengl.sourceforge.net/context/): real Great-Smoky-Mountains
elevation under a runtime multi-layer **splat terrain**, a GPU-instanced forest with
distance **LOD** (real tree *meshes* near you, baked *impostor* billboards far off),
two-layer camera-following grass, terrain sun-shadows and canopy shade, and
first-person **walking** (ground-clamped, blocked by trunks). ~230k trees, comfortably
above 60 fps.

The generic rendering engine lives in OpenGLContext
(`OpenGLContext.scenegraph.terrain`, `OpenGLContext.scenegraph.vegetation`,
`OpenGLContext.move.terrainwalk`). This package is just the **scene**: the biome mix
and the concrete geometry/textures/heightmaps.

## Run it

Once published, no checkout needed:

```bash
uvx --with openglcontext-forest-demo oglc-forest
```

From this repository (engine + demo are local, editable):

```bash
uv pip install -e ./openglcontext -e ./forest-demo
oglc-forest
# or: python -m openglcontext_forest_demo
```

First launch fetches CC0 ground textures from ambientCG (cached afterwards).

**Controls:** W/A/S/D move, mouse-look; arrows / PageUp-Down also navigate.

## Licensing

The demo's source code is MIT (`LICENSE`). The art assets carry their own licenses,
enumerated in [`ASSET-LICENSES.md`](ASSET-LICENSES.md): tree models are **CC-BY 4.0**
(attribution printed on launch), ground textures are CC0, terrain elevation is open
public data. The source `.glb` tree models are kept in `assets-source/` as provenance
and are not shipped in the wheel.

## Layout

```
forest-demo/
  pyproject.toml            # packaging; entry point oglc-forest
  ASSET-LICENSES.md         # every asset + license + provenance
  assets-source/            # downloaded source .glb models (provenance, not packaged)
  src/openglcontext_forest_demo/
    run.py                  # the scene + main()
    assets/                 # baked runtime geometry/textures/heightmaps (packaged)
  tools/                    # bench.py, capture.py, bake_assets.py (dev tools)
```

## Regenerating the tree assets

The runtime `*.npz` meshes, textures and impostor billboards are baked from the
source `.glb` models in `assets-source/`:

```bash
python tools/bake_assets.py                     # regenerate all into the package
python tools/bake_assets.py --only fir real3    # a subset
python tools/bake_assets.py --no-impostors      # geometry + textures only (no GL)
```

The bake reproduces the shipped mesh geometry exactly (unit height, base at y=0, XZ
centroid at origin); impostors are re-rendered front-on. This is the reproducible
provenance chain behind `ASSET-LICENSES.md`.
