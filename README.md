# openglcontext-forest-demo

A walkable, forest demo for
[OpenGLContext](http://pyopengl.sourceforge.net/context/): real Great-Smoky-Mountains
elevation under a runtime multi-layer **splat terrain**, a GPU-instanced forest with
distance **LOD** (real tree *meshes* near you, baked *impostor* billboards far off),
two-layer camera-following grass, terrain sun-shadows and canopy shade, and
**mouse-look walking** with gravity (ground-clamped, blocked by trunks). ~230k trees,
comfortably above 60 fps.

The generic engine lives in OpenGLContext — `OpenGLContext.scenegraph.terrain` and
`.vegetation` for the world, `OpenGLContext.move.terrainwalk` for the avatar that
walks it, and `OpenGLContext.ui` for the menu, settings and key-binding screens. This
package is just the **scene**: the biome mix and the concrete
geometry/textures/heightmaps.

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

## Controls

You start in **mouse-look**: the pointer steers and the forest is walked, not flown.

| | |
|---|---|
| mouse | look around |
| `w` `a` `s` `d`, arrows | move (`a`/`d` strafe) |
| `shift` (held) | run |
| `space` | jump; rise, while flying |
| `c` | sink, while flying |
| `m` | cycle the way you move: mouse-look, walk (`q`/`e` turn), fly |
| `f` | fly (noclip) on/off |
| `g` | hand the camera to the free-fly navigator, and back |
| `escape` | the menu: Resume, Controls, Settings, Asset credits, Quit |
| `F6` / `F10` | the key bindings / the rendering settings |
| `F8` | cycle the render-quality preset (low / medium / high) |
| `F2` | save a screenshot into the working directory |
| `alt` + `f` | the developer overlay |

The keys are not fixed here: each way of moving is a declared `MovementMode` node
carrying its own speeds and bindings, so the `F6` page can rebind them and they are
saved for next time. `F10` edits the rendering the same way — shadows, environment
lighting, anti-aliasing — without a restart. Both are OpenGLContext's own overlay
screens, so this demo, `oglc-view` and `twitch` are driven the same way.

## Quality and performance

`--quality` chooses how much near-field grass the scene carries — the layer that
dominates GPU cost — across `high` (the shipped look), `medium` and `low`:

```bash
oglc-forest --quality medium
```

The default, `--quality auto`, measures the frame rate at start-up and steps the
preset down until the forest holds ~60 fps, so an integrated GPU lands on `medium`
and a discrete one stays at `high`; shadows stay on. `F8` cycles the preset by hand
at any time, which both picks a level and shows what each one costs.

The real-geometry grass clumps carry a distance LOD: full-detail blades close to the
camera, a coarser-mesh set from ~45% of the clump radius out to its edge, cross-fading
at the boundary. Most of the disc area is in the far band, so the coarse set carries
most of the clumps at a fraction of the per-clump vertex cost — the `high` look at close
to the `medium` frame time. `--clump-far-length-samples` tunes the far detail.

The camera-following grass and impostors re-scatter as you move. That scatter runs
on a background thread and the render loop only uploads the finished arrays, so
crossing a streaming boundary no longer lands its work on a single frame.

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
    config.py               # every user-facing knob, and the command line
    scene.py                # build_forest_scene(): the reusable world
    run.py                  # the navigation, the screens and main()
    menu.py                 # the menu Escape raises
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
