# openglcontext-forest-demo

A walkable, forest demo for
[OpenGLContext](http://pyopengl.sourceforge.net/context/): real Great-Smoky-Mountains
elevation under a runtime multi-layer **splat terrain**, a GPU-instanced forest with
distance **LOD** (real tree *meshes* near you, baked *impostor* billboards far off),
a camera-following **forest floor** of grass, fern, nettle, periwinkle and shrub,
terrain sun-shadows and canopy shade, and
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
uv run --with openglcontext-forest-demo oglc-forest
# or, to run it without keeping anything: uvx --from openglcontext-forest-demo oglc-forest
```

`--with` and `--from` both name the *package*; `oglc-forest` is the command it
provides, and uv looks for a package under that name if it is given one.

From a checkout of the workspace (engine and demo local and editable):

```bash
uv pip install -e ./openglcontext -e ./openglcontext-forest
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
| `F2` | save a screenshot into your picture folder |
| `alt` + `f` | the developer overlay |

The keys are not fixed here: each way of moving is a declared `MovementMode` node
carrying its own speeds and bindings, so the `F6` page can rebind them and they are
saved for next time. `F10` edits the rendering the same way — shadows, environment
lighting, anti-aliasing — without a restart. Both are OpenGLContext's own overlay
screens, so this demo, `oglc-view` and `twitch` are driven the same way.

## Quality and performance

`--quality` chooses how much near-field ground cover the scene carries — the layer
that dominates GPU cost — over a five-rung ladder from `high` (the shipped look) down
through `medhigh`, `medium`, `medlow` to `low`:

```bash
oglc-forest --quality medium
```

The default, `--quality auto`, renders the shipped `high` look and measures the frame
rate once at start-up, then picks the **highest rung that holds ~60 fps** (scaling the
measurement by each rung's known relative cost) — so it lands on the fullest level the
GPU can sustain rather than jumping straight to `low`. A capable GPU keeps `high` and
never changes. A GPU that can't sustain it applies its pick **as you next move**,
folded into the grass the field is already restreaming rather than re-scattering the
whole field in place while you stand still (which is what reads as a pop). It decides
once — and **re-measures if you resize or go fullscreen**, since fill cost scales with
pixels (an integrated GPU can hold `high` in a window but not fullscreen). Shadows stay
on. `F8` cycles the rungs by hand at any time.

The distance chain the cover is drawn through is the engine's
(`OpenGLContext.scenegraph.vegetation.GroundCover`), and this demo only says what
grows: real geometry in two levels of detail near you, cards past them, and a
coarse field out to the haze. Most of the geometry disc is covered only by the
decimated level, so it carries most of the plants at a fraction of the vertex cost
— the `high` look at close to the `medium` frame time.

The geometry disc tracks the camera every frame: the scatter is cached over a disc
a little wider than the drawn one, and each frame the drawn subsets are re-selected
from that cache against the live camera position. So the disc's leading edge fades
in through the level-of-detail band as you approach it, rather than the
mid-distance cover stepping up in density each time a streaming boundary recentred
a lagging disc.

## What grows on the floor

The undergrowth is a *set* of plants, not one repeated: grass, fern, nettle,
periwinkle and shrub, baked from CC0 scans published by
[Poly Haven](https://polyhaven.com/) (see [ASSET-LICENSES.md](ASSET-LICENSES.md)).
Each says how densely it grows, how much it gathers into beds, and what depth of
tree cover it grows under — so ferns stand in beds in the shade, shrubs take the
gaps where the trees thin out and the edges of clearings, grass runs through all
of it and periwinkle flowers through the grass. `--cover-density` scales the whole
set at once.

What grows is read from `assets/cover.json`, which `oglc-bake-plants` (in
`openglcontext-editor`) writes beside the models it names, so changing the mix is a
re-bake rather than a code change.

The camera-following cover and impostors re-scatter as you move. That scatter runs
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

## Releasing the art

The art is 66 MB and does not travel in the wheel: it is a **content pack**,
attached to a GitHub release and fetched by
[`OpenGLContext.contentpacks`](https://github.com/mcfletch/openglcontext/blob/main/docs/contentpacks.html)
before the first frame. One command builds it, writes the registry from what it
built, and publishes it:

```bash
./release-assets.py                 # build the archive, write packs.json
./release-assets.py --install       # ...and install it in this machine's store
./release-assets.py --reinstall     # ...over whatever that store already holds
./release-assets.py --push          # ...and attach it to the release tag
```

`--install` is how a content release is driven before it is a release: the demo
finds the pack in its own store and runs as an installed copy would, with
nothing published. It leaves a pack already in the store where it is; `--reinstall` does the same over whatever the store already holds, which is what a rebuilt world needs to be the one that opens. `OPENGLCONTEXT_CONTENT` points a packaged, offline or CI run
at a directory of packs instead.
