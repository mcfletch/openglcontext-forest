# Asset licenses

Every art asset used by the forest demo, its license, and where it came from.
The demo **prints the CC-BY attributions on launch** (a CC-BY requirement).

The demo's own *source code* is MIT (see `LICENSE`). The assets below carry their
own licenses and are **not** relicensed by that MIT grant.

---

## Tree models — CC-BY 4.0 (attribution required)

License for all five: Creative Commons Attribution 4.0,
<http://creativecommons.org/licenses/by/4.0/>.

The bundled runtime files (`*.npz` meshes, bark/branch/leaf textures, and baked
`*_imp*.png` impostor billboards) are **derivative works** baked from the source
`.glb` models below, so they carry the same CC-BY 4.0 license and attribution.

| Model | Author | Source | Source glb (in `assets-source/`) | Baked runtime assets (in `src/openglcontext_forest_demo/assets/`) |
|---|---|---|---|---|
| Fir tree | Georgeous | https://skfb.ly/pA8TG | `fir_tree.glb` | `fir.npz`, `fir_bark.png`, `fir_branch.png`, `fir_imp.png` |
| Noel_Pine_Tree | 3D Error 404 | https://skfb.ly/6XHoJ | `noel_pine_tree.glb` | `noel.npz`, `noel_bark.png`, `noel_branch.png`, `noel_imp.png` |
| Maple trees pack (lowpoly, game ready, LODs) | LOLIPOP | https://skfb.ly/p9tGx | `maple_trees_pack_lowpoly_game_ready_lods.glb` | `maple0..3.npz`, `maple_bark.png`, `maple_leaves.png`, `maple_imp0..3.png` |
| Realistic Trees Collection | Jungle Jim | https://skfb.ly/pDzJR | `realistic_trees_collection.glb` | `real0..6.npz`, `real_br0..6.png`, `real_lf0..6.png`, `imp0..6.png` |
| Low Poly Forest Tree Pack | 99.Miles | https://skfb.ly/pJXrH | `low_poly_forest_tree_pack.glb` | *(not currently used in the scene; kept as provenance)* |

The source `.glb` files are kept in `assets-source/` as provenance for the baked
runtime assets. They are large (~140 MB total) and are **not shipped in the wheel**
— only the baked runtime assets are packaged. The baked assets are reproducible from
the sources with `tools/bake_assets.py` (mesh geometry reproduces exactly; impostors
are re-rendered).

## Ground / detail textures — CC0 (public domain, no attribution required)

Fetched at runtime from [ambientCG](https://ambientcg.com/) and cached under
`~/.cache/openglcontext/cc0/` (not bundled). Materials used as splat layers:
Grass004, Ground042 (forest floor), Ground068 (dirt), Rock030, Snow006, Bark012.
License: CC0 1.0, <https://creativecommons.org/publicdomain/zero/1.0/>.

## Terrain elevation — public / open data

`assets/forest_height.png` is a Great Smoky Mountains heightfield derived from the
public [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
(Terrarium/Mapzen), which aggregate SRTM, USGS 3DEP and other open sources.

## Generated assets — CC0

Created procedurally by/for this demo; released CC0:

- `assets/forest_control.png` — the RGBA splat control map (biome weights).
- `assets/grass.png` — the grass-tuft billboard texture.
