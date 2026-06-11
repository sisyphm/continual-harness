# The cross-repo data contract (continual-harness ⇄ pokemon-worldmodel)

The harness COLLECTS and EXTRACTS; the model repo CONSUMES. Every artifact that crosses the
boundary is listed here with its owner. **Current `schema_version: 2`** — stamped in the corpus
manifest and each conditions sidecar; readers reject mismatches loudly
(`pokemon_worldmodel/aprime/data.py`, `pipelines/build_index.py`).

```
collect (sink) ─▶ run dirs ─▶ precompute_conditions ─▶ conditions/*.npz + manifest + charset
                                                            │
                              model repo: build_index ─▶ precompute_latents ─▶ train
```

## Recorded run directory (writer: `collection/world_model_sink.py` + recorder)

| file | contents |
|---|---|
| `frames_*.npz` chunks + `frames.jsonl` | RGB 160×240×3 per emulated frame + frame→(chunk, offset) map |
| `actions.jsonl` | per-frame `button_vec` (alphabetical keys; canonical order is `core.timing.BUTTON_ORDER`) + `frame_idx`/`next_frame_idx` |
| `semantic.jsonl` | per-frame `{frame, x, y, facing, map, in_battle, objects[]}` — see versions below |
| `ppu_state.bin` + `.idx.json` | full PPU+RAM condition blob per frame (396,288 B = io\|palette\|oam\|vram\|ewram\|iwram), keyframe + XOR-delta, zlib. Walk with `extractors.ram.iter_states` ONLY |
| `coverage_summary.json` | optional; `clip_start_frames` = reset/teleport boundaries clips must not span |

**semantic.jsonl versions** (the v2 cut is 2026-06, commit `de0baf2d`):
- *v1 era (all 119 corpus runs)*: `objects` from the WRONG ObjectEvent layout (garbage; never train
  on it), `facing` from the collector's input tracker (stale after warps/forced turns).
- *v2 (future runs)*: `objects` (NPCs only: `slot, graphics_id, local_id, x, y, facing`, map coords,
  MAP_OFFSET applied) + player `facing` via the validated `extractors.entities`.
- Training is UNAFFECTED by the difference: the A′ precompute re-extracts everything from
  `ppu_state.bin`; `semantic.jsonl` feeds only audits and quick filters.

## data/processed/ (the boundary directory, lives in the model repo's data root)

| artifact | writer | reader | notes |
|---|---|---|---|
| `corpus_manifest.json` | `collection.corpus` (also auto-refreshed by every precompute) | `pipelines/build_index.py` | `{schema_version, total_frames, runs:[{run_key, kind, name, dir(rel to data root), frames, clip_starts}]}`. run_key == conditions npz stem. THE run-discovery interface — the model repo never walks the data tree |
| `conditions/<run_key>.npz` + `.json` | `extractors.conditions.RunConditionWriter` | `aprime.data.RunConditions` | per-GAME-FRAME condition arrays; full key/dtype table in the `conditions.py` docstring. Sidecar: `{schema_version, frames, n_grids, texts[]}` (sidecars without the stamp ARE v2 — the aprime-v1 corpus predates stamping) |
| `conditions/tilesets.json` | `extractors.tilesets` (ROM walk) | `aprime.data` | map `"group,num"` → (primary, secondary) tileset ids; `n_tilesets` |
| `conditions/charset.json` | `extractors.text.export_charset` | model `tests/test_charset_contract.py` | THE Gen-3 charset home (decode byte→char + ctrl + terminator 0xFF). The model's encode table must invert it — pinned by the contract test |
| `aprime_index.json` | model `pipelines/build_index.py` | `aprime.data.APrimeDataset` | clips `{clip_id, run_key, run_dir, start, length}`; lengths 1+4k in [33, 509]; splits only at `clip_starts` |
| `aprime_latents/<clip_id>.npz` | model `pipelines/precompute_latents.py` | `aprime.data.APrimeDataset` | fp16 Wan-VAE latents (T,16,20,30), T = 1+(len−1)/4 |
| `audit/**` | `collection.audits.*` | humans + `audits.report` | the measurement campaign (not a training input) |

## Timing convention (shared, load-bearing)

1 latent frame = 4 game frames, Wan-VAE causal split 1,4,4,… (`core.timing.latent_groups`).
Identity state is sampled at each group's LAST frame; fast scalars carry all 4 sub-frame values.
Conditions are stored per game frame precisely so the model repo owns this aggregation.

## Version history

| schema | meaning |
|---|---|
| 1 | v1 era: legacy semantic objects/facing; overworld-only index; no stamps (implicit) |
| 2 | A′ era: validated extractors; full-modality conditions npz; manifest + charset exported; stamps everywhere new |

Bump rules: any key/dtype/semantic change to a boundary artifact bumps `collection.corpus.
SCHEMA_VERSION` and this file, and the readers' expected version — never change meaning silently.
