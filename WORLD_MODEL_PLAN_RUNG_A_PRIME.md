> **FROZEN ARCHIVE (the A′ design as planned).** The project is now tracked by exactly two living
> documents: `pokemon-worldmodel/docs/PLAN.md` (plans/status) and
> `pokemon-worldmodel/docs/STRUCTURE.md` (codebase/contracts). Do not update this file.

# Rung A′ — Identity-Conditioned Full-Modality World Model (start → Stone Badge)

> Detailed design + execution plan for the next training generation. Child document of
> `WORLD_MODEL_PLAN.md` (north star / ladder / status). Decisions below were locked in discussion
> on 2026-06-10. Last updated: 2026-06-10.

---

## 0. What and why

Train a **fresh, fully-conditioned model of everything Pokémon Emerald shows from game start to the
Stone Badge** — overworld, dialogue, battles, menus, cutscenes, transitions — as the new top anchor
of the conditioning-reduction ladder.

**Rung A′ defined.** The original Rung A conditions on the raw PPU state (provably render-sufficient,
but the model can near-copy the answer). The shipped first model ("naive Rung B") conditioned on a
*learned per-map canvas* + coords — memorization posing as state. A′ is the principled point between:

> **The condition says WHAT is on screen (identities, layout, logic state) — never HOW it looks
> (no pixels, no tile content, no palettes). All appearance lives in learned weights.**

Consequences, all deliberate:
- The model **learns the game's art** (every metatile, sprite sheet, glyph, Pokémon, UI) keyed by
  identity embeddings — and therefore generalizes across maps/modes by construction (no map vocab,
  no `in_vocab` scope guard, one interface for every screen the game can draw).
- A′ is **not provably render-sufficient**: a tail of visual dynamics (move animations, door swings,
  grass rustle, ball shakes…) is conditioned only at event level (IDs + flags) and must be carried by
  **action + visual history** — the *declared learned tail*. This is a feature: it is the first small,
  measured step down the ladder, built in from day one.
- Every reduced rung (B, C, Z) remains an offline mask of the recorded state, and (new) a **training-
  time mask** — see §3 group-dropout.

Out of scope for this generation: anything not reachable before the first badge (Surf/Cut/bike/etc.,
absent from the game segment by construction); literal full-game (8 badges) is a later collection
roadmap, not this plan.

---

## 1. Condition schema v0

To be finalized (v1) by the Phase-0 audit. Sources are RAM structs (pokeemerald symbols); every one
is present in the recorded `ewram`/`iwram` blocks, so extractors are built **offline against existing
data** and the identical parser runs live in the demo.

| group | identity content | source (RAM) | injection |
|---|---|---|---|
| terrain | metatile ID per visible screen cell, keyed `(tileset, local_id)` | map layout grid + map header | channel-concat (embedding → cond ch) |
| camera | per-pixel scroll; sub-tile phase | BG scroll registers (io) | AdaLN scalars (+ phase channels) |
| entities (player = entity 0) | graphics_id, facing, anim/walk frame, **pixel** position, priority | `gObjectEvents` / `gSprites` | **splat**: every covered latent cell gets (gid, part-offset dx,dy, anim, prio) |
| text | char codes + reveal count + window rect (typewriter state) | text-printer structs | **prefix tokens** (typed, per frame) |
| battle | species IDs, HP/level/status, menu+cursor state, move IDs, battle background ID | `gBattleMons` + battle menu state | prefix tokens |
| menus / UI | UI-type ID, cursor index, displayed item/party/page IDs (strings go via text path) | window/task state | prefix tokens |
| effects | fade level (BLDY…), flash, shake, weather, mosaic | io registers + fields | AdaLN, **×4 sub-frame values** |
| action | 10-button vec, **×4 sub-frame values** | input log | AdaLN |
| mode | derived label (overworld / dialogue / battle / menu / transition) | classifier over the above | AdaLN |

Schema rules:
- **Identity only.** If a feature encodes appearance (tile pixels, palette colors), it is excluded.
- **Latent timing.** 1 latent frame = 4 game frames; conditions use the group's representative frame,
  except fast-changing scalars (effects, action) which pass all 4 sub-frame values.
- **The learned tail is explicit.** Anything visual not in the table (move animations' unfolding,
  ambient tile animation, reflections, dust/rustle) is by definition carried by history. The audit
  names and measures this tail; growing the schema to chase it requires a deliberate decision.

---

## 2. Phase 0 — the audit (the spec, before any building)

We hold full RAM for every recorded frame, so completeness and sufficiency are **measured, not argued**.
Run over the whole corpus (~863k frames: 51 storyline segments + 2 exploration seeds):

1. **Mode taxonomy** — classify every frame from RAM (game-state callback / script ctx / battle flags);
   output the per-mode frame-mass histogram.
2. **Identity support** — per group: frames per metatile, per graphics_id, per species, per move,
   per UI screen, per NPC-dialogue; idle-duration histogram; input-pattern counts (bumps, turn-in-place,
   run-toggles, text-advance styles).
3. **Unexplained-dynamics detector** — frames where pixels change but schema-v0 condition doesn't:
   literally a detector for missing state / the learned tail. Inspect, then either add a group or
   declare-and-accept.
4. **VAE text-fidelity gate** — encode→decode dialogue/menu frames through the (finetuned-decoder) VAE;
   measure glyph legibility. Sets expectations for text rendering; NOT a redesign trigger now (user call).
5. **Replay-determinism check** — savestate + input log → bit-identical RAM stream on replay (the
   storage principle that makes every future schema change recollection-free).
   **RESULT (2026-06-10): replay is gameplay-deterministic but NOT frame-exact** — inputs reproduce
   the trajectory (final position exact), but timer/RNG phase drift (a persistent ~15B of io/ewram +
   iwram counters that savestates don't restore) shifts ambient-animation/text timing, so only ~44%
   of frames are pixel-identical. ⇒ Re-derivability holds **via the stored full per-frame state**
   (all six blocks are recorded every frame — any future symbolic field is already in them), NOT via
   replay. New collections must keep storing the full state; never economize assuming replay can
   recreate frames.

**Deliverables:** coverage report; condition schema v1; declared learned tail; **deficit shopping list
with numeric thresholds** (e.g. every pre-badge species ≥ N battle-frames, idle ≥ X% of corpus,
every NPC dialogue ≥ K visits) — the input that drives Phase 2.

Today's corpus already fails known thresholds: idle ≈ 0 frames, all wild battles fled, one starter of
three, menus barely browsed. (The shipped model's won't-stop-when-idle bug is gap #1 made visible.)

---

## 3. Phase 1 — extractors (offline = online)

One module per condition group, written against recorded RAM, validated before any training:
- spot-check fields against independent sources (entity positions vs OAM; text vs the rendered
  tilemap region; battle HP vs visible bar) on sampled frames across all modes;
- the **same parser** reads live emulator memory in the demo (pattern proven by `render_state.py` /
  `memory_reader`); no live/offline skew by construction;
- precompute: conditions for all clips at latent timing → training tensors.

Grubbiest item (flagged): Emerald menus run as tasks with state in task data — the UI group is an
enumeration effort, prioritized **by audit mass** (a 0.1%-mass screen gets the generic
`UI-type + cursor + strings` fallback, not bespoke fields).

---

## 4. Phase 2 — collection (close the loop)

**Principle: audit-driven, closed-loop.** Collection is scheduled from the deficit report; after
collecting, re-audit; iterate until thresholds pass. No more open-loop "run the producer once and
discover gaps from a trained model."

Components:
- **Behavior library** (parameterized primitives, mostly existing harness code): `goto`, `idle(t)`,
  `fidget` (bumps / tap-turns / run-toggles / pauses), `interact`, `battle_policy`
  (fight / switch / item / **catch** / flee, target level — let things faint, level up, evolve),
  `menu_walk`, `text_advance_style` (mash A / hold B / wait mid-box).
- **Deficit scheduler** — composes primitives against the shopping list.
- **Savestate bank** — seed collectors from chain checkpoints across story flags/locations/parties;
  run the storyline chain **3× (one per starter)** to fix starter/party/rival-battle diversity;
  replay one-off scenes (truck intro, clock, naming screen ×N names) for support.
- **QC gates per run** — extractor round-trip spot-checks, stream alignment, mode histogram, dedup;
  reject at produce time.
- **Provenance** — every run stores initial savestate + input log (replay-derivable forever).

Rough volume targets (audit will refine): idle ~50k, fidget ~100k, battles ~300k (≈150–200 varied
encounters), menus+dialogue ~150k → **roughly doubling the corpus**. Emulator is ~27× real-time and
collectors parallelize across processes: days of wall-clock.

---

## 5. Phase 3 — model + training (fresh run)

**Core (unchanged, proven):** latent-space causal video DiT (frozen Wan VAE, 4× temporal — 1 latent
= 4 game frames), diffusion forcing, rectified flow, clean-context KV-cache AR (bit-faithful
streaming inference). Existing test suite carries over.

**Conditioning encoder (replaces `OverworldConditionEncoder`):** one module per schema group;
injection = channel-concat (terrain) + splat (entities) + prefix tokens (text/battle/UI, appended to
spatial tokens per frame, dropped before unpatchify) + AdaLN (scalars). The learned per-map canvas,
map vocab, and `in_vocab` guard are deleted.

**DiT changes:**
- **patch 1** (600 tokens/frame; was 2/150) — glyph-scale fidelity; affordable because inference is
  measured overhead-bound, not compute-bound. *Fallback if training throughput hurts: patch 2 + a
  patch-1 ablation.*
- **window 24** (≈1.6 s context; was 16) — the learned tail is carried by history.
- prefix-token plumbing in spatial attention.
- **size held at ~200M core** (user constraint): re-cut shape within budget (e.g. keep 1024×8 vs a
  deeper-narrower 768×14 — settle with a short probe). Identity embedding tables (metatiles, sprites,
  species, chars) add a few M of *sparse* params on top; they replace, and are much smaller than, the
  old per-map canvases.

**Group-dropout = the ladder in one model.** Sample canonical condition masks per clip:
`full` (upweighted) / `no-text` / `semantic-only` / `coords-only` / `action-only`. One set of weights
spans the rungs; descent experiments become inference-time mask choices + cheap finetunes, and rung
comparisons are controlled (same weights, same data). Hedge: monitor full-mask fidelity vs a no-dropout
control; tune mask weights.

**Training:** fresh WSD run on the full mixture (existing 863k + Phase-2 collection), mixture weights
set from the audit's per-mode masses (re-balance away from 74% walking). Stopping stays user-judged
on visual rollouts (no PSNR auto-cut).

**Eval (per mode, not pooled):** held-out clips of every mode incl. one full unseen storyline segment;
dashboards for short-horizon fidelity (PSNR/LPIPS), AR drift, action-following, and event-rendering
checks (does the right text appear; do HP bars match battle state). Learned-tail effects are scored on
plausibility/stability, not frame-exact timing — they are stochastic by design.

---

## 6. Phase 4 — demo integration

Live condition extraction for all groups (same parsers as Phase 1) in the existing two-process
Blackwell demo. The `in_vocab`/`in_battle` GT-fallback **disappears** — battles, dialogue, menus and
unseen maps render live. GT-vs-generated side-by-side stays as the honesty display.

## 7. Phase 5 — ladder descent (outlook, the thesis payoff)

With masks trained in: evaluate each canonical mask across the full modality suite; first descents —
**drop text content** (box known, words unknown), then **terrain IDs → (map_id, coords)** (forces map
memory), then entities, then coords. Chart where each modality breaks as conditioning is removed;
that chart *is* the research result.

---

## 8. Risks

| risk | mitigation |
|---|---|
| menu/UI extraction grubbiness (task-based state) | prioritize by audit mass; generic UI fallback for the long tail |
| VAE glyph ceiling | measured in Phase 0; expectation-setting now, decoder finetune on text-heavy frames later if needed (encoder stays frozen) |
| patch-1 training cost | fallback to patch 2; keep patch-1 as ablation |
| group-dropout costs full-condition fidelity | upweight `full`; no-dropout control run; tune |
| 200M underfits the full modality set | embeddings carry identity (capacity-light); watch TRAIN-TF fidelity (the proven underfit probe); escalate to user if it reappears |
| learned-tail too stochastic (animations drift) | acceptable by design; if a specific effect matters, promote it into the schema deliberately |
| one-off scenes too thin to learn | scripted replays ×N from the savestate bank |

## 9. Order of work

Phase 0 (audit) → Phase 1 (extractors) → Phase 2 (collection loop, overlaps Phase 1) → Phase 3
(encoder/DiT build + fresh run) → Phase 4 (demo) → Phase 5 (descent). The existing RUSTBORO_L demo
stays live untouched until the A′ model supersedes it.
