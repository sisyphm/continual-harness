# Phase 0 — Coverage & Sufficiency Audit (RESULTS)

> Deliverable of `WORLD_MODEL_PLAN_RUNG_A_PRIME.md` §2, executed 2026-06-10 over the full recorded
> corpus (863,409 frames: 51 storyline segments + 2 exploration seeds). Code:
> `collection/audit_wm{,_ram,_textbox,_framediff,_modes,_report,_replay}.py`. Machine outputs in
> `pokemon-worldmodel/data/processed/audit/` (JSON reports, cb2 contact sheets, artifacts).

---

## 1. Headline findings

1. **`semantic.jsonl` objects are garbage dataset-wide (extractor bug, now understood).**
   `world_model_sink.extract_objects` read `gObjectEvents` with stride 68 and `graphicsId` at +0x03;
   the real struct (validated against live frames) is **36 bytes (0x24)** with `graphicsId` at
   **+0x05**, `localId` +0x08, initial/current/previous `Coords16` at +0x0C/0x10/0x14 in **plain map
   coords (no +7)**, facing-ish nibbles at +0x18; bit0-of-byte0 is NOT usable as the active flag
   (0xFF-cleared slots read active) — validity must be tested structurally. Corrected parsing
   recovers **67 distinct NPC graphics ids** (vs 4 under the bug). ⇒ v1's NPC conditioning channel
   was noise. Fix lands in the Phase-1 extractors; `DATASET.md` needs an erratum; everything is
   re-derivable from the stored ewram (no recollection).
2. **VAE text gate: PASS.** Dialogue frames round-tripped through the (decoder-finetuned) VAE keep
   text legible (text-band PSNR 23.1 dB, full frame 24.4 dB; see
   `audit/vae_text/text_bands_2x.png`). Dialogue rendering is NOT VAE-blocked.
3. **Replay is gameplay-deterministic, NOT frame-exact.** Inputs reproduce the trajectory (final
   position exact), but ~15B of timer/RNG state that savestates don't restore shifts ambient
   animation/text timing — only ~44% of replayed frames are pixel-identical. ⇒ Frame-exact
   re-derivability holds **via the stored full per-frame state**, not via replay. New collections
   must keep storing all six blocks per frame.
4. **Mode taxonomy is tractable: 38 distinct `gMain.callback2` screens** (sheets in
   `audit/cb2_sheets/`, composition in `audit_report.json:cb2_composition`):
   - `0x8085e5d` **CB2_Overworld** (55.4%) — walking, dialogue-over-overworld, START-menu overlay
   - `0x8038421` **battle main** (36.3%, incl. black init/teardown siblings `0x81aad5d`,
     `0x8036fad`, `0x80a933d`, `0x813e3a5`, `0x81aad8d`)
   - `0x8085e51` overworld-transition (3.4%) — battle-intro wipes, map loads
   - `0x802f6b1` intro/Birch presentation (0.5%); `0x8085ffd` + tail = black loads/transitions
   - micro-UIs (naming screen etc.) live in the <0.1% tail — the predicted one-off thinness,
     confirmed
   - dialogue is NOT a cb2 — detected via the validated visual textbox flag (95.8% on dialogue
     segments, FP-checked); START-menu/bag/party need the Phase-1 UI extractor (no audit signal yet)
5. **Battle identity support is the largest deficit, now exact:** player-side species = **2**
   (Mudkip 332,709 battle-frames; tutorial Zigzagoon 4,037) — one starter, zero caught-mon variety;
   enemy species = **16**, heavily skewed (Poochyena 146k → tail 736). `gBattleMons @0x02024084`
   validated.
6. **Input-pattern deficits confirmed at zero:** run-with-B-while-moving = **0 frames in 863k**
   (verify Running-Shoes obtainability pre-badge-1 during Phase 2; if obtainable, collect);
   storyline bumps = 0 (coverage: 1,008); free idle exists only in degenerate clumps (§2).

## 2. Aggregate masses (full corpus: 863,409 frames @60fps)

| mode (precedence-classified) | frames | share |
|---|---|---|
| overworld, free (no battle/dialogue/dark) | 411,486 | **47.7%** |
| battle (incl. black init/teardown) | 336,775 | **39.0%** |
| dialogue (textbox over overworld) | 75,837 | **8.8%** |
| transition (dark warp/load) | 39,311 | 4.6% |
| fade in progress (BLDY>0) | 2,728 | 0.3% |

- **Free idle** (player stationary ∧ free-overworld): 57,153 frames in ≥64f runs (6.6% of corpus) —
  but **degenerately distributed**: 5 runs ≥256f (explorer NPC-wait stalls) hold most of the mass;
  only 556 runs of 1–2s; everyday "stand around and look" (varied spot/facing/duration) is absent.
  The histogram peak is the 8–15f inter-step pause (16,684 runs). The won't-stop demo bug is
  consistent with this: idle support exists only in degenerate clumps. S1 stands, reworded to
  *varied* idle.
- **NPC identity support (corrected)**: **67 distinct graphics ids**; long-tailed but live.
- **Unexplained dynamics (schema-v0 proxy)**: 385,144 frames (**44.6%**) where pixels change but
  the proxy condition is static. Visual inspection (`audit/unexplained_dynamics.png`) names it:
  (a) **NPC sub-tile walk motion** — tile coords static while sprites slide (becomes CONDITIONED
  under A′'s pixel-coord splat + anim phase); (b) **battle animations + menu cursor/text** (menu
  state becomes conditioned via Phase-1 battle extraction; move animations = true learned tail);
  (c) **ambient tile animation** — water, flowers, fountains, sunbeams (true learned tail, carried
  by AR history). So 44.6% is the proxy ceiling; the true post-Phase-1 tail is mostly (c) + move
  animations — re-measure after extractors land.

## 3. The deficit shopping list (drives Phase-2 collection)

Numeric targets; "support" = frames at 60 fps. Re-audit after collection; iterate until all pass.

| # | deficit (measured) | target | collector |
|---|---|---|---|
| S1 | free idle exists only as degenerate NPC-wait clumps | ≥ 40k frames of *varied* idle: ≥20 maps × 4 facings × durations 1–10s, NPCs around | `idle(t)` schedule |
| S2 | run(B) gait = 0 frames | ≥ 30k running frames across routes (if obtainable pre-badge) | `fidget`/`goto(run=True)` |
| S3 | bumps ≈ 0 (storyline) | ≥ 5k bump events, all 4 facings | `fidget` |
| S4 | turn-in-place 8k events total | ≥ 30k events | `fidget` |
| S5 | player-side species = 2 | all 3 starters × levels 5–15; ≥ 5 caught species in-party | chain ×3 + `battle_policy(catch)` |
| S6 | enemy species = 16, skewed | every pre-badge wild+trainer species ≥ 3k battle-frames | `battle_farmer` per route |
| S7 | battle outcomes: flee-dominated | faints, catches (ball shakes), level-ups, ≥1 evolution | `battle_policy` variants |
| S8 | menus barely browsed (no audit signal) | bag/party/summary/options/PC ≥ 10k frames each | `menu_walk` |
| S9 | dialogue input texture: scripted pace only | mash-A / hold-B / mid-box-wait variants over ≥ 50 NPCs | `text_advance_style` |
| S10 | one-off scenes: 1 visit each (truck, clock, naming) | ×10 replays each, varied names/inputs | intro-segment replays |

## 4. Schema-v1 notes (what the audit changes in the condition design)

- **Entities**: extractor = corrected 36-byte `gObjectEvents` parse (structural validity, not bit0);
  facing/anim from the +0x18 nibbles (pin exact encoding in Phase 1); player object localId=0xFF
  needs its own resolution (gPlayerAvatar path) — its record's coord semantics differ.
- **Dialogue/text**: no validated RAM flag yet (DIALOG_STATE/script-ctx addresses are stale for this
  build) — Phase 1 must locate the text-printer state empirically (the visual textbox flag provides
  free labels for that search); until then the visual flag is the audit/mixture signal.
- **Mode token**: derive from cb2 (38 values → ~8 labels) + textbox flag; battle sub-phase from
  `gBattleCommunication`/cb2 siblings.
- **Fades**: BLDY io register confirmed live in the data (mass in §2); ×4 sub-frame values per
  latent frame as planned.
- **Stale-address warning**: every harness address must be revalidated against recorded ewram
  before use (two of three audited addresses were wrong for this build).

## 5. Done / next

Phase 0 complete (all five audit items + this report). **Next: Phase 1 extractors** (corrected
objects, text-printer search, battle/UI state), in parallel with **Phase 2 collection** per §3.
