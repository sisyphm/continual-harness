# Validated RAM/ROM addresses — vanilla US Pokémon Emerald (this build)

The address table behind `collection/extractors/`. **Nothing here is trusted from symbol lists**:
every entry was validated empirically against recorded frames (Phase 0/1 of the A′ plan —
`validate_phase1*.py`, visual overlays, on-screen text/battle readbacks), after two of three
legacy harness addresses proved stale for this build. All reads go through
`extractors.ram.GBAState` (works identically over recorded blobs and the live emulator).

## Overworld / entities (`extractors/entities.py`)

| symbol | address | layout / notes |
|---|---|---|
| `gObjectEvents` | `0x02037230` | 16 × **0x24** bytes. `+0x04` spriteId · `+0x05` graphicsId · `+0x06` movementType · `+0x08` localId (**0xFF = player**) · `+0x0C/+0x10/+0x14` Coords16 initial/current/previous, **map coords +7** (subtract MAP_OFFSET; saveblock player pos has NO +7) · `+0x18` facing nibble (lo) \| moving nibble (hi), 1=DOWN 2=UP 3=LEFT 4=RIGHT. No usable `active` bit — validity is structural |
| `gSaveBlock1Ptr` | `0x03005D8C` | → SaveBlock1 (DMA-shifted): player x `+0x00` u16, y `+0x02` u16, facing byte `+0x04` (0=S 1=N 2=W 3=E — goes STALE; use the object record's nibble) |
| `gSprites` | `0x02020630` | 64 × 0x44; `pos1` x/y s16 at `+0x20/+0x22` in world px |
| camera X / Y | `0x03005DEC` / `0x03005DE8` | s16; **screen anchor = pos1 + cam**; player anchor ≡ (120, 112) every frame; OAM top-left of a 16×32 sprite = anchor + (−8, −56) |

## Terrain / map (`extractors/terrain.py`, `extractors/tilesets.py`)

| symbol | address | notes |
|---|---|---|
| `gBackupMapLayout` | `0x03005DC0` | pinned by cross-frame consensus scan (38/38 overworld frames); buffer is (map_w+15) × (map_h+14) u16 metatiles |
| map bank / number | `0x020322E4` / `0x020322E5` | source of the semantic map names (validated by data) |
| `gMapGroups` (ROM) | `0x08486578` | group ptr → header ptr → layout → tilesets; pure-ROM walk (the RAM gMapHeader walk found 0 maps); 43/43 maps resolved, 16 tilesets |

## Text (`extractors/text.py`)

| symbol | address | notes |
|---|---|---|
| `gStringVar4` | `0x02021FC4` | composed message buffer; **STALE during battles** — never use in battle |
| `gDisplayedStringBattle` | `0x02022E2C` | the battle UI's own message buffer (use when `in_battle`) |
| `sTextPrinters` | `0x020201B0` | 0x24/slot; currentChar ptr `+0x00`, active `+0x1B` — the typewriter reveal |
| `sTempTextPrinter` | `0x0202018C` | last-added printer's ORIGINAL string start |

## Battle (`extractors/battle.py`)

| symbol | address | notes |
|---|---|---|
| `gBattleMons` | `0x02024084` | BattlePokemon[4] × 0x58: species `+0x00` u16 · moves `+0x0C` u16[4] · pp `+0x24` u8[4] · hp `+0x28` u16 · level `+0x2A` u8 · maxHP `+0x2C` u16 · status1 `+0x4C` u32 — verified field-by-field against an on-screen "MUDKIP Lv10 31/33" |
| `gBattleTypeFlags` | `0x02022AAE` | u32 wild/trainer bitflags |
| `gBattleCommunication` | `0x02024A60` | 8-byte battle phase machine (recorded raw) |
| `gMain.inBattle` | `0x030026F9` & `0x02` | the Phase-0-validated battle gate (sits at gMain+0x439, consistent with cb2) |

## Mode / UI (`extractors/ui.py`, `audits/ram_fields.py`)

| symbol | address | notes |
|---|---|---|
| `gMain.callback2` | `0x030022C4` | the game's own screen id; mode labels come from the Phase-0 empirical cb2 taxonomy (contact sheets), e.g. overworld = `0x8085E5D` |
| `sGlobalScriptContext` | `0x02037A58` | first bytes ≠ 0 while a script/dialogue runs |
| BLDCNT / BLDALPHA / BLDY | io `0x04000050/52/54` | the GBA's entire fade/flash vocabulary |
| window mask | BG0 tilemap (vram) | UI geometry = BG0 occupancy vs dominant filler → (20,30) cell mask |

## Known-bad (do NOT use)

| thing | why |
|---|---|
| `DIALOG_STATE` `0x020370B8` | DEAD on this build — constant 0x3FF (legacy harness address) |
| legacy object layout (stride 68, gfx `+0x03`) | wrong struct — produced the garbage v1-era `semantic.objects`; survives only as `world_model_sink.extract_objects_v1_legacy` for v1-demo consistency |
| `semantic.jsonl` `facing` (pre-2026-06 runs) | collector input-tracker; pixel-adjudicated wrong after warps/forced turns |
| saveblock facing byte | goes stale; use the player object record's facing nibble |
