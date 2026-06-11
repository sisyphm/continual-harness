"""World-model data audits — the measurement campaign that gates training (Phase 0/2 of the A′ plan).

The chain, in dependency order (each is a CLI: `.venv/bin/python -m collection.audits.<name>`):

    l1          cheap-stream stats (semantic+actions+brightness): mode mass, idle gap, inputs, NPCs
    ram_fields  streaming RAM extraction per frame (cb2, in_battle, blend, species, objects) -> npz
    modes       cb2-cluster taxonomy: sample contact sheets per callback2 value, label empirically
    textbox     visual textbox detector (band rows 116-152) vs the script/printer signals
    framediff   frame-difference mass — how much pixel change the conditions must explain
    report      folds all of the above into the go/no-go acceptance report
    replay      end-to-end check: recorded PPU state re-rendered == recorded RGB (render_state)

All corpus walking goes through `collection.corpus`; all RAM parsing through
`collection.extractors` (GBAState seam) — audits measure, extractors define truth."""
