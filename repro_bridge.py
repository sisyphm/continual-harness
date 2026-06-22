"""Reproduce the Route-104 bridge crossing and inspect the CONDITIONING (no model needed).
Path: load ROUTE_104_NORTH, hold RIGHT then UP to cross the bridge north. Logs per-frame
map_id / player_xy / facing / mode / in_battle and saves a GT montage. Tests: transition
(map_id change at the bridge), conditioning glitch (garbage coords), layering (what terrain reads).
"""
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, os.getcwd())
from pokemon_env.emulator import EmeraldEmulator
from collection.extractors.conditions import RunConditionWriter
from collection.extractors.ram import GBAState

ROM = "Emerald-GBAdvance/rom.gba"
STATE = "../pokemon-worldmodel/data/storyline_wm/ROUTE_104_NORTH/attempt_000001/final.state"
env = EmeraldEmulator(rom_path=ROM); env.initialize(); env.load_state(STATE)
writer = RunConditionWriter(open(ROM, "rb").read())

seq = ["RIGHT"] * 110 + ["UP"] * 520           # right a few tiles, then north across the bridge
trace = []; shots = []
for i, btn in enumerate(seq):
    env.run_frame_with_buttons([btn.lower()])
    r = writer.row(GBAState.snapshot(env))
    trace.append((i, btn, tuple(int(x) for x in r["map_id"]), tuple(int(x) for x in r["player_xy"]),
                  int(r["player_facing"]), int(r["mode"]), int(r["in_battle"])))
    if i % 35 == 0:
        shots.append(np.asarray(env.get_screenshot().convert("RGB")))

# montage of the crossing
if shots:
    Image.fromarray(np.concatenate(shots, axis=1)).save("../pokemon-worldmodel/runs/repro_bridge.png")

# print sampled trace + flag map_id changes
print(f"{'frame':>5} {'btn':>5} {'map_id':>10} {'player_xy':>12} {'face':>4} {'mode':>4} {'batt':>4}")
prev = None
for (i, btn, mid, xy, fc, md, ib) in trace:
    flag = "  <-- MAP CHANGE" if (prev is not None and mid != prev) else ""
    if i % 20 == 0 or flag:
        print(f"{i:>5} {btn:>5} {str(mid):>10} {str(xy):>12} {fc:>4} {md:>4} {ib:>4}{flag}")
    prev = mid
# summary
maps = sorted(set(t[2] for t in trace))
xs = [t[3][0] for t in trace]; ys = [t[3][1] for t in trace]
print(f"\ndistinct map_ids: {maps}")
print(f"player_xy range: x[{min(xs)}..{max(xs)}] y[{min(ys)}..{max(ys)}]")
print(f"modes seen: {sorted(set(t[5] for t in trace))}  | any in_battle: {any(t[6] for t in trace)}")
print(f"map changes: {sum(1 for k in range(1,len(trace)) if trace[k][2]!=trace[k-1][2])}")
print("montage: runs/repro_bridge.png")
