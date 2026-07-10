"""CLI: collect ONE continuous full playthrough (power-on -> Roxanne), no state loads.

Usage (harness conda python + libmgba on LD_LIBRARY_PATH):
  PYTHONPATH=. python -m collection.collect_playthrough \
      --policy-dir ../pokeagent-solution/expert_policies_by_llm \
      --out data/playthroughs/playthrough__mudkip_s0__smoke000 --seed 0
"""
import argparse
from collection.playthrough.director import run_playthrough


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rom-path", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--starter", default="mudkip")   # only mudkip supported today (policies hardcoded)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--stop-after", default=None, help="stop after this milestone (smoke)")
    ap.add_argument("--blocks", action="store_true", help="inject seeded life blocks (wander)")
    args = ap.parse_args()
    s = run_playthrough(policy_dir=args.policy_dir, out_dir=args.out, rom_path=args.rom_path,
                        starter=args.starter, seed=args.seed, record=not args.no_record,
                        stop_after=args.stop_after, blocks=args.blocks)
    print(f"\nDONE: {s['milestones_passed']}/{s['milestones_total']} milestones | "
          f"{s['total_frames']} frames | {s['wall_s']}s")


if __name__ == "__main__":
    main()
