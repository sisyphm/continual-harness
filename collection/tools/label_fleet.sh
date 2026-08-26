#!/bin/bash
# Label-extraction fleet: extract_ledger over every corpus-v2 run into the
# derived tree (corpus dir is frozen read-only). 3 runs in parallel x 4 workers.
set -u
CH=/root/code/proj-minhyuk-2026/continual-harness
FLEET=/root/data1/proj-minhyuk-2026/w33_fleet
OUT=/root/data1/proj-minhyuk-2026/w33v2_labels
LOG=$OUT/_fleet_logs
mkdir -p "$LOG"
cd "$CH"

RUNS=$(.venv/bin/python -c "
import json
m = json.load(open('$FLEET/CORPUS_MANIFEST.json'))
print('\n'.join(r['run_id'] for r in m['run_table']))")

extract_one() {
    run="$1"
    if [ -f "$OUT/$run/ledger_offline/index.json" ] && [ -f "$OUT/$run/.done" ]; then
        echo "[skip] $run"; return 0
    fi
    rm -rf "$OUT/$run"
    PYTHONPATH=. .venv/bin/python -m collection.extract_ledger \
        --run "$FLEET/$run" --workers 4 --out "$OUT" \
        > "$LOG/$run.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then touch "$OUT/$run/.done"; echo "[done] $run"
    else echo "[FAIL] $run rc=$rc"; fi
}
export -f extract_one
export CH FLEET OUT LOG

echo "$RUNS" | xargs -P 3 -I{} bash -c 'extract_one {}'
echo "FLEET COMPLETE: $(ls "$OUT" | grep -c '^rg_') runs, $(ls "$OUT"/*/.done 2>/dev/null | wc -l) done"
