"""W33 corpus-v2: environment pinning (spec §10.3).

Savestates are emulator-build-sensitive and replay determinism depends on the exact
ROM + harness code + RTC policy. Every recorded run's manifest carries this block so
"reproducible forever" is checkable, not aspirational.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

_ROM_SHA_CACHE: dict[tuple[str, float], str] = {}


def rom_sha256(rom_path: str) -> str:
    p = Path(rom_path)
    key = (str(p.resolve()), p.stat().st_mtime)
    if key not in _ROM_SHA_CACHE:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _ROM_SHA_CACHE[key] = h.hexdigest()
    return _ROM_SHA_CACHE[key]


def harness_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def harness_dirty_state() -> tuple[bool, str | None]:
    """(dirty, diff_sha256): dirty when `git status --porcelain` is non-empty; the
    sha256 of `git diff HEAD` then pins EXACTLY which uncommitted code recorded the
    run (a bare commit hash can't identify a dirty tree)."""
    try:
        cwd = Path(__file__).resolve().parent
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd,
            capture_output=True, text=True, timeout=30)
        if status.returncode != 0 or not status.stdout.strip():
            return False, None
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=cwd, capture_output=True, timeout=60)
        return True, hashlib.sha256(diff.stdout).hexdigest()
    except Exception:
        return False, None


_ENV_HASH_CACHE: str | None = None


def env_hash() -> str:
    """sha256 over the SORTED `pip freeze` of the recording interpreter — pins the
    installed package set alongside the harness commit."""
    global _ENV_HASH_CACHE
    if _ENV_HASH_CACHE is None:
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=120).stdout
            lines = sorted(l.strip() for l in out.splitlines() if l.strip())
            _ENV_HASH_CACHE = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        except Exception:
            _ENV_HASH_CACHE = "unknown"
    return _ENV_HASH_CACHE


def mgba_version() -> str:
    try:
        import mgba

        for attr in ("__version__", "version"):
            v = getattr(mgba, attr, None)
            if v:
                return str(v)
        import mgba.core as _c  # library version string if exposed

        return str(getattr(_c, "PROJECT_VERSION", "unknown"))
    except Exception:
        return "unknown"


def collect_provenance(env, rom_path: str) -> dict:
    """The pin block. `fixed_rtc_value` must be present — DirectEmulatorRunner refuses
    to record without an installed fixed RTC (v1 lesson: wall-clock RTC broke replay
    for every family collected before 2026-06-28)."""
    dirty, diff_sha = harness_dirty_state()
    return {
        "rom_sha256": rom_sha256(rom_path),
        "harness_commit": harness_commit(),
        "harness_dirty": dirty,
        "harness_diff_sha256": diff_sha,
        "mgba_version": mgba_version(),
        "env_hash": env_hash(),
        "python": sys.version.split()[0],
        "fixed_rtc_value": getattr(env, "_pokemon_wm_fixed_rtc_value", None),
    }
