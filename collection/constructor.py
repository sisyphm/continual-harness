"""Data-closure Phase C: the STATE CONSTRUCTOR — build game states instead of playing to them.

Coverage closure needs states gameplay rarely reaches (every species at every level on the player
side, full Poké Ball pockets for catches, level-15 mons for evolutions). Playing there is hours;
CONSTRUCTING the state is milliseconds: load a checkpoint, edit party/bag in emulated RAM, verify,
snapshot to the savestate bank. The ROM renders whatever state we set — for a renderer-rung model,
GT is GT regardless of how the state arose.

Self-validating by design (no symbol trusted untested):
  party    gPlayerParty decode must produce checksum-VALID mons (1/65536 by chance per mon);
  stats    our base-stats table + stat formula + nature table must reproduce the live party
           lead's STORED stats exactly before we are allowed to write anything;
  exp      the live lead's exp must lie inside its level bracket of our scanned exp tables;
  names    the species-name table is pinned by decode(entry[283]) == "MUDKIP";
  bag      the security key must decode the live money and every nonzero bag slot to sane values;
  after    edits are re-read after emulated frames (checksum still valid, species intact, no
           bad-egg flag) before the state is saved.

Usage (one constructed state per invocation; the bank builder loops a spec over this):
  .venv/bin/python -m collection.constructor \
      --base ../pokemon-worldmodel/data/storyline_wm/RIVAL_HOUSE/attempt_000001/final.state \
      --out  ../pokemon-worldmodel/data/processed/state_bank/torchic_l10 \
      --species 280 --level 10 --moves 10,45 --balls 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from itertools import permutations
from pathlib import Path

from collection.extractors.ram import GBAState
from collection.extractors.text import decode

# ---- layout constants (each is PINNED at runtime before use) -----------------------------------
G_PLAYER_PARTY = 0x020244EC            # struct Pokemon[6], 100 B each (fixed EWRAM, not DMA-shifted)
SB1_PTR, SB2_PTR = 0x03005D8C, 0x03005D90
SB2_ENCRYPTION_KEY = 0xAC              # u32 at SaveBlock2 + 0xAC (Emerald bag/money xor key)
SB1_MONEY = 0x490
POCKETS = {"items": (0x560, 30), "keyitems": (0x5D8, 30), "balls": (0x650, 16)}
ITEM_POKE_BALL, MAX_ITEM_ID = 4, 376
N_SPECIES, MUDKIP = 412, 283           # corpus-pinned: player lead of the chain save is Mudkip

# Gen-3 box substructure order: permutations of (Growth, Attacks, EVs, Misc) by personality % 24
ORDERS = list(permutations(range(4)))
NATURE_UP_DOWN = [(n // 5, n % 5) for n in range(25)]      # over (atk, def, spe, spa, spd)


# ---- party codec (pure bytes; no emulator) ------------------------------------------------------

def _xor48(data: bytes, key: int) -> bytes:
    words = struct.unpack("<12I", data)
    return struct.pack("<12I", *(w ^ key for w in words))


def decrypt_box(box: bytes) -> dict:
    """80-byte BoxPokemon -> decoded fields + the 4 decrypted 12-byte substructures (G,A,E,M)."""
    pid, otid = struct.unpack_from("<II", box, 0)
    data = _xor48(box[32:80], pid ^ otid)
    order = ORDERS[pid % 24]
    sub = {name: data[order.index(i) * 12:(order.index(i) + 1) * 12]
           for i, name in enumerate("GAEM")}
    csum = sum(struct.unpack("<24H", data)) & 0xFFFF
    g = struct.unpack_from("<HHIBB", sub["G"], 0)            # species, item, exp, ppBonus, friend
    a = struct.unpack_from("<4H4B", sub["A"], 0)
    iv32 = struct.unpack_from("<I", sub["M"], 4)[0]
    return {"pid": pid, "otid": otid, "nickname": box[8:18], "checksum_ok":
            csum == struct.unpack_from("<H", box, 28)[0], "flags": box[19],
            "species": g[0], "exp": g[2], "moves": list(a[:4]), "pp": list(a[4:8]),
            "ivs": [(iv32 >> (5 * i)) & 0x1F for i in range(6)],
            "evs": list(sub["E"][:6]), "sub": sub}


def encrypt_box(box: bytes, sub: dict) -> bytes:
    """Re-pack edited substructures into the box (order + xor by pid, checksum over plaintext)."""
    pid, otid = struct.unpack_from("<II", box, 0)
    order = ORDERS[pid % 24]
    plain = b"".join(sub["GAEM"[i]] for i in order)
    csum = sum(struct.unpack("<24H", plain)) & 0xFFFF
    out = bytearray(box)
    struct.pack_into("<H", out, 28, csum)
    out[32:80] = _xor48(plain, pid ^ otid)
    return bytes(out)


def compute_stats(base: list[int], level: int, ivs: list[int], evs: list[int], pid: int) -> list[int]:
    """Gen-3 stat formula, stat order (hp, atk, def, spe, spa, spd) as stored in the party struct."""
    up, down = NATURE_UP_DOWN[pid % 25]
    out = [((2 * base[0] + ivs[0] + evs[0] // 4) * level) // 100 + level + 10]
    for i in range(1, 6):
        s = ((2 * base[i] + ivs[i] + evs[i] // 4) * level) // 100 + 5
        if i - 1 == up != down:
            s = s * 110 // 100
        elif i - 1 == down != up:
            s = s * 90 // 100
        out.append(s)
    return out


# ---- ROM tables (scanned structurally, pinned against the live save) ----------------------------

class RomTables:
    def __init__(self, rom: bytes):
        self.rom = rom
        self.base_stats = self._find_base_stats()
        self.exp_tables = self._find_exp_tables()
        self.species_names = self._find_species_names()

    def _u(self, off, n):
        return int.from_bytes(self.rom[off:off + n], "little")

    def _plausible_base(self, off: int) -> bool:
        s = self.rom[off:off + 28]
        return (all(1 <= x <= 255 for x in s[0:6]) and s[6] < 18 and s[7] < 18
                and s[19] < 6 and 0 < s[20] < 16 and 0 < s[21] < 16)

    def _find_base_stats(self) -> int:
        best, best_run = 0, 0
        off = 0
        while off < len(self.rom) - 28 * 50:
            if self._plausible_base(off):
                run = 1
                while self._plausible_base(off + run * 28):
                    run += 1
                if run > best_run:
                    best, best_run = off, run
                off += run * 28
            else:
                off += 4
        assert best_run >= N_SPECIES - 2, f"base-stats scan failed (run {best_run})"
        return best - 28                                     # entry 0 (??????) precedes the run

    def _find_exp_tables(self) -> int:
        """6+ contiguous monotonic u32[101] curves, each starting 0 and ending <= 2,560,000."""
        off = 0
        while off < len(self.rom) - 4 * 101 * 6:
            ok = True
            for t in range(6):
                base = off + t * 404
                if self._u(base, 4) != 0 or not (1 <= self._u(base + 4, 4) <= 16):
                    ok = False; break
                prev = 0
                for lv in range(1, 101):
                    v = self._u(base + lv * 4, 4)
                    if v < prev or v > 2_560_000:
                        ok = False; break
                    prev = v
                if not ok:
                    break
            if ok:
                return off
            off += 4
        raise AssertionError("exp-tables scan failed")

    def _find_species_names(self) -> int:
        """gSpeciesNames: 11-byte entries; pinned by decode(entry[MUDKIP]) == 'MUDKIP'."""
        enc = bytes(0xBB + ord(c) - ord("A") for c in "MUDKIP") + b"\xff"   # gen-3 caps + terminator
        pos = -1
        while True:
            pos = self.rom.find(enc, pos + 1)
            if pos < 0:
                raise AssertionError("species-names scan failed")
            base = pos - MUDKIP * 11
            if base >= 0 and decode(self.rom[base + MUDKIP * 11:][:11]) == "MUDKIP" \
                    and decode(self.rom[base + 11:base + 22]).isupper():    # entry 1 is a name too
                return base

    def base(self, species: int) -> list[int]:
        return list(self.rom[self.base_stats + species * 28:][:6])

    def growth(self, species: int) -> int:
        return self.rom[self.base_stats + species * 28 + 19]

    def exp_at(self, species: int, level: int) -> int:
        return self._u(self.exp_tables + self.growth(species) * 404 + level * 4, 4)

    def name(self, species: int) -> bytes:
        raw = self.rom[self.species_names + species * 11:][:11]
        return raw[:raw.index(0xFF) + 1] if 0xFF in raw else raw


# ---- the constructor -----------------------------------------------------------------------------

class StateConstructor:
    """Load a base savestate, edit party/bag in emulated RAM, validate, snapshot to the bank.
    One instance constructs MANY states: `load_base()` re-arms it (the bank builder loops it)."""

    def __init__(self, rom_path: str, base_state: str | None = None):
        from mgba._pylib import ffi
        from pokemon_env.emulator import EmeraldEmulator
        self.env = EmeraldEmulator(rom_path=rom_path)
        self.env.initialize()
        self.tables = RomTables(Path(rom_path).read_bytes())
        mem_core = self.env.core.memory.u8._core
        size = ffi.new("size_t *")
        ptr = ffi.cast("uint8_t *", mem_core.getMemoryBlock(mem_core, 0x2, size))
        self._ewram = ffi.buffer(ptr, size[0])               # LIVE view: writes hit emulated RAM
        self.report: dict = {"pins": {}, "edits": []}
        if base_state:
            self.load_base(base_state)

    def load_base(self, base_state: str) -> None:
        """Arm on a base savestate: load, settle, re-run every pin (each base re-proves the layout)."""
        self.env.load_state(base_state)
        self.env.run_frame_with_buttons([])
        self.report = {"pins": {}, "edits": [], "base": str(base_state)}
        self._pin_everything()

    # -- raw access (reads via the fresh snapshot seam; writes via the live buffer)
    def _st(self) -> GBAState:
        return GBAState.snapshot(self.env)

    def _write_ewram(self, addr: int, data: bytes) -> None:
        off = addr - 0x02000000
        self._ewram[off:off + len(data)] = data

    # -- pins: refuse to construct unless every assumption reproduces the live save exactly
    def _pin_everything(self) -> None:
        st = self._st()
        mon = st.bytes(G_PLAYER_PARTY, 100)
        d = decrypt_box(mon[:80])
        assert d["checksum_ok"] and 0 < d["species"] < N_SPECIES, "party anchor failed"
        level = mon[84]
        stored = list(struct.unpack_from("<6H", mon, 88))    # maxHP, atk, def, spe, spa, spd
        calc = compute_stats(self.tables.base(d["species"]), level, d["ivs"], d["evs"], d["pid"])
        assert calc == stored, f"stat-formula pin failed: {calc} != {stored}"
        lo = self.tables.exp_at(d["species"], level)
        hi = self.tables.exp_at(d["species"], min(level + 1, 100))
        assert lo <= d["exp"] <= hi or level == 100, "exp-table pin failed"
        key = self._key()
        money = st.u32(st.u32(SB1_PTR) + SB1_MONEY) ^ key
        assert money <= 999_999, "security-key pin failed (money)"
        for name, (off, slots) in POCKETS.items():
            for i in range(slots):
                a = st.u32(SB1_PTR) + off + i * 4
                iid, q = st.u16(a), st.u16(a + 2) ^ (key & 0xFFFF)
                assert iid == 0 or (iid <= MAX_ITEM_ID and q <= 999), f"bag pin failed ({name})"
        self.report["pins"] = {"lead_species": d["species"], "lead_level": level,
                               "stats_formula": "exact", "money": money}

    def _key(self) -> int:
        st = self._st()
        return st.u32(st.u32(SB2_PTR) + SB2_ENCRYPTION_KEY)

    # -- edits
    def set_party_slot(self, slot: int, species: int, level: int,
                       moves: list[int] | None = None, near_levelup: bool = False) -> None:
        st = self._st()
        addr = G_PLAYER_PARTY + slot * 100
        mon = bytearray(st.bytes(addr, 100))
        d = decrypt_box(bytes(mon[:80]))
        assert d["checksum_ok"], f"slot {slot} is not a valid mon (construct onto the lead's clone)"
        sub = dict(d["sub"])
        g = bytearray(sub["G"])
        struct.pack_into("<H", g, 0, species)
        exp = (max(self.tables.exp_at(species, min(level + 1, 100)) - 20,    # one fight away from
                   self.tables.exp_at(species, level))                       # the level-up (evolve
               if near_levelup else self.tables.exp_at(species, level))      # states) or bracket start
        struct.pack_into("<I", g, 4, exp)
        sub["G"] = bytes(g)
        if moves:
            a = bytearray(sub["A"])
            for i in range(4):
                struct.pack_into("<H", a, i * 2, moves[i] if i < len(moves) else 0)
                a[8 + i] = 30 if i < len(moves) else 0       # generous pp
            sub["A"] = bytes(a)
        mon[:80] = encrypt_box(bytes(mon[:80]), sub)
        mon[8:18] = (self.tables.name(species) + b"\xff" * 10)[:10]      # nickname = species name
        mon[84] = level
        stats = compute_stats(self.tables.base(species), level, d["ivs"], d["evs"], d["pid"])
        struct.pack_into("<6H", mon, 88, *stats)
        struct.pack_into("<H", mon, 86, stats[0])            # current HP = maxHP
        self._write_ewram(addr, bytes(mon))
        self.report["edits"].append({"slot": slot, "species": species, "level": level,
                                     "moves": moves, "stats": stats})

    def give_item(self, item_id: int, qty: int, pocket: str = "balls") -> None:
        st = self._st()
        key16 = self._key() & 0xFFFF
        off, slots = POCKETS[pocket]
        base = st.u32(SB1_PTR) + off
        for i in range(slots):
            a = base + i * 4
            iid = st.u16(a)
            if iid in (0, item_id):
                self._write_ewram(a, struct.pack("<HH", item_id, qty ^ key16))
                self.report["edits"].append({"item": item_id, "qty": qty, "pocket": pocket})
                return
        raise AssertionError(f"no free slot in {pocket}")

    # -- verify + snapshot
    def finalize(self, out_dir: str | Path) -> Path:
        for _ in range(4):                                   # let the game run over the edits
            self.env.run_frame_with_buttons([])
        st = self._st()
        for e in self.report["edits"]:
            if "slot" in e:
                mon = st.bytes(G_PLAYER_PARTY + e["slot"] * 100, 100)
                d = decrypt_box(mon[:80])
                assert d["checksum_ok"] and d["species"] == e["species"], "post-edit decode failed"
                # flags byte: bit0 = isBadEgg (must be clear), bit1 = hasSpecies (must be set)
                assert not (d["flags"] & 0x01), "game flagged the mon as a BAD EGG"
                assert d["flags"] & 0x02, "hasSpecies flag lost"
                e["verified"] = True
            else:
                key16 = self._key() & 0xFFFF
                off, slots = POCKETS[e["pocket"]]
                base = st.u32(SB1_PTR) + off
                got = {st.u16(base + i * 4): st.u16(base + i * 4 + 2) ^ key16 for i in range(slots)}
                assert got.get(e["item"]) == e["qty"], "post-edit bag readback failed"
                e["verified"] = True
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        state_path = out / "constructed.state"
        self.env.save_state(str(state_path))
        self.report["sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()[:16]
        (out / "manifest.json").write_text(json.dumps(self.report, indent=1))
        return state_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--base", required=True, help="base savestate (a chain checkpoint final.state)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--species", type=int, default=None)
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--moves", default=None, help="comma-separated move ids")
    ap.add_argument("--balls", type=int, default=0)
    args = ap.parse_args()

    c = StateConstructor(args.rom, args.base)
    c.report["base"] = args.base
    if args.species is not None:
        moves = [int(m) for m in args.moves.split(",")] if args.moves else None
        c.set_party_slot(args.slot, args.species, args.level, moves)
    if args.balls:
        c.give_item(ITEM_POKE_BALL, args.balls, "balls")
    p = c.finalize(args.out)
    print(json.dumps(c.report, indent=1))
    print(f"constructed state -> {p}")


if __name__ == "__main__":
    main()
