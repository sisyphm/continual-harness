"""Corpus-wide invariant: collected Pokemon keep their species name.

Emerald offers a nickname whenever you receive or catch a Pokemon. Every A-mashing
path in the collector (policy dialog-confirm, the walk-and-talk NPC recovery, the
dialog-release and unstick loops) answers that offer with "yes" and then types into
the name keyboard, which is how every v2 run ended up with a party of AAAAAAAAAA.

The keyboard itself is a dead end: measured from a savestate captured on it, B,
START and SELECT are all inert, and the only exit is typing until the field fills
and auto-confirms. So the offer has to be refused one action earlier, at the yes/no
box, where RAM names it exactly:

  * task 0x080E215D is active -- found by diffing the active-task set across a whole
    BIRCH_LAB_VISITED run: present in exactly one sample, the action before the
    keyboard opens, absent from the 350 ordinary-dialog actions before it;
  * and the dialog text contains "nickname".

The text term is what scopes it. The task alone would also match other yes/no boxes
(healing, marts, the PC), and declining those would break real progress; and the
player's own name uses this same keyboard but must be answered, so it must never
trip this guard. Only a genuine nickname offer says "nickname".
"""

from __future__ import annotations

from typing import Any

_NICKNAME_YESNO_TASK = 0x080E215D


def nickname_prompt_open(env: Any) -> bool:
    """True only while the "give a nickname?" yes/no box is up."""
    try:
        from collection.extractors.ledger_panel import _active_task_funcs
        from collection.extractors.ram import GBAState
        # Cheap RAM check first: this runs before every action, and the dialog-text
        # read is far too expensive to pay on actions that cannot be the prompt.
        if _NICKNAME_YESNO_TASK not in set(_active_task_funcs(GBAState(env=env))):
            return False
        from collection.heatz_adapter import _read_dialog_text
        return "nickname" in (_read_dialog_text(env) or "").lower()
    except Exception:
        return False
