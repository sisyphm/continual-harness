"""Continuous full-playthrough collection (E2E corpus spine).

One emulator session, no savestate loads: iterate MILESTONE_ORDER, running each
Heatz policy until its postcondition, handing off to the next in the SAME session.
The per-milestone action loop reuses collect_events' predicates verbatim.
"""
