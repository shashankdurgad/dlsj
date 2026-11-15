"""Residual gate shared by the sim and the deployed controller: DICE-RL (residual + best-of-K) runs only while the
prompt asks to pick up an object. Deterministic text filter, no learned phase detector.

    from sim.twin.rl.gate import is_pick_prompt, Fader
    is_pick_prompt("pick up the brick and lift it up")   # True
    is_pick_prompt("put the brick in the red bowl")      # False
"""
from __future__ import annotations

import re

PICK_VERBS = ("pick up", "pick", "grasp", "lift", "grab")
OBJECT_WORDS = ("brick", "block", "object", "lego", "duplo", "piece", "cube", "it")
_VERB = "|".join(re.escape(v) for v in sorted(PICK_VERBS, key=len, reverse=True))
_OBJ = "|".join(re.escape(o) for o in OBJECT_WORDS)
# a pick verb, then (within a few words) an object word; "it" only right after the verb ("lift it up")
_RE = re.compile(rf"\b(?:{_VERB})\b(?:\s+(?:up|the|a|an|one|that|this|\w+-?\w*)){{0,3}}?\s+(?:{_OBJ})s?\b", re.I)
_NOT = re.compile(r"^\s*(?:put|place|bring|drop|release|move|present|hold)\b", re.I)


def is_pick_prompt(prompt: str | None) -> bool:
    """True when the prompt starts a pick (verb + object). Prompts that START with put / place / bring / drop / ...
    are never pick prompts even if they mention lifting later."""
    if not prompt:
        return False
    p = " ".join(str(prompt).split())
    if _NOT.match(p):
        return False
    return bool(_RE.search(p))


class Fader:
    """Residual gate with a linear fade-out (no command jump when the prompt switches away from picking).
    scale() is 1 while gated on, then falls to 0 over `ticks` control ticks; re-arms when a pick prompt returns."""

    def __init__(self, ticks: int = 6):
        self.ticks, self.k = max(1, ticks), 0
        self.on = False

    def update(self, prompt: str | None) -> float:
        on = is_pick_prompt(prompt)
        if on:
            self.k = self.ticks
        elif self.k > 0:
            self.k -= 1
        self.on = on
        return self.k / self.ticks
