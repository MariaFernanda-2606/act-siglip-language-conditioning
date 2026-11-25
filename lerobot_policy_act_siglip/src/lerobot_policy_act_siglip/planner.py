"""Deterministic high-level planner: task command -> ordered subtasks, advanced on the completion signal."""

from __future__ import annotations

import re

COLORS = ("red", "green", "yellow")
PHASES = ("pick", "move", "open", "place", "close")


def subtask_text(phase: str, colour: str) -> str:
    """Canonical subtask sentences; identical to the manifest fallback texts (present in the table)."""
    return {
        "pick": f"pick up the {colour} cube",
        "move": f"move the {colour} cube to the {colour} drawer",
        "open": f"open the {colour} drawer",
        "place": f"place the {colour} cube inside the {colour} drawer",
        "close": f"close the {colour} drawer",
    }[phase]


def colour_from_command(command: str) -> str:
    found = [c for c in COLORS if c in re.findall(r"[a-z]+", command.lower())]
    if len(set(found)) != 1:
        raise ValueError(f"Expected exactly one colour among {COLORS} in the command, got: {command!r}")
    return found[0]


class SubtaskPlanner:
    """Sends one subtask at a time and moves to the next only when the current one is complete.

    Completion = the policy's completion score stays >= `threshold` for `patience` consecutive
    updates. With completion_head="done" the score is P(instruction already completed); the
    defaults (0.5, 5) were the best on the 15 development episodes in the offline proxy study.
    The order is fixed: subtasks are never skipped or revisited.

    Usage:
        planner = SubtaskPlanner("pick up the green cube and place it inside the green drawer, ...")
        policy.set_instruction(planner.current)
        while not planner.done:
            action = policy.select_action(obs)
            ...
            if planner.update(policy.last_completion):
                policy.set_instruction(planner.current)
    """

    def __init__(self, command: str, threshold: float = 0.5, patience: int = 5):
        self.command = command
        self.colour = colour_from_command(command)
        self.subtasks = [subtask_text(phase, self.colour) for phase in PHASES]
        self.threshold = threshold
        self.patience = patience
        self.step = 0
        self._streak = 0
        self.done = False

    @property
    def current(self) -> str | None:
        return None if self.done else self.subtasks[self.step]

    def update(self, score: float | None) -> bool:
        """Feed the latest completion score. Returns True when the planner moved to a new subtask."""
        if self.done or score is None:
            return False
        self._streak = self._streak + 1 if float(score) >= self.threshold else 0
        if self._streak < self.patience:
            return False
        self._streak = 0
        self.step += 1
        if self.step == len(self.subtasks):
            self.done = True
            return False
        return True
