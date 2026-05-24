from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import copy


@dataclass
class StateSnapshot:
    label: str
    state: Any


class UndoRedoStack:
    """Small framework-agnostic undo/redo stack.

    GUI code can push serializable state snapshots before changing bounds, report
    sections, imported data definitions or project settings. The stack intentionally
    stores deep copies, not live references.
    """
    def __init__(self, max_depth: int = 50):
        self.max_depth = max(1, int(max_depth))
        self._undo: list[StateSnapshot] = []
        self._redo: list[StateSnapshot] = []

    def push(self, label: str, state: Any) -> None:
        self._undo.append(StateSnapshot(label, copy.deepcopy(state)))
        self._undo = self._undo[-self.max_depth:]
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, current_state: Any) -> StateSnapshot | None:
        if not self._undo:
            return None
        snap = self._undo.pop()
        self._redo.append(StateSnapshot('redo', copy.deepcopy(current_state)))
        return snap

    def redo(self, current_state: Any) -> StateSnapshot | None:
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(StateSnapshot('undo', copy.deepcopy(current_state)))
        return snap

    def clear(self) -> None:
        self._undo.clear(); self._redo.clear()

    def status(self) -> dict[str, Any]:
        return {'undo_depth': len(self._undo), 'redo_depth': len(self._redo), 'max_depth': self.max_depth}
