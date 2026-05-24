from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
import time
from typing import Any


class OptimizationStopped(RuntimeError):
    """Raised internally when the user requests a graceful optimizer stop."""


@dataclass
class OptimizationController:
    """Thread-safe pause/stop/checkpoint controller for long optimizations.

    GUI code can keep one controller per run and call pause(), resume() or stop().
    The optimizer calls check() from residual evaluations, so SciPy methods exit
    safely at the next function evaluation rather than killing the worker thread.
    """
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _pause: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    status_message: str = "running"

    def pause(self) -> None:
        with self._lock:
            self.status_message = "paused"
            self._pause.set()

    def resume(self) -> None:
        with self._lock:
            self.status_message = "running"
            self._pause.clear()

    def stop(self) -> None:
        with self._lock:
            self.status_message = "stopping"
            self._stop.set()
            self._pause.clear()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def pause_requested(self) -> bool:
        return self._pause.is_set()

    def check(self, sleep_s: float = 0.15) -> None:
        while self._pause.is_set():
            if self._stop.is_set():
                raise OptimizationStopped("Optimization stopped by user while paused.")
            time.sleep(sleep_s)
        if self._stop.is_set():
            raise OptimizationStopped("Optimization stopped by user.")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return str(path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_safe(obj: Any):
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
