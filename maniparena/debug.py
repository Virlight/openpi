"""Debug formatting helpers for readable action logs."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

_USE_COLOR = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None


class _C:
    RESET = "\033[0m" if _USE_COLOR else ""
    GREY = "\033[90m" if _USE_COLOR else ""
    CYAN = "\033[96m" if _USE_COLOR else ""
    YELLOW = "\033[93m" if _USE_COLOR else ""
    GREEN = "\033[92m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""


def _to_debug_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_debug_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_debug_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return value


def _format_action_vec(vals: list[Any]) -> str:
    if len(vals) != 7:
        return f"{_C.YELLOW}{json.dumps(vals, ensure_ascii=False)}{_C.RESET}"
    body = ", ".join(
        (f"{_C.GREEN}{v}{_C.RESET}" if i == 6 else f"{_C.YELLOW}{v}{_C.RESET}")
        for i, v in enumerate(vals)
    )
    return f"[{body}]"


def format_action_payload_for_debug(result: Any, max_steps: int = 4) -> str:
    """Format action payloads as one colored line per timestep."""
    if not isinstance(result, dict):
        return json.dumps(_to_debug_jsonable(result), ensure_ascii=False)

    action_keys = (
        "follow1_pos",
        "follow2_pos",
        "follow1_joints",
        "follow2_joints",
    )
    present = [k for k in action_keys if k in result]
    if not present:
        return json.dumps(_to_debug_jsonable(result), ensure_ascii=False)

    first_val = result[present[0]]
    if isinstance(first_val, np.ndarray):
        n_steps = first_val.shape[0]
    elif isinstance(first_val, (list, tuple)):
        n_steps = len(first_val)
    else:
        payload = {k: _to_debug_jsonable(result[k]) for k in present}
        return json.dumps(payload, ensure_ascii=False)

    lines = []
    for t in range(min(n_steps, max_steps)):
        parts = []
        for key in present:
            value = result[key]
            if isinstance(value, np.ndarray):
                step_val = np.round(value[t], 4).tolist()
            elif isinstance(value, (list, tuple)):
                step_val = _to_debug_jsonable(value[t])
            else:
                step_val = _to_debug_jsonable(value)
            val_str = (
                _format_action_vec(step_val)
                if isinstance(step_val, list) and len(step_val) == 7
                else f"{_C.YELLOW}{json.dumps(step_val, ensure_ascii=False)}{_C.RESET}"
            )
            parts.append(f"{_C.CYAN}{key}{_C.RESET}{_C.DIM}={_C.RESET}{val_str}")
        prefix = f"{_C.GREY}  step {t:02d} |{_C.RESET} "
        lines.append(prefix + f"  {_C.DIM}|{_C.RESET}  ".join(parts))

    suffix = (
        f"  {_C.GREY}... ({n_steps - max_steps} more steps){_C.RESET}"
        if n_steps > max_steps
        else ""
    )
    return "\n" + "\n".join(lines) + suffix
