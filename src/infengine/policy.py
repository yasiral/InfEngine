"""Policy files for ``infengine apply`` (M13 Phase 5A/5B).

A policy file is a flat mapping of operator control -> value:

    runtime_concurrency: 32
    runtime_max_scheduled_tokens: 1024
    runtime_kv_watermark: 0.1

JSON is accepted directly; YAML is parsed by a tiny dependency-free subset
parser (comments, blank lines, ``key: value``, quoted scalars, ints/floats,
booleans, null) because a policy is a flat map and PyYAML is not a dependency.
An optional top-level ``policy:``/``controls:`` wrapper is unwrapped.

Validation happens against the live capability contract BEFORE any mutation:
unknown, unsupported, out-of-bounds and duplicate/conflicting aliases all fail
closed with field-specific diagnostics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PolicyError(ValueError):
    """A policy file is unreadable or structurally invalid."""


def _scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except (TypeError, ValueError):
            pass
    return text


def _parse_minimal_yaml(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip() != line:
            raise PolicyError(
                f"line {lineno}: nested YAML is not supported in a policy file; "
                "use a flat mapping"
            )
        if ":" not in line:
            raise PolicyError(f"line {lineno}: expected 'key: value', got {raw_line!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        if not key:
            raise PolicyError(f"line {lineno}: empty key")
        if value.strip() == "":
            raise PolicyError(
                f"line {lineno}: control {key!r} has no value (flat mapping required)"
            )
        result[key] = _scalar(value)
    return result


def load_policy(path: str) -> Dict[str, Any]:
    """Load a policy file (JSON or the minimal YAML subset)."""
    p = Path(path)
    if not p.exists():
        raise PolicyError(f"policy file not found: {path}")
    try:
        # utf-8-sig strips a leading BOM (Windows editors write one) and is a
        # no-op otherwise.
        text = p.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise PolicyError(f"policy file unreadable: {exc}") from exc
    stripped = text.strip()
    if not stripped:
        raise PolicyError("policy file is empty")
    data: Any
    if p.suffix.lower() == ".json" or stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise PolicyError(f"invalid JSON policy: {exc}") from exc
    else:
        try:
            data = _parse_minimal_yaml(text)
        except PolicyError:
            raise
    if not isinstance(data, dict):
        raise PolicyError("policy must be a mapping of control -> value")
    # Unwrap an optional top-level policy/controls wrapper.
    for wrapper in ("policy", "controls", "runtime"):
        if len(data) == 1 and wrapper in data and isinstance(data[wrapper], dict):
            data = data[wrapper]
            break
    if not data:
        raise PolicyError("policy is empty")
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            raise PolicyError(
                f"control {key!r} must be a scalar value, got {type(value).__name__}"
            )
    return dict(data)


def validate_policy(
    request: Dict[str, Any], capabilities: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Validate a policy against the capability contract (client-side, read-only).

    Returns ``(resolved_intent_map, errors)``. ``resolved_intent_map`` is keyed
    by the operator intent the server accepts. Errors are field-specific.
    """
    by_intent: Dict[str, Dict[str, Any]] = {}
    by_internal: Dict[str, Dict[str, Any]] = {}
    for entry in capabilities:
        if entry.get("intent"):
            by_intent[entry["intent"]] = entry
        if entry.get("internal_key"):
            by_internal.setdefault(entry["internal_key"], entry)

    resolved: Dict[str, Any] = {}
    seen_internal: Dict[str, Tuple[str, Any]] = {}
    errors: List[Dict[str, Any]] = []
    for key, value in request.items():
        entry = by_intent.get(key) or by_internal.get(key)
        if entry is None:
            errors.append({"control": key, "error": "unknown control for this deployment"})
            continue
        if not entry.get("supported_on_selected_engine"):
            errors.append({"control": key, "error": "unsupported on the selected engine"})
            continue
        internal = entry.get("internal_key") or key
        if internal in seen_internal:
            prev_key, prev_value = seen_internal[internal]
            if prev_value != value:
                errors.append({
                    "control": key,
                    "error": (
                        f"conflicting alias for {internal!r}: {prev_key!r}={prev_value!r} "
                        f"vs {key!r}={value!r}"
                    ),
                })
            continue
        seen_internal[internal] = (key, value)
        lo, hi = entry.get("min"), entry.get("max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if lo is not None and value < lo:
                errors.append({"control": key, "error": f"{value} below minimum {lo}"})
                continue
            if hi is not None and value > hi:
                errors.append({"control": key, "error": f"{value} above maximum {hi}"})
                continue
        resolved[entry.get("intent") or key] = value
    return resolved, errors
