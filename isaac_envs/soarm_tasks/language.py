"""Language instructions for the fastener tasks (plain Python).

The prompt is derived from the fastener name, e.g. ``hex_bolt_M6x20`` gives
"pick up the M6 hex bolt and place it in the bin". The data collector stores it under
``soarm_common.contract.DATASET_TASK_KEY`` and the policy server receives it as the prompt.
"""

from __future__ import annotations

# [start:phrases]
KIND_PHRASES = {
    "hex_bolt": "hex bolt",
    "socket_cap": "socket head cap screw",
    "hex_nut": "hex nut",
    "washer": "washer",
}


def fastener_phrase(name: str) -> str:
    """'hex_bolt_M6x20' -> 'M6 hex bolt' (the length is left out, as a person would say it)."""
    kind, _, rest = name.rpartition("_")
    if kind not in KIND_PHRASES:
        raise ValueError(f"unknown fastener type in {name!r}; expected one of {sorted(KIND_PHRASES)}")
    size = rest.split("x")[0]
    return f"{size} {KIND_PHRASES[kind]}"


def instruction_for(name: str, target: str = "the bin") -> str:
    """'hex_bolt_M6x20' -> 'pick up the M6 hex bolt and place it in the bin'."""
    return f"pick up the {fastener_phrase(name)} and place it in {target}"
# [end:phrases]


def fastener_name_from_attrs(kind: str, size: str, length_mm: float) -> str:
    """Name of a fastener from the ``fastener:*`` attributes its USD carries."""
    if kind in ("hex_bolt", "socket_cap"):
        return f"{kind}_{size}x{length_mm:g}"
    return f"{kind}_{size}"
