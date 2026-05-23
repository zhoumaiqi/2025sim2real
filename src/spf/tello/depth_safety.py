from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


SAFE = "safe"
BLOCKED = "blocked"
UNKNOWN = "unknown"


def attach_depth_safety(
    candidates: Sequence[Dict[str, Any]],
    depth_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not depth_result:
        for candidate in candidates:
            candidate["depth"] = None
            candidate["safety"] = UNKNOWN
        return {
            "available": False,
            "front_risk": False,
            "roi_median_m": None,
        }

    candidate_depth = depth_result.get("candidate_depth") or {}
    candidate_safety = depth_result.get("candidate_safety") or {}

    for candidate in candidates:
        candidate_id = str(candidate.get("id", "")).upper()
        candidate["depth"] = _coerce_optional_float(candidate_depth.get(candidate_id))
        candidate["safety"] = _normalize_safety(candidate_safety.get(candidate_id))

    center_lower_roi = depth_result.get("center_lower_roi") or {}
    return {
        "available": True,
        "front_risk": bool(depth_result.get("front_risk", False)),
        "roi_median_m": _coerce_optional_float(center_lower_roi.get("median_m")),
    }


def format_depth_prompt_hint(
    candidates: Sequence[Dict[str, Any]],
    front_risk: bool,
) -> str:
    candidate_lines = "\n".join(
        f'- {candidate["id"]}: [y, x] = {candidate["point"]}, '
        f'depth={_format_depth(candidate.get("depth"))}, '
        f'safety={candidate.get("safety", UNKNOWN)}'
        for candidate in candidates
    )
    return (
        "\nDepth safety observations:\n"
        f"{candidate_lines}\n"
        f"- front_risk={str(front_risk).lower()}\n"
        "- Prefer candidates marked safety=safe.\n"
        "- Avoid candidates marked safety=blocked unless no other direction is reasonable.\n"
        "- Candidates marked safety=unknown are allowed, but safe is preferred.\n"
    )


def replace_blocked_candidate(
    selected_candidate: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], bool, str]:
    selected = _candidate_by_id(candidates, selected_candidate.get("id")) or selected_candidate
    selected_safety = selected.get("safety", UNKNOWN)
    if selected_safety != BLOCKED:
        return selected, False, "selected_safe_or_unknown"

    replacement = _nearest_candidate_with_safety(selected, candidates, SAFE)
    if replacement is not None:
        return replacement, True, "blocked_to_safe"

    replacement = _nearest_candidate_with_safety(selected, candidates, UNKNOWN)
    if replacement is not None:
        return replacement, True, "blocked_to_unknown"

    center = _candidate_by_id(candidates, "P8")
    if center is not None and center.get("id") != selected.get("id"):
        return center, True, "blocked_to_center"

    return selected, False, "blocked_no_replacement"


def apply_front_risk_to_motion(
    adjusted_depth: float,
    z3d: float,
    front_risk: bool,
    reduce_forward_on_front_risk: bool,
    clamp_downward_on_front_risk: bool,
    min_motion_depth_m: float,
) -> Tuple[float, float, bool, bool]:
    forward_reduced = False
    downward_clamped = False

    if front_risk and reduce_forward_on_front_risk and adjusted_depth > min_motion_depth_m:
        adjusted_depth = min_motion_depth_m
        forward_reduced = True

    if front_risk and clamp_downward_on_front_risk and z3d < 0:
        z3d = 0.0
        downward_clamped = True

    return adjusted_depth, z3d, forward_reduced, downward_clamped


def summarize_failed_depth(depth_result: Optional[Dict[str, Any]]) -> str:
    if not depth_result:
        return "depth_unavailable"

    status = depth_result.get("status")
    if status:
        return str(status)

    return "depth_available"


def _candidate_by_id(
    candidates: Iterable[Dict[str, Any]],
    candidate_id: Any,
) -> Optional[Dict[str, Any]]:
    candidate_id = str(candidate_id or "").upper()
    for candidate in candidates:
        if str(candidate.get("id", "")).upper() == candidate_id:
            return candidate
    return None


def _nearest_candidate_with_safety(
    reference: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    safety: str,
) -> Optional[Dict[str, Any]]:
    ref_x, ref_y = reference.get("pixel", (0, 0))
    matching = [
        candidate
        for candidate in candidates
        if candidate.get("safety") == safety
        and candidate.get("id") != reference.get("id")
    ]
    if not matching:
        return None

    return min(
        matching,
        key=lambda candidate: (
            candidate["pixel"][0] - ref_x
        ) ** 2 + (candidate["pixel"][1] - ref_y) ** 2,
    )


def _normalize_safety(value: Any) -> str:
    value = str(value or "").strip().lower()
    if value in {SAFE, BLOCKED, UNKNOWN}:
        return value
    return UNKNOWN


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_depth(value: Any) -> str:
    depth = _coerce_optional_float(value)
    if depth is None:
        return "unknown"
    return f"{depth:.2f}m"
