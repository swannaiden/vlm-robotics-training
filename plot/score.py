"""Forward kinematics and answer scoring for the Panda probe (copied from the harness that ran it)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np



# Modified (Craig) convention: T_i = RotX(alpha_{i-1}) TransX(a_{i-1}) RotZ(theta_i) TransZ(d_i)
MODIFIED = "modified"
# Standard convention: T_i = RotZ(theta_i) TransZ(d_i) TransX(a_i) RotX(alpha_i)
STANDARD = "standard"


@dataclass(frozen=True)
class Link:
    """One DH row. Rows without joint limits are fixed transforms, such as a flange."""

    a: float
    d: float
    alpha: float
    offset: float = 0.0
    lower: float | None = None
    upper: float | None = None

    @property
    def is_joint(self) -> bool:
        return self.lower is not None


@dataclass(frozen=True)
class Arm:
    key: str
    name: str
    convention: str
    links: tuple[Link, ...]
    # Shown instead of the DH table in the name-only condition.
    name_hint: str = ""

    @property
    def joints(self) -> tuple[Link, ...]:
        return tuple(link for link in self.links if link.is_joint)

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Arm:
        return cls(**{**data, "links": tuple(Link(**link) for link in data["links"])})


def link_transform(convention: str, a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    if convention == MODIFIED:
        return np.array([
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ])
    if convention == STANDARD:
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ])
    raise ValueError(f"unknown DH convention {convention!r}")


def fk(arm: Arm, q) -> np.ndarray:
    """Pose of the end-effector frame (the last DH row) in the base frame, as a 4x4 matrix."""
    q = np.asarray(q, dtype=float)
    if q.shape != (arm.dof,):
        raise ValueError(f"{arm.key} expects {arm.dof} joint angles, got shape {q.shape}")
    T = np.eye(4)
    i = 0
    for link in arm.links:
        theta = link.offset
        if link.is_joint:
            theta += q[i]
            i += 1
        T = T @ link_transform(arm.convention, link.a, link.d, link.alpha, theta)
    return T


def fk_position(arm: Arm, q) -> np.ndarray:
    return fk(arm, q)[:3, 3]


def within_limits(arm: Arm, q, tol: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return (q >= arm.lower - tol) & (q <= arm.upper + tol)


def wrap_angle(x):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_answer(content) -> dict | None:
    if not isinstance(content, str):
        return None
    text = _FENCE.sub("", content.strip())
    for candidate in (text, *re.findall(r"\{[^{}]*\}", text)[::-1]):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _floats(values, n: int) -> np.ndarray | None:
    try:
        arr = np.array([float(v) for v in values])
    except (TypeError, ValueError):
        return None
    return arr if arr.shape == (n,) and np.isfinite(arr).all() else None


def extract_position(obj: dict | None) -> np.ndarray | None:
    if obj is None:
        return None
    if all(k in obj for k in "xyz"):
        return _floats([obj[k] for k in "xyz"], 3)
    if isinstance(obj.get("position"), list):
        return _floats(obj["position"], 3)
    return None


def extract_choice(obj: dict | None) -> str | None:
    if obj is None:
        return None
    letter = str(obj.get("answer", "")).strip().upper()[:1]
    return letter if letter in "ABCD" and letter else None


def extract_joints(obj: dict | None, dof: int) -> np.ndarray | None:
    if obj is None:
        return None
    keys = [f"q{i}" for i in range(1, dof + 1)]
    if all(k in obj for k in keys):
        return _floats([obj[k] for k in keys], dof)
    if isinstance(obj.get("q"), list):
        return _floats(obj["q"], dof)
    return None


def score(task: dict, record: dict, arms: dict[str, Arm]) -> dict:
    row = {key: task[key] for key in ("task_id", "task", "base", "kind", "k", "condition", "item", "arm")}
    row.update(
        spec=record["spec"],
        model=record["model"],
        effort=record.get("effort_used"),
        label=f"{record['model']}@{record.get('effort_used')}",
        error=record.get("error"),
        finish_reason=record.get("finish_reason"),
        reasoning_tokens=record.get("reasoning_tokens"),
        tool_calls=record.get("tool_calls", False),
    )
    obj = parse_answer(record.get("content"))
    arm, twin = arms[task["arm"]], arms[task["twin"]]

    if task["task"] == "fk":
        pos = extract_position(obj)
        row["parsed"] = pos is not None
        if pos is not None:
            row["err_m"] = float(np.linalg.norm(pos - task["answer"]))
            row["err_twin_m"] = float(np.linalg.norm(pos - task["twin_answer"]))
    elif task["task"] == "fk_mc":
        choice = extract_choice(obj)
        row["parsed"] = choice is not None
        row["correct"] = choice == task["answer"]
        row["picked"] = task["option_roles"].get(choice) if choice else None
    elif task["task"] == "ik":
        q = extract_joints(obj, arm.dof)
        row["parsed"] = q is not None
        if q is not None:
            target = np.array(task["target"])
            row["err_m"] = float(np.linalg.norm(fk_position(arm, q) - target))
            row["err_twin_m"] = float(np.linalg.norm(fk_position(twin, q) - target))
            row["limit_violations"] = int((~within_limits(arm, q)).sum())
            row["dist_to_source_rad"] = float(np.linalg.norm(wrap_angle(q - np.array(task["q"]))))
            row["q_answer"] = q.tolist()
    return row


def load_arms(path) -> dict[str, Arm]:
    with open(path) as fh:
        return {key: Arm.from_dict(d) for key, d in json.load(fh).items()}
