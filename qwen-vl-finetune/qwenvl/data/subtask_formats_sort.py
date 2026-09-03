"""Sorting-task QA supervision (0824_prompting): question pools, compressed answers.

Same machinery as subtask_formats_bins.py (phrasing-selects-format, last-N phrasings
eval-held-out, compressed answers for the expert-attends-subtask serve path), for the
0824 task: three trays (left / middle / right), a FIXED pick order (green block, then
grey box, then tape), and a human demo that specifies WHERE each object goes.

Label strings (0824_prompting/*/videos/chunk-000/subtask_labels.json, already
phase-split):  `pick up the {green block,grey box,tape,cup}` / `move to the
{left,middle,right}`.  The 0827 augmentation also adds `pick up the ball` as an
explicitly unprompted, standalone low-level skill.  Triple episodes have 6
segments, complete single-object episodes 2, and standalone skills exactly 1.
Human demo labels use the SAME complete-pair strings, so the pairing key derivation
below is shared by both sides (unlike the bins task, where human labels were
legacy-form).  Standalone skills deliberately have no pairing key or combo.

Design points carried over from the bins task, and why they matter more here:

- The MOVE answer is OBJECT-FREE (`move middle`, not `move the green block to the
  middle`).  The cleaned robot corpus deliberately contains zero green-block ->
  middle trajectories: the five old single-object episodes and their five matching
  human demonstrations were physically removed before this run.  With an
  object-free move phrase the expert's conditioning for
  that held-out joint is a phrase it has already trained on thousands of times; the
  joint (green, middle) never has to appear as text at all.
- `object` and `target` factorize the same decision the `phase` format makes jointly,
  so neither can be answered by memorizing (object, destination) pairs -- the exact
  failure mode the 2xx configuration eval probes.
- `where` is new here: it asks about a NAMED object, not the current one, so the model
  must retrieve an arbitrary entry of the demo's mapping (what a composite prompt at
  test time asks for).  It is also the only pool whose QUESTIONS name objects; every
  other pool keeps object and destination words out of the question entirely, so
  nothing leaks around the demo video.
- `remaining` is the only format that cannot be answered from either the demo alone or
  the scene alone: it needs the demo's mapping AND how far the episode has progressed.

Destination words (left / middle / right) never appear in ANY question.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

# ORDER MATTERS: the final N_HELDOUT entries of each pool are reserved for
# evaluation. Append new training phrasings BEFORE the held-out tail; never
# reorder existing entries (eval comparability across runs).
N_HELDOUT = 2

# {obj} is filled with the full object name ("the tape", "the green block").
QUESTION_POOLS: Dict[str, List[str]] = {
    # Answer: "pick <object>" or "move <destination>". Drives the action expert
    # (phrasing #1 is the serve-time default question).
    "phase": [
        "What should be done now?",
        "What is the current step?",
        "State the action to perform now.",
        "What should the robot do now?",
        "Name the current step of the task.",
        "What is the next action to carry out?",
        # -- held out (eval only) --
        "At this moment, what should be done?",
        "Describe the step to carry out now.",
    ],
    # Answer: the object symbol -- "green" / "grey" / "tape" / "cup" / "ball".
    # Shared verbatim with the bins task's `block` pool (object-neutral wording).
    "object": [
        "Which object should be picked up now?",
        "Which object should you move now?",
        "Name the object to pick up next.",
        "What is the object to move now?",
        "Which object is next to be placed?",
        "State the object to pick up now.",
        # -- held out (eval only) --
        "At this moment, which object should be picked up?",
        "Identify the object that should be moved now.",
    ],
    # Answer: "left" / "middle" / "right" for the object being handled now.
    "target": [
        "Where does the current object go?",
        "Where should the object be placed now?",
        "Name the destination for the object being moved.",
        "Which tray does the current object belong in?",
        "Where should the object being moved be put?",
        "State the destination of the current object.",
        # -- held out (eval only) --
        "At this moment, where should the object go?",
        "Give the destination for the current object.",
    ],
    # Object-conditioned retrieval from the demo. Answer: "left" / "middle" /
    # "right", or NOT_SHOWN when the demo never handled that object.
    "where": [
        "Where does {obj} go?",
        "Where should {obj} be placed?",
        "Name the destination of {obj}.",
        "According to the demonstration, where does {obj} go?",
        "Which tray does {obj} belong in?",
        "State where {obj} should be put.",
        # -- held out (eval only) --
        "In the demonstration, where was {obj} placed?",
        "Give the destination of {obj}.",
    ],
    # Prompt comprehension: the answer describes the DRAWN demo clip, not the robot
    # scene -- "green to left, then grey to middle, then tape to right".
    "demo": [
        "What happened in the demonstration?",
        "What did the demonstration show?",
        "Describe what was demonstrated.",
        "Summarize the demonstration.",
        "State what the demonstration showed.",
        "What task does the demonstration show?",
        # -- held out (eval only) --
        "In the demonstration, what was done?",
        "Give a summary of the demonstrated task.",
    ],
    # Demo mapping + scene progress: the placements from the current one onward.
    "remaining": [
        "What remains to be done?",
        "What still needs to be done?",
        "List the steps still to complete.",
        "Name the remaining placements.",
        "What has not been done yet?",
        "State the rest of the task.",
        # -- held out (eval only) --
        "At this point, what still remains?",
        "Give the placements that remain.",
    ],
}

NOT_SHOWN = "not shown"

_PICK_RE = re.compile(r"^pick up the (green block|grey box|tape|cup|ball)$")
_MOVE_RE = re.compile(r"^move to the (left|middle|right)$")

# Full name (as it appears in the labels and in `where` questions) -> answer symbol.
OBJECT_SYMBOL: Dict[str, str] = {
    "green block": "green",
    "grey box": "grey",
    "tape": "tape",
    "cup": "cup",
    "ball": "ball",
}
OBJECT_NAMES: Tuple[str, ...] = tuple(OBJECT_SYMBOL)
SYMBOL_NAME: Dict[str, str] = {v: k for k, v in OBJECT_SYMBOL.items()}
DESTINATIONS: Tuple[str, ...] = ("left", "middle", "right")

# Objects eligible for *routine training-time* `where` draws.  Ball has robot
# action/QA supervision but intentionally no human prompt during training.  Including
# it here would therefore teach only the negative answer "not shown", an accidental
# channel-specific shortcut.  Explicit `where_question(..., "ball")` and
# `where_answer(..., "ball")` remain supported for frozen-checkpoint evaluation.
WHERE_TRAIN_OBJECT_NAMES: Tuple[str, ...] = (
    "green block", "grey box", "tape", "cup",
)


def is_phase_task(task: str) -> bool:
    task = task.strip()
    return bool(_PICK_RE.match(task) or _MOVE_RE.match(task))


def _standalone_pick_object(tasks: Sequence[str]) -> str:
    """Validate an exact one-segment pick-only episode and return its symbol.

    This is intentionally separate from :func:`phase_context`: a standalone skill
    has no destination, while every phase-context/prompt-pair operation requires a
    complete pick+move pair.
    """
    if isinstance(tasks, (str, bytes)) or len(tasks) != 1:
        raise ValueError(
            f"standalone pick must contain exactly one segment, got {list(tasks)!r}"
            if not isinstance(tasks, (str, bytes)) else
            f"standalone pick tasks must be a sequence, not {tasks!r}"
        )
    task = tasks[0]
    if not isinstance(task, str):
        raise ValueError(f"standalone pick task must be a string, got {task!r}")
    match = _PICK_RE.fullmatch(task.strip())
    if match is None:
        raise ValueError(
            f"standalone skill must be exactly one recognized pick segment: {list(tasks)!r}")
    return OBJECT_SYMBOL[match.group(1)]


def is_standalone_pick(tasks: Sequence[str]) -> bool:
    """Whether ``tasks`` is exactly one recognized pick segment and nothing else."""
    try:
        _standalone_pick_object(tasks)
    except (TypeError, ValueError):
        return False
    return True


def standalone_pick_answer(fmt: str, tasks: Sequence[str]) -> str:
    """Answer an unprompted standalone-pick QA sample.

    Only the destination-free scene formats are truthful for this data: ``phase``
    and ``object``.  Destination and prompt-relative formats fail loudly so a
    pick-only trajectory cannot silently acquire fabricated placement supervision.
    """
    obj = _standalone_pick_object(tasks)
    if fmt == "phase":
        return f"pick {obj}"
    if fmt == "object":
        return obj
    if fmt in ("target", "where", "demo", "remaining"):
        raise ValueError(
            f"the {fmt!r} format requires a destination or human prompt and is "
            "invalid for a standalone pick")
    raise ValueError(f"unknown sorting QA format {fmt!r}")


def standalone_pick_qa_specs(
    tasks: Sequence[str],
) -> Tuple[Tuple[str, str, str], ...]:
    """Truthful visual-QA records for a destination-free pickup episode.

    Each tuple is ``(visual_source, question, compressed_answer)``.  Initial
    questions use the first robot view; full questions use the complete sparse
    robot demonstration.  The natural-language task in question two uses the
    full object name (``green block`` / ``grey box``), while answers retain the
    compact vocabulary used by the ordinary sorting QA stream.

    No question asks where the object goes: standalone recordings contain no
    transport, destination, or release evidence.
    """
    obj = _standalone_pick_object(tasks)
    full_name = SYMBOL_NAME[obj]
    phase = f"pick {obj}"
    return (
        (
            "initial",
            "What object is available for the robot to pick up?",
            obj,
        ),
        (
            "initial",
            f"Given the task is to pick up the {full_name}, what should the "
            "robot do next?",
            phase,
        ),
        (
            "full",
            "What object did the robot pick up in this demonstration?",
            obj,
        ),
        (
            "full",
            "What pickup skill did the robot demonstrate?",
            phase,
        ),
    )


def phase_context(tasks: Sequence[str], idx: int) -> Tuple[str, str, str]:
    """(phase, object symbol, destination) for segment idx. Segments must alternate
    pick, move, ...: the factor a segment's own string omits (the pick's destination,
    the move's object) comes from its partner segment."""
    task = tasks[idx].strip()
    m = _PICK_RE.match(task)
    if m:
        if idx + 1 >= len(tasks) or not _MOVE_RE.match(tasks[idx + 1].strip()):
            raise ValueError(f"pick segment {idx} has no following move segment: {list(tasks)}")
        return "pick", OBJECT_SYMBOL[m.group(1)], _MOVE_RE.match(tasks[idx + 1].strip()).group(1)
    m = _MOVE_RE.match(task)
    if m:
        if idx == 0 or not _PICK_RE.match(tasks[idx - 1].strip()):
            raise ValueError(f"move segment {idx} has no preceding pick segment: {list(tasks)}")
        return "move", OBJECT_SYMBOL[_PICK_RE.match(tasks[idx - 1].strip()).group(1)], m.group(1)
    raise ValueError(f"not a 0824 sorting task string: {task!r}")


def key_at(tasks: Sequence[str], idx: int) -> str:
    """Human-pool pairing key for segment idx: the compressed placement of its
    pick/move pair, e.g. "green to middle". Human demo labels use the same strings,
    so a demo of that placement is a prompt for both of the robot's phases."""
    _, obj, dest = phase_context(tasks, idx)
    return f"{obj} to {dest}"


def combo(tasks: Sequence[str]) -> Tuple[str, ...]:
    """Ordered full-episode pool key: one compressed placement per pick/move pair."""
    if not tasks or not _PICK_RE.match(tasks[0].strip()):
        raise ValueError(f"episode must open with a pick segment: {list(tasks)}")
    if len(tasks) % 2:
        raise ValueError(f"episode must have an even segment count: {list(tasks)}")
    keys = tuple(key_at(tasks, i) for i in range(0, len(tasks), 2))
    # One placement per object. `where_answer` maps an object to a single destination
    # by first match, so a demo that moved the same object twice would answer with the
    # first of two truths and be queried twice as often. No episode does this today;
    # this makes a future relabel fail loudly instead of mislabeling silently.
    objects = [k.partition(" to ")[0] for k in keys]
    if len(set(objects)) != len(objects):
        raise ValueError(f"an object is placed more than once: {keys}")
    return keys


def _join(keys: Sequence[str]) -> str:
    return ", then ".join(keys)


def demo_answer(prompt_keys: Sequence[str]) -> str:
    """'demo' format answer: the DRAWN prompt clip's placements in order. prompt_keys
    is what the dataloader recorded at prompt-draw time (one key for a sub-clip
    prompt, the ordered combo for a full-episode prompt)."""
    if not prompt_keys:
        raise ValueError("the 'demo' format needs the drawn human prompt's placements")
    if isinstance(prompt_keys, str):   # a bare string would join character by character
        raise TypeError(f"prompt_keys must be a sequence of keys, got {prompt_keys!r}")
    return _join(list(prompt_keys))


def remaining_answer(tasks: Sequence[str], idx: int,
                     prompt_keys: Optional[Sequence[str]] = None) -> str:
    """'remaining' format answer: the placements from the CURRENT pair onward (the one
    in progress counts as unfinished).

    Restricted to what the DRAWN demo actually showed. A sub-clip prompt shows one
    placement, so the episode's later placements are not knowable from it -- answering
    with them would supervise a guess. The dataloader forces a full-episode prompt for
    this format, and this intersection keeps the answer honest if it ever cannot."""
    if not 0 <= idx < len(tasks):
        raise ValueError(f"segment index {idx} outside {list(tasks)}")
    keys = combo(tasks)[idx // 2:]
    if prompt_keys is not None:
        shown = set(prompt_keys)
        keys = tuple(k for k in keys if k in shown)
        if not keys:
            raise ValueError(
                f"the drawn prompt {tuple(prompt_keys)} shows none of the placements "
                f"remaining at segment {idx} of {list(tasks)}")
    return _join(keys)


def where_question(phrasing: str, object_name: str) -> str:
    if object_name not in OBJECT_SYMBOL:
        raise ValueError(f"unknown object {object_name!r} (have {OBJECT_NAMES})")
    # str.format on an already-filled question is a silent no-op, which would pair a
    # question about one object with an answer about another.
    if "{obj}" not in phrasing:
        raise ValueError(f"not a 'where' template (no {{obj}} slot): {phrasing!r}")
    return phrasing.format(obj=f"the {object_name}")


def where_answer(prompt_keys: Sequence[str], object_name: str) -> str:
    """Destination the DEMO gave for a named object, or NOT_SHOWN if the demo never
    handled it (the abstention case a composite prompt needs). Accepts the full name
    ("green block") or the answer symbol ("green")."""
    sym = OBJECT_SYMBOL.get(object_name) or (
        object_name if object_name in SYMBOL_NAME else None)
    if sym is None:
        raise ValueError(f"unknown object {object_name!r} (have {OBJECT_NAMES})")
    for key in prompt_keys:
        obj, _, dest = key.partition(" to ")
        if obj == sym:
            return dest
    return NOT_SHOWN


def where_objects(prompt_keys: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Training-time ``where`` candidates as full names, demo order first.

    The absent list intentionally excludes objects, currently ball, that have no
    human-prompt positives.  They remain available through direct calls to
    :func:`where_question` and :func:`where_answer` for explicit evaluation.
    """
    shown = [SYMBOL_NAME[k.partition(" to ")[0]] for k in prompt_keys]
    absent = [n for n in WHERE_TRAIN_OBJECT_NAMES if n not in shown]
    return shown, absent


def answer_at(fmt: str, tasks: Sequence[str], idx: int) -> str:
    """Answer for the scene-relative formats at segment idx. `where` and `demo` are
    prompt-relative and have their own entry points."""
    phase, obj, dest = phase_context(tasks, idx)
    if fmt == "phase":
        return f"pick {obj}" if phase == "pick" else f"move {dest}"
    if fmt == "object":
        return obj
    if fmt == "target":
        return dest
    if fmt in ("where", "demo", "remaining"):
        raise ValueError(f"the {fmt!r} format is prompt-relative; use its own entry point")
    raise ValueError(f"unknown sorting QA format {fmt!r}")


# --- task-module adapter (robot_data selects a module by --subtask_task) -------
# Same API as subtask_formats_bins. The difference that matters: human demo labels for
# this task are phase-split exactly like the robot's, so a sub-clip pool entry must
# span a whole pick+move PAIR -- a prompt showing only the pick half would not name a
# destination, which is the one thing the demo exists to convey.
HAS_WHERE = True
SUPPORTS_ORDER_SAMPLES = False   # the pick order is fixed; there is no order to prompt


def human_pool_entries(segs: Sequence[Dict]) -> List[Tuple[str, int, int]]:
    """(sub-clip pool key, start, end) per pick/move pair of a HUMAN demo episode."""
    tasks = [s["task"] for s in segs]
    combo(tasks)                       # validates pick-first / even segment count
    return [(key_at(tasks, i), int(segs[i]["start"]), int(segs[i + 1]["end"]))
            for i in range(0, len(segs), 2)]


def human_full_key(segs: Sequence[Dict]) -> Tuple[str, ...]:
    return combo([s["task"] for s in segs])


def robot_pool_key(tasks: Sequence[str], idx: int) -> str:
    return key_at(tasks, idx)


def robot_full_key(tasks: Sequence[str]) -> Tuple[str, ...]:
    return combo(tasks)


def train_questions(fmt: str) -> List[str]:
    return QUESTION_POOLS[fmt][:-N_HELDOUT]


def heldout_questions(fmt: str) -> List[str]:
    return QUESTION_POOLS[fmt][-N_HELDOUT:]


def parse_format_mix(spec: str) -> List[Tuple[str, float]]:
    """'phase:0.4,object:0.1,...' -> [('phase', 0.4), ...]. Weights must be positive
    and sum to 1 (+-1e-6); formats must be known; no duplicates."""
    mix = []
    for part in spec.split(","):
        name, _, w = part.strip().partition(":")
        if name not in QUESTION_POOLS:
            raise ValueError(f"unknown sorting QA format {name!r} (have {sorted(QUESTION_POOLS)})")
        weight = float(w)
        # `not (weight > 0)` also rejects NaN, which `weight <= 0` lets through -- a NaN
        # weight passes the sum check too (abs(nan - 1) > 1e-6 is False) and then makes
        # choose_format fall through to the LAST format on every draw, so a typo'd mix
        # would silently train one format at 100%.
        if not (weight > 0) or weight == float("inf"):
            raise ValueError(f"format weight must be positive and finite: {part!r}")
        mix.append((name, weight))
    total = sum(w for _, w in mix)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"format weights must sum to 1, got {total} from {spec!r}")
    if len({n for n, _ in mix}) != len(mix):
        raise ValueError(f"duplicate format in {spec!r}")
    return mix


def choose_format(mix: Sequence[Tuple[str, float]], u: float) -> str:
    """Deterministic pick from a uniform draw u in [0, 1)."""
    acc = 0.0
    for name, w in mix:
        acc += w
        if u < acc:
            return name
    return mix[-1][0]
