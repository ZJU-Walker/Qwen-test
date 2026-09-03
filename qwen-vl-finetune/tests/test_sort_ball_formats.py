"""Pure contract test for unprompted 0827 ball support in sorting formats.

Run from ``qwen-vl-finetune`` with::

    python tests/test_sort_ball_formats.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwenvl.data.subtask_formats_sort import (  # noqa: E402
    NOT_SHOWN,
    OBJECT_NAMES,
    WHERE_TRAIN_OBJECT_NAMES,
    combo,
    is_phase_task,
    is_standalone_pick,
    key_at,
    phase_context,
    standalone_pick_answer,
    standalone_pick_qa_specs,
    where_answer,
    where_objects,
    where_question,
)


BALL_PICK = ["pick up the ball"]
BALL_COMPLETE = ["pick up the ball", "move to the left"]


def must_raise(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__}{args!r} did not raise ValueError")


def test_action_vocabulary_and_standalone_contract():
    assert "ball" in OBJECT_NAMES
    assert is_phase_task("pick up the ball")
    assert is_standalone_pick(BALL_PICK)
    assert standalone_pick_answer("phase", BALL_PICK) == "pick ball"
    assert standalone_pick_answer("object", BALL_PICK) == "ball"

    # The helper is general enough for other explicitly configured pick-only skills;
    # dataset eligibility remains a loader concern.
    assert is_standalone_pick(["pick up the cup"])

    for malformed in (
        [],
        "pick up the ball",
        ["move to the left"],
        ["pick up ball"],
        ["pick up the ball", "move to the left"],
    ):
        assert not is_standalone_pick(malformed), malformed
        must_raise(standalone_pick_answer, "phase", malformed)

    for fmt in ("target", "where", "demo", "remaining", "unknown"):
        must_raise(standalone_pick_answer, fmt, BALL_PICK)


def test_standalone_qa_is_generic_and_destination_free():
    cases = {
        "ball": ("ball", "pick ball"),
        "green block": ("green", "pick green"),
        "grey box": ("grey", "pick grey"),
    }
    for full_name, (obj, phase) in cases.items():
        specs = standalone_pick_qa_specs([f"pick up the {full_name}"])
        assert [visual for visual, _, _ in specs] == [
            "initial", "initial", "full", "full"
        ]
        assert [answer for _, _, answer in specs] == [obj, phase, obj, phase]
        joined_questions = " ".join(question for _, question, _ in specs)
        assert f"pick up the {full_name}" in joined_questions
        assert not any(
            destination in joined_questions.lower()
            for destination in ("left", "middle", "right", "tray")
        )


def test_complete_pair_apis_still_require_a_destination():
    # Standalone data must not receive a human-prompt key or a fabricated target.
    must_raise(phase_context, BALL_PICK, 0)
    must_raise(key_at, BALL_PICK, 0)
    must_raise(combo, BALL_PICK)

    # Ball remains legal if complete ball placement data is intentionally added later.
    assert phase_context(BALL_COMPLETE, 0) == ("pick", "ball", "left")
    assert phase_context(BALL_COMPLETE, 1) == ("move", "ball", "left")
    assert combo(BALL_COMPLETE) == ("ball to left",)


def test_ball_is_not_a_negative_only_where_candidate():
    legacy_prompt = ("green to left", "grey to middle", "tape to right")
    shown, absent = where_objects(legacy_prompt)
    assert shown == ["green block", "grey box", "tape"]
    assert absent == ["cup"]
    assert "ball" not in WHERE_TRAIN_OBJECT_NAMES
    assert "ball" not in absent

    # Direct lookup stays available for a future frozen-model evaluation harness.
    assert where_question("Where does {obj} go?", "ball") == "Where does the ball go?"
    assert where_answer(legacy_prompt, "ball") == NOT_SHOWN
    assert where_answer(("ball to right",), "ball") == "right"


if __name__ == "__main__":
    test_action_vocabulary_and_standalone_contract()
    test_standalone_qa_is_generic_and_destination_free()
    test_complete_pair_apis_still_require_a_destination()
    test_ball_is_not_a_negative_only_where_candidate()
    print("PASS: standalone ball format contract")
