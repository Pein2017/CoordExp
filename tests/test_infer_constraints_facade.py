from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


def test_constraints_facade_import_is_lightweight_without_torch() -> None:
    code = r'''
import builtins
import importlib
import unittest

real_import = builtins.__import__

def _block_optional_deps(name, globals=None, locals=None, fromlist=(), level=0):
    if name in {"torch", "transformers"}:
        raise ModuleNotFoundError(f"No module named {name!r}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _block_optional_deps
constraints = importlib.import_module("src.infer.constraints")
assert "CompactFullGrammarIds" not in vars(constraints)
with unittest.TestCase().assertRaisesRegex(
    AttributeError,
    "CompactFullGrammarIds",
):
    getattr(constraints, "CompactFullGrammarIds")
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_constraints_facade_exports_stop_pressure_only() -> None:
    import src.infer.constraints as constraints

    assert set(constraints.__all__) == {
        "STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN",
        "STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT",
        "STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN",
        "STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE",
        "STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY",
        "STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY",
        "STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY",
        "STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY",
        "STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY",
        "STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN",
        "build_terminating_token_suppression_logits_processor",
        "build_array_branch_continuation_steering_logits_processor",
        "build_bbox_tail_closure_steering_logits_processor",
        "build_bbox_tail_then_object_open_steering_logits_processor",
        "build_bbox_tail_then_object_open_once_steering_logits_processor",
    }


def test_constraints_facade_does_not_expose_compact_grammar_processors() -> None:
    import src.infer.constraints as constraints

    for name in (
        "CompactFullGrammarIds",
        "CompactFullGrammarLogitsProcessor",
        "build_compact_full_grammar_logits_processor",
        "build_compact_grammar_logits_processor",
    ):
        assert name not in constraints.__all__
        with pytest.raises(AttributeError):
            getattr(constraints, name)


def test_constraints_facade_real_env_exports_match_stop_pressure_impl() -> None:
    import src.infer.constraints as constraints
    import src.infer._constraints_impl as constraints_impl

    assert (
        constraints.build_terminating_token_suppression_logits_processor
        is constraints_impl.build_terminating_token_suppression_logits_processor
    )


def test_legacy_constraint_modules_are_removed() -> None:
    import importlib.util

    assert importlib.util.find_spec("src.infer.compact_grammar") is None
    assert importlib.util.find_spec("src.infer.stop_pressure") is None
