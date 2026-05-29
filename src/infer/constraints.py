"""Public decode-constraint facade for shared inference runtime callers."""

from __future__ import annotations

from importlib import import_module
import sys

STOP_PRESSURE_MODE_MIN_NEW_TOKENS_AFTER_OBJECT_OPEN = (
    "min_new_tokens_after_object_open"
)
STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_OPEN = "raw_text_object_open"
STOP_PRESSURE_MODE_SUPPRESS_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY = (
    "suppress_terminating_tokens_after_object_boundary"
)
STOP_PRESSURE_MODE_SUPPRESS_SPECIAL_TERMINATING_TOKENS_AFTER_OBJECT_BOUNDARY = (
    "suppress_special_terminating_tokens_after_object_boundary"
)
STOP_PRESSURE_MODE_SUPPRESS_FIRST_STRUCTURAL_CLOSURE_AFTER_OBJECT_BOUNDARY = (
    "suppress_first_structural_closure_after_object_boundary"
)
STOP_PRESSURE_MODE_STEER_FIRST_ARRAY_BRANCH_TO_NEXT_OBJECT_AFTER_OBJECT_BOUNDARY = (
    "steer_first_array_branch_to_next_object_after_object_boundary"
)
STOP_PRESSURE_MODE_STEER_BBOX_TAIL_CLOSURE_TO_NEXT_OBJECT = (
    "steer_bbox_tail_closure_to_next_object"
)
STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN = (
    "steer_bbox_tail_then_object_open"
)
STOP_PRESSURE_MODE_STEER_BBOX_TAIL_THEN_OBJECT_OPEN_ONCE = (
    "steer_bbox_tail_then_object_open_once"
)
STOP_PRESSURE_TRIGGER_RULE_RAW_TEXT_OBJECT_BOUNDARY = "raw_text_object_boundary"

_EXPORT_MODULES = {
    "CompactFullGrammarIds": "src.infer._constraints_impl",
    "CompactFullGrammarLogitsProcessor": "src.infer._constraints_impl",
    "build_compact_full_grammar_logits_processor": "src.infer._constraints_impl",
    "build_compact_grammar_logits_processor": "src.infer._constraints_impl",
    "build_array_branch_continuation_steering_logits_processor": (
        "src.infer._constraints_impl"
    ),
    "build_bbox_tail_closure_steering_logits_processor": "src.infer._constraints_impl",
    "build_bbox_tail_then_object_open_once_steering_logits_processor": (
        "src.infer._constraints_impl"
    ),
    "build_bbox_tail_then_object_open_steering_logits_processor": (
        "src.infer._constraints_impl"
    ),
    "build_terminating_token_suppression_logits_processor": "src.infer._constraints_impl",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    module_name = _EXPORT_MODULES[name]
    try:
        value = getattr(import_module(module_name), name)
    except ModuleNotFoundError as exc:
        _clear_failed_constraint_import(module_name)
        raise ModuleNotFoundError(
            f"{name} requires optional decode constraint dependencies "
            f"while importing {module_name}: {exc}"
        ) from exc
    globals()[name] = value
    return value


def _clear_failed_constraint_import(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    for optional_root in ("torch", "transformers"):
        for loaded_name in tuple(sys.modules):
            if loaded_name == optional_root or loaded_name.startswith(optional_root + "."):
                sys.modules.pop(loaded_name, None)


__all__ = [
    "CompactFullGrammarIds",
    "CompactFullGrammarLogitsProcessor",
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
    "build_array_branch_continuation_steering_logits_processor",
    "build_bbox_tail_closure_steering_logits_processor",
    "build_bbox_tail_then_object_open_once_steering_logits_processor",
    "build_bbox_tail_then_object_open_steering_logits_processor",
    "build_compact_full_grammar_logits_processor",
    "build_compact_grammar_logits_processor",
    "build_terminating_token_suppression_logits_processor",
]
