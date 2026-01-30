"""High-confidence junk `desc` labels to drop for Visual Genome (VG).

VG object names are open-vocabulary and sometimes include deictic/pronoun
placeholders (e.g., "this") that are not meaningful supervision targets.

This file intentionally keeps the list conservative ("high confidence").
If you want more aggressive cleanup (e.g., mapping synonyms), do that in a
separate, explicitly-reviewed step to avoid deleting legitimate labels.
"""

from __future__ import annotations

import re
from typing import Iterable


_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\"'`\\(\\)\\[\\]\\{\\}<>\\s]+|[\"'`\\(\\)\\[\\]\\{\\}<>\\s]+$")


def normalize_desc_for_match(desc: str) -> str:
    """Normalize a raw desc string for exact-match filtering."""
    s = str(desc).strip().lower()
    s = _EDGE_PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# Demonstratives / pronouns / placeholders that are never a useful object label.
HIGH_CONF_JUNK_DESCS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "it's",
    "itself",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "here",
    "there",
    # Generic placeholders (kept conservative).
    "something",
    "someone",
    "somebody",
    "anything",
    "anyone",
    "anybody",
    "everything",
    "everyone",
    "everybody",
    "nothing",
    "nobody",
    "noone",
    "unknown",
    "n/a",
    "na",
    "nil",
    "none",
}


def is_high_conf_junk_desc(desc: str) -> bool:
    return normalize_desc_for_match(desc) in HIGH_CONF_JUNK_DESCS


def filter_high_conf_junk_descs(descs: Iterable[str]) -> list[str]:
    out: list[str] = []
    for d in descs:
        if not is_high_conf_junk_desc(d):
            out.append(d)
    return out
