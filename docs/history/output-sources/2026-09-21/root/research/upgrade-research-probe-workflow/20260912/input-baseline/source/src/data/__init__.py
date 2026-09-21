"""Data loading and raw-example validation."""

from src.data.examples import ImageRef, RawExample, RawObject, SourceProvenance
from src.data.jsonl import iter_raw_examples, load_raw_examples

__all__ = [
    "ImageRef",
    "RawExample",
    "RawObject",
    "SourceProvenance",
    "iter_raw_examples",
    "load_raw_examples",
]
