from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.training.supervision.context import (
    SemanticMetadata,
    freeze_semantic_metadata,
    validate_optional_semantic_string,
)
from src.training.supervision.spans import SupervisionSpan


@dataclass(frozen=True, slots=True)
class SupervisionBatch:
    """Light semantic holder for supervision spans.

    :param spans: Supervision spans included in this batch.
    :param batch_id: Optional semantic batch identifier.
    :param metadata: Optional scalar semantic metadata.
    """

    spans: Sequence[SupervisionSpan] = field(default_factory=tuple)
    batch_id: str | None = None
    metadata: SemanticMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze semantic batch contents."""

        if isinstance(self.spans, (str, bytes, Mapping)) or not isinstance(
            self.spans,
            Sequence,
        ):
            raise TypeError("supervision batch spans must be a sequence")

        spans = tuple(self.spans)
        for span in spans:
            if type(span) is not SupervisionSpan:
                raise TypeError("supervision batch spans must be SupervisionSpan entries")

        object.__setattr__(self, "spans", spans)
        object.__setattr__(
            self,
            "batch_id",
            validate_optional_semantic_string(self.batch_id, field_name="batch_id"),
        )
        object.__setattr__(self, "metadata", freeze_semantic_metadata(self.metadata))
