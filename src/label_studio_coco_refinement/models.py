"""Label Studio runtime paths; working-data values live in working_dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.working_dataset.models import Split, _validate_split

RUNTIME_RELATIVE_ROOT = Path(
    ".local/state/label_studio_coco_refinement/rescale_32_1024_bbox_len12000"
)

SHARED_IMAGE_RELATIVE_ROOT = Path("public_data/coco/rescale_32_1024_bbox/images")

SELECTED_SOURCE_RELATIVE_ROOT = Path(
    "public_data/coco/rescale_32_1024_bbox_len12000"
)

@dataclass(frozen=True)
class RefinementRuntimeLayout:
    """Exact repository-bound one-instance/two-split mutable runtime paths."""

    repository_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_root",
            Path(self.repository_root).expanduser().resolve(),
        )

    @classmethod
    def under_repository(cls, repository_root: Path) -> "RefinementRuntimeLayout":
        return cls(Path(repository_root))

    @property
    def root(self) -> Path:
        return self.repository_root / RUNTIME_RELATIVE_ROOT

    @property
    def image_root(self) -> Path:
        return self.repository_root / SHARED_IMAGE_RELATIVE_ROOT

    @property
    def label_studio_state(self) -> Path:
        return self.root / "label-studio" / "state"

    def split_root(self, split: Split) -> Path:
        return self.root / _validate_split(split)

    def project_manifest(self, split: Split) -> Path:
        return self.split_root(split) / "project.json"

    def working_norm(self, split: Split) -> Path:
        return self.split_root(split) / "working.norm.jsonl"

    def working_coord(self, split: Split) -> Path:
        return self.split_root(split) / "working.coord.jsonl"

    def journal(self, split: Split) -> Path:
        return self.split_root(split) / "journal.jsonl"

    def images_link(self, split: Split) -> Path:
        return self.split_root(split) / "images"

    def commit_lock(self, split: Split) -> Path:
        return self.split_root(split) / ".commit.lock"

    def selected_source(self, split: Split) -> Path:
        return self.repository_root / SELECTED_SOURCE_RELATIVE_ROOT / f"{_validate_split(split)}.norm.jsonl"
