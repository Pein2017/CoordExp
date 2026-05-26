from __future__ import annotations

from src.trainers.rollout_runtime.swift_coord_row_patch import (
    apply_coord_row_patch_for_rollout_server,
)


def main() -> None:
    apply_coord_row_patch_for_rollout_server()

    from swift.pipelines import rollout_main

    rollout_main()


if __name__ == "__main__":
    main()
