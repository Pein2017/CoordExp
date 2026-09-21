"""CPU-only compositor and scorer-contract checks."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from .compositor import compose
from .reduce import selfcheck as reduce_selfcheck


def main() -> None:
    reduce_selfcheck()
    with tempfile.TemporaryDirectory(prefix="visual-binding-selfcheck-") as directory:
        root = Path(directory)
        source = root / "source.png"
        image = Image.new("RGB", (40, 40), (10, 20, 30))
        for x in range(8, 16):
            for y in range(8, 16):
                image.putpixel((x, y), (200, 100, 50))
        image.save(source)
        receipt = compose(source, {"pixel_xyxy": [8, 8, 16, 16]}, root / "out", name="ablate_A")
        assert receipt["changed_pixel_count"] == 64
        assert receipt["unchanged_complement_pixel_count"] == 1536
    print("visual_instance_binding compositor selfcheck: PASS")


if __name__ == "__main__":
    main()
