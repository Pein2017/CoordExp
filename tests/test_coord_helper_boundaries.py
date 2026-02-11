import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COORD_UTILS = PROJECT_ROOT / "src/common/geometry/coord_utils.py"
DATASET_GEOMETRY = PROJECT_ROOT / "src/datasets/geometry.py"
EVAL_PARSING = PROJECT_ROOT / "src/eval/parsing.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_coord_utils_is_import_light_and_cycle_safe():
    imports = _imports(COORD_UTILS)
    assert all(not name.startswith("src.datasets") for name in imports)
    assert all(not name.startswith("src.eval") for name in imports)


def test_dataset_and_eval_layers_do_not_import_each_other_directly():
    dataset_imports = _imports(DATASET_GEOMETRY)
    eval_imports = _imports(EVAL_PARSING)

    assert all(not name.startswith("src.eval") for name in dataset_imports)
    assert all(not name.startswith("src.datasets") for name in eval_imports)
