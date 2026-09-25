"""Static dependency contracts; never import or execute experimental producers."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {'route_learning', 'recurrence_dynamics', 'readout_geometry',
            'coordinate_representation', 'visual_grounding'}


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            yield node.module or ''


def test_base_never_imports_a_research_family_or_cli():
    bad = [(str(path.relative_to(ROOT)), name) for path in (ROOT/'src').rglob('*.py')
           for name in imports(path) if name.startswith(('probes.', 'scripts.'))]
    assert bad == []


def test_new_families_use_named_profiles_not_other_experiment_runners():
    bad = []
    for family in sorted(FAMILIES):
        for path in (ROOT/'probes'/family).rglob('*.py'):
            if 'tests' in path.parts or path.name.startswith('test_'):
                continue
            for name in imports(path):
                if name.startswith('probes.') and name.split('.')[1] not in {family, 'model_profiles'}:
                    bad.append((str(path.relative_to(ROOT)), name))
    assert bad == []


def test_retired_catchall_and_document_artifact_interface_have_no_code():
    assert list((ROOT/'probes/training_set_completion').rglob('*.py')) == []
    assert not (ROOT/'src/artifacts/source_locations.py').exists()


def test_knowledge_implementation_is_model_independent_and_cli_is_thin():
    implementation = ROOT/'scripts/tools/research_knowledge.py'
    assert not any(name.startswith(('probes.', 'torch', 'transformers', 'src.'))
                   for name in imports(implementation))
    cli = ast.parse((ROOT/'scripts/research/check_research_knowledge.py').read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in cli.body)
