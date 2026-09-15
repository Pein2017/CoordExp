"""CPU-only synthetic contracts for research-source preservation and routing."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "research_knowledge_checks", ROOT / "scripts/research/check_research_knowledge.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class KnowledgeContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="research-knowledge-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.program = CHECK.PROGRAM
        self.unit = "2026-01-01-example"
        self.original = f"research/investigations/qwen3-vl-dense-enumeration/experiments/{self.unit}/unit.md"
        self.archive_prefix = "docs/history/capture/investigations"
        self.archived = self.original.replace("research/investigations", self.archive_prefix, 1)
        data = b"# Frozen protocol\n"
        self.write(self.archived, data)
        self.write("research/ideas/example.md", b"# Retained idea\n")
        self.write("research/index.md", b"# Live research entry\n")
        self.write("docs/history/capture/old-index.md", b"# Old entry\n")
        self.manifest = {
            "canonical_root": "/original/repository",
            "source_prefix": "research/investigations",
            "archive_prefix": self.archive_prefix,
            "files": [
                {"source": self.original, "archive": self.archived,
                 "sha256": CHECK.digest(data), "bytes": len(data),
                 "action": "relocated_byte_exact", "git_tracked_at_capture": True},
                {"source": "research/index.md", "archive": "docs/history/capture/old-index.md",
                 "sha256": CHECK.digest(b"# Old entry\n"), "bytes": len(b"# Old entry\n"),
                 "action": "snapshot_before_authorized_edit", "git_tracked_at_capture": True},
            ],
        }
        self.state_path = str(self.program / "experiments" / self.unit / "state.json")
        self.result = str(self.program / "experiments" / self.unit / "results.md")
        self.write(self.result, b"# Accepted bounded result\n")
        self.state = {
            "schema_version": 1, "unit_id": self.unit, "lifecycle": "paused",
            "evidence": "accepted", "disposition": "stage_incomplete",
            "state_as_of": "2026-09-15", "protocol": self.archived,
            "result": self.result, "state_source": self.archived,
            "boundary": "No further model work authorized.", "not_authorized": ["GPU work"],
            "next_action": "Discuss the accepted result.",
        }
        self.write(self.state_path, json.dumps(self.state).encode())
        self.write(str(self.program / "questions" / "example.md"), b"# A question\n")
        self.row = {
            "id": self.unit, "topics": ["example"],
            "record_root": str(Path(self.archived).parent),
            "protocols": [self.archived], "result_records": [],
            "reading_entry": self.result, "tracking": "current", "state": self.state_path,
        }

    def write(self, name: str, data: bytes) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def resolve(self, document: str, target: str) -> dict:
        return CHECK.resolve_reference(self.root, self.manifest, document, target)

    def test_preserved_source_bytes_pass(self) -> None:
        self.assertEqual(CHECK.check_sources(self.root, self.manifest), [])

    def test_changed_source_bytes_fail(self) -> None:
        self.write(self.archived, b"A silently changed result\n")
        self.assertTrue(any("source bytes changed" in x for x in CHECK.check_sources(self.root, self.manifest)))

    def test_duplicate_source_mapping_fails(self) -> None:
        self.manifest["files"].append(copy.deepcopy(self.manifest["files"][0]))
        self.assertTrue(any("duplicate manifest" in x for x in CHECK.check_sources(self.root, self.manifest)))

    def test_original_relative_link_uses_original_coordinates(self) -> None:
        ref = self.resolve(self.original, "../../../../ideas/example.md")
        self.assertEqual(ref["path"], "research/ideas/example.md")
        self.assertTrue(ref["exists"])

    def test_archived_relative_link_uses_original_coordinates(self) -> None:
        ref = self.resolve(self.archived, "../../../../ideas/example.md")
        self.assertEqual(ref["path"], "research/ideas/example.md")
        self.assertTrue(ref["exists"])

    def test_absolute_original_path_maps_to_archive(self) -> None:
        ref = self.resolve(self.archived, "/original/repository/" + self.original)
        self.assertEqual(ref["path"], self.archived)
        self.assertTrue(ref["exists"])

    def test_external_artifact_remains_unverified(self) -> None:
        ref = self.resolve(self.archived, "/some/other/project/outputs/receipt.json")
        self.assertEqual(ref["kind"], "external")
        self.assertIsNone(ref["exists"])

    def test_relative_escape_rejected(self) -> None:
        ref = self.resolve("research/index.md", "../../outside.txt")
        self.assertEqual(ref["kind"], "invalid")
        with self.assertRaises(ValueError):
            CHECK.local_path(self.root, "../outside.txt")

    def test_live_link_does_not_route_to_old_router_snapshot(self) -> None:
        ref = self.resolve(str(self.program / "current.md"), "../index.md")
        self.assertEqual(ref["path"], "research/index.md")

    def test_historical_self_reference_keeps_snapshot(self) -> None:
        ref = self.resolve("docs/history/capture/old-index.md", "#old-entry")
        self.assertEqual(ref["path"], "docs/history/capture/old-index.md")

    def test_accepted_incomplete_paused_state_is_valid(self) -> None:
        self.assertEqual(CHECK.check_state(self.root, self.state, self.unit), [])

    def test_accepted_evidence_requires_result(self) -> None:
        self.state["result"] = None
        self.assertTrue(any("lacks a result" in x for x in CHECK.check_state(self.root, self.state, self.unit)))

    def test_invalid_state_identity_and_axis_fail(self) -> None:
        self.state.update(unit_id="different", evidence="running", lifecycle="verified")
        self.assertGreaterEqual(len(CHECK.check_state(self.root, self.state, self.unit)), 3)

    def test_catalog_complete_fixture_passes(self) -> None:
        self.assertEqual(CHECK.check_catalog(self.root, [self.row], self.manifest), [])

    def test_duplicate_catalog_id_fails(self) -> None:
        errors = CHECK.check_catalog(self.root, [self.row, copy.deepcopy(self.row)], self.manifest)
        self.assertTrue(any("duplicate catalog id" in x for x in errors))

    def test_missing_imported_protocol_fails(self) -> None:
        self.row["protocols"] = []
        errors = CHECK.check_catalog(self.root, [self.row], self.manifest)
        self.assertTrue(any("uncatalogued imported protocols" in x for x in errors))

    def test_handoff_cannot_own_current_route(self) -> None:
        handoff = str(self.program / "handoff.md")
        self.write(handoff, b"# Temporary transport\n")
        self.row["reading_entry"] = handoff
        errors = CHECK.check_catalog(self.root, [self.row], self.manifest)
        self.assertTrue(any("handoff cannot" in x for x in errors))

    def test_orphan_state_fails(self) -> None:
        self.write(str(self.program / "experiments/other/state.json"), b"{}")
        errors = CHECK.check_catalog(self.root, [self.row], self.manifest)
        self.assertTrue(any("state/catalog mismatch" in x for x in errors))

    def test_missing_question_target_fails(self) -> None:
        self.row["topics"] = ["absent"]
        errors = CHECK.check_catalog(self.root, [self.row], self.manifest)
        self.assertTrue(any("missing question page" in x for x in errors))

    def test_code_example_is_not_a_live_markdown_link(self) -> None:
        text = "# Sample\n```text\n[not a link](missing.md)\n```\n[real](actual.md)\n"
        self.assertEqual(CHECK.links_in(text), ["actual.md"])

    def test_broken_live_link_is_an_error(self) -> None:
        path = self.root / self.program / "current.md"
        self.write(str(path.relative_to(self.root)), b"[missing](absent.md)\n")
        errors, _, _ = CHECK.check_live_links(self.root, self.manifest, [path])
        self.assertTrue(any("broken live link" in x for x in errors))


if __name__ == "__main__":
    unittest.main()
