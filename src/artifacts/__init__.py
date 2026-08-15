"""Canonical artifact exports with dependency-isolated lazy loading."""

from importlib import import_module

__all__ = [
    "CheckpointWriteResult",
    "CheckpointWriter",
    "AbsoluteExecutableBinding",
    "DirectoryTreeBinding",
    "ExecutionEvidenceJournal",
    "JournalAttemptView",
    "JournalInspection",
    "JournalRecordView",
    "JournalSnapshot",
    "RegularFileBinding",
    "ResearchProbeAdmission",
    "ResearchProbeAdmissionError",
    "ReservedOutputPath",
    "ResolvedDataFileBinding",
    "RunWriter",
    "StageEvidence",
    "StrictValueBinding",
    "canonical_json_bytes",
    "capture_binding_manifest",
    "json_sha256",
    "load_canonical_json",
    "validate_json_value",
]


_EXPORT_MODULES = {
    "CheckpointWriteResult": "src.artifacts.checkpoints",
    "CheckpointWriter": "src.artifacts.checkpoints",
    "AbsoluteExecutableBinding": "src.artifacts.research_probe_admission",
    "DirectoryTreeBinding": "src.artifacts.research_probe_admission",
    "ExecutionEvidenceJournal": "src.artifacts.evidence_journal",
    "JournalAttemptView": "src.artifacts.evidence_journal",
    "JournalInspection": "src.artifacts.evidence_journal",
    "JournalRecordView": "src.artifacts.evidence_journal",
    "JournalSnapshot": "src.artifacts.evidence_journal",
    "RegularFileBinding": "src.artifacts.research_probe_admission",
    "ResearchProbeAdmission": "src.artifacts.research_probe_admission",
    "ResearchProbeAdmissionError": "src.artifacts.research_probe_admission",
    "ReservedOutputPath": "src.artifacts.research_probe_admission",
    "ResolvedDataFileBinding": "src.artifacts.research_probe_admission",
    "RunWriter": "src.artifacts.run_writer",
    "StageEvidence": "src.artifacts.research_probe_admission",
    "StrictValueBinding": "src.artifacts.research_probe_admission",
    "canonical_json_bytes": "src.artifacts.json_values",
    "capture_binding_manifest": "src.artifacts.research_probe_admission",
    "json_sha256": "src.artifacts.json_values",
    "load_canonical_json": "src.artifacts.json_values",
    "validate_json_value": "src.artifacts.json_values",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
