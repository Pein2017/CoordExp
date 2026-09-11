"""Keep research vocabulary out of stable journal and inference owners."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Stable journal and inference owners that SHALL NOT contain research vocabulary.
STABLE_JOURNAL_AND_INFERENCE_OWNERS = (
    REPOSITORY_ROOT / "src" / "artifacts",
    REPOSITORY_ROOT / "src" / "inference",
    REPOSITORY_ROOT / "src" / "config",
    REPOSITORY_ROOT / "configs" / "coordexp_infras" / "infer",
)
# Admission deliberately names a fixed claim boundary to deny scientific authority;
# its typed contract tests own that research-specific schema.
RESEARCH_SPECIFIC_ARTIFACT_OWNERS = frozenset(
    {REPOSITORY_ROOT / "src" / "artifacts" / "research_probe_admission.py"}
)
SCANNED_TEXT_SUFFIXES = frozenset({".json", ".py", ".toml", ".yaml", ".yml"})

# Research vocabulary that belongs to callers only, never to stable owners.
# These patterns are scanned in code and config, avoiding false positives from:
# - "arm" in CPU architecture names (ARM, x86_arm64)
# - "condition" in try/except or conditional logic
# - "treat" in "retreat", "treat_as", "treat_missing"
# - "match" in regex patterns or file matching
# - "threshold" in signal processing or ML thresholds without effect interpretation
#
# To avoid false positives, we scan for more distinctive research patterns:
# - Compound research terms (e.g., "effect_threshold", "treatment_effect", "arm_name")
# - Research workflow keywords combined with interpretation (e.g., "intervention",
#   "estimand", "unmatched_outcome", "stopping_rule")
# - Claims about effect or validity that belong to research, not to mechanics

RESEARCH_PROBE_ARM_IDENTIFIERS = (
    # Research arm/treatment assignment and comparison
    "probe_arm",
    "probe_arms",
    "arm_name",
    "arm_id",
    "arm_assignment",
    "arm_effect",
)

RESEARCH_COHORT_IDENTIFIERS = (
    # Sample grouping with research meaning (avoid generic "cohort" to prevent
    # false positives in "cohort backend" or similar unrelated uses)
    "cohort_id",
    "cohort_name",
    "cohort_assignment",
    "cohort_condition",
    "cohort_stratification",
)

RESEARCH_CONDITIONING_IDENTIFIERS = (
    # Causal conditioning or eligibility restriction (not program control flow)
    "conditioning_on",
    "conditioned_on",
    "eligibility_condition",
    "inclusion_condition",
    "exclusion_condition",
    "stratification_condition",
)

RESEARCH_INTERVENTION_IDENTIFIERS = (
    # Causal treatment or policy assignment (specific research patterns)
    "intervention",
    "interventions",
    "intervention_id",
    "intervention_name",
    "treatment_assignment",
    "treatment_arm",
    "policy_arm",
    "manipulate_outcome",
)

RESEARCH_ESTIMAND_IDENTIFIERS = (
    # Causal target quantity or effect of interest
    "estimand",
    "estimands",
    "target_effect",
    "effect_of_interest",
    "causal_effect",
    "causal_target",
    "treatment_effect",
    "effect_estimate",
)

RESEARCH_MATCHING_IDENTIFIERS = (
    # Causal matching for paired analysis (not regex/file matching)
    "matched_pair",
    "matched_sample",
    "unmatched_outcome",
    "unmatched_result",
    "matching_condition",
    "match_quality",
)

RESEARCH_UNMATCHED_OUTCOME_IDENTIFIERS = (
    # Results with explicit research meaning in matched analysis
    "unmatched_outcome",
    "unmatched_result",
    "unmatched_sample",
)

RESEARCH_THRESHOLD_IDENTIFIERS = (
    # Decision thresholds for effect interpretation or trial stopping
    "stopping_threshold",
    "stopping_rule",
    "effect_threshold",
    "significance_threshold",
    "futility_threshold",
    "interim_threshold",
    "decision_threshold_effect",
)

RESEARCH_SCIENTIFIC_VALIDITY_IDENTIFIERS = (
    # Claims about measurement, experiment, or result validity
    "internal_validity",
    "external_validity",
    "construct_validity",
    "validity_threat",
    "validity_check_effect",
    "replicability_claim",
)

RESEARCH_CLAIM_IDENTIFIERS = (
    # Statements of effect or proof
    "causal_claim",
    "effect_claim",
    "treatment_claim",
    "evidence_of_effect",
    "claim_scope",
    "claim_boundary",
    "prove_effect",
    "proof_of_mechanism",
)

RESEARCH_STOP_RULE_IDENTIFIERS = (
    # Trial termination policy based on interim effect evidence
    "stop_rule",
    "stopping_rule",
    "stopping_time",
    "early_stopping_rule",
    "futility_rule",
    "interim_analysis_rule",
    "stop_criteria_effect",
)

RESEARCH_VOCABULARY = {
    "probe arm": RESEARCH_PROBE_ARM_IDENTIFIERS,
    "cohort": RESEARCH_COHORT_IDENTIFIERS,
    "conditioning": RESEARCH_CONDITIONING_IDENTIFIERS,
    "intervention": RESEARCH_INTERVENTION_IDENTIFIERS,
    "estimand": RESEARCH_ESTIMAND_IDENTIFIERS,
    "matching": RESEARCH_MATCHING_IDENTIFIERS,
    "unmatched outcome": RESEARCH_UNMATCHED_OUTCOME_IDENTIFIERS,
    "threshold": RESEARCH_THRESHOLD_IDENTIFIERS,
    "scientific validity": RESEARCH_SCIENTIFIC_VALIDITY_IDENTIFIERS,
    "claim": RESEARCH_CLAIM_IDENTIFIERS,
    "stop rule": RESEARCH_STOP_RULE_IDENTIFIERS,
}


def test_stable_journal_and_inference_owners_exclude_research_vocabulary() -> None:
    """Reject experiment design vocabulary from stable journal and inference owners.

    The execution-evidence journal and inference artifact systems are mechanics-only
    infrastructure. They do NOT own research interpretation, design, or semantics.
    Per the stable-schema requirements:
    - Journal SHALL NOT define cohort membership, conditioning, intervention, arm
      meaning, estimand, matching, unmatched meaning, thresholds, uncertainty,
      scientific validity, claim scope, or stop rules.
    - Research vocabulary MUST remain caller-owned and out of stable surfaces.
    """

    for root in STABLE_JOURNAL_AND_INFERENCE_OWNERS:
        assert root.exists(), (
            f"Declared stable root does not exist: {root.relative_to(REPOSITORY_ROOT)}"
        )
    for path in RESEARCH_SPECIFIC_ARTIFACT_OWNERS:
        assert path.is_file(), (
            f"Declared research-specific owner does not exist: "
            f"{path.relative_to(REPOSITORY_ROOT)}"
        )

    violations: list[str] = []
    for path in _stable_text_files():
        relative_path = path.relative_to(REPOSITORY_ROOT)
        path_text = relative_path.as_posix().casefold()
        lines = path.read_text(encoding="utf-8").splitlines()
        for vocabulary_class, identifiers in RESEARCH_VOCABULARY.items():
            for identifier in identifiers:
                normalized_identifier = identifier.casefold()
                if normalized_identifier in path_text:
                    violations.append(
                        f"{relative_path}: path contains {vocabulary_class} "
                        f"identifier {identifier!r}"
                    )
                for line_number, line in enumerate(lines, start=1):
                    if normalized_identifier in line.casefold():
                        violations.append(
                            f"{relative_path}:{line_number}: contains "
                            f"{vocabulary_class} identifier {identifier!r}"
                        )

    assert not violations, (
        "Research vocabulary leaked into stable journal and inference owners:\n"
        + "\n".join(violations)
    )


def _stable_text_files() -> list[Path]:
    return sorted(
        path
        for root in STABLE_JOURNAL_AND_INFERENCE_OWNERS
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SCANNED_TEXT_SUFFIXES
        if path not in RESEARCH_SPECIFIC_ARTIFACT_OWNERS
    )
