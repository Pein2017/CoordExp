"""Derive consumer-required aliases only from frozen exact visual groups."""
from __future__ import annotations

from typing import Any


ALIAS_FIELDS = ("same_image_aliases", "execution_aliases_same_visual_group")


def derive_required_alias_provenance(
    rows: list[dict[str, Any]], groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mutate missing alias fields and return an exact addition ledger.

    A frozen visual group was formed from one exact image/c/w visual key.
    Therefore every job in that group is an execution alias. Multiple jobs in
    that same exact group are also same-image aliases of one another; the row
    itself is not listed as its own alias. A singleton therefore receives no
    same-image alias. Merely sharing an image across different groups creates
    no alias claim. Existing reviewed alias provenance is retained verbatim
    and is not reinterpreted by this missing-field repair.
    """
    by_job = {row["job_id"]: row for row in rows}
    by_group = {group["visual_group_id"]: group for group in groups}
    if len(by_job) != len(rows) or len(by_group) != len(groups):
        raise ValueError("duplicate frozen job/group identity")

    additions = []
    for group_id, group in by_group.items():
        job_ids = group["job_ids"]
        if not job_ids or len(job_ids) != len(set(job_ids)):
            raise ValueError(f"invalid exact visual group membership: {group_id}")
        for job_id in job_ids:
            row = by_job.get(job_id)
            if row is None or row["visual_group_id"] != group_id:
                raise ValueError(f"row/group membership mismatch: {group_id}/{job_id}")
            if int(row["image_id"]) != int(group["image_id"]):
                raise ValueError(f"row/group image mismatch: {group_id}/{job_id}")

    for row in rows:
        group = by_group[row["visual_group_id"]]
        exact_group_jobs = list(group["job_ids"])
        expected_execution = exact_group_jobs
        expected_same_image = [job_id for job_id in exact_group_jobs if job_id != row["job_id"]]
        added: dict[str, Any] = {
            "job_id": row["job_id"], "visual_group_id": row["visual_group_id"],
            "image_id": int(row["image_id"]), "added": {},
        }

        if "execution_aliases_same_visual_group" in row:
            if row["execution_aliases_same_visual_group"] != expected_execution:
                raise ValueError(f"existing execution aliases disagree with frozen group: {row['job_id']}")
        else:
            row["execution_aliases_same_visual_group"] = expected_execution
            added["added"]["execution_aliases_same_visual_group"] = expected_execution

        if "same_image_aliases" in row:
            aliases = row["same_image_aliases"]
            if (not isinstance(aliases, list) or len(aliases) != len(set(aliases))
                    or any(not isinstance(alias, str) or alias not in by_job for alias in aliases)):
                raise ValueError(f"invalid existing same-image aliases: {row['job_id']}")
        else:
            row["same_image_aliases"] = expected_same_image
            added["added"]["same_image_aliases"] = expected_same_image

        if added["added"]:
            added["basis"] = "same frozen exact visual_group_id and image_id; no co-image-only inference"
            additions.append(added)
    return additions
