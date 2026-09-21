"""Saved-row accounting shared by the readout/history experiment family.

This operation preserves the September 16 scorer's evaluation semantics.
It does not assign physical identity to unmatched proposals, load models,
read experiment directories, or choose a scientific denominator.
"""

import collections
import re

from src.inference.parsing import parse_compact_object_box_closed
from probes.training_set_completion import paired_evaluation as match
from probes.training_set_completion import source256_evaluation as metrics


PAT = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>"
    + r"<\|coord_(\d+)\|>" * 4
    + r"<\|box_end\|>"
)


def score(raw, case, bank):
    """Return the frozen known-owner, validity, and recurrence accounting."""
    native = parse_compact_object_box_closed(
        raw["text"],
        image_width=case["image_width"],
        image_height=case["image_height"],
        row_id=case["row_id"],
        row_index=case["row_index"],
    ).to_artifact_dict()
    valid, drops = match._matchable_rows_with_geometry_debt(
        {**native, "pred": native["predictions"]}
    )
    for prediction in valid:
        prediction["prediction_id"] = f"P{prediction['generated_order']}"
    ledger = match._ledger_image(bank, valid, threshold=0.5)
    owners = {str(owner["owner_id"]): owner for owner in bank}
    predictions = {prediction["prediction_id"]: prediction for prediction in valid}
    ledger["label_string_compatible_matches"] = [
        dict(owner_id=item["reference_owner_id"], prediction_id=item["prediction_id"])
        for item in ledger["matches"]
        if owners[item["reference_owner_id"]]["description"].strip().lower()
        == predictions[item["prediction_id"]]["description"].strip().lower()
    ]
    spans = [
        dict(row=index + 1, description=item[1], box=list(map(int, item.groups()[1:])))
        for index, item in enumerate(PAT.finditer(raw["text"]))
    ]
    counts = collections.Counter((item["description"], tuple(item["box"])) for item in spans)
    seen = set()
    first_repeat = None
    runs = []
    for item in spans:
        key = (item["description"], tuple(item["box"]))
        if key in seen and first_repeat is None:
            first_repeat = item
        seen.add(key)
        if runs and (runs[-1]["description"], runs[-1]["box"]) == (
            item["description"], item["box"]
        ):
            runs[-1]["length"] += 1
        else:
            runs.append(dict(start_row=item["row"], length=1,
                             description=item["description"], box=item["box"]))
    repeats = metrics._strict_repeat_rows(valid)
    reasons = collections.Counter(item.get("drop_reason", item.get("reason")) for item in drops)
    coordinates = [token - 151670 for token in raw["token_ids"] if 151670 <= token <= 152669]
    return dict(
        matches=ledger,
        token_count=len(raw["token_ids"]),
        stop=raw["stop"],
        burden=dict(
            complete_rows=len(spans),
            valid=len(valid),
            invalid=sum(not (item["box"][0] < item["box"][2]
                            and item["box"][1] < item["box"][3]) for item in spans),
            malformed=sum(count for reason, count in reasons.items()
                          if "geometry" not in str(reason) and "bbox" not in str(reason)),
            drop_reasons=dict(reasons),
            strict_valid_repeats=len(repeats),
            literal_repeats=sum(count - 1 for count in counts.values()),
            literal_invalid_repeats=sum(
                count - 1 for (_, box), count in counts.items()
                if not (box[0] < box[2] and box[1] < box[3])
            ),
            unknown=len(ledger["annotation_unmatched_prediction_ids"]),
            cap=int(raw["stop"] == "length"),
            eos=int(raw["stop"] == "im_end"),
        ),
        endpoint_occupancy=dict(
            total=len(coordinates),
            zero=coordinates.count(0),
            last=coordinates.count(999),
            fraction=sum(value in [0, 999] for value in coordinates) / len(coordinates)
            if coordinates else None,
            per_role={
                role: dict(total=len(spans),
                           zero=sum(item["box"][index] == 0 for item in spans),
                           last=sum(item["box"][index] == 999 for item in spans))
                for index, role in enumerate(["x1", "y1", "x2", "y2"])
            },
        ),
        first_literal_repeat=first_repeat,
        first_strict_repeat=repeats[0] if repeats else None,
        longest_exact_run=max(runs, key=lambda item: item["length"]) if runs else None,
        exact_runs=runs,
        complete_rows=spans,
        native_parse=native,
        valid_predictions=valid,
        drops=drops,
    )
