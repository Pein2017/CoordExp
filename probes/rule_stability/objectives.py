"""Frozen chronological rule credit and certified own-prefix geometry loss.

All trajectory positions are zero-based generated-action indices. Replay uses
the causal row ``prompt_length + position - 1``. Caller owns policy-logit
normalization, image averaging, and the finite-value gate.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
import re

import torch

from probes import rollout_row_credit as rows
from probes.online_row_credit import max_geometry_margin
from src.eval.saved_rows import iou_xyxy
from src.inference.token_text import decode_literal_ids
from src.losses.token_scores import aligned_token_logprobs


HORIZON = 3084
DUPLICATE_IOU = .9
_EOS_REASONS = frozenset(("eos", "im_end", "<|im_end|>"))
_CAP_REASONS = frozenset(("length", "max_new_tokens", "max_tokens"))


def trajectory_diagnostics(record: dict) -> dict:
    """Tokenizer-free parser diagnostics; every overlap partner remains visible.

    ``unique_rows`` counts distinct literal valid rows. The separate
    ``nonduplicate_valid_rows`` counts valid rows minus strict-overlap events.
    Geometry-invalid and malformed rows cannot supply duplicate events.
    """
    parsed = rows.parse(record)
    complete = [dict(row, valid=True, order=row["generated_order"],
                     bbox=list(row["coord_bins"]), raw_text=row["raw_span_text"])
                for row in parsed.predictions]
    malformed = []
    for drop in parsed.dropped_predictions:
        if drop["reason"] == "geometry_invalid":
            coordinate_bins = []
            for span in drop["coord_token_spans"]:
                match = re.fullmatch(r"<\|coord_(\d+)\|>", span["text"])
                if match is None:
                    break
                coordinate_bins.append(int(match.group(1)))
            description = drop["raw_text"].split("<|object_ref_start|>", 1)[-1].split("<|object_ref_end|>", 1)[0]
            complete.append(dict(drop, valid=False, order=drop["generated_order"],
                                 bbox=coordinate_bins, description=description,
                                 parser_drop=drop))
        else:
            censored = (record["stop_reason"] in _CAP_REASONS and
                        drop["char_end"] == len(record["text"]) and
                        not drop["raw_text"].endswith("<|box_end|>"))
            malformed.append(dict(drop, censored=censored))
    complete.sort(key=lambda row: row["order"])
    valid = [row for row in complete if row["valid"]]
    pairs, events = [], []
    for index, later in enumerate(valid):
        partners = []
        for earlier in valid[:index]:
            overlap = iou_xyxy(earlier["bbox"], later["bbox"])
            pair = dict(earlier_order=earlier["order"], later_order=later["order"],
                        iou=overlap, same_category=earlier["description"] == later["description"])
            pairs.append(pair)
            if overlap > DUPLICATE_IOU:
                partners.append(pair)
        if partners:
            events.append(dict(order=later["order"], partner_orders=[p["earlier_order"] for p in partners],
                               max_iou=max(p["iou"] for p in partners)))
    event_orders = {event["order"] for event in events}
    longest, burst = 0, 0
    # Generated wrapper order retains interruptions by invalid/malformed rows.
    for order in range(max((row["order"] for row in complete), default=-1) + 1):
        burst = burst + 1 if order in event_orders else 0
        longest = max(longest, burst)
    literal = Counter(row["raw_text"] for row in complete)
    valid_literal = Counter(row["raw_text"] for row in valid)
    invalid_literal = Counter(row["raw_text"] for row in complete if not row["valid"])
    overlap_values = sorted(pair["iou"] for pair in pairs)
    distribution = dict(pair_count=len(pairs), strict_gt_09_pairs=sum(v > .9 for v in overlap_values),
                        exact_09_pairs=sum(v == .9 for v in overlap_values),
                        positive_pairs=sum(v > 0 for v in overlap_values),
                        max=max(overlap_values, default=None),
                        mean=sum(overlap_values) / len(overlap_values) if overlap_values else None,
                        sorted_ious=overlap_values)
    burdens = dict(duplicate_events=len(events), longest_event_burst=longest,
                   literal_repeat_rows=sum(n - 1 for n in literal.values()),
                   literal_repeat_valid_rows=sum(n - 1 for n in valid_literal.values()),
                   literal_repeat_invalid_rows=sum(n - 1 for n in invalid_literal.values()),
                   invalid_rows=len(complete) - len(valid), malformed_outputs=len(malformed),
                   censored_outputs=sum(row["censored"] for row in malformed),
                   valid_rows=len(valid), unique_rows=len(valid_literal),
                   nonduplicate_valid_rows=len(valid) - len(events), overlap_pairs=len(pairs),
                   generated_tokens=len(record["token_ids"]), eos=record["stop_reason"] in _EOS_REASONS,
                   cap=record["stop_reason"] in _CAP_REASONS, empty=not complete)
    return dict(rows=complete, duplicate_events=events, overlap_pairs=pairs,
                overlap_distribution=distribution, malformed=malformed, burdens=burdens,
                parser_status=parsed.parse_status)


def trajectory_analysis(record: dict, tokenizer) -> dict:
    """Certify literal token alignment, row completion, and illegal greedy sites.

    Decoded-text validity never depends on special action IDs. Original causal
    prefixes locate row completion, including a close ending inside an action.
    Geometry uses an unambiguous literal causal frame and actual action family;
    a type escape is scored once before further slot certification stops.
    """
    result = trajectory_diagnostics(record)
    boundaries, markers, mapping = _literal_action_boundaries(record, tokenizer, result["rows"])
    coordinate_ids = tuple(tokenizer.convert_tokens_to_ids(f"<|coord_{i}|>") for i in range(1000))
    if len(set(coordinate_ids)) != 1000:
        raise ValueError("coordinate token family is not independent")
    bins_by_id = {token: index for index, token in enumerate(coordinate_ids)}
    for row in result["rows"]:
        begin, end = boundaries[row["char_start"] + 1], boundaries[row["char_end"]]
        row["positions"] = list(range(begin, end + 1))
        row["completion_position"] = end
        coordinate_positions = [pos for pos in row["positions"] if record["token_ids"][pos] in bins_by_id]
        is_coordinate_only = (len(coordinate_positions) == 4 and
                              coordinate_positions == list(range(coordinate_positions[0], coordinate_positions[0] + 4)) and
                              [bins_by_id[record["token_ids"][pos]] for pos in coordinate_positions] == row["bbox"])
        row["coordinate_positions"] = coordinate_positions if is_coordinate_only else []
        row["coordinate_action_family"] = "coordinate_only" if is_coordinate_only else "ordinary_or_mixed"
    by_order = {row["order"]: row for row in result["rows"]}
    result["event_positions"] = [by_order[event["order"]]["completion_position"] for event in result["duplicate_events"]]
    escapes = [marker for marker in markers if not marker["canonical_action"]]
    family = dict(marker_escapes=len(escapes),
                  coordinate_marker_escapes=sum(marker["marker"].startswith("<|coord_") for marker in escapes),
                  ordinary_or_mixed_coordinate_rows=sum(row["coordinate_action_family"] == "ordinary_or_mixed" for row in result["rows"]))
    sites, empty, dispositions = _causal_geometry_sites(record, tokenizer, bins_by_id, boundaries, mapping)
    result.update(geometry_sites=sites, empty_legal_sets=empty, geometry_dispositions=dispositions,
                  action_family_dispositions=escapes, action_family_burdens=family, action_mapping=mapping)
    return result


def _literal_action_boundaries(record: dict, tokenizer, complete_rows: list) -> tuple[dict, list, dict]:
    """Map needed decoded boundaries to the earliest original completing action.

    Canonical structural IDs retain the maintained attribution fast path. When
    ordinary pieces spell markers, decode each causal prefix once, in order.
    This fallback stores only one decoded prefix and integer boundary indices;
    it does not require additive single-token byte/Unicode decoding.
    """
    ids, text = record["token_ids"], record["text"]
    if decode_literal_ids(tokenizer, ids) != text:
        raise ValueError("saved token/text alignment differs")
    special = {token: value for token, value in tokenizer.get_added_vocab().items() if token.startswith("<|")}
    by_id = {value: token for token, value in special.items()}
    pattern = re.compile("|".join(re.escape(token) for token in sorted(special, key=len, reverse=True)))
    markers = list(pattern.finditer(text))
    structural_ids = [(position, by_id[value]) for position, value in enumerate(ids) if value in by_id]
    targets = sorted({boundary for marker in markers for boundary in (marker.start() + 1, marker.end())} |
                     {boundary for row in complete_rows for boundary in (row["char_start"] + 1, row["char_end"])})
    mapping = dict(method="canonical_structural_ids", full_decode_calls=1, prefix_decode_calls=0,
                   fallback_prefix_decode_calls=0, frame_decode_calls=0,
                   decoded_id_visits=len(ids), boundary_count=len(targets))
    boundaries = {}
    if [marker.group() for marker in markers] == [token for _, token in structural_ids]:
        for marker, (position, _) in zip(markers, structural_ids, strict=True):
            boundaries[marker.start() + 1] = boundaries[marker.end()] = position
    else:
        mapping["method"] = "causal_prefix_decode"
        pending = 0
        for position in range(len(ids)):
            prefix = decode_literal_ids(tokenizer, ids[:position + 1])
            mapping["prefix_decode_calls"] += 1
            mapping["fallback_prefix_decode_calls"] += 1
            mapping["decoded_id_visits"] += position + 1
            while pending < len(targets) and targets[pending] <= len(prefix) and prefix.startswith(text[:targets[pending]]):
                boundaries[targets[pending]] = position
                pending += 1
            if pending == len(targets):
                break
    if any(target not in boundaries for target in targets):
        raise ValueError("uncertified original-action row or marker boundary")
    dispositions = [dict(marker=marker.group(), char_start=marker.start(), char_end=marker.end(),
                         action_start=boundaries[marker.start() + 1], action_end=boundaries[marker.end()],
                         canonical_action=boundaries[marker.start() + 1] == boundaries[marker.end()] and
                         ids[boundaries[marker.end()]] == special[marker.group()]) for marker in markers]
    return boundaries, dispositions, mapping


def _causal_geometry_sites(record: dict, tokenizer, bins_by_id: dict, boundaries: dict, mapping: dict) -> tuple[list, list, list]:
    """Certify actual slots until a type escape makes subsequent slots ambiguous."""
    ids, text = record["token_ids"], record["text"]
    start = "<|object_ref_start|>"
    literal = re.compile(re.escape(start) + r"(?P<description>.*?)" +
                         re.escape("<|object_ref_end|><|box_start|>"), re.DOTALL)
    sites, empty, dispositions = [], [], []
    for opener in re.finditer(re.escape(start), text):
        begin = boundaries[opener.start() + 1]
        match = literal.match(text, opener.start())
        if match is None or not match.group("description").strip() or "<|" in match.group("description"):
            dispositions.append(dict(wrapper=begin, reason="ambiguous_or_incomplete_description"))
            continue
        frame_end = boundaries[match.end()]
        prefix = decode_literal_ids(tokenizer, ids[:frame_end + 1])
        mapping["prefix_decode_calls"] += 1
        mapping["frame_decode_calls"] += 1
        mapping["decoded_id_visits"] += frame_end + 1
        if prefix != text[:match.end()]:
            dispositions.append(dict(wrapper=begin, position=frame_end, reason="box_start_action_crosses_causal_frame"))
            continue
        first = frame_end + 1
        for slot in range(4):
            pos = first + slot
            if pos >= len(ids):
                dispositions.append(dict(wrapper=begin, slot=slot, reason="incomplete_or_censored"))
                break
            actual = ids[pos]
            lo, hi = 0, 999
            if slot >= 2:
                predecessor = bins_by_id.get(ids[pos - 2])
                if predecessor is None or predecessor == 999:
                    empty.append(dict(wrapper=begin, slot=slot, position=pos,
                                      offending_start_position=pos - 2,
                                      reason="unknown_start" if predecessor is None else "empty_end"))
                    if actual not in bins_by_id:
                        dispositions.append(dict(wrapper=begin, slot=slot, position=pos, reason="type_escape",
                                                 legal_set_unavailable=True, unavailable_slots=list(range(slot + 1, 4))))
                        break
                    continue
                lo, hi = predecessor + 1, 1000
            value = bins_by_id.get(actual)
            if value is None or not lo <= value < hi:
                sites.append(dict(position=pos, slot=slot, lo=lo, hi=hi, token_id=actual,
                                  reason="type" if value is None else "geometry"))
            if value is None:
                dispositions.append(dict(wrapper=begin, slot=slot, position=pos, reason="type_escape",
                                         unknown_start=slot < 2, unavailable_slots=list(range(slot + 1, 4))))
                break
    return sites, empty, dispositions


def duplicate_advantages(sample_event_positions: Sequence[int], greedy_event_positions: Sequence[int],
                         sample_length: int, greedy_length: int, horizon: int = HORIZON) -> torch.Tensor:
    """Detached reward-to-go at absolute generated indices, including actual EOS."""
    if horizon != HORIZON or any(type(length) is not int or not 0 <= length <= horizon
                                 for length in (sample_length, greedy_length)):
        raise ValueError("trajectory length or frozen horizon drift")
    events = []
    for supplied, length in ((sample_event_positions, sample_length), (greedy_event_positions, greedy_length)):
        values = list(supplied)
        if any(type(pos) is not int or not 0 <= pos < length for pos in values):
            raise ValueError("duplicate event positions must be actual actions")
        events.append(sorted(values))
    sample, greedy = events
    values = [(-(len(sample) - bisect_left(sample, t)) +
               (len(greedy) - bisect_left(greedy, t) if t < greedy_length else 0)) / horizon
              for t in range(sample_length)]
    return torch.tensor(values, dtype=torch.float32)


def duplicate_loss(policy_logits: torch.Tensor, token_ids: Sequence[int] | torch.Tensor,
                   advantages: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Per-image action sum; caller supplies causally aligned normalized scores."""
    if policy_logits.ndim == 3 and policy_logits.shape[0] == 1:
        policy_logits = policy_logits[0]
    if policy_logits.ndim != 2:
        raise ValueError("duplicate policy logits must be [actions,vocabulary]")
    targets = torch.as_tensor(token_ids, dtype=torch.long, device=policy_logits.device)
    credit = torch.as_tensor(advantages, dtype=torch.float32, device=policy_logits.device).detach()
    if targets.ndim != 1 or credit.ndim != 1 or len(targets) != policy_logits.shape[0] or len(credit) != len(targets):
        raise ValueError("duplicate actions, advantages and causal scores differ")
    return -(credit * aligned_token_logprobs(policy_logits, targets)).sum()


def geometry_objective(logits: torch.Tensor, positions: Sequence[int], sites: Sequence[dict],
                       prompt_length: int, coordinate_ids: Sequence[int]) -> tuple[torch.Tensor, dict]:
    """Image mean over actual illegal decisions, using the full-vocabulary hinge."""
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3 or logits.shape[0] != 1 or len(positions) != logits.shape[1]:
        raise ValueError("geometry compact logits/position shape drift")
    if len(set(positions)) != len(positions) or prompt_length < 1:
        raise ValueError("geometry causal positions drift")
    if len(coordinate_ids) != 1000 or len(set(coordinate_ids)) != 1000 or any(
            not 0 <= token < logits.shape[-1] for token in coordinate_ids):
        raise ValueError("geometry coordinate vocabulary drift")
    lookup = {position: index for index, position in enumerate(positions)}
    seen, losses, details = set(), [], []
    for site in sites:
        pos, lo, hi = site["position"], site["lo"], site["hi"]
        causal = prompt_length + pos - 1
        if pos in seen or pos < 0 or not 0 <= lo < hi <= 1000 or causal not in lookup:
            raise ValueError("geometry site has duplicate, empty or unavailable causal support")
        legal = tuple(coordinate_ids[lo:hi])
        if site["token_id"] in legal:
            raise ValueError("geometry site is a legal emitted action")
        seen.add(pos)
        z = logits[0, lookup[causal]].float()
        value = max_geometry_margin(z, legal)
        losses.append(value)
        details.append(dict(position=pos, causal_logits_position=causal, lo=lo, hi=hi,
                            emitted_token_id=site["token_id"], replay_argmax=int(z.detach().argmax()),
                            emission_replay_disagrees=int(z.detach().argmax()) != site["token_id"],
                            loss=float(value.detach())))
    loss = torch.stack(losses).mean() if losses else logits.sum() * 0
    return loss, dict(certified_illegal_decisions=len(losses), sites=details)
