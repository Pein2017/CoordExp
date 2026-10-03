import copy
import math
import unittest

import torch

from probes import online_row_credit as maintained
from probes.rule_stability import objectives as objective


def render_row(box, description="person"):
    return "<|object_ref_start|>" + description + "<|object_ref_end|><|box_start|>" + "".join(
        f"<|coord_{value}|>" for value in box
    ) + "<|box_end|>"


class RuleStabilityObjectivesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = maintained.r.frontend().tokenizer
        cls.coordinate_ids = tuple(cls.tokenizer.convert_tokens_to_ids(f"<|coord_{i}|>") for i in range(1000))

    def record(self, text, stop="im_end"):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        raw = dict(request_id="rule-stability-cpu", image_id=11, arm="greedy", text=text,
                   token_ids=ids, generated_tokens=len(ids), stop_reason=stop, width=1000,
                   height=1000, crop=[0, 0, 1000, 1000], prompt_token_ids=[10, 20, 30],
                   media_sha256="cpu-fixture", image_grid_thw=[[1, 1, 1]])
        return maintained.seal(raw, dict(kind="historical_CPU_fixture", update=0,
                                        parameter_sha256="not-live-cpu"))

    def analyze(self, text, stop="im_end"):
        record = self.record(text, stop)
        return record, objective.trajectory_analysis(record, self.tokenizer)

    def record_ids(self, ids, stop="im_end"):
        raw = self.record(self.tokenizer.decode(ids, skip_special_tokens=False), stop)
        return maintained.seal(dict(raw, token_ids=list(ids), generated_tokens=len(ids)), raw["producer"])

    def ordinary(self, text):
        return [token for char in text for token in self.tokenizer.encode(char, add_special_tokens=False)]

    def test_strict_cross_category_chronology_and_one_cost_per_row(self):
        boxes = [[0, 0, 100, 100], [0, 0, 90, 100], [0, 0, 91, 100]]
        record, result = self.analyze("".join(render_row(b, d) for b, d in zip(boxes, ["person", "cup", "chair"])) + "<|im_end|>")
        self.assertEqual([event["order"] for event in result["duplicate_events"]], [2])
        self.assertEqual(result["duplicate_events"][0]["partner_orders"], [0, 1])
        self.assertEqual(result["overlap_pairs"][0]["iou"], .9)
        self.assertEqual(result["event_positions"], [result["rows"][2]["positions"][-1]])
        self.assertEqual(result["burdens"]["duplicate_events"], 1)
        # Inclusive comparison, category filtering and pair-counting all disagree.
        self.assertEqual(sum(pair["iou"] >= .9 for pair in result["overlap_pairs"]), 3)
        self.assertEqual(sum(pair["iou"] > .9 and pair["same_category"] for pair in result["overlap_pairs"]), 0)
        self.assertEqual(sum(pair["iou"] > .9 for pair in result["overlap_pairs"]), 2)
        pure = objective.trajectory_diagnostics(record)
        self.assertEqual(pure["burdens"], result["burdens"])
        _, repeated = self.analyze(render_row(boxes[0]) * 3 + "<|im_end|>")
        self.assertEqual(repeated["burdens"]["duplicate_events"], 2)
        self.assertEqual(repeated["burdens"]["longest_event_burst"], 2)
        self.assertEqual(repeated["burdens"]["literal_repeat_rows"], 2)

    def test_invalid_malformed_and_censored_outputs_remain_visible(self):
        valid = render_row([10, 20, 100, 200])
        invalid = render_row([10, 20, 10, 200])
        truncated = "<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_4|>"
        record, result = self.analyze(valid + valid + invalid + invalid + truncated, "max_new_tokens")
        self.assertEqual(result["burdens"]["duplicate_events"], 1)
        self.assertEqual(result["burdens"]["invalid_rows"], 2)
        self.assertEqual(result["burdens"]["malformed_outputs"], 1)
        self.assertEqual(result["burdens"]["literal_repeat_rows"], 2)
        self.assertEqual(result["burdens"]["literal_repeat_invalid_rows"], 1)
        self.assertEqual(result["burdens"]["censored_outputs"], 1)
        self.assertTrue(result["burdens"]["cap"])
        self.assertFalse(result["burdens"]["eos"])
        self.assertEqual(len(record["token_ids"]), result["burdens"]["generated_tokens"])
        corrupt = copy.deepcopy(record)
        corrupt["token_ids"][-1] = 0
        with self.assertRaises(ValueError):
            objective.trajectory_analysis(corrupt, self.tokenizer)

    def test_actual_row_completion_and_eos_are_different_actions(self):
        record, result = self.analyze(render_row([10, 20, 100, 200]) * 2 + "\n<|im_end|>")
        event = result["event_positions"][0]
        self.assertEqual(record["token_ids"][event], self.tokenizer.convert_tokens_to_ids("<|box_end|>"))
        self.assertLess(event, len(record["token_ids"]) - 1)
        self.assertEqual(record["token_ids"][-1], self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.assertTrue(result["burdens"]["eos"])
        self.assertFalse(result["burdens"]["cap"])
        empty = objective.trajectory_diagnostics(self.record("<|im_end|>"))
        self.assertTrue(empty["burdens"]["empty"])
        self.assertEqual(empty["burdens"]["generated_tokens"], 1)

    def test_fixed_horizon_absolute_baseline_and_actual_actions(self):
        advantages = objective.duplicate_advantages([2, 4], [1, 5], 7, 6)
        self.assertTrue(torch.allclose(advantages, torch.tensor([0., 0., -1., 0., 0., 1., 0.]) / 3084))
        # Sampled EOS at t=2 receives the later greedy event's baseline, not EOS masking.
        short = objective.duplicate_advantages([], [4], 3, 5)
        self.assertEqual(len(short), 3)
        self.assertTrue(torch.equal(short, torch.ones(3) / 3084))
        capped = objective.duplicate_advantages([2], [], 3, 0)
        self.assertEqual(len(capped), 3)  # no invented EOS action
        self.assertTrue(torch.equal(capped, -torch.ones(3) / 3084))
        with self.assertRaises(ValueError):
            objective.duplicate_advantages([3], [], 3, 0)
        shared_action = objective.duplicate_advantages([2, 2], [], 3, 0)
        self.assertTrue(torch.equal(shared_action, -torch.ones(3) * 2 / 3084))
        with self.assertRaises(ValueError):
            objective.duplicate_advantages([True], [], 3, 0)

    def test_duplicate_loss_sum_eos_and_detached_advantages(self):
        logits = torch.tensor([[1., 0.], [0., 2.], [3., 0.]], requires_grad=True)
        advantages = torch.tensor([.25, -.5, .75], requires_grad=True)
        value = objective.duplicate_loss(logits, [0, 1, 1], advantages)
        expected = -(advantages.detach() * logits.log_softmax(-1)[range(3), [0, 1, 1]]).sum()
        self.assertAlmostEqual(float(value), float(expected), places=6)
        self.assertNotAlmostEqual(float(value), float(expected / 3), places=4)
        value.backward()
        self.assertIsNone(advantages.grad)
        self.assertGreater(float(logits.grad[-1].abs().sum()), 0.)  # actual final EOS action retained
        empty_logits = torch.zeros(0, 2, requires_grad=True)
        zero = objective.duplicate_loss(empty_logits, [], torch.zeros(0))
        zero.backward()
        self.assertEqual(float(zero), 0.)
        with self.assertRaises(ValueError):
            objective.duplicate_loss(logits, [0, 1], advantages)

    def test_empty_end_support_corrects_actual_start_and_full_complement(self):
        record, result = self.analyze(render_row([999, 20, 999, 20]) + "<|im_end|>")
        sites = result["geometry_sites"]
        rows = result["rows"]
        coord_positions = rows[0]["coordinate_positions"]
        self.assertEqual([(s["position"], s["lo"], s["hi"]) for s in sites],
                         [(coord_positions[0], 0, 999), (coord_positions[3], 21, 1000)])
        self.assertEqual([item["position"] for item in result["empty_legal_sets"]], [coord_positions[2]])
        n = len(record["prompt_token_ids"])
        positions = tuple(n + s["position"] - 1 for s in sites) + (n + coord_positions[2] - 1,)
        vocab_size = max(self.coordinate_ids) + 1
        logits = torch.zeros(1, len(positions), vocab_size)
        escape = self.tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
        logits[0, 0, escape] = 7.
        logits[0, 1, escape] = 4.
        logits[0, 2, escape] = 1000.  # empty-support end has no loss or repaired history
        logits.requires_grad_()
        value, details = objective.geometry_objective(logits, positions, sites, n, self.coordinate_ids)
        self.assertAlmostEqual(float(value), (math.log1p(math.exp(8)) + math.log1p(math.exp(5))) / 2, places=5)
        self.assertEqual(details["certified_illegal_decisions"], 2)
        value.backward()
        self.assertGreater(float(logits.grad[0, 0, escape]), 0.)
        self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.)
        with self.assertRaises(ValueError):
            objective.geometry_objective(logits, tuple(p + 1 for p in positions), sites, n, self.coordinate_ids)
        bad = dict(sites[0], lo=1000, hi=1000)
        with self.assertRaises(ValueError):
            objective.geometry_objective(logits, positions, [bad], n, self.coordinate_ids)

    def test_actual_type_escape_and_no_error_image_zero(self):
        text = "<|object_ref_start|>person<|object_ref_end|><|box_start|>person<|coord_20|><|coord_30|><|coord_40|><|box_end|><|im_end|>"
        record, result = self.analyze(text)
        self.assertEqual(len(result["geometry_sites"]), 1)
        self.assertEqual(result["geometry_sites"][0]["reason"], "type")
        self.assertEqual(result["empty_legal_sets"], [])
        self.assertEqual(result["geometry_dispositions"][0]["reason"], "type_escape")
        self.assertTrue(result["geometry_dispositions"][0]["unknown_start"])
        _, okay = self.analyze(render_row([0, 0, 999, 999]) + "<|im_end|>")
        self.assertEqual(okay["geometry_sites"], [])
        logits = torch.zeros(1, 1, 1002, requires_grad=True)
        zero, details = objective.geometry_objective(logits, (2,), [], 3, tuple(range(1000)))
        zero.backward()
        self.assertEqual(float(zero), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)
        self.assertEqual(details["certified_illegal_decisions"], 0)

    def test_unfinished_actual_prefix_has_start_loss_without_invented_end(self):
        prefix = "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        record, capped = self.analyze(prefix + "<|coord_999|>", "max_new_tokens")
        self.assertEqual(len(capped["geometry_sites"]), 1)
        self.assertEqual(capped["geometry_sites"][0]["position"], len(record["token_ids"]) - 1)
        self.assertEqual(capped["geometry_sites"][0]["slot"], 0)
        self.assertEqual(capped["empty_legal_sets"], [])
        self.assertEqual(capped["event_positions"], [])
        # No completion requirement: an actual non-coordinate action is illegal too.
        _, typed = self.analyze(prefix + "person", "max_new_tokens")
        self.assertEqual([site["reason"] for site in typed["geometry_sites"]], ["type"])
        # Encountered x2 has empty support; correct only its actual offending start.
        _, dead = self.analyze(prefix + "<|coord_999|><|coord_20|><|coord_400|>", "max_new_tokens")
        self.assertEqual(len(dead["geometry_sites"]), 1)
        self.assertEqual([item["slot"] for item in dead["empty_legal_sets"]], [2])
        # These are actual illegal actions at the certified slot, then alignment stops.
        for ending in ("<|box_end|>", "<|im_end|>"):
            _, boundary = self.analyze(prefix + ending)
            self.assertEqual(len(boundary["geometry_sites"]), 1)
            self.assertEqual(boundary["geometry_sites"][0]["reason"], "type")
            self.assertEqual(boundary["geometry_sites"][0]["slot"], 0)
        _, ambiguous = self.analyze("<|object_ref_start|>person<|coord_999|><|object_ref_end|><|box_start|><|coord_999|>")
        self.assertEqual(ambiguous["geometry_sites"], [])
        _, later_malformed = self.analyze(prefix + "<|coord_999|><|object_ref_start|>bad", "max_new_tokens")
        self.assertEqual(len(later_malformed["geometry_sites"]), 2)
        self.assertEqual(later_malformed["geometry_sites"][0]["slot"], 0)
        self.assertEqual([(site["slot"], site["reason"]) for site in later_malformed["geometry_sites"]], [(0, "geometry"), (1, "type")])

    def test_original_sampled_lexical_ids_are_not_retokenized(self):
        t = self.tokenizer
        lexical = t.encode("a", add_special_tokens=False) + t.encode("b", add_special_tokens=False)
        self.assertNotEqual(lexical, t.encode(t.decode(lexical), add_special_tokens=False))
        row_ids = (t.encode("<|object_ref_start|>", add_special_tokens=False) + lexical +
                   t.encode("<|object_ref_end|><|box_start|><|coord_10|><|coord_20|><|coord_100|><|coord_200|><|box_end|>", add_special_tokens=False))
        actual = row_ids * 2 + [t.convert_tokens_to_ids("<|im_end|>")]
        raw = self.record(t.decode(actual, skip_special_tokens=False))
        raw = maintained.seal(dict(raw, token_ids=actual, generated_tokens=len(actual)), raw["producer"])
        # Existing retokenization contract fails for this legitimate full-support history.
        with self.assertRaises(ValueError):
            maintained.r.aligned_tokens(raw, t)
        result = objective.trajectory_analysis(raw, t)
        self.assertEqual(result["event_positions"], [len(row_ids) * 2 - 1])
        self.assertEqual(result["rows"][0]["positions"], list(range(len(row_ids))))
        self.assertEqual(result["rows"][1]["positions"], list(range(len(row_ids), 2 * len(row_ids))))
        self.assertEqual(result["burdens"]["duplicate_events"], 1)
        self.assertEqual(result["geometry_sites"], [])

    def test_ordinary_close_is_eligible_only_at_true_completing_action(self):
        t = self.tokenizer
        row = render_row([10, 20, 100, 200])
        first = t.encode(row, add_special_tokens=False)
        second = t.encode(row.removesuffix("<|box_end|>"), add_special_tokens=False)
        close = self.ordinary("<|box_end|>")
        self.assertNotIn(t.convert_tokens_to_ids("<|box_end|>"), close)
        before = objective.trajectory_analysis(self.record_ids(first + second + close[:-1], "max_new_tokens"), t)
        self.assertEqual(before["burdens"]["duplicate_events"], 0)
        ids = first + second + close
        completed = objective.trajectory_analysis(self.record_ids(ids, "max_new_tokens"), t)
        self.assertEqual(completed["burdens"]["valid_rows"], 2)
        self.assertEqual(completed["event_positions"], [len(ids) - 1])
        self.assertEqual(completed["action_family_burdens"]["marker_escapes"], 1)
        self.assertEqual(completed["burdens"]["malformed_outputs"], 0)
        suffix = t.encode(" unrelated<|im_end|>", add_special_tokens=False)
        later = objective.trajectory_analysis(self.record_ids(ids + suffix), t)
        self.assertEqual(later["event_positions"], completed["event_positions"])
        corrupt = self.record_ids(ids)
        corrupt["text"] += "corruption"
        with self.assertRaisesRegex(ValueError, "saved token/text alignment"):
            objective.trajectory_analysis(corrupt, t)

    def test_close_endpoint_inside_original_action_and_later_suffix_stability(self):
        t = self.tokenizer
        row = render_row([10, 20, 100, 200])
        prefix = t.encode(row + row.removesuffix("<|box_end|>"), add_special_tokens=False)
        close_fragment = self.ordinary("<|box_end|")
        crossing = t.encode("><", add_special_tokens=False)
        self.assertEqual(crossing, [1784])
        ids = prefix + close_fragment + crossing
        before = objective.trajectory_analysis(self.record_ids(ids[:-1], "max_new_tokens"), t)
        self.assertEqual(before["event_positions"], [])
        completed = objective.trajectory_analysis(self.record_ids(ids, "max_new_tokens"), t)
        self.assertEqual(completed["event_positions"], [len(ids) - 1])
        self.assertEqual(completed["rows"][1]["completion_position"], len(ids) - 1)
        later = objective.trajectory_analysis(self.record_ids(ids + t.encode("suffix<|im_end|>", add_special_tokens=False)), t)
        self.assertEqual(later["event_positions"], completed["event_positions"])

    def test_ordinary_coordinate_keeps_parser_eligibility_and_separate_family_burden(self):
        t = self.tokenizer
        canonical = t.encode(render_row([10, 20, 100, 200]), add_special_tokens=False)
        second = list(canonical)
        index = second.index(t.convert_tokens_to_ids("<|coord_10|>"))
        second[index:index + 1] = self.ordinary("<|coord_10|>")
        record = self.record_ids(canonical + second + [t.convert_tokens_to_ids("<|im_end|>")])
        result = objective.trajectory_analysis(record, t)
        self.assertEqual(result["burdens"], objective.trajectory_diagnostics(record)["burdens"])
        self.assertEqual(result["burdens"]["valid_rows"], 2)
        self.assertEqual(result["burdens"]["duplicate_events"], 1)
        self.assertEqual(result["burdens"]["malformed_outputs"], 0)
        self.assertEqual(result["rows"][1]["coordinate_action_family"], "ordinary_or_mixed")
        self.assertEqual(result["action_family_burdens"]["coordinate_marker_escapes"], 1)
        self.assertEqual(result["action_family_burdens"]["ordinary_or_mixed_coordinate_rows"], 1)
        self.assertEqual([(site["slot"], site["reason"]) for site in result["geometry_sites"]], [(0, "type")])

    def test_capped_action_vs_actual_eos_and_escape_alignment_stop(self):
        t = self.tokenizer
        prefix = "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
        _, capped = self.analyze(prefix + "<|coord_10|>", "max_new_tokens")
        self.assertEqual(capped["geometry_sites"], [])
        record, ended = self.analyze(prefix + "<|coord_10|><|im_end|>")
        self.assertEqual([(site["slot"], site["position"]) for site in ended["geometry_sites"]], [(1, len(record["token_ids"]) - 1)])
        ids = t.encode(prefix, add_special_tokens=False) + self.ordinary("xyz") + t.encode("<|coord_20|><|coord_30|><|coord_40|><|box_end|><|im_end|>", add_special_tokens=False)
        escaped = objective.trajectory_analysis(self.record_ids(ids), t)
        self.assertEqual([(site["slot"], site["token_id"]) for site in escaped["geometry_sites"]], [(0, self.ordinary("x")[0])])
        self.assertEqual(escaped["empty_legal_sets"], [])
        self.assertEqual(escaped["geometry_dispositions"][0]["unavailable_slots"], [1, 2, 3])
        # An ordinary wrapper can establish a literal frame at a real action boundary.
        wrapper = t.encode("<|object_ref_start|>person<|object_ref_end|>", add_special_tokens=False) + self.ordinary("<|box_start|>")
        ordinary = objective.trajectory_analysis(self.record_ids(wrapper + [t.convert_tokens_to_ids("<|im_end|>")]), t)
        self.assertEqual([(site["slot"], site["reason"]) for site in ordinary["geometry_sites"]], [(0, "type")])
        # Crossing beyond the box-start endpoint cannot manufacture a coordinate slot.
        crossing = wrapper[:-1] + t.encode("><", add_special_tokens=False) + [t.convert_tokens_to_ids("<|im_end|>")]
        ambiguous = objective.trajectory_analysis(self.record_ids(crossing), t)
        self.assertEqual(ambiguous["geometry_sites"], [])
        self.assertEqual(ambiguous["geometry_dispositions"][0]["reason"], "box_start_action_crosses_causal_frame")

    def test_marker_fallback_handles_nonadditive_byte_pieces(self):
        t = self.tokenizer
        byte_ids = [t.convert_tokens_to_ids("Ã"), t.convert_tokens_to_ids("©")]
        self.assertEqual(t.decode(byte_ids, skip_special_tokens=False), "é")
        self.assertNotEqual("".join(t.decode([token], skip_special_tokens=False) for token in byte_ids), "é")
        prefix = t.encode("<|object_ref_start|>", add_special_tokens=False) + byte_ids + t.encode("<|object_ref_end|><|box_start|><|coord_10|><|coord_20|><|coord_100|><|coord_200|>", add_special_tokens=False)
        first = prefix + [t.convert_tokens_to_ids("<|box_end|>")]
        ids = first + prefix + self.ordinary("<|box_end|>")
        result = objective.trajectory_analysis(self.record_ids(ids, "max_new_tokens"), t)
        self.assertEqual(result["event_positions"], [len(ids) - 1])
        self.assertEqual(result["action_mapping"]["method"], "causal_prefix_decode")
        self.assertEqual(result["burdens"]["duplicate_events"], 1)


if __name__ == "__main__":
    unittest.main()
