import hashlib
import json

import pytest

from probes.unmatched_judge.infer import (
    atomic_json, parse_answer, read_requests, render_prompt, timing_summary,
)


@pytest.mark.parametrize("raw", [
    '{"entity":"yes","box":"no","reason":"extra"}',
    '{"entity":"YES","box":"no"}',
    '{"entity":true,"box":"no"}',
    '{"entity":"yes","entity":"no","box":"no"}',
    '```json\n{"entity":"yes","box":"no"}\n```',
    '{"entity":"yes"}', '[]', 'garbage',
])
def test_malformed_answers_are_not_repaired(raw):
    parsed, error = parse_answer(raw)
    assert parsed is None and error


def test_valid_answer():
    assert parse_answer(' {"box":"unknown", "entity":"yes"} ') == (
        {"box": "unknown", "entity": "yes"}, None)


def test_requests_fail_on_changed_image_or_duplicate(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    row = dict(case_id="case1", image_path=str(image),
               image_sha256=hashlib.sha256(b"image").hexdigest(),
               system_prompt="system", user_prompt="user")
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps(row) + "\n")
    assert read_requests(requests) == [row]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_requests(requests)
    image.write_bytes(b"image")
    requests.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate case_id"):
        read_requests(requests)


def test_native_template_contract():
    class Processor:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            assert messages == [
                {"role": "system", "content": "system"},
                {"role": "user", "content": [
                    {"type": "image", "image": "/image.png"},
                    {"type": "text", "text": "user"},
                ]},
            ]
            return "native prompt"
    assert render_prompt(Processor(), dict(system_prompt="system", user_prompt="user",
                                           image_path="/image.png")) == "native prompt"


def test_batch_throughput_not_hot_latency():
    batches = [dict(phase="cold_first_request", case_ids=["cold"], wall_seconds=100)]
    batches += [dict(phase="hot_individual", case_ids=[str(i)], wall_seconds=i)
                for i in range(1, 6)]
    batches += [dict(phase="remaining_batch", case_ids=["batch"], wall_seconds=999)]
    summary = timing_summary(batches)
    assert summary["hot_individual_median_seconds"] == 3
    assert summary["hot_individual_p95_seconds"] == pytest.approx(4.8)
    assert summary["hot_case_ids"] == [str(i) for i in range(1, 6)]


def test_atomic_json_no_temporary_left(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    assert not path.with_suffix(".json.tmp").exists()
