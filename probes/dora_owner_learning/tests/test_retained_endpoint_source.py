from pathlib import Path

import pytest

from probes.dora_owner_learning import margin_preserved_endpoint as margin
from probes.dora_owner_learning import positive_branch_endpoint as endpoint


def test_current_endpoint_uses_ordinary_maintained_modules():
    assert margin.old_endpoint() is endpoint
    assert "outputs" not in Path(endpoint.__file__).parts
    assert all("outputs" not in Path(module.__file__).parts for module in endpoint._witness_modules())


def test_explicit_retained_packet_preserves_frozen_population():
    packet = margin.load_old_packet()
    assert sum(map(len, packet["eval_shards"])) == 384
    assert len(packet["conditional_cases"]) == 3
    assert len(packet["conditional_jobs"]) == 6


def test_retained_reader_does_not_accept_a_different_packet(tmp_path):
    path = tmp_path / "packet.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="retained endpoint packet bytes changed"):
        endpoint.read_retained_packet(path, expected_sha256=margin.OLD_PACKET_SHA256,
                                      archive_manifest=tmp_path / "missing.json")
