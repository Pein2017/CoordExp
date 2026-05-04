import numpy as np

import src.eval.detection as detection


class _PatchedEncoder:
    calls = []

    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        self.__class__.calls.append(
            {
                "model_name": model_name,
                "device": device,
                "batch_size": batch_size,
            }
        )

    def encode_norm_texts(self, texts):
        return {str(text): np.array([1.0, 0.0], dtype=float) for text in texts}


def test_detection_facade_semantic_encoder_patch_controls_moved_paths(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(detection, "SemanticDescEncoder", _PatchedEncoder)
    _PatchedEncoder.calls = []

    options = detection.EvalOptions(output_dir=tmp_path, semantic_threshold=0.5)

    mapping = detection._build_semantic_desc_mapping(
        [(1, [{"desc": "kitten"}])],
        {"cat": 1},
        options=options,
        counters=detection.EvalCounters(),
    )
    embeddings = detection._try_build_semantic_embeddings(["kitten"], options=options)

    assert mapping["kitten"][0] == "cat"
    assert "kitten" in embeddings
    assert len(_PatchedEncoder.calls) == 2
