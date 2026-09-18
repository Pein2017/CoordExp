"""Full-plan work must not repeat once per emitted pack."""
from types import SimpleNamespace

from src.packing.planner import PackPlan
from src.training import cache_workflow, micro_step_assembler


def test_complete_plan_digest_is_computed_once(monkeypatch):
    examples = tuple(SimpleNamespace(example_id=str(i), input_ids=(i,),
        supervised_token_spans=(), ignored_token_spans=()) for i in range(16))
    config = SimpleNamespace(packing=SimpleNamespace(global_max_length=2,
        policy='source_order_next_fit', window_size=None, lookahead=None, seed=17,
        worker_count=1, cursor_byte_budget=65536, fragment_item_budget=1024,
        fragment_byte_budget=4194304))
    original = PackPlan.canonical_sha256.fget
    calls = []
    def digest(plan):
        calls.append(plan)
        return original(plan)
    monkeypatch.setattr(PackPlan, 'canonical_sha256', property(digest))
    packs, receipt, identities = cache_workflow._materialize_pack_plan(config, examples)
    assert len(packs) == 8
    assert [s.example_id for p in packs for s in p.segments] == [str(i) for i in range(16)]
    assert set(identities.values()) == {receipt['plan_sha256']}
    assert len(calls) == 1


def test_assembly_indexes_full_corpus_once(monkeypatch):
    class CountedExamples(list):
        iterations = 0
        def __iter__(self):
            self.iterations += 1
            return super().__iter__()
    examples = CountedExamples(SimpleNamespace(example_id=str(i)) for i in range(16))
    packs = tuple(SimpleNamespace(pack_index=i,
        segments=(SimpleNamespace(example_id=str(15-i)),)) for i in range(16))
    monkeypatch.setattr(micro_step_assembler, 'build_qwen_position_inputs',
                        lambda *args, **kwargs: 'positions')
    monkeypatch.setattr(micro_step_assembler, 'build_token_sequence_from_packed_supervision',
                        lambda *args: 'supervision')
    result = micro_step_assembler.assemble_micro_steps(
        SimpleNamespace(training=SimpleNamespace(precision='bf16'),
                        model=SimpleNamespace(fa2_branch_proof='first_forward')),
        SimpleNamespace(token_identity=SimpleNamespace(tokenizer_vocab_size=1000)),
        'vocab', split='train', packs=packs, encoded_examples=examples,
        augmentation_receipt={}, pack_plan_receipt={'plan_sha256':'same'},
        fragment_by_pack={i:'same' for i in range(16)}, token_atoms_by_pack={})
    assert len(result) == 16
    assert all(step.encoded_examples == (examples[15-i],) for i,step in enumerate(result))
    assert examples.iterations == 1
