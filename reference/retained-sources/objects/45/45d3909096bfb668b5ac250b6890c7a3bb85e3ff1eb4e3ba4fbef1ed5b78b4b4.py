"""CPU-only comparison and finite-precision KL fixtures; no model construction."""
import json
import math
from pathlib import Path
import numpy as np
import torch

from probes.dora_owner_learning.selective_preservation_dense import OUTPUT as DENSE, WIDE, AUTONOMOUS
from probes.dora_owner_learning.route_access import publish
from probes.dora_owner_learning.candidate_opportunity import file_hash

OUT = AUTONOMOUS / 'dense48-parity-diagnosis'
assert not torch.cuda.is_initialized()
assert not (OUT / 'summary.json').exists()
sources = {}
def read(path):
    sources[str(path)] = file_hash(path)
    return json.loads(path.read_text())

def gradient_fixture(values, dtype, *, exact_same_forward):
    x = torch.tensor(np.asarray(values).copy(), dtype=dtype)
    # For exact_same_forward, source and student log-softmax execute on identical logits.
    # Otherwise the retained GPU logp is the reference and is replayed as CPU logit input.
    reference = torch.log_softmax(x, -1).detach() if exact_same_forward else x.detach().clone()
    student_logits = x.clone().requires_grad_(True)
    student = torch.log_softmax(student_logits, -1)
    kl = (reference.exp() * (reference - student)).sum()
    kl.backward()
    g = student_logits.grad
    return dict(dtype=str(dtype), KL=float(kl.detach()),
        source_student_logp_exact_equal=bool(torch.equal(reference, student.detach())),
        reference_probability_sum=float(reference.exp().sum()),
        gradient_linf=float(g.abs().max()), gradient_l2=float(g.double().norm()),
        gradient_nonzero_elements=int(torch.count_nonzero(g)),
        vocabulary_size=g.numel(), reference_has_gradient=reference.grad is not None)

vectors = []
raw_fixtures = []
for image in ('368', '7116'):
    paths = [root / 'scores' / f'step-{step:02d}-{image}.npy' for root in (DENSE, WIDE) for step in (0, 1)]
    for path in paths: sources[str(path)] = file_hash(path)
    d0, d1, w0, w1 = [np.load(path, allow_pickle=False) for path in paths]
    diff = d1.astype(np.float64) - w1.astype(np.float64)
    dd, wd = d1.astype(np.float64)-d0, w1.astype(np.float64)-w0
    readouts = {}
    for label, root in [('dense', DENSE), ('wide', WIDE)]:
        for step in (0, 1):
            all_scores = read(root / 'scores' / f'step-{step:02d}.json')
            cid = next(k for k in all_scores if int(k.split(':')[0].split('_')[-1]) == int(image))
            readouts[f'{label}_step{step}'] = {k: all_scores[cid][k] for k in
                ['probability','logprob','A_vs_best_other_margin','A_vs_B_margin','top1_id','rank_min','rank_max','target_id']}
    vectors.append(dict(image_id=image, step0_exact_equal=bool(np.array_equal(d0,w0)),
        step1_max_abs=float(abs(diff).max()), step1_RMS=float(np.sqrt(np.mean(diff**2))),
        step1_mean_signed=float(diff.mean()),
        step1_abs_quantiles={str(q):float(np.quantile(abs(diff),q)) for q in [0,.25,.5,.75,.9,.95,.99,.999,1]},
        step1_positions_exceeding_1e5=int((abs(diff)>1e-5).sum()),
        largest_difference_vocab_id=int(abs(diff).argmax()),
        dense_update_l2=float(np.linalg.norm(dd)), wide_update_l2=float(np.linalg.norm(wd)),
        update_cosine=float(np.dot(dd,wd)/(np.linalg.norm(dd)*np.linalg.norm(wd))),
        readouts=readouts,
        dense_minus_wide_step1={k:readouts['dense_step1'][k]-readouts['wide_step1'][k]
                              for k in ['probability','logprob','A_vs_best_other_margin','A_vs_B_margin','rank_min']}))
    raw_fixtures.append(dict(image_id=image, source='Retained initial full-vocabulary isolated-entry raw logits',
        fp32=gradient_fixture(d0,torch.float32,exact_same_forward=True),
        fp64=gradient_fixture(d0,torch.float64,exact_same_forward=True)))

cards = []
for rank in range(8): cards += read(DENSE / 'ranks' / f'rank{rank}' / 'references.json')
assert len(cards)==50
common_checks=[]
for card in cards:
    path=Path(card['path']); image=path.stem; old=WIDE/'references'/f'{image}-logp.npy'
    if old.exists():
        sources[str(path)]=card['sha256'];sources[str(old)]=file_hash(old)
        current_values=np.load(path,mmap_mode='r',allow_pickle=False)
        old_values=np.load(old,mmap_mode='r',allow_pickle=False)
        common_checks.append(dict(image_id=image,exact_equal=bool(np.array_equal(current_values,old_values)),
                                  shape=list(current_values.shape)))
assert len(common_checks)==33

reference_fixtures=[]
for image in ('152252','177167','7116'):
    card=next(c for c in cards if Path(c['path']).stem==image)
    path=Path(card['path']);sources[str(path)]=file_hash(path)
    arr=np.load(path,mmap_mode='r',allow_pickle=False)
    for index in sorted({0,len(arr)//2,len(arr)-1}):
        vals=np.array(arr[index],copy=True)
        fixture_path=OUT/f'reference-{image}-row{index}.npy'
        with fixture_path.open('xb') as stream: np.save(stream,vals,allow_pickle=False)
        reference_fixtures.append(dict(image_id=image,reference_matrix_row=index,shape=list(arr.shape),
            fixture_path=str(fixture_path),fixture_sha256=file_hash(fixture_path),
            retained_reference_as_logit_input=gradient_fixture(vals,torch.float32,exact_same_forward=False),
            equal_recomputed_logp_fp32=gradient_fixture(vals,torch.float32,exact_same_forward=True),
            equal_recomputed_logp_fp64=gradient_fixture(vals,torch.float64,exact_same_forward=True)))

wide_step=read(WIDE/'update-01.json')
failure=read(DENSE/'failure_receipt.json')
report=dict(schema='dense48_parity_cpu_diagnosis.v1',status='CPU_diagnosis_only_no_gate_revision',
    vector_comparisons=vectors,common33_reference_cache_checks=common_checks,
    raw_logit_KL_fixtures=raw_fixtures,saved_reference_KL_fixtures=reference_fixtures,
    durable_optimizer_evidence=dict(wide_step1={k:wide_step[k] for k in
        ['raw_gradient_norm','clipped_gradient_norm','source_parameter_delta_l2','step_parameter_delta_l2']},
        dense_step1_parameter_movement=None,dense_step1_gradient_hash=None,dense_step1_optimizer_hash=None,
        limitation='Dense live assertions passed before the failing score gate, but hash/movement payloads and optimizer/adapter tensors were not durably published. They cannot be reconstructed from saved output vectors.'),
    observations=['Initial Source entry vectors match exactly. Entry top1 and rank agree after update1 despite failed full-vector tolerance.',
      'Exact zero-valued KL on identical FP32 log-softmax forwards can have nonzero CPU backward gradients on real retained Source raw-logit fixtures.',
      'Saved Source logp fixtures show normalization/backward rounding at finite precision; FP64 arithmetic on the same retained values reduces cancellation residuals.'],
    inference='Finite-precision KL cancellation plus altered support/accumulation and Adam epsilon is a plausible explanation, not an identified cause. No model Jacobian or dense parameter-gradient tensor is retained to quantify its contribution.',
    weighting_evidence='Source code multiplies globally normalized losses by8, DDP averages, and clipping occurs after synchronization; deterministic CPU tests detect missing compensation/rank means/local clipping. Output proximity alone cannot rule out uniform scaling errors: global clipping and first-step Adam largely remove global scale.',
    fixture_limitations='CPU kernels are not the failed GPU kernels. Reusing saved logp as logits introduces renormalization, explicitly separated from exact-same-forward fixtures. Raw entry logits are real Source distributions but the corrected entries themselves were excluded from preservation KL.',
    cheapest_next_discriminant=dict(status='PROPOSAL_NOT_LAUNCHED',arm='same-objective dense48 serial first-update control',
      model_loads=1,reference_forwards=50,train_forwards=50,isolated_score_forwards=4,total_forwards=104,
      order='Original two entries then all48 support images in frozen numeric-ID order; exactly one joint globally normalized backward accumulation/clip/AdamW step on Source.',
      preserve_before_gate=['Selected raw gradient norm/hash and tensor snapshot','Clipped gradient norm/hash','Step1 adapter tensors/optimizer state and hashes','Frozen/reference identities','Both full step0/step1 entry vectors'],
      comparison='Compare serial dense48 with failed DDP dense48 and retained wide31 at identical entries. Serial-dense agreement with DDP-dense but not wide31 supports changed-support numerical effects; persistent same-objective serial/DDP discrepancy narrows the issue to execution/accumulation, but needs retained gradient evidence rather than a threshold change.',
      authorization='Root must freeze the packet and authorize this model read; no execution performed here.'),
    sources=sources,torch_version=torch.__version__,cuda_initialized=torch.cuda.is_initialized())
assert not report['cuda_initialized']
publish(OUT/'summary.json',report)
print(json.dumps(dict(vectors=[{k:v for k,v in x.items() if k!='readouts'} for x in vectors],
    common_reference_exact=sum(x['exact_equal'] for x in common_checks),common_reference_count=len(common_checks),
    raw_logit_KL_fixtures=raw_fixtures,
    saved_reference_exactKL_nonzero_grad_count=sum(x['equal_recomputed_logp_fp32']['KL']==0 and x['equal_recomputed_logp_fp32']['gradient_nonzero_elements']>0 for x in reference_fixtures),
    summary=str(OUT/'summary.json'),sha256=file_hash(OUT/'summary.json'))))
