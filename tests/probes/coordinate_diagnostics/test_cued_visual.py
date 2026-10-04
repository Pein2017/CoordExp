"""Frozen CPU-only entry/consumer qualification; no Qwen/GPT2 construction."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from probes.coordinate_diagnostics import cued_visual as probe
from src.qwen.untied_embeddings import SelectedDeltaOutputHead, SpecialTokenSelection


@pytest.fixture(autouse=True)
def bounded_threads():
    before=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


class SyntheticModel(torch.nn.Module):
    device=torch.device('cpu')
    config=SimpleNamespace()

    def __init__(self):
        super().__init__()
        base=torch.nn.Linear(2,probe.COORD_START+1000,bias=False,dtype=torch.bfloat16)
        with torch.no_grad():base.weight.fill_(1)
        base.requires_grad_(False)
        selection=SpecialTokenSelection(token_strings=tuple(f'<|coord_{i}|>' for i in range(1000)),token_ids=tuple(probe.COORD_IDS))
        self.head=SelectedDeltaOutputHead(base,selection,torch.nn.Parameter(torch.zeros(1000,2),requires_grad=False))
        self.input_head=SimpleNamespace(shared_embed_delta=torch.zeros(1000,2))

    def get_output_embeddings(self):return self.head
    def get_input_embeddings(self):return self.input_head

    def generate(self,**kwargs):
        ids=kwargs['input_ids'].clone();raw=[];processed=[];cell=self.cell
        incoming=torch.get_num_threads()
        for pos in range(kwargs['max_new_tokens']):
            if self.interrupt and cell['condition']==self.interrupt and pos==cell['position']+2:
                raise RuntimeError('synthetic interrupted generation')
            scores=torch.full((1,probe.COORD_START+1000),-12.,dtype=torch.float32)
            winner=cell['reference_median_winners'][pos]
            scores[0,winner]=8
            if pos in cell['observations']:
                target=probe.COORD_START+cell['target_box'][pos-cell['position']]
                scores[0,target]+={'clean':0,'target':.25,'background':-.25}[cell['input_region']]
            if self.early_eos==cell['condition'] and pos==cell['position']+1:
                scores[0,kwargs['eos_token_id']]=20
            raw.append(scores)
            decision=kwargs['logits_processor'](ids,scores);processed.append(decision)
            assert torch.get_num_threads()==incoming
            token=decision.argmax(-1).reshape(1,1);ids=torch.cat((ids,token),1)
            if int(token)==kwargs['eos_token_id']:break
        return SimpleNamespace(sequences=ids,logits=tuple(raw),scores=tuple(processed))


class FixtureSession(probe.NativeSession):
    q_frontend=None
    break_anchor=None
    early_eos=None
    interrupt=None
    loads=[]

    def __init__(self,checkpoint,packet):
        from unittest.mock import patch
        q=replace(self.q_frontend,model=SyntheticModel())
        self.loads.append(checkpoint.name)
        with patch('probes.rule_stability.runner.native_components',return_value=(q,None,dict(compute='CPU_FIXTURE',research_model_loaded=False,research_forwards=0))):
            super().__init__(checkpoint,packet)

    def generate(self,cell):
        from unittest.mock import patch
        self.q.model.cell=cell;self.q.model.early_eos=self.early_eos;self.q.model.interrupt=self.interrupt
        autocast=torch.autocast
        with patch('torch.autocast',side_effect=lambda *args,**kwargs:autocast('cpu',enabled=False)):
            record=super().generate(cell)
        if self.break_anchor==cell['condition']:
            record['median_steps'][0]['argmax']+=1
        return record

    def close(self):
        self.q=self.norm=self.batches=self.delta=None


@pytest.fixture(scope='module')
def prepared():
    from probes import rollout_row_credit as retained
    destination=Path(os.environ['CUED_VISUAL_CPU_OUTPUT'])
    q=retained.frontend();assert q.model is None;FixtureSession.q_frontend=q
    folder=destination/'prepared'
    if not folder.exists():assert probe.main(['prepare','--output',str(folder)])==0
    _,packet=probe.load_packet(folder/'native-proposal.json',cpu=True)
    return folder,packet


def saved(prepared,name='bottle-cued-clean'):
    folder,packet=prepared
    record=probe.a.load(folder.parent/'cpu-positive'/(name+'.json'))
    cell=next(c for c in packet['conditions'] if c['condition']==name)
    return record,cell,packet


def test_actual_entry_publication_fresh_consumer(prepared):
    folder,packet=prepared;output=folder.parent/'cpu-positive';FixtureSession.loads=[]
    assert probe.main(['run','--config',str(folder/'native-proposal.json'),'--output',str(output)],cpu_factory=FixtureSession)==0
    report=probe.a.load(output/'readback.json')
    assert report['complete'] and report['completed_requests']==12 and report['generated_actions']==1530
    assert FixtureSession.loads==['checkpoint-16']
    assert report['requested_denominators']==dict(states=2,modes=2,image_variants=3,clean_anchors=4,cued_cells=6,uncued_cells=6)
    assert all(report['control_qualified'].values())
    assert len(report['contrasts'])==4 and all(c['status']=='available' for c in report['contrasts'])
    assert sum(r['analysis']['actual_free_actions'] for r in report['conditions'])==54
    assert all(r['physical_recovery_credit'] is False for c in report['conditions'] for r in c['analysis']['rows'])
    for c in report['contrasts']:
        if c['mode']=='cued':assert len(c['first_free_y1'])==2 and all(y['prefix_exact'] for y in c['first_free_y1'])
    p=subprocess.run([sys.executable,'-m','probes.coordinate_diagnostics.cued_visual','readback','--output',str(output)],cwd=probe.ROOT,capture_output=True,text=True)
    (folder.parent/'fresh-consumer.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr
    fresh=json.loads(p.stdout);semantic=copy.deepcopy(report);semantic.pop('compute')
    assert semantic==fresh
    incoming=probe.a.load(output/'invocation.json')['cpu_threads_incoming']
    fresh_incoming=int(next(x.split('=',1)[1] for x in p.stderr.splitlines() if x.startswith('cued_visual_cpu_threads_incoming=')))
    assert incoming==2 and fresh_incoming!=incoming
    probe.a.write(folder.parent/'semantic-report-equality.json',dict(exact=True,producer_incoming_threads=incoming,
        fresh_consumer_incoming_threads=fresh_incoming,only_removed_field='compute',research_forwards=0,
        CUDA_initialized=torch.cuda.is_initialized()))
    with pytest.raises(FileExistsError):probe.main(['run','--config',str(folder/'native-proposal.json'),'--output',str(output)],cpu_factory=FixtureSession)


@pytest.mark.parametrize('mutation',['cue','boundary','order','request','budget','reference'])
def test_frozen_matrix_input_and_cue_rejection(prepared,mutation):
    _,original=prepared;p=copy.deepcopy(original);c=p['conditions'][2]
    if mutation=='cue':c['forced_actions']['14']+=1
    if mutation=='boundary':c['forced_actions']['15']=probe.COORD_START+30
    if mutation=='order':p['conditions'][0],p['conditions'][1]=p['conditions'][1],p['conditions'][0]
    if mutation=='request':p['requests']['351017']['target']['request_id']+='-changed'
    if mutation=='budget':p['bounds']['actions']+=1
    if mutation=='reference':c['expected_ids'][15]+=1
    with pytest.raises(ValueError):probe.validate_conditions(p)
    assert sum(len(c['forced_actions']) for c in original['conditions'])==1476


def test_canonical_corruption_rejects_gpu_metadata_does_not(prepared):
    r,c,p=saved(prepared);probe.validate_record(r,c,p)
    changed=copy.deepcopy(r);changed['raw_observations'][0]['full_log_normalizer']+=1e-6
    with pytest.raises(ValueError,match='canonical CPU'):probe.validate_record(changed,c,p)
    changed=copy.deepcopy(r);changed['raw_observations'][0]['coordinate_scores'][30]+=1
    with pytest.raises(ValueError,match='canonical CPU'):probe.validate_record(changed,c,p)
    changed=copy.deepcopy(r);changed['raw_observations'][0]['native_device_full_log_normalizer']+=10
    probe.validate_record(changed,c,p)
    assert probe.continuity.score_summary(changed['raw_observations'][0],changed['raw_steps'][14])==probe.continuity.score_summary(r['raw_observations'][0],r['raw_steps'][14])
    clean=probe.a.load(Path(r['score_tensors']['path']).parent/'readback.json')
    assert clean['complete']
    # Optional GPU diagnostic never enters the coordinate or behavior consumer.
    assert probe.causal_roles(changed,c)==probe.causal_roles(r,c)


def test_canonical_thread_convention_restores_caller_and_is_exact(prepared):
    from safetensors.torch import load_file
    r,c,p=saved(prepared,'person-cued-clean');v=load_file(r['score_tensors']['path'])['raw_232']
    before=torch.get_num_threads();target=probe.COORD_START+633
    reference=probe.canonical(v,p,target)
    try:
        torch.set_num_threads(4)
        assert probe.canonical(v,p,target)==reference and torch.get_num_threads()==4
        with pytest.raises(IndexError):probe.canonical(v,p,p['vocabulary_size']+1)
        assert torch.get_num_threads()==4
        with pytest.raises(RuntimeError,match='intentional CPU diagnostic'):
            with probe.cpu_diagnostics():
                assert torch.get_num_threads()==1
                raise RuntimeError('intentional CPU diagnostic')
        assert torch.get_num_threads()==4
    finally:torch.set_num_threads(before)


def test_score_token_request_shift_rejects(prepared):
    r,c,p=saved(prepared)
    for mutation in ('position','prefix','request','action','vector'):
        changed=copy.deepcopy(r)
        if mutation=='position':changed['median_observations'][0]['position']+=1
        if mutation=='prefix':changed['median_observations'][1]['prefix_token_ids'][-1]+=1
        if mutation=='request':changed['request_id']+='-wrong'
        if mutation=='action':changed['raw_steps'][15]['requested_token_id']+=1
        if mutation=='vector':changed['raw_observations'][0]['vector_key']='median_14'
        with pytest.raises(ValueError):probe.validate_record(changed,c,p)


@pytest.mark.parametrize('anchor',['bottle-uncued-clean','bottle-cued-clean','person-uncued-clean','person-cued-clean'])
def test_each_clean_anchor_holds_its_state_only(prepared,monkeypatch,anchor):
    folder,_=prepared;monkeypatch.setattr(FixtureSession,'break_anchor',anchor)
    output=folder.parent/('hold-'+anchor)
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==2
    report=probe.a.load(output/'readback.json')
    assert report['attempted_requests']==8 and report['completed_requests']==8
    held=[r['condition'] for r in report['conditions'] if r['status']=='skipped-HOLD']
    assert len(held)==4 and all(n.startswith(anchor.split('-')[0]) for n in held)
    assert sum(r['status']=='completed' and r['condition'].endswith('-clean') for r in report['conditions'])==4


def test_same_prefix_and_later_feedback_are_distinct(prepared):
    folder,packet=prepared;report=probe.a.load(folder.parent/'cpu-positive/readback.json')
    rows={r['condition']:r for r in report['conditions']}
    left=rows['bottle-cued-clean'];right=copy.deepcopy(rows['bottle-cued-target'])
    assert all(x['prefix_exact'] for x in probe.compare(left,right))
    right['record']['token_ids'][15]=probe.COORD_START+31
    for ch in ('raw','median'):
        for o in right['record'][ch+'_observations']:
            o['prefix_token_ids']=right['record']['token_ids'][:o['position']]
    comparisons=probe.compare(left,right)
    assert all(x['prefix_exact'] for x in comparisons if x['position']==15)
    assert all(not x['prefix_exact'] for x in comparisons if x['position']>15)
    with pytest.raises(ValueError,match='same literal prefix'):
        wrong=copy.deepcopy(rows['bottle-cued-target']);wrong['record']['median_observations'][1]['prefix_token_ids'][-1]+=1
        probe.contrasts([wrong if r['condition']==wrong['condition'] else r for r in report['conditions']])
    cell=next(c for c in packet['conditions'] if c['condition']=='bottle-cued-target')
    right['record']['token_ids'][15]=packet['eos_id']
    assert '16' not in probe.causal_roles(right['record'],cell)


def test_early_eos_keeps_actual_y1_and_null_box(prepared,monkeypatch):
    folder,_=prepared;monkeypatch.setattr(FixtureSession,'early_eos','bottle-cued-target')
    output=folder.parent/'early-eos'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==0
    report=probe.a.load(output/'readback.json');row=next(r for r in report['conditions'] if r['condition']=='bottle-cued-target')
    assert row['analysis']['selected_row'] is None and row['analysis']['emitted_eos']
    assert row['causal_roles']['15']=='y1' and row['analysis']['actual_free_actions']==1
    contrast=next(c for c in report['contrasts'] if c['image_id']==351017 and c['mode']=='cued')
    assert contrast['behavior']['target_iou'] is None and contrast['behavior']['target_minus_clean'] is None
    assert all(y['prefix_exact'] for y in contrast['first_free_y1'])


def test_partial_acquisition_settles_without_successor(prepared,monkeypatch):
    folder,_=prepared;monkeypatch.setattr(FixtureSession,'interrupt','bottle-cued-target')
    output=folder.parent/'interrupted'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==2
    partial=probe.a.load(output/'partial-generation.json')
    assert partial['condition']=='bottle-cued-target' and len(partial['selected_unconfirmed'])==16
    terminal=probe.a.load(output/'terminal.json');assert terminal['counts']['attempted_requests']==6
    assert terminal['counts']['completed_requests']==5
    assert all(e['status']=='skipped-HOLD' for e in probe.a.load(output/'conditions.json')[5:])


def test_full_family_mass_and_conditional_distances(prepared):
    r,c,p=saved(prepared);o=r['median_observations'][1]
    summary=probe.continuity.score_summary(o,r['median_steps'][15])
    x=torch.tensor(o['coordinate_scores'],dtype=torch.float64)
    assert summary['coordinate_family_mass']==pytest.approx(float(torch.exp(x.logsumexp(0)-o['full_log_normalizer'])))
    other=copy.deepcopy(o);other['full_log_normalizer']+=2
    diff=probe.continuity.score_difference(o,other)
    assert diff['coordinate_conditional_TV']==0 and diff['coordinate_conditional_W1_bins']==0
    assert diff['family_mass_left']!=diff['family_mass_right']


def test_native_release_rejection_before_model_load(prepared,monkeypatch):
    folder,_=prepared;loads=[]
    monkeypatch.setattr(probe,'NativeSession',lambda *args:loads.append(args))
    assert probe.run(folder/'native-proposal.json',folder.parent/'unreleased')==2
    assert not loads and not torch.cuda.is_initialized()
    assert 'exact clean lead release' in probe.a.load(folder.parent/'unreleased/error.json')['message']
