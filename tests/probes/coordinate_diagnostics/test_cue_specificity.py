"""Model-free qualification of the six-cell public entry and saved consumer."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import test_cued_visual as fixtures
from test_cued_visual import bounded_threads
from probes.coordinate_diagnostics import cue_specificity as probe


class FixtureModel(fixtures.SyntheticModel):
    def generate(self,**kwargs):
        original=self.cell
        self.cell=dict(original,input_region={'clean':'clean','mask-A':'target','mask-B':'background'}[original['input_region']])
        try:return super().generate(**kwargs)
        finally:self.cell=original


class FixtureSession(fixtures.FixtureSession):
    def __init__(self,*args):
        with patch.object(fixtures,'SyntheticModel',FixtureModel):super().__init__(*args)

    def generate(self,cell):
        anchor=self.packet['conditions'][0]
        # B-clean has no scientific reference. Only synthetic computation uses
        # this deliberate poor-B sequence; no production fidelity field added.
        cell=dict(cell,reference_median_winners=anchor['reference_median_winners'])
        return super().generate(cell)


@pytest.fixture(scope='module')
def prepared():
    from probes import rollout_row_credit as retained
    q=retained.frontend();assert q.model is None;FixtureSession.q_frontend=q
    folder=Path(os.environ['CUE_SPECIFICITY_CPU_OUTPUT'])/'prepared'
    if not folder.exists():assert probe.main(['prepare','--output',str(folder)])==0
    _,packet=probe.load_packet(folder/'native-proposal.json',cpu=True)
    return folder,packet


def test_actual_entry_publication_fresh_consumer(prepared):
    folder,packet=prepared;out=folder.parent/'cpu-positive';FixtureSession.loads=[]
    assert probe.main(['run','--config',str(folder/'native-proposal.json'),'--output',str(out)],cpu_factory=FixtureSession)==0
    report=probe.a.load(out/'readback.json');terminal=probe.a.load(out/'terminal.json')
    assert report['complete'] and report['completed_requests']==6 and report['generated_actions']==114
    assert sum(len(c['forced_actions']) for c in packet['conditions'])==90
    assert sum(r['analysis']['actual_free_actions'] for r in report['conditions'])==24
    assert FixtureSession.loads==['checkpoint-16'] and terminal['counts']['checkpoint_loads']==0
    assert all(len(r['analysis']['selected_row']['same_category_overlaps'])==37 for r in report['conditions'])
    assert all(r['analysis']['selected_row']['physical_recovery_credit'] is False for r in report['conditions'])
    b=next(r for r in report['conditions'] if r['condition']=='B-clean')
    assert not b['analysis']['selected_row']['valid'] and b['control']['quality_gate'] is False
    result=subprocess.run([sys.executable,'-m','probes.coordinate_diagnostics.cue_specificity','readback','--output',str(out)],cwd=probe.ROOT,capture_output=True,text=True)
    (folder.parent/'fresh-consumer.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    semantic=copy.deepcopy(report);semantic.pop('compute');assert json.loads(result.stdout)==semantic
    incoming=probe.a.load(out/'invocation.json')['cpu_threads_incoming']
    fresh=int(next(x.split('=',1)[1] for x in result.stderr.splitlines() if x.startswith('cue_specificity_cpu_threads_incoming=')))
    assert incoming==2 and fresh!=incoming and not torch.cuda.is_initialized()
    probe.a.write(folder.parent/'semantic-report-equality.json',dict(exact=True,producer_incoming_threads=incoming,
        fresh_consumer_incoming_threads=fresh,only_removed_field='compute',CUDA_initialized=False,research_forwards=0))
    with pytest.raises(FileExistsError):probe.main(['run','--config',str(folder/'native-proposal.json'),'--output',str(out)],cpu_factory=FixtureSession)


def saved(prepared,name='A-clean'):
    folder,packet=prepared
    return probe.a.load(folder.parent/'cpu-positive'/(name+'.json')),next(c for c in packet['conditions'] if c['condition']==name),packet


@pytest.mark.parametrize('mutation',['cue','history','boundary','order','request','budget','role','mask'])
def test_frozen_inputs_and_action_boundary_reject(prepared,mutation):
    _,original=prepared;p=copy.deepcopy(original);cell=p['conditions'][1]
    if mutation=='cue':cell['forced_actions']['14']=probe.reused.COORD_START+186
    if mutation=='history':cell['forced_actions']['5']+=1
    if mutation=='boundary':cell['forced_actions']['15']=probe.reused.COORD_START+396
    if mutation=='order':p['conditions'][0],p['conditions'][1]=p['conditions'][1],p['conditions'][0]
    if mutation=='request':p['requests']['351017']['mask-B']['media_sha256']='changed'
    if mutation=='budget':p['bounds']['actions']+=1
    if mutation=='role':cell['position']+=1
    if mutation=='mask':p['pixels']['mask-B']['mask']['sha256']='changed'
    with pytest.raises(ValueError):probe.verify_inputs(p)
    assert sum(c['budget'] for c in original['conditions'])==114
    assert sum(len(c['forced_actions']) for c in original['conditions'])==90
    assert all(len(c['expected_ids'])==14 for c in original['conditions'][1:])


def test_regenerated_mask_has_exact_complement_and_preflight(prepared):
    import numpy as np
    _,packet=prepared;clean=np.load(packet['pixels']['clean']['array']['path'],allow_pickle=False)
    rgb,mask,statistics=probe.regenerated_B(clean)
    assert np.array_equal(rgb[~mask],clean[~mask]) and mask.sum()==10920
    assert np.array_equal(rgb,np.load(packet['bindings']['B-array']['path'],allow_pickle=False))
    assert statistics==packet['B_statistics'] and np.all(rgb[mask]==probe.FILL)
    changed=clean.copy();changed[0,0,0]^=1
    rgb_bad,_,_=probe.regenerated_B(changed)
    assert not np.array_equal(rgb_bad,rgb) and rgb_bad[0,0,0]!=rgb[0,0,0]


@pytest.mark.parametrize('mutation',['summary','coordinate','position','prefix','request','action','vector','schema'])
def test_saved_vector_summary_and_action_identity_reject(prepared,mutation):
    record,cell,packet=saved(prepared);changed=copy.deepcopy(record)
    if mutation=='summary':changed['raw_observations'][0]['full_log_normalizer']+=1e-6
    if mutation=='coordinate':changed['median_observations'][1]['coordinate_scores'][396]+=1
    if mutation=='position':changed['median_observations'][1]['position']+=1
    if mutation=='prefix':changed['median_observations'][1]['prefix_token_ids'][-1]+=1
    if mutation=='request':changed['request_id']+='-wrong'
    if mutation=='action':changed['raw_steps'][15]['requested_token_id']+=1
    if mutation=='vector':changed['raw_observations'][0]['vector_key']='median_14'
    if mutation=='schema':changed['schema']=probe.reused.SCHEMA
    with pytest.raises(ValueError):probe.validate_record(changed,cell,packet)


def test_vector_bytes_corruption_and_optional_gpu_diagnostic(prepared):
    from safetensors.torch import load_file,save_file
    record,cell,packet=saved(prepared);probe.validate_record(record,cell,packet)
    changed=copy.deepcopy(record);path=Path(record['score_tensors']['path']).parent/'corrupt-vector.safetensors'
    vectors=load_file(record['score_tensors']['path']);vectors['raw_15'][probe.reused.COORD_START+396]+=1
    save_file(vectors,str(path));changed['score_tensors']['path']=str(path)
    with pytest.raises(ValueError,match='vector bytes'):probe.validate_record(changed,cell,packet)
    changed=copy.deepcopy(record);changed['raw_observations'][0]['native_device_full_log_normalizer']+=10
    probe.validate_record(changed,cell,packet)
    before=probe.reused.continuity.score_summary(record['raw_observations'][0],record['raw_steps'][14])
    assert probe.reused.continuity.score_summary(changed['raw_observations'][0],changed['raw_steps'][14])==before
    assert probe.reused.causal_roles(changed,cell)==probe.reused.causal_roles(record,cell)


def test_A_anchor_holds_all_five_before_execution(prepared,monkeypatch):
    folder,_=prepared;monkeypatch.setattr(FixtureSession,'break_anchor','A-clean')
    output=folder.parent/'hold-A'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==2
    report=probe.a.load(output/'readback.json');counts=probe.a.load(output/'terminal.json')['counts']
    assert counts['attempted_requests']==counts['completed_requests']==1
    assert [r['condition'] for r in report['conditions'] if r['status']=='skipped-HOLD']==['B-clean','A-mask-A','B-mask-A','A-mask-B','B-mask-B']
    assert not report['A_anchor_qualified'] and not report['complete']
    assert all(v is None for v in report['endpoint']['E'].values())


def test_first_y1_matrix_margins_and_conditional_normalization(prepared):
    folder,_=prepared;report=probe.a.load(folder.parent/'cpu-positive/readback.json');rows=report['conditions']
    result=probe.endpoints(rows);assert result==report['endpoint']
    indexed={r['condition']:r for r in rows}
    with probe.reused.cpu_diagnostics():
        for region in ('mask-A','mask-B'):
            for cue in ('A','B'):
                left,right=[indexed[cue+'-'+r] for r in ('clean',region)]
                x,y=[torch.tensor(r['record']['median_observations'][1]['coordinate_scores'],dtype=torch.float64) for r in (left,right)]
                tv=float(.5*(x.softmax(0)-y.softmax(0)).abs().sum())
                assert result['E'][region+','+cue]==tv
                for ch in ('raw','median'):
                    lo,ro=[r['record'][ch+'_observations'][1]['coordinate_scores'] for r in (left,right)]
                    for v in (30,396):assert result['fixed_margin_mask_minus_clean'][region+','+cue][ch][str(v)]==(ro[v]-ro[0])-(lo[v]-lo[0])
        assert result['D_A']==result['E']['mask-A,A']-result['E']['mask-A,B']
        assert result['D_B']==result['E']['mask-B,B']-result['E']['mask-B,A']
        left=indexed['A-clean']['record']['median_observations'][1];right=copy.deepcopy(left)
        right['full_log_normalizer']+=2
        difference=probe.reused.continuity.score_difference(left,right)
        assert difference['coordinate_conditional_TV']==difference['coordinate_conditional_W1_bins']==0
        assert difference['family_mass_left']!=difference['family_mass_right']
    incomplete=probe.endpoints([r for r in rows if r['condition']!='B-mask-B'])
    assert incomplete['E']['mask-B,B'] is None and incomplete['D_B'] is None and incomplete['D_A']==result['D_A']
    # This complete production-shaped fixture already has a negative D_B.
    assert report['complete'] and result['D_B']<0 and result['signs_are_not_acceptance_gates']


def test_same_cue_pairing_role_and_thread_scope(prepared):
    from safetensors.torch import load_file
    folder,packet=prepared;rows=probe.a.load(folder.parent/'cpu-positive/readback.json')['conditions']
    record,cell,_=saved(prepared);vector=load_file(record['score_tensors']['path'])['raw_15']
    target=probe.reused.COORD_START+30;reference=probe.reused.canonical(vector,packet,target)
    baseline=probe.endpoints(rows);before=torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        assert probe.reused.canonical(vector,packet,target)==reference
        assert probe.endpoints(rows)==baseline and torch.get_num_threads()==4
        wrong=copy.deepcopy(rows);next(r for r in wrong if r['condition']=='A-mask-A')['record']['median_observations'][1]['prefix_token_ids'][-1]=probe.reused.COORD_START+495
        with pytest.raises(ValueError,match='paired prefix'):probe.endpoints(wrong)
        assert torch.get_num_threads()==4
        with pytest.raises(RuntimeError,match='intentional'):
            with probe.reused.cpu_diagnostics():
                assert torch.get_num_threads()==1
                raise RuntimeError('intentional diagnostics exception')
        assert torch.get_num_threads()==4
    finally:torch.set_num_threads(before)
    changed=copy.deepcopy(record);changed['token_ids'][15]=packet['eos_id']
    assert probe.reused.causal_roles(changed,cell)['15']=='y1' and '16' not in probe.reused.causal_roles(changed,cell)


def test_early_y1_EOS_is_outcome_and_box_unavailable(prepared,monkeypatch):
    folder,_=prepared;monkeypatch.setattr(FixtureSession,'early_eos','B-mask-B')
    output=folder.parent/'early-eos'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==0
    report=probe.a.load(output/'readback.json');row=report['conditions'][-1]
    assert row['analysis']['selected_row'] is None and row['analysis']['emitted_eos']
    assert row['analysis']['actual_free_actions']==1 and row['causal_roles']['15']=='y1'
    assert report['endpoint']['E']['mask-B,B'] is not None
    assert any(x.get('status')=='unavailable_semantic_alignment' for x in report['endpoint']['paired_scores']['mask-B,B']) is False


def test_partial_acquisition_settles_without_successor(prepared,monkeypatch):
    folder,_=prepared;monkeypatch.setattr(FixtureSession,'interrupt','A-mask-A')
    output=folder.parent/'interrupted'
    assert probe.run(folder/'native-proposal.json',output,cpu_factory=FixtureSession)==2
    partial=probe.a.load(output/'partial-generation.json');terminal=probe.a.load(output/'terminal.json')
    assert partial['condition']=='A-mask-A' and len(partial['selected_unconfirmed'])==16
    assert terminal['counts']['attempted_requests']==3 and terminal['counts']['completed_requests']==2
    assert all(r['status']=='skipped-HOLD' for r in probe.a.load(output/'conditions.json')[2:])


def test_native_release_rejects_before_model_or_cuda(prepared,monkeypatch):
    folder,_=prepared;loads=[]
    monkeypatch.setattr(probe,'NativeSession',lambda *args:loads.append(args))
    output=folder.parent/'unreleased'
    assert probe.run(folder/'native-proposal.json',output)==2
    assert not loads and not torch.cuda.is_initialized()
    assert 'exact clean lead release' in probe.a.load(output/'error.json')['message']
