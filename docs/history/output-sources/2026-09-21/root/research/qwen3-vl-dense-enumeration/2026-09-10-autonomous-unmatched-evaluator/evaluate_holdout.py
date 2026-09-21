"""Replay frozen predictions, then compare with the sealed provisional reference."""
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from profile_rule import judge

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonl(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]

def reference_label(r):
    if r['confidence']=='low':return 'gray'
    if r['entity']=='unsupported' or r['category']=='incorrect' or r['geometry'] in ['too_tight','too_loose','shifted','multi_instance']:return 'defective'
    if r['entity']=='supported' and r['category']=='correct' and r['geometry']=='acceptable':return 'clean'
    return 'gray'

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]

def main():
    out=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'holdout-evaluation-v1');out.mkdir(exist_ok=False)
    candidates=jsonl(ROOT/'holdout-v1/candidates.jsonl')
    detector=json.loads((ROOT/'codetr-context-holdout-v1/decisions.json').read_text())
    obs=jsonl(ROOT/'semantic-holdout-v1/responses.jsonl');by_obs=collections.defaultdict(list)
    for o in obs:by_obs[o['case_id']].append(o)
    by_detector={d['case_id']:d for d in detector}
    assert len(candidates)==len(detector)==64 and len(by_detector)==64
    assert len({r['image_id'] for r in candidates})==64
    assert {c['case_id'] for c in candidates}==set(by_detector)
    for case_id,rows in by_obs.items():
        assert len(rows)==2 and {r['form'] for r in rows}=={'full','fixed_context'}
        assert by_detector[case_id]['decision']=='accept'
    eligible={d['case_id'] for d in detector if d['decision']=='accept'}
    assert set(by_obs)==eligible
    decisions=[judge(c,by_detector[c['case_id']],by_obs[c['case_id']]) for c in candidates]
    (out/'decisions-before-unblinding.json').write_text(json.dumps(decisions,indent=2)+'\n')
    # Only now open the independent reference; never pass it to judge().
    refs=jsonl(ROOT/'holdout-v1/labels-a.jsonl')+jsonl(ROOT/'holdout-v1/labels-b.jsonl')
    by_ref={r['case_id']:r for r in refs}
    assert len(refs)==len(by_ref)==64 and set(by_ref)==set(by_detector)
    override_file=Path(sys.argv[2]) if len(sys.argv)>2 else None
    reference_changes=[]
    if override_file:
        for override in jsonl(override_file):
            case_id=override['case_id'];assert case_id in by_ref
            before=reference_label(by_ref[case_id])
            for key,value in override.items():
                if key=='viewed_paths':
                    by_ref[case_id][key]=sorted(set(by_ref[case_id][key]+value))
                elif key!='case_id':by_ref[case_id][key]=value
            reference_changes.append({'case_id':case_id,'before':before,'after':reference_label(by_ref[case_id])})
    enums={'entity':{'supported','unsupported','uncertain'},'category':{'correct','incorrect','uncertain'},'geometry':{'acceptable','too_tight','too_loose','shifted','multi_instance','uncertain'},'confidence':{'high','medium','low'}}
    for c in candidates:
        r=by_ref[c['case_id']]
        for key,values in enums.items():assert r[key] in values
        assert {c['image_path'],c['crop_path']}<=set(r['viewed_paths'])
        assert sha(c['image_path'])==c['image_sha256']
    cases=[dict(d,reference=by_ref[d['case_id']],reference_class=reference_label(by_ref[d['case_id']])) for d in decisions]
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    counts=collections.Counter((r['reference_class'],r['decision']) for r in cases)
    a=counts['clean','accept_candidate'];b=counts['defective','accept_candidate'];g=counts['gray','accept_candidate'];accepted=a+b+g
    clean=sum(r['reference_class']=='clean' for r in cases)
    detector_rt=json.loads((ROOT/'codetr-context-holdout-v1/runtime.json').read_text());semantic_rt=json.loads((ROOT/'semantic-holdout-v1/runtime.json').read_text())
    latency=(sum(detector_rt['image_latencies'])+sum(b['wall_seconds'] for b in semantic_rt['batches']))/64
    metrics={'definite_accept_precision':a/(a+b) if a+b else None,'definite_accept_count':a+b,'clean_retention':a/clean if clean else None,'accept_coverage':accepted/64,'gray_share_among_accepts':g/accepted if accepted else None,'conservative_verified_precision':a/accepted if accepted else None,'definite_precision_wilson95':wilson(a,a+b),'hot_amortized_seconds_per_candidate':latency,'cold_initializations_seconds_sum':detector_rt['initialization_seconds']+semantic_rt['initialization_seconds'],'allocated_gpu_hours':detector_rt['allocated_gpu_hours']+semantic_rt['allocated_gpu_hours']}
    gates={'precision':metrics['definite_accept_precision'] is not None and metrics['definite_accept_precision']>=.90,'definite_accept_n':a+b>=10,'clean_retention':metrics['clean_retention'] is not None and metrics['clean_retention']>=.25,'accept_coverage':metrics['accept_coverage']>=.15,'gray_accept_share':metrics['gray_share_among_accepts'] is not None and metrics['gray_share_among_accepts']<=.20,'hot_latency':latency<=5,'cold_initialization':metrics['cold_initializations_seconds_sum']<=180}
    summary={'status':'candidate_measurement_not_lead_acceptance','population':64,'reference_counts':dict(collections.Counter(r['reference_class'] for r in cases)),'decision_by_reference':{f'{k[0]} / {k[1]}':v for k,v in counts.items()},'accepted_clean':a,'accepted_defective':b,'accepted_gray':g,'metrics':metrics,'gates':gates,'all_gates_pass':all(gates.values()),'scope':'single frozen64image-balanced real-candidate holdout; provisional blinded visual reference, not ground truth; no cross-prediction duplicate evaluation','timing_scope':'actual model stages: sequential Co-DETR crop forwards plus conditional batched8B point observations; subprocess and input-materialization overhead not included','hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'selected-candidate-v1.json',ROOT/'profile_rule.py',ROOT/'holdout-reference-predicate.json',ROOT/'holdout-v1/candidates.jsonl',ROOT/'holdout-v1/labels-a.jsonl',ROOT/'holdout-v1/labels-b.jsonl',ROOT/'codetr-context-holdout-v1/decisions.json',ROOT/'semantic-holdout-v1/responses.jsonl',out/'decisions-before-unblinding.json']}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if override_file:
        summary['reference_adjudication']={'path':str(override_file),'sha256':sha(override_file),'changes':reference_changes,'timing':'after prediction unblinding; all14 accepted cases checked, method and thresholds unchanged; original reference/evaluation retained'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='hashes'},indent=2))
    print('accepted cases',[(r['case_id'],r['reference_class'],r['reference']['geometry'],r['reference']['confidence']) for r in cases if r['decision']=='accept_candidate'])

if __name__=='__main__':main()
