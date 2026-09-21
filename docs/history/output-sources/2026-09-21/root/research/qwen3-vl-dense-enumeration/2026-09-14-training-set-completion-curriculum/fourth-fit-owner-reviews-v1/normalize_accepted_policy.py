"""Root policy projection; retain original worker judgments unchanged."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE/'fourth-fit-accepted-reviews-v3'
def read(p): return json.loads(p.read_text())
def bind(p): return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p, obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(obj,indent=2)+'\n'
    if p.exists(): assert p.read_text()==text, ('immutable projection changed',p)
    else: p.write_text(text)

def main():
    mp=HERE/'match-inheritance-v2.json'
    annotation=BASE/'annotations-with-unlabeled-v3/annotations.jsonl'
    classes={}
    for line in annotation.read_text().splitlines():
        image= json.loads(line)
        for row in image['objects']:
            classes[(image['image_id'],str(row['coco_ann_id']))]={'class_status':'verified','category_name':row['category_name']}
        for row in image['unlabeled']:
            classes[(image['image_id'],row['stable_owner_id'])]={'class_status':row['class_status'],'category_name':row['category_name']}
    changes=[]
    phases={}
    rulings_path=BASE/'fourth-fit-root-rulings-v1/raw-rulings.json'
    root_patches={(r['image_id'],r['phase'],r['prediction_id']):r for r in read(rulings_path)['patches']}
    for rule in read(mp)['rows']:
        image_id,phase=rule['image_id'],rule['phase']
        source=HERE/f'image-{image_id:012d}'/f'{phase}-review.json'
        original=read(source);result=copy.deepcopy(original)
        packet=read(Path(rule['source_packet']['path']))
        raw={r['prediction_id']:r for r in packet['full_raw_rows']}
        v4={r['owner_id'] for r in packet['target_catalog_references']}
        v5={r['owner_id'] for r in packet['annotation_catalog_v5_references']}
        inherited={r['prediction_id']:r for r in rule['fixed_v4_matches']+rule['v5_additional_matches']}
        seen=set();covered=set();annotated=set();counts=Counter()
        for decision in result['decisions']:
            before=copy.deepcopy(decision);pid=decision['prediction_id']
            patch=root_patches.get((image_id,phase,pid))
            if patch:
                assert pid not in inherited, 'Root unmatched ruling cannot veto inherited match'
                decision.update({k:patch[k] for k in ('owner_id','physical_status','extent','class','reason')})
                decision['root_unmatched_ruling']=bind(rulings_path)
            if pid in inherited:
                decision.update(owner_id=inherited[pid]['reference_owner_id'],extent='reasonable',physical_status='true_unique',basis='iou_matched_inherited')
                decision['reason']='User v2: inherit existing IoU50 owner and extent. Prior reviewer explanation retained: '+before['reason']
            owner=decision['owner_id']
            category=classes.get((image_id,owner))
            if decision['physical_status'] in ('true_unique','repeat'):
                decision['physical_status']='repeat' if owner in seen else 'true_unique'
                seen.add(owner)
                if category and category['class_status']=='verified' and (raw[pid].get('description') or '').strip().casefold()==category['category_name'].strip().casefold():
                    decision['class']='verified'
                    decision['class_reference']={'annotation':bind(annotation),'stable_owner_id':owner,**category}
            good=decision['physical_status'] in ('true_unique','repeat') and decision['extent']=='reasonable'
            decision['coverage_eligible']=good and owner in v4
            decision['annotation_coverage_eligible']=good and owner in v5
            if decision['coverage_eligible']:covered.add(owner)
            if decision['annotation_coverage_eligible']:annotated.add(owner)
            positive=good and decision['physical_status']=='true_unique' and owner in v5
            decision['direct_CE']={'bbox':'positive' if positive else 'mask','description':'positive' if positive and decision['class']=='verified' else 'mask'}
            counts[decision['physical_status']]+=1;counts['class_'+decision['class']]+=1
            keys=('owner_id','extent','physical_status','class','coverage_eligible','annotation_coverage_eligible','direct_CE')
            if any(before.get(k)!=decision.get(k) for k in keys):
                changes.append({'image_id':image_id,'phase':phase,'prediction_id':pid,'before':{k:before.get(k) for k in keys},'after':{k:decision.get(k) for k in keys},'source_review':bind(source)})
        summary=result['summary']
        summary.update(target_owner_count=len(v4),covered_owner_ids=sorted(covered),missing_owner_ids=sorted(v4-covered),covered_owner_count=len(covered),missing_owner_count=len(v4-covered),annotation_target_owner_count=len(v5),annotation_covered_owner_ids=sorted(annotated),annotation_missing_owner_ids=sorted(v5-annotated),physical_repeat_row_count=counts['repeat'],confirmed_false_row_count=counts['false'],physical_unknown_row_count=counts['unknown'],invalid_output_row_count=counts['invalid_output'],class_wrong_row_count=counts['class_wrong'],class_unknown_row_count=counts['class_unknown'])
        phases[(image_id,phase)]=covered
        if phase=='final256':
            prior=phases[(image_id,'parent64')]
            summary.update(retained_parent_owner_ids=sorted(prior&covered),lost_parent_owner_ids=sorted(prior-covered),newly_covered_fixed_owner_ids=sorted(covered-prior))
        result.update(status='candidate_ready',matching_inheritance=bind(mp),source_review=bind(source),view_notes=bind(source.parent/'review-notes.jsonl'),root_policy_projection={'producer':bind(Path(__file__)),'rule':'Current user IoU50 inheritance; reuse verified owner category from accepted annotation when literal exactly agrees. No new visual judgment or guessed class.'})
        write(OUT/f'image-{image_id:012d}'/f'{phase}-review.json',result)
    write(OUT/'policy-projection-changes.json',{'changes':changes,'producer':bind(Path(__file__)),'annotation_source':bind(annotation),'matching_inheritance':bind(mp),'root_unmatched_rulings':bind(rulings_path)})
    print(json.dumps({'reviews':len(phases),'changed_decisions':len(changes),'output':str(OUT)}))

if __name__=='__main__':main()
