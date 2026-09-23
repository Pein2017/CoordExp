"""Small saved-only views of the fixed three-loss comparison."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path
from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.coordinate_codebook_alignment.scale_reduce import _aggregate, _flags

CONDITIONS=('source','ce_only_epoch16','three_loss_epoch8','three_loss_epoch16')

def band(count):
    return '1-4' if count<=4 else '5-9' if count<=9 else '10-19' if count<=19 else '20-39' if count<=39 else '40+'

def burden(rows):
    spans=[s for r in rows for s in r.get('format_summary',{}).get('malformed_spans',[])]
    return {'images':len(rows),'bad_images':sum(_flags(r)['bad'] for r in rows),
            'parser_drops':sum(r.get('parser_dropped',0) for r in rows),
            'invalid_geometry':sum(r.get('invalid_geometry',0) for r in rows),
            'dropped_span_characters':sum(len(s.get('raw_span_text','')) for s in spans),
            'drop_reasons':dict(Counter(s.get('reason','unknown') for s in spans)),
            'exact_revisits':sum(r.get('repeat_proxy',{}).get('exact_row_revisit_count',0) for r in rows),
            'owner_revisits':sum(r.get('repeat_proxy',{}).get('owner_revisit_count_iou50',0) for r in rows),
            'max_owner_run':max((r.get('repeat_proxy',{}).get('owner_max_run_iou50',0) for r in rows),default=0)}

def span_causes(span):
    text=span.get('raw_span_text','')
    causes=set()
    for body in re.findall(r"<\|box_start\|>(.*?)(?:<\|box_end\|>|$)",text,re.S):
        bins=[int(x) for x in re.findall(r"<\|coord_(\d+)\|>",body)]
        remainder=re.sub(r"<\|coord_\d+\|>","",body)
        if remainder.strip(): causes.add('coordinate_family_departure_text_or_token')
        if re.search(r"(?<![A-Za-z])\d",re.sub(r"<\|[^>]*\|>","",remainder)):
            causes.add('literal_digit_in_box')
        if len(bins)==4:
            x1,y1,x2,y2=bins
            if x1>x2:causes.add('reversed_x')
            if y1>y2:causes.add('reversed_y')
            if x1==x2:causes.add('degenerate_x')
            if y1==y2:causes.add('degenerate_y')
    if span.get('reason')=='malformed_object_span':causes.add('malformed_wrapper_text_or_span')
    if span.get('reason')=='geometry_invalid':causes.add('parser_invalid_geometry')
    return sorted(causes)


def taxonomy(rows):
    per_image=[]; totals={}
    for row in rows:
        causes=Counter(c for span in row.get('format_summary',{}).get('malformed_spans',[]) for c in span_causes(span))
        flags=_flags(row)
        repeat=row.get('repeat_proxy',{})
        if causes or any(flags.values()) or repeat.get('exact_row_revisit_count',0):
            per_image.append({'row_id':row['row_id'],'condition':row['condition'],'panel':row['panel'],'group':row['group'],'stratum':row['stratum'],'cell_path':row['cell_path'],'dropped_span_cause_counts':dict(causes),'flags':flags,'exact_revisits':repeat.get('exact_row_revisit_count',0),'owner_revisits':repeat.get('owner_revisit_count_iou50',0),'max_owner_run':repeat.get('owner_max_run_iou50',0)})
        key=row['condition']+':'+row['panel']
        t=totals.setdefault(key,{'affected_images':{},'affected_spans':{}})
        for c,n in causes.items():
            t['affected_images'][c]=t['affected_images'].get(c,0)+1
            t['affected_spans'][c]=t['affected_spans'].get(c,0)+n
    return {'scope':'Overlapping structural causes in parser-dropped spans. One malformed span can contain multiple rows; span counts are not row or object counts. Exact and annotation-owner recurrence remain distinct; UNKNOWN is excluded.','totals':totals,'per_image':per_image}


def summarize(path):
    d=json.loads(path.read_text());rows=d['per_image'];complete=[r for r in rows if r['status']=='complete']
    by={(r['condition'],r['row_id']):r for r in complete}
    sentinel={r['row_id'] for r in rows if r['condition']=='three_loss_epoch8'}
    assert len(sentinel)==96
    common={c:_aggregate([by[c,i] for i in sorted(sentinel) if (c,i) in by],96) for c in CONDITIONS}
    bands={};paired={};severity={}
    for panel,n in [('train',1024),('validation',256)]:
        ids={r['row_id'] for r in rows if r['condition']=='source' and r['panel']==panel}
        assert len(ids)==n
        bands[panel]={}
        for c in ('source','ce_only_epoch16','three_loss_epoch16'):
            bands[panel][c]={}
            for b in ('1-4','5-9','10-19','20-39','40+'):
                wanted={i for i in ids if ('source',i) in by and band(by['source',i]['target_count'])==b}
                bands[panel][c][b]=_aggregate([by[c,i] for i in wanted if (c,i) in by],len(wanted))
        paired[panel]={};severity[panel]={}
        for baseline in ('source','ce_only_epoch16'):
            pairs=[(by[baseline,i],by['three_loss_epoch16',i]) for i in sorted(ids) if (baseline,i) in by and ('three_loss_epoch16',i) in by]
            result={'expected':n,'complete':len(pairs),'failures':{},'coverage':{}}
            for flag in ('bad','cap','owner_recurrent','severe'):
                result['failures'][flag]={k:sum(predicate(_flags(a)[flag],_flags(b)[flag]) for a,b in pairs) for k,predicate in [('repaired',lambda a,b:a and not b),('persistent',lambda a,b:a and b),('new',lambda a,b:not a and b)]}
            for metric in ('iou50_class_consistent','iou80_class_consistent'):
                changes=[len(b[metric]['covered_owner_ids'])-len(a[metric]['covered_owner_ids']) for a,b in pairs]
                result['coverage'][metric]={'improved_images':sum(x>0 for x in changes),'worse_images':sum(x<0 for x in changes),'equal_images':sum(x==0 for x in changes),'net_matches':sum(changes),'gains':sum(max(0,x) for x in changes),'losses':sum(max(0,-x) for x in changes)}
            paired[panel][baseline]=result
            severity[panel][baseline]={}
            for name,predicate in [('baseline_bad',lambda a,b:_flags(a)['bad']),('new_bad',lambda a,b:not _flags(a)['bad'] and _flags(b)['bad'])]:
                selected=[(a,b) for a,b in pairs if predicate(a,b)]
                severity[panel][baseline][name]={'before':burden([a for a,b in selected]),'after':burden([b for a,b in selected]),'row_ids':[b['row_id'] for a,b in selected]}
    return {'status':d['status'],'reduction':binding(path),'denominator':d['denominator'],'common96':common,'failure_taxonomy':taxonomy(complete),'target_bands':bands,'paired':paired,'severity':severity,'guardrails':d['guardrails'],'limitations':['Known-annotation coverage and owner recurrence are proxies; UNKNOWN is unmatched, not physically false.','Plain teacher CE is distinct from composite training objective.','Single joint three-loss recipe contrast does not isolate individual losses or address component.','Mature SFT includes959/1024 train and248/256 validation identities.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reduction',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as out:json.dump(summarize(a.reduction),out,indent=2,sort_keys=True)
