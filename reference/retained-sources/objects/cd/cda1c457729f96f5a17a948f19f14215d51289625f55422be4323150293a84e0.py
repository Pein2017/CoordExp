import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
EVAL=ROOT.parent/"evaluation-v1"
rows=[]
for arm in ("A", "B"):
    for step in (0,64,128,256):
        score=json.loads((EVAL/f"{arm}-step-{step:03d}.json").read_text())
        agg=score["aggregate"]; m=agg["annotation_relative_micro_scoped218"]; raw=agg["raw_and_physical"]
        rows.append(dict(arm=arm,new_updates=step,tp=m["tp"],fn=m["fn_scoped218"],f1=m["f1"],valid_predictions=m["prediction_denominator_all_valid_parsed_rows"],invalid=raw["parser_dropped_total"],geometry_invalid=raw["geometry_invalid"],malformed=raw["malformed_non_geometry"],eos_images=11-raw["eos_debt"],capped_images=raw["cap_debt"]))
with (ROOT/"endpoint-metrics.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout="constrained")
for arm,color,label in [("A","#1764ab","A: fourth-fit step256 start"),("B","#d06b10","B: original step2444 start")]:
    values=[r for r in rows if r["arm"]==arm]
    for ax,key in zip(axes,("fn","f1","invalid")):
        ax.plot([r["new_updates"] for r in values],[r[key] for r in values],marker="o",color=color,label=label,linewidth=1.8,alpha=.85)
for ax,title in zip(axes,("Scoped218 false negatives","Scoped218 annotation-relative F1","Malformed + geometry-invalid rows")):
    ax.set_title(title,fontsize=10);ax.set_xlabel("New optimizer updates");ax.set_xticks([0,64,128,256]);ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
axes[1].set_ylim(0,1.05);axes[0].set_ylim(bottom=-4);axes[2].set_ylim(bottom=-20)
axes[0].legend(fontsize=8)
fig.suptitle("11 training images; same218-owner teacher; fresh AdamW; final256 is the decision endpoint",fontsize=11)
fig.savefig(ROOT/"dual-start-curves.png",dpi=180)
fig.savefig(ROOT/"dual-start-curves.pdf")
print(ROOT/"dual-start-curves.png")
