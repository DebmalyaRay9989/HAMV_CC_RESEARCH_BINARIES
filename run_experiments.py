import argparse,json
from pathlib import Path
from src.hamvcc.runner import run_one, ABLATIONS

def main():
    p=argparse.ArgumentParser(description='Matched-seed HAMV-CC research runner')
    p.add_argument('--config',required=True); p.add_argument('--seeds',required=True)
    p.add_argument('--output',default=None); p.add_argument('--ablations',default='full')
    p.add_argument('--conditions',default=None)
    a=p.parse_args(); cfg=json.load(open(a.config))
    def parse_seeds(spec):
        vals=[]
        for part in spec.split(','):
            part=part.strip()
            if not part: continue
            if '-' in part and part.count('-') == 1:
                lo,hi=part.split('-',1); vals.extend(range(int(lo),int(hi)+1))
            else:
                vals.append(int(part))
        if not vals: raise ValueError('No seeds supplied')
        return list(dict.fromkeys(vals))
    seeds=parse_seeds(a.seeds)
    ablations=[x.strip() for x in a.ablations.split(',') if x.strip()]
    conditions=[x.strip() for x in (a.conditions.split(',') if a.conditions else [cfg.get('missing_view_condition','full')])]
    out=a.output or cfg.get('output_dir','runs')
    summary=[]
    for abl in ablations:
        for cond in conditions:
            for seed in seeds:
                c=dict(cfg); c['output_dir']=out
                rec=run_one(c,seed,condition=cond,ablation=abl,out_dir=out)
                summary.append(rec); print(json.dumps(rec,sort_keys=True))
    Path(out).mkdir(parents=True,exist_ok=True)
    json.dump(summary,open(Path(out)/'run_summary.json','w'),indent=2)

if __name__=='__main__': main()
