import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score

def bh(pvals,q=0.05):
    p=np.asarray(pvals,float); order=np.argsort(p); ranked=p[order]; m=len(p); adj=np.empty(m)
    vals=ranked*m/np.arange(1,m+1); vals=np.minimum.accumulate(vals[::-1])[::-1]; adj[order]=np.minimum(vals,1)
    return adj, adj<=q

def bootstrap_mean(x,B=5000,seed=123):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); means=[]
    for _ in range(B): means.append(rng.choice(x,size=len(x),replace=True).mean())
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def load_records(root):
    rows=[]
    for p in Path(root).rglob('run.json'):
        try: rows.append(json.load(open(p)))
        except Exception: pass
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--bootstrap',type=int,default=5000); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    df=load_records(a.input); df.to_csv(out/'all_runs.csv',index=False)
    if df.empty: raise SystemExit('No run.json files found')
    metrics=[c for c in ['ari','nmi','silhouette','calinski_harabasz','davies_bouldin','duration_sec'] if c in df]
    success=df[df.status=='success'].copy(); rows=[]
    for keys,g in success.groupby(['ablation','missing_view_condition']):
        for m in metrics:
            x=g[m].dropna().to_numpy()
            if not len(x): continue
            lo,hi=bootstrap_mean(x,a.bootstrap)
            rows.append({"ablation":keys[0],"condition":keys[1],"metric":m,"n":len(x),"mean":x.mean(),"std":x.std(ddof=1) if len(x)>1 else np.nan,"ci95_t":(x.mean()-1.96*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else np.nan,"ci95_boot_lo":lo,"ci95_boot_hi":hi})
    stats=pd.DataFrame(rows); stats.to_csv(out/'summary_statistics.csv',index=False)
    # Matched-seed paired comparisons against full condition within each missing-view condition.
    paired=[]
    for cond in success.missing_view_condition.dropna().unique():
        base=success[(success.ablation=='full')&(success.missing_view_condition==cond)][['seed']+metrics].set_index('seed')
        for abl in success.ablation.unique():
            if abl=='full': continue
            comp=success[(success.ablation==abl)&(success.missing_view_condition==cond)][['seed']+metrics].set_index('seed')
            common=base.index.intersection(comp.index)
            for m in metrics:
                if len(common)<2: continue
                d=comp.loc[common,m]-base.loc[common,m]
                try: stat,pv=wilcoxon(d,alternative='two-sided',zero_method='wilcox')
                except ValueError: stat,pv=np.nan,np.nan
                sd=d.std(ddof=1)
                paired.append({"condition":cond,"comparison":f"{abl}-vs-full","metric":m,"n":len(d),"mean_diff":d.mean(),"median_diff":d.median(),"std_diff":sd,"standardized_paired_effect":d.mean()/sd if sd>0 else np.nan,"wilcoxon_p":pv})
    pair=pd.DataFrame(paired)
    if not pair.empty:
        pair['q_bh'],pair['reject_bh']=bh(pair.wilcoxon_p.fillna(1).to_numpy())
    pair.to_csv(out/'paired_comparisons.csv',index=False)
    # Matched-seed assignment stability: mean ARI across all seed pairs in each condition.
    stability=[]
    for (abl,cond),g in success.groupby(['ablation','missing_view_condition']):
        labs=[]
        for _,r in g.iterrows():
            lp=Path(a.input)/r['run_id']/'labels.npy'
            if lp.exists(): labs.append((int(r['seed']),np.load(lp)))
        vals=[adjusted_rand_score(labs[i][1],labs[j][1]) for i in range(len(labs)) for j in range(i+1,len(labs))]
        if vals:
            lo,hi=bootstrap_mean(vals,a.bootstrap)
            stability.append({'ablation':abl,'condition':cond,'n_seeds':len(labs),'n_pairs':len(vals),'pairwise_seed_ari_mean':float(np.mean(vals)),'pairwise_seed_ari_std':float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,'ci95_boot_lo':lo,'ci95_boot_hi':hi})
    pd.DataFrame(stability).to_csv(out/'seed_stability.csv',index=False)
    json.dump({"n_runs":len(df),"n_success":int((df.status=='success').sum()),"n_failed":int((df.status=='failed').sum())},open(out/'aggregation_manifest.json','w'),indent=2)
    print(f'Aggregated {len(df)} runs; success={int((df.status=="success").sum())}; failed={int((df.status=="failed").sum())}')
if __name__=='__main__': main()
