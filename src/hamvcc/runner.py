import json, os, platform, time, traceback, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from .data import load_npz, apply_condition
from .model import HAMVCC, info_nce, cross_view_loss, cluster_consistency, balance_loss
from .metrics import internal_metrics, external_metrics

ABLATIONS = ["full","no_reliability","no_missing_mask","no_cross_view","no_cluster_consistency","fixed_k"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def choose_k(X, ks, seed):
    scores=[]
    for k in ks:
        km=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(X)
        m=internal_metrics(X,km.labels_)
        scores.append((k,m["silhouette"],m["calinski_harabasz"],m["davies_bouldin"]))
    # Direction-normalized rank average; higher silhouette/CH and lower DB.
    arr=np.array([[s[1],s[2],-s[3]] for s in scores],float)
    ranks=np.zeros_like(arr)
    for j in range(3): ranks[:,j]=arr[:,j].argsort().argsort()
    best=scores[int(np.argmax(ranks.mean(1)))]
    return int(best[0]), scores

def run_one(cfg, seed, condition=None, ablation=None, out_dir=None):
    condition=condition or cfg.get("missing_view_condition","full")
    ablation=ablation or cfg.get("ablation","full")
    out=Path(out_dir or cfg.get("output_dir","runs")); out.mkdir(parents=True,exist_ok=True)
    run_id=f"{cfg.get('run_id_prefix','run')}_seed{seed}_{ablation}_{condition}"
    rd=out/run_id; rd.mkdir(parents=True,exist_ok=True)
    t0=time.time(); seed_all(seed)
    record={"run_id":run_id,"seed":seed,"ablation":ablation,"missing_view_condition":condition,"status":"started","started_at":time.time()}
    try:
        data=apply_condition(load_npz(cfg["data_npz"]),condition)
        device=cfg.get("device","auto")
        if device=="auto": device="cuda" if torch.cuda.is_available() else "cpu"
        device=torch.device(device)
        masks=data.masks.copy()
        if ablation=="no_missing_mask": masks=np.ones_like(masks)
        dims={k:v.shape[1] for k,v in data.views.items()}
        model=HAMVCC(dims,cfg.get("hidden_dim",256),cfg.get("latent_dim",128)).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.get("lr",1e-3),weight_decay=cfg.get("weight_decay",1e-4))
        tensors={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in data.views.items()}
        mt=torch.tensor(masks,dtype=torch.float32,device=device)
        hist=[]; epochs=cfg.get("epochs",100)
        lw=cfg.get("loss_weights",{})
        for ep in range(epochs):
            model.train(); opt.zero_grad(); z,w,fused=model(tensors,mt,uniform_weights=(ablation=="no_reliability"))
            li=info_nce(fused,cfg.get("temperature",0.2)) if ablation!="no_instance" else fused.sum()*0
            lcv=cross_view_loss(z,mt) if ablation!="no_cross_view" else fused.sum()*0
            lcc=cluster_consistency(z,mt,cfg.get("n_clusters") or 3) if ablation!="no_cluster_consistency" else fused.sum()*0
            lb=balance_loss(fused,cfg.get("n_clusters") or 3)
            loss=lw.get("instance",1.0)*li + lw.get("cross_view",0.5)*lcv + lw.get("cluster",0.5)*lcc + lw.get("balance",0.1)*lb
            loss.backward(); opt.step()
            hist.append({"epoch":ep+1,"loss":float(loss.detach().cpu()),"instance":float(li.detach().cpu()),"cross_view":float(lcv.detach().cpu()),"cluster":float(lcc.detach().cpu()),"balance":float(lb.detach().cpu())})
        model.eval()
        with torch.no_grad(): z,w,fused=model(tensors,mt,uniform_weights=(ablation=="no_reliability"))
        X=fused.detach().cpu().numpy()
        # Persist the trained representation model so external/frozen benchmark
        # evaluation can use the actual learned checkpoint rather than a
        # newly initialized (random) encoder.
        torch.save({
            "state_dict": model.state_dict(),
            "view_dims": dims,
            "hidden_dim": int(cfg.get("hidden_dim",256)),
            "latent_dim": int(cfg.get("latent_dim",128)),
            "seed": int(seed),
            "ablation": ablation,
            "missing_view_condition": condition,
            "config": cfg,
        }, rd/"model_checkpoint.pt")
        if ablation=="fixed_k" or cfg.get("n_clusters") is not None:
            k=int(cfg.get("n_clusters") or 3); k_selection=[]
        else:
            k,k_selection=choose_k(X,cfg.get("k_search",[2,3,4,5]),seed)
        km=KMeans(n_clusters=k,n_init=cfg.get("kmeans_n_init",20),random_state=seed).fit(X)
        labels=km.labels_
        im=internal_metrics(X,labels); em=external_metrics(data.labels,labels)
        np.save(rd/"labels.npy",labels); np.save(rd/"embeddings.npy",X); np.save(rd/"per_view_weights.npy",w.detach().cpu().numpy())
        pd.DataFrame(hist).to_csv(rd/"loss_history.csv",index=False)
        with open(rd/"k_selection.json","w") as f: json.dump(k_selection,f,indent=2)
        record.update({"status":"success","duration_sec":time.time()-t0,"n_samples":data.n,"n_clusters":k,**im,**em,"device":str(device),"python":platform.python_version(),"torch":torch.__version__})
        with open(rd/"run.json","w") as f: json.dump(record,f,indent=2)
        return record
    except Exception as e:
        record.update({"status":"failed","duration_sec":time.time()-t0,"error":str(e),"traceback":traceback.format_exc()})
        with open(rd/"failures.json","w") as f: json.dump(record,f,indent=2)
        with open(rd/"run.json","w") as f: json.dump(record,f,indent=2)
        return record
