"""S3 v3 — do it right.
Phase A: diagnose per-seed S2 variance (the seed5 anomaly).
Phase B: MODEL SELECTION by TRAIN-side health only (reconstruction + dead count),
         BLIND to the S2 outcome. This is not fishing.
Phase C: take the selected SAE, probe held-out seeds ONCE. Report honestly.
"""
import glob, os, pickle, json, numpy as np, sys
sys.path.insert(0, os.path.expanduser("~/elastic-mt-refactor"))
from sae.snapshot import aggregate_regions, N_BINS
from sae.probe import s2_from_snapshot
from sae.model import SAE
root=os.path.expanduser("~/mt_ablation")
CONDS=["baseline","no_zipper","zipper_to_cross","no_induced_catas","always_catas"]
def load(seeds):
    X=[];meta=[]
    for c in CONDS:
        for s in seeds:
            for f in sorted(glob.glob(f"{root}/{c}_seed{s}/orderp_seed{s}_*.pickle")):
                try:d=pickle.load(open(f,"rb"))
                except Exception:continue
                a=aggregate_regions(d)
                if a is None:continue
                for row in np.atleast_2d(np.asarray(a,dtype=np.float32)):
                    if row.sum()>0:X.append(row);meta.append((c,s))
    return np.array(X,dtype=np.float32),meta
Xtr,mtr=load([1,2,3]); Xte,mte=load([4,5])
s2tr=np.array([s2_from_snapshot(x) for x in Xtr]); s2te=np.array([s2_from_snapshot(x) for x in Xte])

print("=== Phase A: per-seed S2 spread ===")
for sd in [1,2,3,4,5]:
    X,_=load([sd]); v=np.array([s2_from_snapshot(x) for x in X])
    print(f"  seed{sd}: n={len(v)} S2 mean={v.mean():.3f} std={v.std():.3f} range[{v.min():.3f},{v.max():.3f}]")

def train(nf, sc, epochs=250, lr=2e-3, seed=0):
    sae=SAE(input_dim=Xtr.shape[1], n_features=nf, sparsity_coef=sc, seed=seed)
    rng=np.random.default_rng(seed)
    for e in range(epochs):
        perm=rng.permutation(len(Xtr))
        for i in range(0,len(Xtr),128): sae.step(Xtr[perm[i:i+128]], lr=lr)
    h=sae.encode(Xtr); recon=float(((sae.decode(h)-Xtr)**2).mean())
    dead=int((h.sum(0)==0).sum()); act=float((h>0).mean())
    return sae, dict(nf=nf, sc=sc, recon=recon, dead=dead, active_frac=act)

print("\n=== Phase B: model selection by TRAIN recon + dead-count (blind to S2) ===")
cands=[]
for nf in [8,16,24]:
    for sc in [0.0005,0.001,0.003]:
        sae,st=train(nf,sc); cands.append((sae,st))
        print(f"  nf={nf} sc={sc}: recon={st['recon']:.5f} dead={st['dead']}/{nf} active_frac={st['active_frac']:.3f}")
# selection rule (pre-stated, outcome-blind): among configs with dead==0, lowest recon.
healthy=[c for c in cands if c[1]['dead']==0]
pool = healthy if healthy else cands
sae, st = min(pool, key=lambda c:(c[1]['dead'], c[1]['recon']))
print(f"\nSELECTED (blind): nf={st['nf']} sc={st['sc']} recon={st['recon']:.5f} dead={st['dead']} active_frac={st['active_frac']:.3f}")

print("\n=== Phase C: probe the selected SAE ONCE on held-out seeds ===")
Htr=sae.encode(Xtr); Hte=sae.encode(Xte)
def r2(yh,y):ss=((y-y.mean())**2).sum();return float(1-((y-yh)**2).sum()/ss)
ctr=np.array([np.corrcoef(Htr[:,f],s2tr)[0,1] if Htr[:,f].std()>1e-8 else 0 for f in range(st['nf'])])
b=int(np.argmax(np.abs(ctr)))
bt=float(np.corrcoef(Hte[:,b],s2te)[0,1])
def scorr(sd):
    idx=[i for i,(c,s) in enumerate(mte) if s==sd];h=Hte[idx,b];y=s2te[idx]
    return round(float(np.corrcoef(h,y)[0,1]),3) if len(idx)>2 and h.std()>1e-8 else None
A=np.c_[Htr,np.ones(len(Htr))];w,_,_,_=np.linalg.lstsq(A,s2tr,rcond=None)
dense=r2(np.c_[Hte,np.ones(len(Hte))]@w,s2te)
res=dict(selected=st, best_feature=b, best_feature_test_corr=round(bt,3),
         seed4=scorr(4), seed5=scorr(5), dense_probe_test_r2=round(dense,3),
         top3=[(int(f),round(float(ctr[f]),3)) for f in np.argsort(-np.abs(ctr))[:3]])
print(json.dumps(res,indent=2))
ok = abs(bt)>0.7 and (res['seed4'] is None or abs(res['seed4'])>0.6) and (res['seed5'] is None or abs(res['seed5'])>0.6)
res['verdict']= "CONFIRMED single-feature" if ok else ("dense-only (kill)" if dense>0.7 else "weak")
print("VERDICT:",res['verdict'])
json.dump(res,open(os.path.expanduser("~/s3_v3_result.json"),"w"),indent=2)
