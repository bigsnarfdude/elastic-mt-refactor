"""S3 — does an SAE rediscover S2 as a single sparse feature?
Trains on S1 ablation orderp snapshots (held-out seeds), probes features vs S2,
and runs the anti-trivial single-feature-vs-dense-probe comparison.
"""
import glob, os, pickle, json, numpy as np, sys
sys.path.insert(0, os.path.expanduser("~/elastic-mt-refactor"))
from sae.snapshot import aggregate_regions, N_BINS
from sae.probe import s2_from_snapshot, omega_from_snapshot
from sae.retrain_multiseed import train_sae, probe_orientation_tuning

root = os.path.expanduser("~/mt_ablation")
CONDS = ["baseline","no_zipper","zipper_to_cross","no_induced_catas","always_catas"]

def load(seeds):
    X=[]; meta=[]
    for c in CONDS:
        for s in seeds:
            for f in sorted(glob.glob(f"{root}/{c}_seed{s}/orderp_seed{s}_*.pickle")):
                try: data=pickle.load(open(f,"rb"))
                except Exception: continue
                arr=aggregate_regions(data)
                if arr is None: continue
                arr=np.atleast_2d(np.asarray(arr, dtype=np.float32))
                for row in arr:
                    if row.sum()>0:
                        X.append(row); meta.append((c,s))
    return np.array(X, dtype=np.float32), meta

Xtr, mtr = load([1,2,3])
Xte, mte = load([4,5])
print(f"train {Xtr.shape}  test {Xte.shape}", flush=True)
s2_tr = np.array([s2_from_snapshot(x) for x in Xtr])
s2_te = np.array([s2_from_snapshot(x) for x in Xte])
print(f"S2 range train [{s2_tr.min():.3f},{s2_tr.max():.3f}]  test [{s2_te.min():.3f},{s2_te.max():.3f}]", flush=True)

sae = train_sae(Xtr, n_features=32, sparsity_coef=0.02, epochs=100, seed=0)

# per-feature corr with S2 on TEST (held-out seeds 4,5)
pr = probe_orientation_tuning(sae, Xte, None)
corrs = pr["corrs_s2"]; alive = pr["alive"]
# pick best single feature by |corr| on TRAIN, then report its honest TEST corr
Htr = sae.encode(Xtr); Hte = sae.encode(Xte)
corr_tr = np.array([np.corrcoef(Htr[:,f], s2_tr)[0,1] if Htr[:,f].std()>1e-8 else 0.0 for f in range(32)])
best = int(np.argmax(np.abs(corr_tr)))
best_test_corr = float(np.corrcoef(Hte[:,best], s2_te)[0,1])

# seed-stability: best feature's corr on seed4 and seed5 separately
def seed_corr(sd):
    idx=[i for i,(c,s) in enumerate(mte) if s==sd]
    h=Hte[idx,best]; y=s2_te[idx]
    return float(np.corrcoef(h,y)[0,1]) if len(idx)>2 and h.std()>1e-8 else None
s4=seed_corr(4); s5=seed_corr(5)

# anti-trivial: dense linear probe from ALL sae features -> S2 (fit train, eval test R2)
def r2(yhat,y):
    ss=((y-y.mean())**2).sum(); return float(1-((y-yhat)**2).sum()/ss)
A=np.c_[Htr, np.ones(len(Htr))]; w,_,_,_=np.linalg.lstsq(A, s2_tr, rcond=None)
dense_r2=r2(np.c_[Hte,np.ones(len(Hte))]@w, s2_te)
# trivial upper bound: dense probe straight from raw input histogram -> S2
B=np.c_[Xtr, np.ones(len(Xtr))]; wb,_,_,_=np.linalg.lstsq(B, s2_tr, rcond=None)
raw_r2=r2(np.c_[Xte,np.ones(len(Xte))]@wb, s2_te)
# single-feature R2 (best feature linear fit)
a=np.polyfit(Htr[:,best], s2_tr, 1); single_r2=r2(np.polyval(a,Hte[:,best]), s2_te)

res=dict(
  train_shape=list(Xtr.shape), test_shape=list(Xte.shape),
  n_dead=int((~alive).sum()), n_features=32,
  best_feature=best, best_feature_pref_angle_deg=float(pr["pref_angles_deg"][best]),
  best_feature_test_corr=best_test_corr, best_feature_test_r2=single_r2,
  best_feature_corr_seed4=s4, best_feature_corr_seed5=s5,
  dense_probe_test_r2=dense_r2, raw_input_probe_test_r2=raw_r2,
  top5_features_by_abscorr=[(int(f), round(float(corr_tr[f]),3)) for f in np.argsort(-np.abs(corr_tr))[:5]],
)
print("\n=== RESULT ==="); print(json.dumps(res, indent=2))
thr=0.7
verdict = ("CONFIRMED" if abs(best_test_corr)>thr and (s4 is None or abs(s4)>0.6) and (s5 is None or abs(s5)>0.6)
           else "KILL/trivial" if dense_r2>0.7 else "WEAK")
print(f"\nVERDICT: {verdict}  (best single-feature |test corr|={abs(best_test_corr):.3f} vs threshold {thr}; dense R2={dense_r2:.3f})")
json.dump({**res,"verdict":verdict}, open(os.path.expanduser("~/s3_result.json"),"w"), indent=2)
sae.save(os.path.expanduser("~/s3_sae.npz"))
print("saved ~/s3_result.json  ~/s3_sae.npz")
