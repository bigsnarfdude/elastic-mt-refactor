"""S3 diagnostic — is S2 seed-dependent because features are angle-tuned but S2 is
rotation-invariant? Test: (1) per-seed S2 AND dominant angle Omega. (2) does the best
single feature track S2 only when the array orders near ITS preferred angle?
(3) does a ROTATION-INVARIANT readout of the angle-tuned dictionary recover S2 stably?
"""
import glob, os, pickle, json, numpy as np, sys
sys.path.insert(0, os.path.expanduser("~/elastic-mt-refactor"))
from sae.snapshot import aggregate_regions, N_BINS
from sae.probe import s2_from_snapshot, omega_from_snapshot
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
Xtr,_=load([1,2,3]); Xte,mte=load([4,5])
s2tr=np.array([s2_from_snapshot(x) for x in Xtr]); s2te=np.array([s2_from_snapshot(x) for x in Xte])

print("=== per-seed S2 and dominant angle Omega ===")
for sd in [1,2,3,4,5]:
    X,_=load([sd]); v=np.array([s2_from_snapshot(x) for x in X]); om=np.degrees([omega_from_snapshot(x) for x in X])
    print(f" seed{sd}: S2 mean={v.mean():.3f} std={v.std():.3f} | Omega mean={om.mean():.1f}deg std={om.std():.1f}")

# reproduce the selected SAE (nf=8, sc=0.0005, seed=0)
sae=SAE(input_dim=36,n_features=8,sparsity_coef=0.0005,seed=0)
rng=np.random.default_rng(0)
for e in range(250):
    p=rng.permutation(len(Xtr))
    for i in range(0,len(Xtr),128): sae.step(Xtr[p[i:i+128]],lr=2e-3)
Htr=sae.encode(Xtr); Hte=sae.encode(Xte)
ctr=np.array([np.corrcoef(Htr[:,f],s2tr)[0,1] if Htr[:,f].std()>1e-8 else 0 for f in range(8)])
b=int(np.argmax(np.abs(ctr)))
pref_bins=sae.W_d.argmax(0); pref_deg=(pref_bins+0.5)*180/N_BINS
print(f"\nbest single feature = {b}, preferred angle = {pref_deg[b]:.1f} deg")

# (2) conditional: feature b vs S2, split by |Omega - pref_angle| of each held-out snapshot
om_te=np.degrees([omega_from_snapshot(x) for x in Xte])
d=np.abs(((om_te-pref_deg[b]+90)%180)-90)   # circular distance in [0,90]
near=d<30; far=d>=30
def cc(mask):
    h=Hte[mask,b]; y=s2te[mask]
    return (round(float(np.corrcoef(h,y)[0,1]),3), int(mask.sum())) if mask.sum()>2 and h.std()>1e-8 else (None,int(mask.sum()))
print(f"feature{b} vs S2 | array orders NEAR its angle (<30deg): {cc(near)}")
print(f"feature{b} vs S2 | array orders FAR from its angle (>=30deg): {cc(far)}")

# (3) ROTATION-INVARIANT readout: 2nd-Fourier magnitude of angle-tuned activations
th=np.radians(pref_deg)                       # preferred angle per feature (radians, 0..pi)
def s2hat(H):
    w=H/ (H.sum(1,keepdims=True)+1e-9)
    cx=(w*np.cos(2*th)).sum(1); sx=(w*np.sin(2*th)).sum(1)
    return np.sqrt(cx*cx+sx*sx)
ri_tr=s2hat(Htr); ri_te=s2hat(Hte)
print(f"\nrotation-invariant readout vs S2:  train r={np.corrcoef(ri_tr,s2tr)[0,1]:.3f}  test r={np.corrcoef(ri_te,s2te)[0,1]:.3f}")
for sd in [4,5]:
    idx=[i for i,(c,s) in enumerate(mte) if s==sd]
    print(f"   seed{sd}: r={np.corrcoef(ri_te[idx],s2te[idx])[0,1]:.3f} (n={len(idx)})")
res=dict(best_feature=b, pref_angle_deg=round(float(pref_deg[b]),1),
         feat_vs_s2_near=cc(near)[0], feat_vs_s2_far=cc(far)[0],
         rotinv_test_r=round(float(np.corrcoef(ri_te,s2te)[0,1]),3),
         rotinv_seed4=round(float(np.corrcoef(ri_te[[i for i,(c,s) in enumerate(mte) if s==4]],
                              s2te[[i for i,(c,s) in enumerate(mte) if s==4]])[0,1]),3),
         rotinv_seed5=round(float(np.corrcoef(ri_te[[i for i,(c,s) in enumerate(mte) if s==5]],
                              s2te[[i for i,(c,s) in enumerate(mte) if s==5]])[0,1]),3))
json.dump(res,open(os.path.expanduser("~/s3_diag_result.json"),"w"),indent=2)
print("\n",json.dumps(res))
