"""S3.5 (corrected) — reconstruct h from top-k atoms INCLUDING the decoder bias,
then apply the real s2_from_snapshot to the reconstruction. Answers: how many atoms
does the rotation-invariant order S2 actually need in this learned basis?
"""
import glob, os, pickle, numpy as np, json, sys
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
Xtr,_=load([1,2,3]); Xte,mte=load([4,5])
s2te=np.array([s2_from_snapshot(x) for x in Xte])
sae=SAE(input_dim=36,n_features=8,sparsity_coef=0.0005,seed=0)
rng=np.random.default_rng(0)
for e in range(250):
    p=rng.permutation(len(Xtr))
    for i in range(0,len(Xtr),128): sae.step(Xtr[p[i:i+128]],lr=2e-3)
Hte=sae.encode(Xte); Wd=sae.W_d; bd=sae.b_d
live=list(np.where(Hte.sum(0)>0)[0])
b=np.arange(N_BINS); e2=np.exp(2j*(b+0.5)*np.pi/N_BINS)
c2=np.abs(Wd.T@e2); order=sorted(live,key=lambda f:-c2[f])
def recon_s2(atoms):
    H=Hte.copy()
    mask=np.zeros(H.shape[1],bool); mask[atoms]=True
    R=(H*mask)@Wd.T + bd            # (N,36) reconstruction w/ bias
    return np.array([s2_from_snapshot(r) for r in R])
def rr(v):
    d={'all':round(float(np.corrcoef(v,s2te)[0,1]),3)}
    for sd in [4,5]:
        idx=[i for i,(c,s) in enumerate(mte) if s==sd]; d[sd]=round(float(np.corrcoef(v[idx],s2te[idx])[0,1]),3)
    return d
res={}
for k in [1,2,3,len(order)]:
    v=recon_s2(order[:k]); res[f"k={k}"]={"atoms":[int(x) for x in order[:k]],**rr(v)}
    print(f"k={k} atoms={order[:k]}: {rr(v)}")
# also full reconstruction (all live) as ceiling
print("\nceiling (full SAE recon -> s2):", rr(recon_s2(live)))
json.dump(res,open(os.path.expanduser("~/s3_atoms_v2_result.json"),"w"),indent=2)
print(json.dumps(res))
