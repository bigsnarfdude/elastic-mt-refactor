"""Does the S2-recovery ceiling rise with SAE capacity? Theory says NO: S2 lives in a
2D subspace (2nd Fourier harmonic), so 2 spanning atoms are exact and more can't beat it.
Train SAE at n_features in {8,32,64}; report the k-atom recovery curve + the ceiling."""
import glob, os, pickle, numpy as np, sys
sys.path.insert(0, os.path.expanduser("~/elastic-mt-refactor"))
from sae.snapshot import aggregate_regions, N_BINS
from sae.probe import s2_from_snapshot
from sae.model import SAE
root=os.path.expanduser("~/mt_ablation")
CONDS=["baseline","no_zipper","zipper_to_cross","no_induced_catas","always_catas"]
def load(seeds):
    X=[]
    for c in CONDS:
        for s in seeds:
            for f in sorted(glob.glob(f"{root}/{c}_seed{s}/orderp_seed{s}_*.pickle")):
                try:d=pickle.load(open(f,"rb"))
                except Exception:continue
                a=aggregate_regions(d)
                if a is None:continue
                for row in np.atleast_2d(np.asarray(a,dtype=np.float32)):
                    if row.sum()>0:X.append(row)
    return np.array(X,dtype=np.float32)
Xtr=load([1,2,3]); Xte=load([4,5])
s2te=np.array([s2_from_snapshot(x) for x in Xte])
b=np.arange(N_BINS); e2=np.exp(2j*(b+0.5)*np.pi/N_BINS)
def train(nf,epochs=300,sc=0.0005):
    sae=SAE(input_dim=36,n_features=nf,sparsity_coef=sc,seed=0); rng=np.random.default_rng(0)
    for e in range(epochs):
        p=rng.permutation(len(Xtr))
        for i in range(0,len(Xtr),128): sae.step(Xtr[p[i:i+128]],lr=2e-3)
    return sae
for nf in [8,32,64]:
    sae=train(nf); Hte=sae.encode(Xte); Wd=sae.W_d; bd=sae.b_d
    live=list(np.where(Hte.sum(0)>0)[0])
    c2=np.abs(Wd.T@e2); order=sorted(live,key=lambda f:-c2[f])
    # how concentrated is the 2nd-harmonic energy?
    cc=np.array([c2[f] for f in order]); frac=cc/cc.sum() if cc.sum()>0 else cc
    def recon_r(k):
        mask=np.zeros(Hte.shape[1],bool); mask[order[:k]]=True
        R=(Hte*mask)@Wd.T+bd
        v=np.array([s2_from_snapshot(r) for r in R])
        return round(float(np.corrcoef(v,s2te)[0,1]),3)
    curve={f"k={k}":recon_r(k) for k in range(1,min(7,len(order)+1))}
    ceil=max(curve.values())
    print(f"nf={nf:2d}: live={len(live):2d}  2nd-harm energy in top-2 atoms={frac[:2].sum()*100:.0f}%  "
          f"ceiling={ceil}  curve={curve}")
