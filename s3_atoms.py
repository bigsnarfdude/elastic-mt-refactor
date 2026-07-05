"""S3.5 — are the SAE's live atoms the nematic Fourier components?
If S2 = |2nd-harmonic| of the angle histogram, and the SAE reconstructs h from a few atoms,
test whether the 2nd-harmonic content concentrates in ~2 atoms (the cos2th/sin2th director
components) so that a 2-ATOM rotation-invariant readout recovers S2 seed-stably.
Uses only the reconstruction identity (no fitted probe weights) -> interpretable, not dense.
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
Xtr,_=load([1,2,3]); Xte,mte=load([4,5])
s2te=np.array([s2_from_snapshot(x) for x in Xte])
sae=SAE(input_dim=36,n_features=8,sparsity_coef=0.0005,seed=0)
rng=np.random.default_rng(0)
for e in range(250):
    p=rng.permutation(len(Xtr))
    for i in range(0,len(Xtr),128): sae.step(Xtr[p[i:i+128]],lr=2e-3)
Hte=sae.encode(Xte)
live=np.where(Hte.sum(0)>0)[0]
# Fourier basis over nematic angle bins theta_b in [0,pi)
b=np.arange(N_BINS); th=(b+0.5)*np.pi/N_BINS
e2=np.exp(2j*th)                       # 2nd harmonic
Wd=sae.W_d                             # (36, nf): decoder columns = atom shapes
dc=Wd.sum(0)                           # DC content per atom
c2=Wd.T@e2                             # complex 2nd-harmonic coeff per atom  (nf,)
print("=== live atoms: 2nd-harmonic content |c2| and DC ===")
order=sorted(live, key=lambda f:-abs(c2[f]))
for f in order:
    print(f" atom {f}: |c2|={abs(c2[f]):.3f} angle(c2)/2={np.degrees(np.angle(c2[f])/2)%180:5.1f}deg  DC={dc[f]:.3f}")
# k-atom rotation-invariant readout via the reconstruction identity:
# S2hat = |sum_f a_f c2_f| / |sum_f a_f dc_f|
def s2hat(atoms):
    num=np.abs(Hte[:,atoms]@c2[atoms]); den=np.abs(Hte[:,atoms]@dc[atoms])+1e-9
    return num/den
def rperseed(v):
    out={}
    for sd in [4,5]:
        idx=[i for i,(c,s) in enumerate(mte) if s==sd]
        out[sd]=round(float(np.corrcoef(v[idx],s2te[idx])[0,1]),3)
    return out
res={}
for k in [1,2,3,len(order)]:
    a=order[:k]; v=s2hat(a); r=round(float(np.corrcoef(v,s2te)[0,1]),3)
    res[f"k={k}"]=dict(atoms=[int(x) for x in a], test_r=r, per_seed=rperseed(v))
    print(f" k={k} atoms={a}: test_r={r} per_seed={rperseed(v)}")
json.dump(res,open(os.path.expanduser("~/s3_atoms_result.json"),"w"),indent=2)
print("\nRESULT:",json.dumps(res))
