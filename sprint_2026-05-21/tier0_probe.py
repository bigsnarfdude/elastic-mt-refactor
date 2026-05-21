#!/usr/bin/env python3
"""Tier 0 classifier probe: can we identify the collision rule from
the (S2, density)(t) trajectory it produces?

Three probes:
  1. In-distribution accuracy via leave-one-out CV (5 classifiers)
  2. OOD predictions on 4 unseen-condition trajectories with confidence
  3. Decision boundary via synthetic baseline ↔ no_zipper morphing
  4. Noise robustness on the chosen baseline

Data sources:
  - /tmp/mt_results_raw.json — trajectories extracted from nigel overnight pickles
  - /tmp/all_ood.json — OOD trajectories (baseline_isotropic, gated_v2, etc.)

To regenerate the data, run extract_s2() in step3_verify_gated.py
or the equivalent pickle-reader on nigel:~/mt_phase1/{condition}_seed{N}/.

Findings: see tier0_classifier_findings.md
"""
import json
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def features(s2, dens):
    """Hand-engineered 16-d feature vector per trajectory."""
    s2 = np.array(s2)
    d  = np.array(dens)
    n = len(s2)
    third = n // 3
    return [
        n,
        float(s2[-1]), float(s2[0]),
        float(s2.mean()), float(s2.std()),
        float(s2.max() - s2.min()),
        float((s2[-1] - s2[0]) / max(1, n - 1)),
        float(s2[:third].mean())  if third else float(s2.mean()),
        float(s2[-third:].mean()) if third else float(s2.mean()),
        float(s2[-third:].mean() - s2[:third].mean()) if third else 0.0,
        float(d[-1]),
        float(d.mean()), float(d.std()), float(d.max() - d.min()),
        float(d[-1] / max(1, d[0])),
        float(s2[-1] / max(1, d[-1] / 100)),
    ]


FEATURE_NAMES = [
    'n_pts', 's2_final', 's2_first', 's2_mean', 's2_std', 's2_range',
    's2_slope', 's2_q1_mean', 's2_q3_mean', 's2_diff_q3_q1',
    'dens_final', 'dens_mean', 'dens_std', 'dens_range',
    'dens_growth_ratio', 's2_at_density',
]


def build_xy(traj_dict, min_len=5):
    """Build (X, y, names) from {trajectory_name: {'s2_traj', 'dens_traj', ...}}."""
    X, y, names = [], [], []
    for k, v in traj_dict.items():
        if v.get('s2_final') is None or len(v.get('s2_traj', [])) < min_len:
            continue
        X.append(features(v['s2_traj'], v['dens_traj']))
        y.append('_'.join(k.split('_')[:-1]))  # strip _sNN
        names.append(k)
    return np.array(X, dtype=np.float64), np.array(y), names


def run_loo_cv(X, y):
    """Leave-one-out CV with 5 classifiers."""
    clfs = {
        'RandomForest':     RandomForestClassifier(n_estimators=200, random_state=0),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=0),
        'KNN(k=1)':         make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
        'KNN(k=3)':         make_pipeline(StandardScaler(), KNeighborsClassifier(3)),
        'LogReg':           make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=1000, random_state=0)),
    }
    loo = LeaveOneOut()
    results = {}
    for name, clf in clfs.items():
        preds, trues = [], []
        for tr, te in loo.split(X):
            clf.fit(X[tr], y[tr])
            preds.append(clf.predict(X[te])[0])
            trues.append(y[te][0])
        acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
        results[name] = dict(acc=acc, preds=preds, trues=trues)
    return results


def ood_probe(Xtr, ytr, ood_traj):
    """Classify OOD trajectories with confidence."""
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(Xtr, ytr)
    classes = list(clf.classes_)
    out = {}
    for name, v in ood_traj.items():
        if len(v.get('s2_traj', [])) < 3:
            continue
        feats = features(v['s2_traj'], v['dens_traj'])
        pred = clf.predict([feats])[0]
        proba = clf.predict_proba([feats])[0]
        out[name] = dict(pred=pred, conf=float(proba.max()),
                         dist={c: float(p) for c, p in zip(classes, proba)})
    return out, classes


def morph_probe(Xtr, ytr, traj_a, traj_b, alphas=None):
    """Classify linear interpolations a*A + (1-a)*B."""
    if alphas is None:
        alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(Xtr, ytr)
    classes = list(clf.classes_)
    a, b = np.array(traj_a['s2_traj']), np.array(traj_b['s2_traj'])
    ad, bd = np.array(traj_a['dens_traj']), np.array(traj_b['dens_traj'])
    K = min(len(a), len(b))
    a, b, ad, bd = a[:K], b[:K], ad[:K], bd[:K]
    out = []
    for alpha in alphas:
        s2_mix = (alpha * a + (1 - alpha) * b).tolist()
        d_mix  = (alpha * ad + (1 - alpha) * bd).tolist()
        feats = features(s2_mix, d_mix)
        pred = clf.predict([feats])[0]
        proba = clf.predict_proba([feats])[0]
        out.append(dict(alpha=alpha, pred=pred, conf=float(proba.max()),
                        dist={c: float(p) for c, p in zip(classes, proba)}))
    return out


def noise_robustness(Xtr, ytr, traj, sigma_pairs=None, n_trials=20, seed=42):
    """How much Gaussian noise can we add before classification changes?"""
    if sigma_pairs is None:
        sigma_pairs = [(0.0, 0), (0.01, 50), (0.02, 100), (0.05, 200),
                       (0.1, 400), (0.15, 600), (0.2, 800)]
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(Xtr, ytr)
    rng = np.random.default_rng(seed)
    s2_clean = np.array(traj['s2_traj'])
    d_clean  = np.array(traj['dens_traj'])
    out = []
    for sig_s2, sig_d in sigma_pairs:
        preds, confs = [], []
        for _ in range(n_trials):
            s2_n = np.clip(s2_clean + rng.normal(0, sig_s2, len(s2_clean)), 0, 1)
            d_n  = np.maximum(d_clean + rng.normal(0, sig_d, len(d_clean)), 1)
            feats = features(s2_n.tolist(), d_n.tolist())
            preds.append(clf.predict([feats])[0])
            confs.append(float(clf.predict_proba([feats])[0].max()))
        mode_pred = Counter(preds).most_common(1)[0]
        out.append(dict(sigma_s2=sig_s2, sigma_dens=sig_d,
                        mode_pred=mode_pred[0], mode_count=mode_pred[1],
                        mean_conf=float(np.mean(confs))))
    return out


if __name__ == '__main__':
    import sys
    train_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/mt_results_raw.json'
    ood_path   = sys.argv[2] if len(sys.argv) > 2 else '/tmp/all_ood.json'

    with open(train_path) as f:
        train = json.load(f)
    with open(ood_path) as f:
        ood = json.load(f)

    Xtr, ytr, names = build_xy(train)
    print(f"Training: n={len(ytr)}, classes={dict(Counter(ytr))}")

    print("\n--- LOO CV ---")
    cv = run_loo_cv(Xtr, ytr)
    for name, r in cv.items():
        print(f"  {name:20s} acc={r['acc']*100:5.1f}%")

    print("\n--- OOD probe ---")
    ood_out, classes = ood_probe(Xtr, ytr, ood)
    for name, r in ood_out.items():
        d = '  '.join(f'{c}={p:.2f}' for c, p in r['dist'].items())
        print(f"  {name:<28s} pred={r['pred']:<14s} conf={r['conf']:.3f}  {d}")

    print("\n--- Morph probe (baseline_s43 ↔ no_zipper_s43) ---")
    morph = morph_probe(Xtr, ytr, train['baseline_s43'], train['no_zipper_s43'])
    for r in morph:
        print(f"  α={r['alpha']:.1f}  pred={r['pred']:<14s} conf={r['conf']:.3f}")

    print("\n--- Noise robustness (baseline_s43) ---")
    noise = noise_robustness(Xtr, ytr, train['baseline_s43'])
    for r in noise:
        print(f"  σ_S2={r['sigma_s2']:.2f}  σ_dens={r['sigma_dens']:.0f}  "
              f"pred={r['mode_pred']:<14s} conf={r['mean_conf']:.3f}")
