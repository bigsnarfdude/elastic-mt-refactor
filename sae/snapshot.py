"""Turn a single order_hist() output into a fixed-size feature vector.

The simulator's ``plotting.order_hist(mt_list, t)`` returns
``(angles, lengths, mt_len, theta)`` where:
  - angles : list of MT segment angles
  - lengths: list of MT segment lengths (same len as angles)
  - mt_len : total MT length (scalar)
  - theta  : list of cell-surface angular coordinates (currently unused)

This module encodes that into a length-weighted angle histogram — the natural
fixed-size representation that lets us compare snapshots across simulations.
"""
import numpy as np
from math import pi


N_BINS = 36   # 5° resolution for nematic angles (treats θ and θ+π as same)


def snapshot_vector(geom, n_bins: int = N_BINS) -> np.ndarray:
    """Length-weighted angle histogram, normalized to a unit-sum vector.

    Treats angle and angle+π as the same orientation (nematic), so the
    histogram covers [0, π) only.

    Parameters
    ----------
    geom : (angles, lengths, mt_len, theta)
        The order_hist() output.
    n_bins : int
        Number of histogram bins covering [0, π).

    Returns
    -------
    np.ndarray of shape (n_bins,), summing to 1 (or zeros if no MTs).
    """
    angles, lengths, mt_len, _ = geom[0], geom[1], geom[2], geom[3]
    if not lengths or mt_len <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    # Fold to [0, π) for nematic equivalence
    nematic = np.mod(np.asarray(angles, dtype=np.float64), pi)
    weights = np.asarray(lengths, dtype=np.float64)
    hist, _ = np.histogram(nematic, bins=n_bins, range=(0, pi), weights=weights)
    s = hist.sum()
    return (hist / s).astype(np.float32) if s > 0 else hist.astype(np.float32)


def aggregate_regions(order_pickle_data) -> np.ndarray:
    """Aggregate a per-time, per-region order_hist into a single (T, n_bins) array.

    Parameters
    ----------
    order_pickle_data : (order, order_t)
        The contents of an ``orderp_seed*_*.pickle`` file.

    Returns
    -------
    snapshots : np.ndarray of shape (n_time, n_bins)
        One length-weighted angle histogram per saved time point, pooled
        across all spatial regions.
    """
    order, _ = order_pickle_data
    snapshots = []
    for region_list in order:
        # Pool by summing length-weighted contributions across regions, then renormalize
        combined = np.zeros(N_BINS, dtype=np.float32)
        total = 0.0
        for geom in region_list:
            if geom and len(geom) >= 3 and geom[2] > 0:
                snap = snapshot_vector(geom)
                weight = geom[2]
                combined += snap * weight
                total += weight
        if total > 0:
            combined /= total
        snapshots.append(combined)
    return np.array(snapshots, dtype=np.float32)


def load_trajectory(pickle_path) -> np.ndarray:
    """Load one orderp_*.pickle and convert to a (T, n_bins) snapshot array.

    Returns one aggregated histogram per saved time point, pooled across
    spatial regions. Good for trajectory-level analysis.
    """
    import pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return aggregate_regions(data)


def load_per_region(pickle_path, min_mt_len=0.01) -> tuple:
    """Load per-region snapshots — each spatial region becomes its own sample.

    This gives orders of magnitude more training data than ``load_trajectory``
    because each saved time point contributes ~grid_l × grid_w regions instead
    of one aggregated histogram.

    Parameters
    ----------
    pickle_path : Path
        Path to an orderp_*.pickle file.
    min_mt_len : float
        Skip regions with less than this total MT length. Avoids feeding
        empty / near-empty regions to the SAE (their histograms are all-zero
        and uninformative).

    Returns
    -------
    snaps : np.ndarray of shape (N_regions, n_bins)
        Per-region length-weighted histograms.
    meta : list of dict
        Per-snapshot metadata: {'time': float, 'region_idx': int, 'mt_len': float}
    """
    import pickle
    with open(pickle_path, 'rb') as f:
        order, order_t = pickle.load(f)

    snaps, meta = [], []
    for t_idx, region_list in enumerate(order):
        t = float(order_t[t_idx]) if t_idx < len(order_t) else float('nan')
        for r_idx, geom in enumerate(region_list):
            if not (geom and len(geom) >= 3 and geom[2] > min_mt_len):
                continue
            snaps.append(snapshot_vector(geom))
            meta.append({'time': t, 'region_idx': r_idx, 'mt_len': float(geom[2])})
    return np.array(snaps, dtype=np.float32), meta
