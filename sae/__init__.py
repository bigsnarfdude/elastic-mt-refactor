"""Sparse autoencoder pipeline for cortical MT trajectory features.

The hypothesis: an SAE trained on per-snapshot MT geometry should rediscover
the order parameters biologists hand-design (nematic S₂, dominant angle Ω,
defect density, etc.) — and potentially find features they didn't.

Pipeline:
  1. ``snapshot.py`` — turn an order_hist() output into a fixed-size feature vector
  2. ``model.py``    — minimal SAE in numpy (no torch dependency for portability)
  3. ``train.py``    — train on N trajectory snapshots, save weights
  4. ``probe.py``    — for each learned feature, report what it correlates with
"""
