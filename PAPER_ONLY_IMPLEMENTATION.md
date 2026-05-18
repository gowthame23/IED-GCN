# Paper-Only IED-GCN

This path is only for the paper-style implementation.

Files:

- [paper_ied_gcn/data.py](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/paper_ied_gcn/data.py)
- [paper_ied_gcn/model.py](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/paper_ied_gcn/model.py)
- [paper_ied_gcn/train.py](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/paper_ied_gcn/train.py)
- [run_paper_ied_gcn.py](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/run_paper_ied_gcn.py)
- [requirements-paper.txt](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/requirements-paper.txt)

Input expectation:

- `features.npy`: `(H, W, C)`
- `labels.npy`: `(H, W)` with unlabeled pixels as `-1`

Main paper-line stages:

1. SLIC superpixels
2. Internal pixel graphs
3. Internal GCN
4. External superpixel graph
5. Edge decoupling
6. Internal/external fusion
7. Superpixel classification
8. Raster reconstruction

Run in a compatible environment:

```powershell
python -m pip install -r requirements-paper.txt
python run_paper_ied_gcn.py --features paper_ied_gcn_inputs/features.npy --labels paper_ied_gcn_inputs/labels.npy
```

This path is intentionally separate from the current working fallback pipeline.
