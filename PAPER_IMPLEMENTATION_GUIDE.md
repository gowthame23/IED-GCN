# Paper-Oriented IED-GCN Path

This project currently has two IED-GCN-related paths:

1. `train_ied_gcn.py`
   Working fallback used by the current one-click pipeline.
   It is operational in the current Python 3.14 environment but is not a true paper implementation.

2. `ied_gcn_paper_closer.py`
   A paper-closer prototype that follows the paper structure more closely:
   - SLIC superpixels
   - internal pixel graphs
   - external superpixel graph
   - edge decoupling
   - internal/external fusion

## Why this is separate

The current environment is Python 3.14. The true paper-style stack depends on packages that are not available here:

- `torch`
- `torch-geometric`
- `scikit-image`

Because of that, the one-click workflow stays on the fallback path for now.

## Files

- [prepare_paper_ied_gcn_inputs.py](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/prepare_paper_ied_gcn_inputs.py)
- [ied_gcn_paper_closer.py](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/ied_gcn_paper_closer.py)
- [paper_ied_gcn_inputs](/C:/Users/gowth/OneDrive/Desktop/ADD_PV/paper_ied_gcn_inputs)

## How to use in a compatible environment

Recommended Python version:

- `Python 3.10` or `Python 3.11`

Then install:

```powershell
python -m pip install numpy rasterio scikit-learn scipy pandas matplotlib pyproj scikit-image
python -m pip install torch torchvision torchaudio
python -m pip install torch-geometric
```

Prepare inputs:

```powershell
python prepare_paper_ied_gcn_inputs.py
```

Run the paper-closer prototype:

```powershell
python ied_gcn_paper_closer.py
```

## What this improves

Compared with the fallback path, this paper-closer path adds:

- true SLIC superpixel segmentation
- internal graph convolution over pixels inside each superpixel
- external graph convolution over adjacent superpixels
- edge decoupling
- fused superpixel classification

## Current limitation

This is still a practical reproduction, not the authors' official code. It is closer to the paper structure, but not guaranteed to be identical to every implementation detail in the publication.
