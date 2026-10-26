# IED-GCN landslide susceptibility: paper model vs our model

Reimplementation of *"IED-GCN: An Internal and External Decoupled Graph Convolutional Network for
Landslide Susceptibility Assessment"* (Li et al., IEEE TGRS, 2025; `ADD_PROJECT.pdf`), plus a variant
with two changes of our own. Both models run on the same data, superpixels, splits and settings, so
the comparison isolates what the changes do.

This is a **static susceptibility map** (which places are prone to landslides). It does not use live
data and is not an early-warning system.

## Folder

```
ADD_PROJECT.pdf               the paper
_extracted_dataset/           raw factor rasters (16 layers, 28.5 m grid)
Global_Landslide_Catalog_...  NASA landslide catalog export (not used by the models)
ied_gcn_project/
  data/                       features.npy (556x1885x16), labels.npy (1523 landslide + 1523 non-landslide pixels)
  common.py                   shared code: superpixels, graphs, network, training, metrics, maps
  paper_model.py              MODEL 1: the paper as published
  ours_model.py               MODEL 2: paper + our two changes
  run_paper.py                click Run -> trains the paper model     -> results/paper/
  run_ours.py                 click Run -> trains our model           -> results/ours/
  run_compare.py              click Run (after both) -> results/comparison.md
  run_experiments.py, compare.py   the code behind the three run files
```

## How to run

Open a run file in VS Code and click Run (or from the `ADD_PV` folder:
`.venv\Scripts\python.exe ied_gcn_project\run_paper.py`). Each model trains 5 seeds on 2 splits.
Each model takes about 20 minutes. Needs the
packages in `ied_gcn_project/requirements.txt`; the project's `.venv` already has them.

Splits: **random** = the paper's random 70/30 split; **spatial** = whole ~4 km blocks held out, a
stricter test that the paper does not report.

## The two models

| | Paper model | Our model |
|---|---|---|
| Superpixels | SLIC, np=10 (about 116k) | same |
| Network | 1 internal GCN layer, mean pool, 4 external GCN layers, edge decoupling 10%, H_in + H_ex, MLP | same |
| Training | Adam 1e-3, weight decay 1e-3, L1, gradient clipping, 150 epochs | same |
| Categorical factors (aspect, land use, lithology) | fed as numbers | one-hot encoded |
| Five map classes | natural breaks on the probability | natural breaks on the logit of the probability |

## Results (mean over 5 seeds; full tables in `ied_gcn_project/results/comparison.md`)

| Model | Split | AUC | Accuracy | Recall | Kappa |
|---|---|---|---|---|---|
| Paper, as published (different data) | random | 0.9838 | 0.9838 | 0.9907 | 0.9675 |
| Paper model (ours, reproduced) | random | 0.9998 | 0.9949 | 0.9961 | 0.9894 |
| Our model | random | 0.9999 | 0.9949 | 0.9956 | 0.9894 |
| Paper model | spatial | 0.9991 | 0.9857 | 0.9755 | 0.9691 |
| Our model | spatial | 0.9997 | 0.9900 | 0.9844 | 0.9777 |

What the results show:

- The reproduction reaches the paper's level on the scores. Trends match the paper: np=10 is far better than
  large superpixels, and internal + external graphs beat the internal graph alone.
- **Our model is not more accurate than the paper model.** On the random split the scores are identical
  (kappa 0.9894 both). On the spatial split ours is about 0.9 points higher in kappa (0.9777 vs 0.9691), but the
  run-to-run spread is 1-3 points, so this is a hint, not a proof. An ensemble of 3 networks was also tried and
  gave no measurable gain, so it was dropped.
- **The map is where our model differs.** Landslide density on held-out landslides (random split, seed 0, measured
  over the real study area) rises 0.00, 0.01, 0.05, 0.61, 4.15 across the five classes for our model against
  0.00, 0.00, 0.07, 0.05, 2.35 for the paper model.
- **Both maps are far less selective than the paper's.** High + Very High covers 38.4% of the study area for our
  model and 50.4% for the paper model; the paper reports 10.25% (holding 97.81% of landslides). This is the main
  weakness of the project (see caveats).

## Caveats

- About 54% of the raster rectangle is blank no-data (every factor 0: empty corners and the river surface). The real
  study area is 481,929 pixels = 391 km2, close to the paper's 396 km2. Map statistics are computed over it only.
- Our scores are higher than the paper's mostly because our data is easier, not because the model is
  better. A plain logistic regression reaches AUC 0.998 and a random forest matches the IED-GCN on the
  random split (reference rows in `comparison.md`).
- The paper's other claims (edge-discard sweep, sample-ratio sweep, CNN/ResNet/DenseNet/miniGCN
  baselines, redrawn non-landslide samples) are not reproduced here.
- Edge decoupling showed no gain on this data, unlike in the paper.
- Training and test samples come from one study area. The model must be retrained for any other area.
