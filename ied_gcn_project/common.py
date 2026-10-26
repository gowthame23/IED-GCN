"""Shared building blocks for the paper model and our model: data, graphs, network, metrics, maps.

Both models use exactly the same data, superpixels, splits and training loop. They differ only in
what their own module (paper_model.py / our_model.py) changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic
from sklearn import metrics as skm

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
RESULTS_DIR = PROJECT / "results"

# Order of the 16 channels in data/features.npy.
FACTORS = ["aspect", "elevation", "fault", "landuse", "lithology", "ndvi", "plan", "profile",
           "rainfall", "river", "road", "rough", "slope", "spi", "terrain", "twi"]
CATEGORICAL = ["aspect", "landuse", "lithology"]
CLASS_NAMES = ["VeL", "Lo", "Mo", "Hi", "VeH"]
SPATIAL_BLOCK_PX = 140  # ~4 km blocks at 28.5 m, for the spatial hold-out split

# Paper hyper-parameters (Sec. III-D): 128 hidden units, 1 internal + 4 external layers, Adam 1e-3,
# weight decay 1e-3, 150 epochs, edge discard 0.1, np=10, gradient clipping, L1 penalty.
@dataclass
class HyperParams:
    np_per_superpixel: int = 10
    compactness: float = 1.0
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-3
    edge_drop: float = 0.1
    l1_lambda: float = 1e-5
    clip: float = 2.0
    hidden: int = 128
    ext_layers: int = 4
    train_fraction: float = 0.7


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Return min-max scaled factors (H, W, 16) and labels (H, W) with -1 = unlabeled."""
    features = np.load(DATA_DIR / "features.npy").astype(np.float32)
    labels = np.load(DATA_DIR / "labels.npy").astype(np.int64)
    lo = features.min(axis=(0, 1), keepdims=True)
    hi = features.max(axis=(0, 1), keepdims=True)
    return (features - lo) / np.maximum(hi - lo, 1e-8), labels


def one_hot_categoricals(features: np.ndarray) -> np.ndarray:
    """Replace the categorical channels (aspect, landuse, lithology) by one indicator channel per class."""
    keep = [i for i, name in enumerate(FACTORS) if name not in CATEGORICAL]
    channels = [features[..., i] for i in keep]
    for name in CATEGORICAL:
        channel = features[..., FACTORS.index(name)]
        values = np.unique(channel)
        if len(values) > 20:
            raise ValueError(f"{name} has {len(values)} distinct values; expected a categorical raster")
        channels += [(channel == v).astype(np.float32) for v in values]
    return np.stack(channels, -1)


def valid_pixels(features: np.ndarray) -> np.ndarray:
    """True where the pixel is inside the study area. The raster rectangle is padded with blank
    filler (every factor at its minimum, 0 after scaling); about 54% of it. No label lies there."""
    return ~(features == 0).all(axis=-1)


@dataclass
class GraphData:
    segments: np.ndarray  # (H, W) superpixel id per pixel
    n: int
    ax: torch.Tensor  # (H*W, C) model features after one internal propagation step (A_hat X)
    flat: torch.Tensor  # (H*W,) superpixel id per pixel
    counts: torch.Tensor  # (n, 1) pixels per superpixel
    ext_edges: torch.Tensor  # (2, E) superpixel adjacency, both directions
    sp_labels: np.ndarray  # (n,) label per superpixel, -1 when unlabeled
    centroids: np.ndarray  # (n, 2) row/col centroid
    mixed_label_superpixels: int


def build_graph_data(slic_features: np.ndarray, model_features: np.ndarray, labels: np.ndarray, hp: HyperParams) -> GraphData:
    """Superpixels are always cut from `slic_features`, so both models see identical superpixels."""
    h, w, _ = slic_features.shape
    c = model_features.shape[2]
    segments = slic(slic_features, n_segments=max(1, h * w // hp.np_per_superpixel), compactness=hp.compactness,
                    start_label=0, channel_axis=-1, enforce_connectivity=True)
    n = int(segments.max()) + 1
    flat = segments.ravel()
    counts = np.bincount(flat, minlength=n)

    # Internal graph (Fig. 5): 4-neighbour pixel edges inside a superpixel, plus self loops.
    idx = np.arange(h * w).reshape(h, w)
    src, dst = [], []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        a, b = a.ravel(), b.ravel()
        keep = flat[a] == flat[b]
        src += [a[keep], b[keep]]
        dst += [b[keep], a[keep]]
    src = np.concatenate(src + [np.arange(h * w)])
    dst = np.concatenate(dst + [np.arange(h * w)])
    deg = np.bincount(src, minlength=h * w).astype(np.float32)
    weight = torch.tensor((deg[src] ** -0.5) * (deg[dst] ** -0.5))
    xt = torch.tensor(model_features.reshape(-1, c))
    # With K=1 (Eq. 6) A_hat X W == (A_hat X) W, so A_hat X is computed once.
    ax = torch.zeros_like(xt).index_add_(0, torch.tensor(dst), xt[torch.tensor(src)] * weight[:, None])

    # External graph (Fig. 6): superpixels that touch.
    pairs = []
    for a, b in ((segments[:, :-1], segments[:, 1:]), (segments[:-1, :], segments[1:, :])):
        diff = a != b
        pairs.append(np.stack([a[diff], b[diff]], 0))
    pairs = np.concatenate(pairs, 1).astype(np.int64)
    pairs = np.concatenate([pairs, pairs[::-1]], 1)
    ext = np.unique(pairs[0] * n + pairs[1])
    ext_edges = torch.tensor(np.stack([ext // n, ext % n], 0))

    sp_labels = np.full(n, -1, dtype=np.int64)
    mixed = 0
    rows, cols = np.nonzero(labels >= 0)
    for sid, val in zip(segments[rows, cols], labels[rows, cols]):
        if sp_labels[sid] >= 0 and sp_labels[sid] != val:
            mixed += 1
        sp_labels[sid] = val
    grid_r, grid_c = np.repeat(np.arange(h), w), np.tile(np.arange(w), h)
    centroids = np.stack([np.bincount(flat, grid_r, n) / counts, np.bincount(flat, grid_c, n) / counts], 1)
    return GraphData(segments, n, ax, torch.tensor(flat), torch.tensor(counts, dtype=torch.float32)[:, None],
                     ext_edges, sp_labels, centroids, mixed)


def valid_superpixels(g: GraphData, valid: np.ndarray) -> np.ndarray:
    """A superpixel counts as inside the study area when most of its pixels do."""
    inside = np.bincount(g.segments.ravel(), weights=valid.ravel().astype(float), minlength=g.n)
    return inside / g.counts.numpy().ravel() > 0.5


def normalized_adjacency(edges: torch.Tensor, n: int) -> torch.Tensor:
    index = torch.cat([edges, torch.arange(n).repeat(2, 1)], 1)
    value = torch.ones(index.shape[1])
    inv_sqrt = torch.zeros(n).scatter_add_(0, index[0], value).pow(-0.5)
    return torch.sparse_coo_tensor(index, inv_sqrt[index[0]] * value * inv_sqrt[index[1]], (n, n)).coalesce()


class IEDGCN(nn.Module):
    """Internal GCN layer -> mean pool -> 4 external GCN layers (edge decoupling) -> H_in + H_ex -> MLP."""

    def __init__(self, in_channels: int, hp: HyperParams):
        super().__init__()
        self.edge_drop = hp.edge_drop
        self.internal = nn.Linear(in_channels, hp.hidden)
        self.external = nn.ModuleList([nn.Linear(hp.hidden, hp.hidden) for _ in range(hp.ext_layers)])
        self.classifier = nn.Sequential(nn.Linear(hp.hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, g: GraphData) -> torch.Tensor:
        pixel = F.relu(self.internal(g.ax))
        h_in = torch.zeros(g.n, pixel.shape[1]).index_add_(0, g.flat, pixel) / g.counts  # Eq. 5, mean pool
        edges = g.ext_edges
        if self.training and self.edge_drop > 0:  # edge decoupling, Eqs. 11-12
            edges = edges[:, torch.rand(edges.shape[1]) > self.edge_drop]
        adj = normalized_adjacency(edges, g.n)
        h_ex = h_in
        for layer in self.external:
            h_ex = F.relu(torch.sparse.mm(adj, layer(h_ex)))  # Eq. 10
        return self.classifier(h_in + h_ex)  # Eqs. 13-14


def train_and_predict(g: GraphData, train_idx: np.ndarray, hp: HyperParams, seed: int, tag: str = "") -> np.ndarray:
    """Train one IED-GCN and return P(landslide) for every superpixel."""
    torch.manual_seed(seed)
    model = IEDGCN(g.ax.shape[1], hp)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    y, train_t, started = torch.tensor(g.sp_labels), torch.tensor(train_idx), time.time()
    for epoch in range(hp.epochs):
        model.train()
        loss = F.cross_entropy(model(g)[train_t], y[train_t]) + hp.l1_lambda * sum(p.abs().sum() for p in model.parameters())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
        optimizer.step()
        if epoch % 50 == 0 or epoch == hp.epochs - 1:
            print(f"  {tag} epoch {epoch:03d} loss {loss.item():.4f} ({time.time() - started:.0f}s)", flush=True)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(g), 1)[:, 1].numpy()


def split_nodes(g: GraphData, split: str, seed: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """'random' = paper protocol (random 70/30). 'spatial' = whole ~4 km blocks held out."""
    labeled = np.flatnonzero(g.sp_labels >= 0)
    rng = np.random.default_rng(seed)
    if split == "random":
        order = rng.permutation(labeled)
        k = int(len(order) * train_fraction)
        return order[:k], order[k:]
    block = (g.centroids[:, 0] // SPATIAL_BLOCK_PX).astype(int) * 1000 + (g.centroids[:, 1] // SPATIAL_BLOCK_PX).astype(int)
    blocks = np.unique(block[labeled])
    held = rng.choice(blocks, max(1, int(round(len(blocks) * (1 - train_fraction)))), replace=False)
    is_test = np.isin(block[labeled], held)
    return labeled[~is_test], labeled[is_test]


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = prob > 0.5
    return {
        "auc": float(skm.roc_auc_score(y_true, prob)),
        "oa": float(skm.accuracy_score(y_true, pred)),
        "precision": float(skm.precision_score(y_true, pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, pred, zero_division=0)),
        "mcc": float(skm.matthews_corrcoef(y_true, pred)),
        "kappa": float(skm.cohen_kappa_score(y_true, pred)),
    }


def jenks_breaks(values: np.ndarray, n_classes: int = 5, max_points: int = 1500) -> np.ndarray:
    """Fisher-Jenks natural breaks (the paper's classing) on a quantile subsample; returns inner thresholds."""
    data = np.sort(np.asarray(values, dtype=np.float64))
    if data.size > max_points:
        data = np.quantile(data, np.linspace(0, 1, max_points))
    n = data.size
    cum1, cum2 = np.concatenate([[0.0], np.cumsum(data)]), np.concatenate([[0.0], np.cumsum(data**2)])

    def sse(i: int, j: int) -> float:
        s1 = cum1[j] - cum1[i]
        return (cum2[j] - cum2[i]) - s1 * s1 / (j - i)

    cost = np.full((n_classes + 1, n + 1), np.inf)
    back = np.zeros((n_classes + 1, n + 1), dtype=np.int64)
    for j in range(1, n + 1):
        cost[1, j] = sse(0, j)
    for k in range(2, n_classes + 1):
        for j in range(k, n + 1):
            best, arg = np.inf, k - 1
            for i in range(k - 1, j):
                v = cost[k - 1, i] + sse(i, j)
                if v < best:
                    best, arg = v, i
            cost[k, j], back[k, j] = best, arg
    cuts, j = [], n
    for k in range(n_classes, 1, -1):
        j = back[k, j]
        cuts.append(data[j])
    return np.array(sorted(cuts))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def landslide_density(class_map: np.ndarray, label_pixels: tuple[np.ndarray, np.ndarray], valid: np.ndarray) -> list[dict]:
    """Paper Eq. 21: LD = PL / PC per susceptibility class. PC is measured over the study area only."""
    area = np.bincount(class_map[valid], minlength=5) / valid.sum()
    hit = np.bincount(class_map[label_pixels], minlength=5)
    share = hit / max(1, hit.sum())
    return [{"class": CLASS_NAMES[i], "PC": float(area[i]), "PL": float(share[i]),
             "LD": float(share[i] / area[i]) if area[i] > 0 else float("nan")} for i in range(5)]


def save_map_png(path: Path, class_map: np.ndarray, positives: tuple[np.ndarray, np.ndarray], title: str, valid: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("white")  # blank filler outside the study area
    image = ax.imshow(np.ma.masked_where(~valid, class_map), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.scatter(positives[1], positives[0], s=1, c="black", alpha=0.4, label="mapped landslide samples")
    ax.set_xticks([]), ax.set_yticks([]), ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, ticks=range(5))
    cbar.ax.set_yticklabels(CLASS_NAMES)
    ax.legend(loc="lower right", markerscale=6)
    fig.savefig(path, dpi=200)
    plt.close(fig)
