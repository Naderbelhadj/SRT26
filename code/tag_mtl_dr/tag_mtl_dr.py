"""TAG-MTL-DR: topology-aware graph, multi-task diabetic retinopathy grading.

Corrected version of the original prototype. The design is kept -- EfficientNet-B3
lesion features, a U-Net vessel branch, a GAT over the vessel graph, cross-attention
fusion and an ordinal loss -- but each part now does what its name says:

* multi-task: the U-Net is supervised (BCE + Dice) by vessel masks, either annotated
  (DRIVE/FIVES/CHASE) or classical pseudo-labels (``pseudo_vessel_mask``); the original
  U-Net received no gradient at all;
* a real U-Net (skip connections);
* a topological vessel graph: nodes are skeleton end-points and junctions, edges are the
  vessel segments between them (8-connectivity tracing), with normalised features;
  the original linked pixels in raster order;
* one graph per image, batched with ``torch_geometric.data.Batch`` (the original used the
  graph of the first image for the whole batch);
* cross-attention between the spatial lesion tokens (one per feature-map cell) and the
  graph nodes; with a single key, as in the original, softmax = 1 and the lesion
  features were discarded;
* a CORAL ordinal head (rank-consistent) by default; the CE + MSE loss is kept as an option;
* ImageNet normalisation, QWK / Sev-NR / PDR-NR evaluation, patient-grouped folds.

Run ``python tag_mtl_dr.py --smoke`` to check the whole pipeline on random data.
"""

from __future__ import annotations

import argparse
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import exposure, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

NUM_CLASSES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NODE_FEATS = 5          # y, x, degree/4, is_end, is_junction
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ============================================================ vessel pseudo-labels
def pseudo_vessel_mask(rgb: np.ndarray, radius: int = 7, min_size: int = 50) -> np.ndarray:
    """Classical vessel mask (green channel, CLAHE, black top-hat, Otsu).

    Use it as a training target when no annotated vessel masks are available.
    ``rgb``: H x W x 3, uint8 or float in [0, 1].
    """
    g = rgb[..., 1].astype(np.float32)
    g = g / 255.0 if g.max() > 1.0 else g
    g = exposure.equalize_adapthist(g, clip_limit=0.02)
    th = morphology.black_tophat(g, morphology.disk(radius))   # vessels are darker than background
    if th.max() <= 0:
        return np.zeros(g.shape, bool)
    m = th > threshold_otsu(th)
    try:                                    # scikit-image >= 0.26
        return morphology.remove_small_objects(m, max_size=min_size - 1)
    except TypeError:
        return morphology.remove_small_objects(m, min_size=min_size)


# ============================================================ vessel graph
def skeleton_graph(mask: np.ndarray, max_nodes: int = 256) -> Data:
    """Topological graph of a binary vessel mask.

    Nodes: skeleton end-points (degree 1) and junctions (degree >= 3); a closed loop
    without such points gets one node. Edges: skeleton segments between two nodes,
    with the segment length as edge attribute. Node features: normalised position,
    degree / 4, end-point and junction flags. If the graph has more than ``max_nodes``
    nodes, the largest connected components are kept.
    """
    H, W = mask.shape
    skel = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(skel)
    empty = Data(x=torch.zeros(1, NODE_FEATS), edge_index=torch.zeros(2, 0, dtype=torch.long),
                 edge_attr=torch.zeros(0, 1))
    if len(ys) < 2:
        return empty

    pix = {(y, x): i for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist()))}
    nbrs = [[pix[(y + dy, x + dx)] for dy, dx in _OFFS if (y + dy, x + dx) in pix]
            for y, x in zip(ys.tolist(), xs.tolist())]
    deg = np.array([len(n) for n in nbrs])
    is_key = deg != 2

    # closed loops without end-points or junctions: mark one pixel per loop
    seen = np.zeros(len(ys), bool)
    for s in range(len(ys)):
        if seen[s]:
            continue
        comp, q, has_key = [], deque([s]), False
        seen[s] = True
        while q:
            u = q.popleft()
            comp.append(u)
            has_key |= bool(is_key[u])
            for v in nbrs[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        if not has_key:
            is_key[comp[0]] = True

    key_ids = np.flatnonzero(is_key)
    node_of = {int(p): k for k, p in enumerate(key_ids)}

    # trace every segment from each key pixel
    edges, used = {}, set()
    for p in key_ids.tolist():
        for first in nbrs[p]:
            if (p, first) in used:
                continue
            prev, cur, length = p, first, 1
            used.add((p, first))
            while not is_key[cur]:
                nxt = [v for v in nbrs[cur] if v != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                length += 1
            used.add((cur, prev))
            if is_key[cur]:
                a, b = node_of[p], node_of[cur]
                if a != b:
                    k = (min(a, b), max(a, b))
                    edges[k] = min(edges.get(k, length), length)

    n = len(key_ids)
    keep = np.arange(n)
    if n > max_nodes:                                  # keep the largest components
        parent = list(range(n))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, b in edges:
            parent[find(a)] = find(b)
        roots = np.array([find(i) for i in range(n)])
        order = sorted(set(roots.tolist()), key=lambda r: -(roots == r).sum())
        chosen = []
        for r in order:
            members = np.flatnonzero(roots == r).tolist()
            if len(chosen) + len(members) > max_nodes:
                members = members[: max_nodes - len(chosen)]
            chosen += members
            if len(chosen) >= max_nodes:
                break
        keep = np.array(sorted(chosen))
    remap = -np.ones(n, int)
    remap[keep] = np.arange(len(keep))

    kp = key_ids[keep]
    feats = np.stack([ys[kp] / H, xs[kp] / W, np.minimum(deg[kp], 4) / 4.0,
                      (deg[kp] == 1).astype(float), (deg[kp] >= 3).astype(float)], 1)
    src, dst, attr = [], [], []
    for (a, b), L in edges.items():
        if remap[a] >= 0 and remap[b] >= 0:
            src += [remap[a], remap[b]]
            dst += [remap[b], remap[a]]
            attr += [L / max(H, W)] * 2
    return Data(x=torch.tensor(feats, dtype=torch.float),
                edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
                edge_attr=torch.tensor(attr, dtype=torch.float).reshape(-1, 1))


# ============================================================ networks
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net with skip connections; returns vessel logits (B, 1, H, W)."""

    def __init__(self, c=(32, 64, 128)):
        super().__init__()
        self.e1, self.e2, self.b = ConvBlock(3, c[0]), ConvBlock(c[0], c[1]), ConvBlock(c[1], c[2])
        self.d2, self.d1 = ConvBlock(c[2] + c[1], c[1]), ConvBlock(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        b = self.b(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class GraphEncoder(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.g1 = GATConv(NODE_FEATS, dim // heads, heads=heads, edge_dim=1)
        self.g2 = GATConv(dim, dim, heads=1, edge_dim=1)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, data: Batch):
        h = F.elu(self.n1(self.g1(data.x, data.edge_index, data.edge_attr)))
        h = self.n2(self.g2(h, data.edge_index, data.edge_attr))
        return h


class CoralHead(nn.Module):
    """K-1 rank logits sharing one weight vector (rank-consistent, Cao et al. 2020)."""

    def __init__(self, dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, h):
        return self.w(h) + self.b


class TAGMTLDR(nn.Module):
    def __init__(self, pretrained=True, dim=256, heads=4, max_nodes=256, head="coral",
                 use_graph=True, fusion="cross", backbone="efficientnet_b3"):
        """Ablation switches: ``use_graph=False`` removes the vessel graph (and the
        cross-attention); ``fusion='concat'`` replaces the cross-attention by a
        concatenation of pooled lesion and graph features. The segmentation loss is
        switched off in training with ``lam_seg=0``."""
        super().__init__()
        if fusion not in ("cross", "concat"):
            raise ValueError(fusion)
        self.use_graph, self.fusion = use_graph, fusion
        import timm
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        c = self.encoder.num_features
        self.lesion_proj = nn.Linear(c, dim)
        self.unet = UNet()
        self.graph = GraphEncoder(dim, heads)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(0.3))
        self.head_type = head
        self.head = CoralHead(dim) if head == "coral" else nn.Linear(dim, NUM_CLASSES)
        self.max_nodes = max_nodes

    def graphs_from_masks(self, vessel_logits: torch.Tensor) -> Batch:
        masks = (torch.sigmoid(vessel_logits.detach()) > 0.5).squeeze(1).cpu().numpy()
        return Batch.from_data_list([skeleton_graph(m, self.max_nodes) for m in masks])

    def forward(self, x, graphs: Batch | None = None):
        fmap = self.encoder.forward_features(x)                       # (B, C, h, w)
        tokens = self.lesion_proj(fmap.flatten(2).transpose(1, 2))     # (B, h*w, d)
        lesion_global = tokens.mean(1)
        vessel_logits = self.unet(x)
        B, d = x.size(0), tokens.size(-1)
        attn = None
        if not self.use_graph:
            graph_global = torch.zeros(B, d, device=x.device)
            fused = lesion_global
        else:
            if graphs is None:                                         # inference / predicted-mask training
                graphs = self.graphs_from_masks(vessel_logits)
            graphs = graphs.to(x.device)
            nodes = self.graph(graphs)                                 # (sum N_i, d)
            graph_global = global_mean_pool(nodes, graphs.batch, size=B)
            if self.fusion == "cross":
                dense, valid = to_dense_batch(nodes, graphs.batch, batch_size=B)
                attended, attn = self.cross(tokens, dense, dense, key_padding_mask=~valid)
                fused = self.norm(tokens + attended).mean(1)           # residual keeps lesion information
            else:
                fused = lesion_global
        h = self.fuse(torch.cat([fused, lesion_global, graph_global], 1))
        return self.head(h), vessel_logits, attn


# ============================================================ losses and metrics
def coral_loss(logits, y):
    levels = (y.unsqueeze(1) > torch.arange(NUM_CLASSES - 1, device=y.device)).float()
    return F.binary_cross_entropy_with_logits(logits, levels)


def ce_mse_loss(logits, y, w=0.5):            # the original ordinal loss, kept as an option
    probs = logits.softmax(1)
    expected = (probs * torch.arange(NUM_CLASSES, device=y.device)).sum(1)
    return F.cross_entropy(logits, y) + w * F.mse_loss(expected, y.float())


def seg_loss(vessel_logits, target):
    bce = F.binary_cross_entropy_with_logits(vessel_logits, target)
    p = torch.sigmoid(vessel_logits)
    dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def predict_grade(logits, head="coral"):
    return (torch.sigmoid(logits) > 0.5).sum(1) if head == "coral" else logits.argmax(1)


def metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nr = lambda g: float((y_pred[y_true == g] <= 1).mean()) if (y_true == g).any() else float("nan")
    return {"accuracy": accuracy_score(y_true, y_pred),
            "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "sev_nr": nr(3), "pdr_nr": nr(4),
            "grade_0_4_confusions": int(((y_true == 0) & (y_pred == 4)).sum() + ((y_true == 4) & (y_pred == 0)).sum())}


# ============================================================ data
class FundusDataset(torch.utils.data.Dataset):
    """Items: (image path, grade, optional vessel-mask path). Without a mask path the
    target is ``pseudo_vessel_mask``. With ``reference_graph=True`` the vessel graph is
    built from the target mask in the DataLoader workers (faster training)."""

    def __init__(self, items, size=300, augment=False, reference_graph=False, max_nodes=256,
                 with_mask=True):
        self.items, self.size, self.augment, self.with_mask = items, size, augment, with_mask
        self.reference_graph, self.max_nodes = reference_graph, max_nodes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, grade, mask_path = (list(self.items[i]) + [None])[:3]
        img = np.asarray(Image.open(path).convert("RGB").resize((self.size, self.size), Image.BILINEAR))
        if not self.with_mask:                      # baselines: no vessel target needed
            m = np.zeros((self.size, self.size), bool)
        elif mask_path:
            m = np.asarray(Image.open(mask_path).convert("L").resize((self.size, self.size), Image.NEAREST)) > 127
        else:
            m = pseudo_vessel_mask(img)
        if self.augment and np.random.rand() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
        mask = torch.from_numpy(np.ascontiguousarray(m)).float()[None]
        g = skeleton_graph(np.ascontiguousarray(m), self.max_nodes) if self.reference_graph else None
        return x, int(grade), mask, g


def collate(batch):
    xs, ys, ms, gs = zip(*batch)
    graphs = Batch.from_data_list(list(gs)) if gs[0] is not None else None
    return torch.stack(xs), torch.tensor(ys), torch.stack(ms), graphs


def patient_folds(grades, patients, n_splits=5, seed=42):
    """Patient-grouped stratified folds: the two eyes of a patient stay together."""
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
                .split(np.zeros(len(grades)), grades, groups=patients))


# ============================================================ training
def train_epoch(model, loader, opt, device, lam_seg=0.5, head="coral"):
    model.train()
    total = 0.0
    for x, y, m, g in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits, vlog, _ = model(x, g)
        loss = (coral_loss(logits, y) if head == "coral" else ce_mse_loss(logits, y)) + lam_seg * seg_loss(vlog, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, head="coral"):
    """Inference uses the graph of the *predicted* vessel mask, as in deployment."""
    model.eval()
    ys, ps = [], []
    for x, y, _, _ in loader:
        logits, _, _ = model(x.to(device))
        ys += y.tolist()
        ps += predict_grade(logits, head).cpu().tolist()
    return metrics(ys, ps)


# ============================================================ smoke test
def _smoke():
    torch.manual_seed(0)
    # graph builder on known shapes
    ring = np.zeros((64, 64), bool)
    yy, xx = np.ogrid[:64, :64]
    r = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    ring[(r > 18) & (r < 21)] = True
    g = skeleton_graph(ring)
    assert g.x.shape[1] == NODE_FEATS and g.num_nodes >= 1, "ring"
    tee = np.zeros((64, 64), bool)
    tee[10:13, 5:60] = True
    tee[10:55, 31:34] = True
    g = skeleton_graph(tee)
    ends, junc = int(g.x[:, 3].sum()), int(g.x[:, 4].sum())
    assert ends >= 3 and junc >= 1, f"T shape: {ends} ends, {junc} junctions"
    assert skeleton_graph(np.zeros((32, 32), bool)).edge_index.shape == (2, 0)

    model = TAGMTLDR(pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    x = torch.randn(2, 3, 300, 300)
    y = torch.tensor([1, 3])
    m = (torch.rand(2, 1, 300, 300) > 0.9).float()
    ref = Batch.from_data_list([skeleton_graph(tee), skeleton_graph(ring)])
    for graphs in (ref, None):                  # reference graphs, then predicted-mask graphs
        logits, vlog, attn = model(x, graphs)
        loss = coral_loss(logits, y) + 0.5 * seg_loss(vlog, m)
        opt.zero_grad()
        loss.backward()
        assert model.unet.out.weight.grad is not None and model.unet.out.weight.grad.abs().sum() > 0
        assert model.encoder.conv_stem.weight.grad.abs().sum() > 0
        opt.step()
    assert logits.shape == (2, NUM_CLASSES - 1) and vlog.shape == (2, 1, 300, 300)
    assert attn.shape[0] == 2 and attn.shape[1] == 100      # 10 x 10 lesion tokens attend to nodes
    print("smoke test passed: graph builder, U-Net gradient, encoder gradient, batching, CORAL head")
    print("predicted grades:", predict_grade(logits).tolist(), "| metrics example:",
          {k: round(v, 3) if isinstance(v, float) and not math.isnan(v) else v
           for k, v in metrics([0, 1, 3, 4], [0, 2, 1, 4]).items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        _smoke()
