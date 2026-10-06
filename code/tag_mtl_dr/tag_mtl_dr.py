"""TAG-MTL-DR v2: lesion-aware vessel graph + vessel-path recurrence for DR grading.

Components (each can be switched off for the ablation study):

1. CNN lesion branch: EfficientNet-B3 feature map, one token per cell.
2. Lesion-aware vessel graph (GNN/GAT): nodes are the end-points and junctions of the
   vessel skeleton, edges the vessel segments (length, tortuosity). Each node carries its
   geometry *and* the CNN feature sampled at its position, so the graph attention network
   propagates lesion evidence along the vessels.
3. Vessel-path recurrence (RNN): the vessel tree is rooted at the optic disc; every
   root-to-leaf path is read in anatomical order by a bidirectional GRU over the GAT node
   embeddings, and the paths are pooled by attention. It models how lesions and geometry
   change along a vessel, from the disc to the periphery.
4. Cross-attention from lesion tokens to graph nodes and path embeddings (residual).
5. CORAL ordinal head (rank-consistent) and an auxiliary U-Net vessel segmentation (multi-task).

Speed: images, vessel masks, optic-disc positions, graphs and paths are computed once
(``build_cache``, multi-process) and read from a memory-mapped cache; resizing, flips and
normalisation run on the GPU. Vessel graphs come from a deterministic vessel extraction
applied to every image (training and test alike), so no CPU work is left in the training loop.

Novelty note: to our knowledge, sampling CNN lesion features onto a vessel-topology graph
and reading optic-disc-to-periphery vessel paths with a recurrent network have not been
combined for DR grading; verify with a literature search before claiming it.

``python tag_mtl_dr.py --smoke`` checks every component on random data.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage import exposure, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import softmax as seg_softmax
from torch_geometric.utils import to_dense_batch

NUM_CLASSES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NODE_FEATS = 7          # y, x, degree/4, is_end, is_junction, distance to optic disc, is_root
EDGE_FEATS = 2          # segment length, tortuosity
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ============================================================ classical vessel extraction
def pseudo_vessel_mask(rgb: np.ndarray, radius: int = 7, min_size: int = 50) -> np.ndarray:
    """Classical vessel mask (green channel, CLAHE, black top-hat, Otsu)."""
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


def optic_disc_center(rgb: np.ndarray) -> tuple[float, float]:
    """Brightest smoothed region, normalised (y, x) in [0, 1]."""
    g = rgb.astype(np.float32).mean(2)
    s = ndimage.gaussian_filter(g, sigma=max(g.shape) / 25)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return y / g.shape[0], x / g.shape[1]


# ============================================================ vessel graph and paths
class VesselGraph(Data):
    """Graph with root-to-leaf paths. ``paths`` holds local node indices (-1 = padding);
    ``path_batch`` is the graph index of each path after batching."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "paths":
            return 0
        if key == "path_batch":
            return 1
        return super().__inc__(key, value, *args, **kwargs)


def skeleton_graph(mask: np.ndarray, od: tuple[float, float] = (0.5, 0.5), max_nodes: int = 256,
                   max_paths: int = 32, path_len: int = 16) -> VesselGraph:
    """Topological vessel graph rooted at the optic disc.

    Nodes: skeleton end-points (degree 1) and junctions (degree >= 3); one node per loop
    without such points. Edges: skeleton segments, with length and tortuosity
    (length / chord). Paths: root-to-leaf node sequences of the BFS tree of each
    component, rooted at the node closest to the optic disc, longest first.
    """
    H, W = mask.shape
    skel = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(skel)
    if len(ys) < 2:
        return VesselGraph(x=torch.zeros(1, NODE_FEATS), pos=torch.full((1, 2), 0.5),
                           edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, EDGE_FEATS),
                           paths=torch.zeros(1, path_len, dtype=torch.long).fill_(-1).index_fill_(1, torch.tensor([0]), 0),
                           path_batch=torch.zeros(1, dtype=torch.long))

    pix = {(y, x): i for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist()))}
    nbrs = [[pix[(y + dy, x + dx)] for dy, dx in _OFFS if (y + dy, x + dx) in pix]
            for y, x in zip(ys.tolist(), xs.tolist())]
    deg = np.array([len(n) for n in nbrs])
    is_key = deg != 2
    seen = np.zeros(len(ys), bool)
    for s in range(len(ys)):                       # loops without key points get one
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
                    chord = max(np.hypot(ys[p] - ys[cur], xs[p] - xs[cur]), 1.0)
                    k = (min(a, b), max(a, b))
                    if k not in edges or length < edges[k][0]:
                        edges[k] = (length, length / chord)

    n = len(key_ids)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    comp_id = -np.ones(n, int)
    comps = []
    for s in range(n):                             # connected components of the key-point graph
        if comp_id[s] >= 0:
            continue
        q, members = deque([s]), [s]
        comp_id[s] = len(comps)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp_id[v] < 0:
                    comp_id[v] = len(comps)
                    members.append(v)
                    q.append(v)
        comps.append(members)
    comps.sort(key=len, reverse=True)
    keep, total = [], 0
    for c in comps:                                # keep the largest components
        if total + len(c) > max_nodes:
            break
        keep += c
        total += len(c)
    if not keep:
        keep = comps[0][:max_nodes]
    keep = sorted(keep)
    remap = -np.ones(n, int)
    remap[keep] = np.arange(len(keep))

    py, px = ys[key_ids[keep]] / H, xs[key_ids[keep]] / W
    d_od = np.hypot(py - od[0], px - od[1]) / np.sqrt(2)
    kdeg = deg[key_ids[keep]]
    roots = set()
    paths = []
    sub_adj = [[remap[v] for v in adj[u] if remap[v] >= 0] for u in keep]
    seen = np.zeros(len(keep), bool)
    for s in range(len(keep)):                     # BFS tree of each kept component, rooted near the disc
        if seen[s]:
            continue
        q, members = deque([s]), [s]
        seen[s] = True
        while q:
            u = q.popleft()
            for v in sub_adj[u]:
                if not seen[v]:
                    seen[v] = True
                    members.append(v)
                    q.append(v)
        root = min(members, key=lambda i: d_od[i])
        roots.add(root)
        parent = {root: -1}
        order, q = [root], deque([root])
        while q:
            u = q.popleft()
            for v in sub_adj[u]:
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    q.append(v)
        children = {u: 0 for u in order}
        for v, u in parent.items():
            if u >= 0:
                children[u] += 1
        leaves = [u for u in order if children[u] == 0] or [root]
        for leaf in leaves:
            path, u = [], leaf
            while u != -1:
                path.append(u)
                u = parent[u]
            paths.append(path[::-1])               # root -> leaf (disc -> periphery)
    paths.sort(key=len, reverse=True)
    paths = paths[:max_paths]
    P = -torch.ones(len(paths), path_len, dtype=torch.long)
    for i, p in enumerate(paths):
        if len(p) > path_len:                      # uniform resampling along the path, ends kept
            p = [p[j] for j in np.linspace(0, len(p) - 1, path_len).round().astype(int)]
        P[i, :len(p)] = torch.tensor(p)

    feats = np.stack([py, px, np.minimum(kdeg, 4) / 4.0, (kdeg == 1).astype(float), (kdeg >= 3).astype(float),
                      d_od, np.isin(np.arange(len(keep)), list(roots)).astype(float)], 1)
    src, dst, attr = [], [], []
    for (a, b), (L, tort) in edges.items():
        if remap[a] >= 0 and remap[b] >= 0:
            src += [remap[a], remap[b]]
            dst += [remap[b], remap[a]]
            attr += [[L / max(H, W), min(tort, 5.0) / 5.0]] * 2
    return VesselGraph(x=torch.tensor(feats, dtype=torch.float),
                       pos=torch.tensor(np.stack([py, px], 1), dtype=torch.float),
                       edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
                       edge_attr=torch.tensor(attr, dtype=torch.float).reshape(-1, EDGE_FEATS),
                       paths=P, path_batch=torch.zeros(len(paths), dtype=torch.long))


def flip_graph(g: VesselGraph, horizontal: bool) -> VesselGraph:
    """Mirror node positions (and features) consistently with an image flip."""
    g = g.clone()
    j = 1 if horizontal else 0
    g.pos[:, j] = 1 - g.pos[:, j]
    g.x[:, j] = 1 - g.x[:, j]
    return g


# ============================================================ preprocessing cache (speed)
def _prep_one(args):
    path, mask_path, size = args
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))
    if mask_path:
        m = np.asarray(Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)) > 127
    else:
        m = pseudo_vessel_mask(img)
    od = optic_disc_center(img)
    g = skeleton_graph(m, od)
    return img, np.packbits(m), od, {k: v for k, v in g.to_dict().items()}


def build_cache(items, cache_dir, size=300, workers=None):
    """Pre-compute every image once: resized pixels (uint8 memmap), vessel mask (bit-packed),
    optic disc, vessel graph and paths. ``items``: list of (image_path, mask_path or None)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    N = len(items)
    imgs = np.lib.format.open_memmap(cache / "images.npy", mode="w+", dtype=np.uint8, shape=(N, size, size, 3))
    masks = np.lib.format.open_memmap(cache / "masks.npy", mode="w+", dtype=np.uint8,
                                      shape=(N, (size * size + 7) // 8))
    ods, graphs = np.zeros((N, 2), np.float32), []
    with Pool(workers or os.cpu_count()) as pool:
        for i, (img, m, od, g) in enumerate(pool.imap(_prep_one, [(p, mp, size) for p, mp in items], chunksize=8)):
            imgs[i], masks[i], ods[i] = img, m, od
            graphs.append(g)
            if (i + 1) % 1000 == 0:
                print(f"  cached {i + 1}/{N}", flush=True)
    imgs.flush()
    masks.flush()
    np.save(cache / "optic_disc.npy", ods)
    torch.save(graphs, cache / "graphs.pt")
    json.dump({"size": size, "n": N, "images": [p for p, _ in items]}, open(cache / "index.json", "w"))
    return cache


class CachedFundus(torch.utils.data.Dataset):
    """Reads the cache; returns uint8 image (C, S, S), grade, mask (1, S, S) and graph.
    Random horizontal/vertical flips are applied to the image, mask and graph together."""

    def __init__(self, cache_dir, indices, grades, augment=False, need_mask=True, need_graph=True):
        self.cache, self.idx, self.grades = Path(cache_dir), np.asarray(indices), np.asarray(grades)
        self.augment, self.need_mask, self.need_graph = augment, need_mask, need_graph
        meta = json.load(open(self.cache / "index.json"))
        self.size = meta["size"]
        self.imgs = self.masks = self.graphs = None

    def _open(self):
        self.imgs = np.load(self.cache / "images.npy", mmap_mode="r")
        self.masks = np.load(self.cache / "masks.npy", mmap_mode="r")
        self.graphs = torch.load(self.cache / "graphs.pt", weights_only=False) if self.need_graph else None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        if self.imgs is None:
            self._open()
        i = int(self.idx[k])
        img = np.array(self.imgs[i])
        S = self.size
        m = (np.unpackbits(self.masks[i])[: S * S].reshape(S, S) if self.need_mask
             else np.zeros((1, 1), np.uint8))
        g = VesselGraph(**self.graphs[i]) if self.need_graph else None
        if self.augment:
            if np.random.rand() < 0.5:
                img, m = img[:, ::-1], (m[:, ::-1] if self.need_mask else m)
                g = flip_graph(g, True) if g is not None else None
            if np.random.rand() < 0.5:
                img, m = img[::-1], (m[::-1] if self.need_mask else m)
                g = flip_graph(g, False) if g is not None else None
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(m)).float()[None]
        return x, int(self.grades[k]), mask, g


def collate(batch):
    xs, ys, ms, gs = zip(*batch)
    graphs = Batch.from_data_list(list(gs)) if gs[0] is not None else None
    return torch.stack(xs), torch.tensor(ys), torch.stack(ms), graphs


def gpu_preprocess(x_uint8, size, device):
    """uint8 (B, 3, S, S) -> normalised float on the GPU, resized to the model input size."""
    x = x_uint8.to(device, non_blocking=True).float().div_(255)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=device)[:, None, None]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


# ============================================================ networks
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net with skip connections (auxiliary vessel segmentation)."""

    def __init__(self, c=(16, 32, 64)):
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


class CoralHead(nn.Module):
    """K-1 rank logits sharing one weight vector (rank-consistent, Cao et al. 2020)."""

    def __init__(self, dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, h):
        return self.w(h) + self.b


class TAGMTLDR(nn.Module):
    def __init__(self, pretrained=True, dim=256, heads=4, head="coral", backbone="efficientnet_b3",
                 use_graph=True, use_paths=True, lesion_aware=True, fusion="cross", use_seg=True):
        """Ablation switches: use_graph (vessel graph, GAT), use_paths (vessel-path GRU),
        lesion_aware (CNN features sampled at graph nodes), fusion ('cross' | 'concat'),
        use_seg (auxiliary U-Net)."""
        super().__init__()
        import timm
        if fusion not in ("cross", "concat"):
            raise ValueError(fusion)
        self.use_graph, self.use_paths = use_graph, use_paths and use_graph
        self.lesion_aware, self.fusion, self.use_seg = lesion_aware, fusion, use_seg
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        c = self.encoder.num_features
        self.lesion_proj = nn.Linear(c, dim)
        self.unet = UNet() if use_seg else None
        self.node_geo = nn.Sequential(nn.Linear(NODE_FEATS, dim), nn.GELU())
        self.node_lesion = nn.Linear(c, dim)
        self.g1 = GATConv(dim, dim // heads, heads=heads, edge_dim=EDGE_FEATS)
        self.g2 = GATConv(dim, dim, heads=1, edge_dim=EDGE_FEATS)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.path_score = nn.Linear(dim, 1)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Dropout(0.3))
        self.head = CoralHead(dim) if head == "coral" else nn.Linear(dim, NUM_CLASSES)

    def _node_embeddings(self, fmap, g):
        h = self.node_geo(g.x)
        if self.lesion_aware:                                          # CNN evidence sampled at each node
            pos, valid = to_dense_batch(g.pos, g.batch, batch_size=fmap.size(0))
            grid = pos[..., [1, 0]].mul(2).sub(1).unsqueeze(2)         # (B, Nmax, 1, 2) as (x, y) in [-1, 1]
            samp = F.grid_sample(fmap.float(), grid.float(), align_corners=False).squeeze(-1).transpose(1, 2)
            h = h + self.node_lesion(samp[valid].to(h.dtype))
        h = F.elu(self.n1(self.g1(h, g.edge_index, g.edge_attr)))
        return self.n2(self.g2(h, g.edge_index, g.edge_attr))

    def _paths(self, nodes, g, B):
        if g.paths.numel() == 0:
            return nodes.new_zeros(0, nodes.size(1)), g.path_batch, nodes.new_zeros(B, nodes.size(1))
        offs = g.ptr[g.path_batch].unsqueeze(1)
        idx = torch.where(g.paths >= 0, g.paths + offs, torch.zeros_like(g.paths))
        seq = nodes[idx] * (g.paths >= 0).unsqueeze(-1)
        lengths = (g.paths >= 0).sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                                        # (2, P, d/2)
        emb = torch.cat([h[0], h[1]], 1)
        a = seg_softmax(self.path_score(emb).squeeze(-1), g.path_batch, num_nodes=B)
        pooled = torch.zeros(B, emb.size(1), device=emb.device, dtype=emb.dtype).index_add_(
            0, g.path_batch, emb * a.unsqueeze(-1))
        return emb, g.path_batch, pooled

    def forward(self, x, graphs: Batch | None = None):
        fmap = self.encoder.forward_features(x)                        # (B, C, h, w)
        tokens = self.lesion_proj(fmap.flatten(2).transpose(1, 2))      # (B, h*w, d)
        B, d = x.size(0), tokens.size(-1)
        lesion_global = tokens.mean(1)
        vessel_logits = self.unet(x) if self.use_seg else None
        zeros = tokens.new_zeros(B, d)
        attn, graph_global, path_global, fused = None, zeros, zeros, lesion_global
        if self.use_graph:
            g = graphs.to(x.device)
            nodes = self._node_embeddings(fmap, g).to(tokens.dtype)
            graph_global = global_mean_pool(nodes, g.batch, size=B)
            keys, valid = to_dense_batch(nodes, g.batch, batch_size=B)
            if self.use_paths:
                pe, pb, path_global = self._paths(nodes, g, B)
                if pe.numel():
                    pk, pv = to_dense_batch(pe.to(tokens.dtype), pb, batch_size=B)
                    keys, valid = torch.cat([keys, pk], 1), torch.cat([valid, pv], 1)
            if self.fusion == "cross":
                attended, attn = self.cross(tokens, keys, keys, key_padding_mask=~valid)
                fused = self.norm(tokens + attended).mean(1)           # residual keeps lesion information
        h = self.fuse(torch.cat([fused, lesion_global, graph_global, path_global.to(tokens.dtype)], 1))
        return self.head(h), vessel_logits, attn


# ============================================================ losses and metrics
def coral_loss(logits, y):
    levels = (y.unsqueeze(1) > torch.arange(NUM_CLASSES - 1, device=y.device)).float()
    return F.binary_cross_entropy_with_logits(logits, levels)


def ce_mse_loss(logits, y, w=0.5):
    probs = logits.softmax(1)
    expected = (probs * torch.arange(NUM_CLASSES, device=y.device)).sum(1)
    return F.cross_entropy(logits, y) + w * F.mse_loss(expected, y.float())


def seg_loss(vessel_logits, target):
    if target.shape[-1] != vessel_logits.shape[-1]:
        target = F.interpolate(target, size=vessel_logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(vessel_logits, target)
    p = torch.sigmoid(vessel_logits)
    dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def class_probs(logits, head, T=1.0):
    """(B, 5) class probabilities for every head type, with temperature T."""
    logits = logits / T
    if head == "coral":
        s = torch.cummin(torch.sigmoid(logits), dim=1).values           # P(y > k), monotone
        ones, zeros = torch.ones_like(s[:, :1]), torch.zeros_like(s[:, :1])
        cum = torch.cat([ones, s, zeros], 1)
        return (cum[:, :-1] - cum[:, 1:]).clamp_min(0)
    return logits.softmax(1)


def metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nr = lambda g: float((y_pred[y_true == g] <= 1).mean()) if (y_true == g).any() else float("nan")
    return {"accuracy": accuracy_score(y_true, y_pred),
            "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "sev_nr": nr(3), "pdr_nr": nr(4),
            "grade_0_4_confusions": int(((y_true == 0) & (y_pred == 4)).sum() + ((y_true == 4) & (y_pred == 0)).sum())}


def patient_folds(grades, patients, n_splits=5, seed=42):
    """Patient-grouped stratified folds: the two eyes of a patient stay together."""
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
                .split(np.zeros(len(grades)), grades, groups=patients))


# ============================================================ smoke test
def _smoke():
    import time
    torch.manual_seed(0)
    ring = np.zeros((64, 64), bool)
    yy, xx = np.ogrid[:64, :64]
    r = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    ring[(r > 18) & (r < 21)] = True
    g = skeleton_graph(ring)
    assert g.x.shape[1] == NODE_FEATS and g.num_nodes >= 1, "ring"
    tee = np.zeros((64, 64), bool)
    tee[10:13, 5:60] = True
    tee[10:55, 31:34] = True
    g = skeleton_graph(tee, od=(0.0, 0.0))
    ends, junc = int(g.x[:, 3].sum()), int(g.x[:, 4].sum())
    assert ends >= 3 and junc >= 1, f"T shape: {ends} ends, {junc} junctions"
    assert g.paths.shape[0] >= 2 and (g.paths[:, 0] == g.paths[0, 0]).all(), "paths share the disc root"
    assert g.edge_attr.shape[1] == EDGE_FEATS
    e = skeleton_graph(np.zeros((32, 32), bool))
    assert e.edge_index.shape == (2, 0) and e.paths.shape[1] == 16
    f = flip_graph(g, True)
    assert torch.allclose(f.pos[:, 1], 1 - g.pos[:, 1])
    b = Batch.from_data_list([g, skeleton_graph(ring), e])
    assert b.path_batch.max().item() == 2 and b.paths.max().item() < max(g.num_nodes, 1) + 64

    # timing of the per-image CPU work removed from the training loop by the cache
    rgb = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    t0 = time.time()
    for _ in range(3):
        skeleton_graph(pseudo_vessel_mask(rgb), optic_disc_center(rgb))
    cpu_ms = (time.time() - t0) / 3 * 1000

    for flags in ({}, {"use_paths": False}, {"lesion_aware": False}, {"fusion": "concat"}, {"use_graph": False},
                  {"use_seg": False}):
        model = TAGMTLDR(pretrained=False, **flags)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        x = torch.randn(3, 3, 128, 128)
        y = torch.tensor([1, 3, 4])
        m = (torch.rand(3, 1, 128, 128) > 0.9).float()
        logits, vlog, attn = model(x, b)
        loss = coral_loss(logits, y) + (0.5 * seg_loss(vlog, m) if vlog is not None else 0)
        opt.zero_grad()
        loss.backward()
        assert model.encoder.conv_stem.weight.grad.abs().sum() > 0
        if flags.get("use_graph", True):
            assert model.g1.lin.weight.grad.abs().sum() > 0 if hasattr(model.g1, "lin") else True
            if flags.get("use_paths", True):
                assert model.gru.weight_ih_l0.grad.abs().sum() > 0, "GRU receives gradient"
            if flags.get("lesion_aware", True):
                assert model.node_lesion.weight.grad.abs().sum() > 0, "lesion-aware nodes receive gradient"
        if vlog is not None:
            assert model.unet.out.weight.grad.abs().sum() > 0
        assert logits.shape == (3, NUM_CLASSES - 1)
        P = class_probs(logits.detach(), "coral", T=1.5)
        assert torch.allclose(P.sum(1), torch.ones(3), atol=1e-5)
        print("  ok", flags or "full model")
    print(f"smoke test passed (per-image CPU graph extraction cached once: about {cpu_ms:.0f} ms/image saved per epoch)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        _smoke()
