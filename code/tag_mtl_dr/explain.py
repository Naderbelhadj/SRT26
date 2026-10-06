"""Explanation figures for TAG-MTL-DR v2 (one panel per image).

For each image: (a) the fundus, (b) the vessel graph rooted at the optic disc, nodes
coloured by the attention they receive from the lesion tokens, (c) the lesion-token
attention map (how much each image region attends to the vessel structures), and
(d) the predicted grade probabilities. Intended for the qualitative figure of the paper
and for failure analysis (always show errors, not only successes).

Usage
-----
    python explain.py --checkpoint model.pt --cache cache/ --manifest manifest.csv \
        --images 0 15 42 --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tag_mtl_dr import (IMAGENET_MEAN, IMAGENET_STD, CachedFundus, TAGMTLDR, class_probs, collate,
                        gpu_preprocess)

GRADES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


@torch.no_grad()
def explain(model, x_uint8, graph, size, device):
    model.eval()
    x = gpu_preprocess(x_uint8[None], size, device)
    logits, _, attn = model(x, graph)
    P = class_probs(logits.float().cpu(), "coral")[0].numpy()
    if attn is None:
        return P, None, None
    a = attn[0].float().cpu().numpy()                    # (tokens, keys) averaged over heads
    n_nodes = graph.num_nodes
    node_att = a[:, :n_nodes].mean(0)                    # attention received by each vessel node
    tok = a[:, :n_nodes].sum(1)                          # share of each token's attention on vessel nodes
    h = int(np.sqrt(len(tok)))
    return P, node_att, tok.reshape(h, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="state_dict of a trained TAGMTLDR")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", nargs="+", type=int, required=True, help="row indices of the manifest")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--backbone", default="efficientnet_b3")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(a.manifest)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAGMTLDR(pretrained=False, backbone=a.backbone)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    model.to(device)
    ds = CachedFundus(a.cache, a.images, [int(rows[i]["grade"]) for i in a.images])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(a.images):
        x, y, _, g = ds[k]
        _, _, _, gb = collate([(x, y, torch.zeros(1), g)])
        P, node_att, tok = explain(model, x, gb, a.size, device)
        img = x.permute(1, 2, 0).numpy()
        S = img.shape[0]
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        ax[0].imshow(img)
        ax[0].set_title(f"reference grade {y} ({GRADES[y]})")
        ax[1].imshow(img, alpha=0.35)
        pos = g.pos.numpy() * S
        for s_, d_ in g.edge_index.t().numpy():
            ax[1].plot(pos[[s_, d_], 1], pos[[s_, d_], 0], color="white", lw=0.8)
        c = node_att if node_att is not None else np.zeros(len(pos))
        sc = ax[1].scatter(pos[:, 1], pos[:, 0], c=c, s=18, cmap="magma")
        root = g.x[:, 6].numpy() > 0
        ax[1].scatter(pos[root, 1], pos[root, 0], marker="*", s=120, c="cyan", label="root (optic disc side)")
        ax[1].legend(loc="lower right", fontsize=7)
        fig.colorbar(sc, ax=ax[1], fraction=0.046)
        ax[1].set_title("vessel graph: attention received")
        if tok is not None:
            ax[2].imshow(img)
            ax[2].imshow(np.kron(tok, np.ones((S // tok.shape[0] + 1,) * 2))[:S, :S], alpha=0.5, cmap="jet")
        ax[2].set_title("lesion tokens: attention to vessels")
        ax[3].bar(range(5), P, color=["#999"] * 5)
        ax[3].bar([int(P.argmax())], [P.max()], color="#1F4E79")
        ax[3].set_xticks(range(5), GRADES, rotation=30)
        ax[3].set_ylim(0, 1)
        ax[3].set_title("predicted probabilities")
        for axx in ax[:3]:
            axx.axis("off")
        fig.tight_layout()
        fig.savefig(Path(a.out) / f"explain_{i}.pdf")
        plt.close(fig)
        print(f"image {i}: grade {y}, predicted {int(P.argmax())} (p = {P.max():.2f})")


if __name__ == "__main__":
    main()
