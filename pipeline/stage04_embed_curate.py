"""Stage 4 - Embedding-space curation (distribution enrichment).

Embeds every annotated person crop into a pretrained feature space
(ResNet-50, ImageNet), clusters it, and rebalances the dataset so rare visual
modes are up-weighted and dominant ones are down-weighted. This counters the
skew of web-crawled data and yields per-sample weights consumed by the trainer
via a WeightedRandomSampler.

Output: ``work/04_balanced/`` -> embeddings.npy, clusters.json,
sample_weights.json, distribution.png.
"""

import argparse
import os.path as osp

import numpy as np
import cv2
import torch
import torchvision
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (CURATED_DIR, ANNOTATED_DIR, BALANCED_DIR, normalize_patch,
                    generate_patch_image, process_bbox, ensure_dir, save_json,
                    load_json, get_logger)

log = get_logger("embed")


def _load_encoder():
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()  # -> 2048-d global features
    net.eval()
    return net


def embed_curate(curated_dir=CURATED_DIR, ann_dir=ANNOTATED_DIR,
                 out_dir=BALANCED_DIR, n_clusters=0):
    ensure_dir(out_dir)
    coco = load_json(osp.join(ann_dir, "annotations.json"))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    anns = coco["annotations"]
    if not anns:
        raise RuntimeError("No annotations to curate.")

    encoder = _load_encoder()

    feats, ann_ids = [], []
    cache = {}
    for a in anns:
        fn = id2file[a["image_id"]]
        if fn not in cache:
            cache[fn] = cv2.imread(osp.join(curated_dir, fn))
        bbox = process_bbox(np.array(a["bbox"], dtype=np.float32),
                            cache[fn].shape[1], cache[fn].shape[0])
        patch, _ = generate_patch_image(cache[fn], bbox)
        x = normalize_patch(patch).unsqueeze(0)
        with torch.no_grad():
            feats.append(encoder(x)[0].numpy())
        ann_ids.append(a["id"])
    feats = np.asarray(feats, dtype=np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    np.save(osp.join(out_dir, "embeddings.npy"), feats)

    n = len(feats)
    k = n_clusters or max(2, min(8, n // 4))
    k = min(k, n)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats)
    labels = km.labels_

    # Inverse-frequency weights -> rebalanced sampling distribution.
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    cluster_w = (counts.sum() / (counts + 1e-6))
    cluster_w /= cluster_w.mean()
    weights = {int(ann_ids[i]): float(cluster_w[labels[i]]) for i in range(n)}

    save_json({"k": int(k), "assignments": {int(ann_ids[i]): int(labels[i]) for i in range(n)},
               "cluster_sizes": counts.astype(int).tolist()},
              osp.join(out_dir, "clusters.json"))
    save_json(weights, osp.join(out_dir, "sample_weights.json"))

    # Visualize: PCA scatter + before/after distribution.
    pca = PCA(n_components=2).fit_transform(feats)
    eff_after = counts * cluster_w           # expected counts after reweighting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax[0].scatter(pca[:, 0], pca[:, 1], c=labels, cmap="tab10", s=24)
    ax[0].set_title(f"Person-crop embedding space (PCA), k={k}")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    fig.colorbar(sc, ax=ax[0], label="cluster")
    idx = np.arange(k)
    ax[1].bar(idx - 0.2, counts, width=0.4, label="raw")
    ax[1].bar(idx + 0.2, eff_after, width=0.4, label="rebalanced (expected)")
    ax[1].set_title("Cluster distribution before / after rebalancing")
    ax[1].set_xlabel("cluster"); ax[1].set_ylabel("effective samples"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "distribution.png"), dpi=110)
    plt.close(fig)

    log.info("Embedded %d crops -> %d clusters; wrote weights + distribution.png", n, k)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embedding-space dataset curation.")
    ap.add_argument("--n_clusters", type=int, default=0, help="0 = auto")
    args = ap.parse_args()
    embed_curate(n_clusters=args.n_clusters)
