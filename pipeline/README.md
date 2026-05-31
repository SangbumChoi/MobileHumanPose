# MobileHumanPose — End-to-End Pipeline

A self-contained, **CPU-runnable** pipeline that takes the original
[MobileHumanPose](../README.md) model from raw web images all the way to a
deployable web demo. It reuses the genuine `LpNetSkiConcat` backbone from
`common/backbone`, but removes the multi-GPU / huge-dataset assumptions of the
original training code so everything runs on a laptop.

```
crawl ──▶ curate ──▶ annotate ──▶ embed-curate ──▶ train ──▶ deploy
  1          2           3              4              5         6
```

| Stage | Script | What it does | Key pretrained model |
|-------|--------|--------------|----------------------|
| 1. Crawl | `stage01_crawl.py` | Acquire person images (Wikimedia → DuckDuckGo → bundled fallback) | — |
| 2. Curate | `stage02_curate.py` | Dedup (pHash) + quality + person-presence filter | Faster-RCNN MobileNetV3 |
| 3. Annotate | `stage03_annotate.py` | Auto-label COCO-17 keypoints → `annotations.json` | KeypointRCNN-R50 |
| 4. Embed-curate | `stage04_embed_curate.py` | Embed crops → cluster → rebalance dataset distribution | ResNet-50 (ImageNet) |
| 5. Train | `stage05_train.py` | Train `LpNetSkiConcat` (2D, soft-argmax + L1 loss) | — (trains from scratch) |
| 6a. Deploy (server) | `stage06_demo.py` | Gradio app: detect + pose, multi-person | Faster-RCNN + trained LpNet |
| 6b. Deploy (static) | `export_onnx.py` + `web/` | ONNX + ONNX-Runtime-Web in-browser demo | trained LpNet (ONNX) |

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r pipeline/requirements.txt
```

## Run everything

```bash
cd pipeline
python run_pipeline.py --query "person full body" --limit 60 --epochs 20
python stage06_demo.py            # launch the Gradio demo (http://localhost:7860)
```

Each stage is also runnable on its own (they read/write `pipeline/work/`):

```bash
python stage01_crawl.py --query "yoga pose" --limit 40
python stage02_curate.py
python stage03_annotate.py
python stage04_embed_curate.py
python stage05_train.py --epochs 25
python export_onnx.py             # writes web/pose_model.onnx
```

## Outputs (`pipeline/work/`, git-ignored)

```
01_raw/        crawled images + manifest.json
02_curated/    filtered images + curate_report.json
03_annotated/  annotations.json (COCO format) + previews/
04_balanced/   embeddings.npy, clusters.json, sample_weights.json, distribution.png
05_models/     pose_model.pth, pose_model.onnx, train_curve.png
06_demo/       inference test outputs
web/           index.html + demo.js + pose_model.onnx  (static demo, committed)
```

## Notes & honest limitations

- This sandbox is **CPU-only**. With a small crawled set and few epochs the
  trained model is a *functional demonstration* (loss decreases, valid
  checkpoint, runnable demo) — **not** research-grade accuracy. For real
  results, increase `--limit` and `--epochs` and run on a GPU.
- The auto-annotations are 2D (COCO-17). The LpNet loss masks the depth term
  via `have_depth=0`, so this trains the model in 2D-only mode. The
  architecture still outputs the full 3D heatmap and can be supervised in 3D
  when depth-labeled data is available.
- The static `web/` demo assumes a single, roughly-centered person (no
  in-browser detector); the Gradio app handles detection + multiple people.
