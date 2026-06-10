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
| 5. Train | `stage05_train.py` | Trains via the **original repo `base.Trainer`** on the `CrawlPipeline` dataset (see [INTEGRATION.md](INTEGRATION.md)) | — (trains from scratch) |
| 6a. Deploy (server) | `stage06_demo.py` | Gradio app: detect + pose, multi-person | Faster-RCNN + trained LpNet |
| 6b. Deploy (static) | `export_onnx.py` + `web/` | ONNX + ONNX-Runtime-Web in-browser demo | trained LpNet (ONNX) |
| 6c. Deploy (HF Space) | `hf_space/` | Gradio Space (onnxruntime + detector), ready to push to Hugging Face | LpNet (ONNX) |
| 6d. Deploy (mobile) | `export_mobile.py` | TFLite (float32/float16, parity-checked) + Core ML `.mlpackage` | LpNet |

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

## Mock datasets for every repo format (2D + 3D)

`make_mock_datasets.py` writes a small **real** dataset in each repo format
(MSCOCO, MPII, Human3.6M, MuCo, MuPoTS) — real person images + real KeypointRCNN
keypoints, with camera-consistent 3D for the 3D sets — so every loader runs
without multi-GB downloads. `verify_datasets.py` loads all five through the
repo's own `DatasetLoader` and runs a mixed 2D+3D `Trainer` step. See
[INTEGRATION.md](INTEGRATION.md) §4.

```bash
python pipeline/make_mock_datasets.py   # generate data/<Dataset>/...
python pipeline/verify_datasets.py      # 2D + 3D load/train checks
```

## Integration with the original repo

Stage 5 does **not** reimplement training — it drives the repo's own
`common/base.py::Trainer` via a new `data/CrawlPipeline` dataset (a 2D COCO
loader modeled on `data/MSCOCO`). Several original-repo bugs that prevented it
from running on CPU were fixed in the process (`cfg.set_args`, `soft_argmax`
device, `Tester/Transformer` `.cuda()`, `train.py` arg parsing, a `ToTensor`
normalization bug). Full details: **[INTEGRATION.md](INTEGRATION.md)**. The
original commands now also run on CPU, e.g. `python main/train.py --backbone LPSKI`.

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
