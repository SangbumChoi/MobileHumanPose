# How the pipeline integrates with the original repository

This documents what the **original** MobileHumanPose code does, the bugs that
prevented it from running, and how the pipeline now *reuses* that code instead
of sitting beside it.

## 1. The original repository, as it actually is

The repo is **dataset-class driven**:

- `main/config.py` (`cfg`) names the datasets (`trainset_3d`, `trainset_2d`,
  `testset`) and holds all hyper-params.
- `common/base.py` dynamically `exec("from <Name> import <Name>")` for each
  named dataset, then `Trainer` / `Tester` build a `DatasetLoader`
  (`data/dataset.py`) over them and run training / testing.
- Each dataset class (`data/Human36M`, `data/MSCOCO`, `data/MuCo`, `data/MuPoTS`,
  `data/Dummy`) exposes `self.data` (a list of dicts: `img_path`, `bbox`,
  `joint_img` `[x,y,depth]`, `joint_vis`, ...) and an `evaluate()`.
- `main/model.py` builds `CustomNet(backbone)`; the backbone (`common/backbone/
  lpnet_*`) outputs a `joint_num*32` heatmap; `soft_argmax` turns it into
  `(joint_num, 3)` coords. The loss masks depth with a `have_depth` flag, so
  **2D-only datasets are already supported** (this is exactly what `MSCOCO`
  does: `joints_have_depth=False`, `joint_img[:,2]=0`).

So the repo *already* contains a 2D-keypoint training path. My first attempt
ignored it and reimplemented training — this revision fixes that.

## 2. Bugs that stopped the original code from running (now fixed)

| File | Problem | Fix |
|------|---------|-----|
| `main/config.py` | `test.py` / `demo.py` call `cfg.set_args(...)`, but **no such method existed** → instant crash. | Added `Config.set_args(gpu_ids, continue_train)` with **CPU fallback** (`use_cuda`, `num_gpus`). |
| `main/model.py` | `soft_argmax` built `torch.arange(...)` with **no device** → crashes on GPU (device mismatch); the GPU-correct line was commented out. | `arange` now created on `heatmaps.device`. Runs on CPU **and** GPU. |
| `common/base.py` | `Tester` / `Transformer` call `.cuda()` **unconditionally** → unusable on CPU; `torch.load` had no `map_location`. | Guarded `.cuda()`, added `map_location='cpu'`, plus `_match_state_dict` to add/strip the `module.` prefix when loading across DataParallel/CPU. |
| `main/train.py` | The README documents `--gpu` / `--backbone`, but `train.py` **parsed no arguments**. | Added `argparse` (`--gpu`, `--backbone`, `--continue`) and `cfg.set_args`. |
| `data/dataset.py` | A float `[0,255]` patch was passed to `transforms.ToTensor()`, which only divides by 255 for **uint8** → `Normalize` got `[0,255]` (wrong stats). | Cast to `uint8` before `ToTensor` (train + test paths). |

With these, the original command runs on CPU:

```bash
python main/train.py --backbone LPSKI   # trains on the bundled Dummy dataset
```

## 3. The integration: `data/CrawlPipeline`

`data/CrawlPipeline/CrawlPipeline.py` is a thin specialisation of the repo's
`MSCOCO` *train* loader. The crawler already emits COCO-17 person keypoints, so
this dataset:

- reads `pipeline/work/03_annotated/annotations.json` + `work/02_curated/`,
- adds Thorax/Pelvis exactly like `MSCOCO` → the repo's canonical **19-joint,
  depth-less** target,
- optionally **oversamples by the Stage-4 embedding weights**, so the original
  (shuffle-based) `DataLoader` trains on the rebalanced distribution with **zero
  changes to `base.py`**.

`pipeline/stage05_train.py` no longer has its own training loop. It configures
`cfg` to use `CrawlPipeline` and runs the **original `base.Trainer`**
(`get_pose_net` → `CustomNet` → soft-argmax → L1 loss). It writes the repo-format
`output/model_dump/snapshot_*.pth.tar` (loadable by `main/test.py`) plus a small
`pipeline/work/05_models/pose_model.pth` (backbone weights + meta) consumed by
the demo / ONNX export.

## 4. Mock datasets for every format (2D + 3D)

`pipeline/make_mock_datasets.py` generates a small but **real** dataset in each
repo format so every loader runs out-of-the-box (no multi-GB downloads):

| Dataset | Type | Joints | On-disk format produced |
|---------|------|--------|--------------------------|
| MSCOCO  | 2D | 17→19 | `person_keypoints_train2017.json` + `images/train2017/` |
| MPII    | 2D | 16 | `annotations/train.json` + `images/` |
| Human3.6M | 3D | 17→18 | `Human36M_subject{1,5,6,7,8}_{data,camera,joint_3d}.json` + images |
| MuCo    | 3D | 21 | `MuCo-3DHP.json` (per-img `f,c`; per-ann `keypoints_cam/img`) |
| MuPoTS  | 3D test | 17→21 | `data/MuPoTS-3D.json` (per-img `intrinsic`) + `MultiPersonTestSet/` |

- **Real images** come from the committed, license-clean fallback assets.
- **2D keypoints** are real KeypointRCNN detections, remapped to each dataset's
  joint convention (deriving Pelvis/Thorax/Neck/Spine/Head_top, mapping
  hands→wrists and toes→ankles where a dataset has joints COCO lacks).
- **3D is camera-consistent**: with focal `f` and principal point `c`, each
  joint gets a plausible root-relative depth (anatomical template) and X,Y are
  back-projected so the 3D skeleton **reprojects exactly onto the real 2D
  detection** — a genuine "3D human in the image", not random noise.

Each loader's hardcoded research path now falls back to its repo-local mock
directory when the absolute path is absent (real-data use is unchanged).
`pipeline/verify_datasets.py` (and `tests/test_datasets.py`) load all five
through the repo's `DatasetLoader`, run a **mixed 2D+3D training step via the
original `Trainer`**, and a test-set pass via `Tester`.

```bash
python pipeline/make_mock_datasets.py   # (re)generate the mock data
python pipeline/verify_datasets.py      # load + train-step check (2D and 3D)
```

## 5. What stayed pipeline-specific (and why)

- **Stages 1-4** (crawl / curate / annotate / embed-curate) have no equivalent
  in the original repo — they are genuinely new and feed COCO-format data the
  repo can consume.
- The **demo** (`stage06_demo.py`, `web/`) reuses the repo backbone via
  `common.build_model` (the bare `LpNetSkiConcat`) + the same fixed `soft_argmax`
  math; it loads the backbone sub-state-dict the Trainer produced.
- Joint convention across `common.py`, `inference.py`, ONNX and the web demo is
  the repo's 19-joint MSCOCO set, so a model trained by the original Trainer is
  consumed unchanged. (Raw Stage-3 annotations remain plain COCO-17.)
