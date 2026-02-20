# MobileHumanPose

> PyTorch implementation of **MobileHumanPose: Toward real-time 3D human pose estimation in mobile devices** (CVPRW 2021).

**Last updated:** 2026-02-20

## Changelog

| Date       | Change |
| ---------- | ------ |
| 2026-02-20 | Web video inference (Gradio), pre-commit+ruff, README overhaul |
| 2022-05-19 | Dummy dataloader for fast PoC |
| 2021-11-23 | Initial release |

## Introduction

Official implementation of [MobileHumanPose (CVPRW 2021)](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Choi_MobileHumanPose_Toward_Real-Time_3D_Human_Pose_Estimation_in_Mobile_Devices_CVPRW_2021_paper.html). Edit `src/config.py` for backbone, dataset. Use `torchrun` for multi-GPU (see `runs/train.sh`).

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Generate dummy data (PoC without full dataset)
python scripts/generate_dummy_data.py

# 3. Train (single GPU)
python -m src.train

# 4. Test
python -m src.test --test_epoch 0

# 5. Web video inference
python demo/web_video_inference.py   # Open http://localhost:7860
```

## Directory

```
${ROOT}
├── assets/           # Images, sample videos
│   └── videos/       # Web inference output (pose_*.mp4)
├── common/           # Backbone, utils, base Trainer/Tester
├── data/             # Dataset loaders (Human36M, MuCo, MSCOCO, MPII, MuPoTS, Dummy)
├── demo/             # demo.py, video_inference.py, web_video_inference.py
├── docs/             # CoreML/ONNX conversion guide
├── src/              # config.py, train.py, test.py, model
├── runs/             # train.sh (torchrun)
├── scripts/          # generate_dummy_data, test_config_combinations, export_*_img_names
├── tool/             # Data preprocessing (optional)
├── vis/              # Legacy MATLAB 2D/3D; Python vis in common.utils.vis
└── output/           # log, model_dump, result, vis
```

## Data

Download links from [original README](https://github.com/SangbumChoi/MobileHumanPose#data):

| Dataset | Links |
| --------| ----- |
| Human3.6M | [data](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK?usp=sharing) |
| MPII | [images](http://human-pose.mpi-inf.mpg.de/), [annotations](https://drive.google.com/drive/folders/1MmQ2FRP0coxHGk0Ntj0JOGv9OxSNuCfK?usp=sharing) |
| MuCo | [data](https://drive.google.com/drive/folders/1yL2ey3aWHJnh8f_nhWP--IyC9krAPsQN?usp=sharing) |
| MuPoTS | [images](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/), [annotations](https://drive.google.com/drive/folders/1WmfQ8UEj6nuamMfAdkxmrNcsQTrTfKK_?usp=sharing) |

**Dummy data (PoC):** `python scripts/generate_dummy_data.py` creates minimal valid sets for all datasets.

## Training

Edit `src/config.py`:

- `backbone`: LPSKI | LPRES | LPWO
- `trainset_3d`: Human36M | MuCo | Dummy
- `trainset_2d`: MSCOCO | MPII
- `testset`: Human36M | MuPoTS | MSCOCO | Dummy

```bash
python -m src.train                 # Single GPU
python -m src.train --continue      # Resume
bash runs/train.sh                  # Multi-GPU (NPROC=2 default)
NPROC=8 bash runs/train.sh          # 8 GPUs
```

## Testing

```bash
python -m src.test --test_epoch 0
python -m src.test --epochs 20-21
```

## Inference

### Web Video Inference

Short video → pose overlay → saved to `assets/videos/`:

```bash
python demo/web_video_inference.py
# Open http://localhost:7860
```

### CLI Video

```bash
python demo/video_inference.py -i video.mp4 -m output/model_dump/snapshot_0.pth.tar
```

### Image Demo

```bash
python demo/demo.py -m output/model_dump/snapshot_0.pth.tar -i image.jpg
```

## Export (ONNX / CoreML / TFLite)

See [docs/COREML_ONNX_CONVERSION.md](docs/COREML_ONNX_CONVERSION.md).

```bash
python -m src.pytorch2onnx --joint 18 --modelpath output/model_dump/snapshot_0.pth.tar
python -m src.pytorch2coreml --joint 18 --modelpath output/model_dump/snapshot_0.pth.tar
```

## Linting

```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run ruff --all-files
```

## Citation

```bibtex
@InProceedings{Choi_2021_CVPR,
    author    = {Choi, Sangbum and Choi, Seokeon and Kim, Changick},
    title     = {MobileHumanPose: Toward Real-Time 3D Human Pose Estimation in Mobile Devices},
    booktitle = {CVPR Workshops},
    year      = {2021},
    pages     = {2328-2338}
}
```

## License

MIT
