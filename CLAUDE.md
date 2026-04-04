# MobileHumanPose

Official PyTorch implementation of **"Toward Real-Time 3D Human Pose Estimation in Mobile Devices"** (CVPRW 2021). Lightweight 3D human pose estimation pipeline optimized for mobile deployment.

## What This Project Does

Estimates 3D human body joint positions from single RGB images using a lightweight backbone (LpNet) with soft-argmax decoding. The full inference pipeline: person detection (YOLOv8) -> 3D pose estimation (LpNet + soft-argmax) -> root depth estimation (RootNet).

## Project Structure

```
src/3dpose_estimator/   # Core model: CustomNet, training, testing, export
src/distance_estimator/ # RootNet for absolute depth estimation
src/person_detector/    # Person bounding box detection
common/backbone/        # LpNet variants (lpnet_ski_concat, lpnet_res_concat, lpnet_wo_concat)
common/utils/           # Pose utilities, visualization, directory helpers
common/base.py          # Base Trainer/Tester classes
data/                   # Dataset implementations (Human36M, MuCo, MuPoTS, MSCOCO, MPII, Dummy)
data/dataset.py         # DatasetLoader with augmentation pipeline
demo/                   # Inference scripts (image, video, webcam, web UI)
demo/models/            # Pre-trained ONNX models (person_detector, rootnet, pose_3d)
vis/                    # Visualization tools (Python + legacy MATLAB)
scripts/                # Helper scripts (data generation, model export)
runs/                   # Training shell scripts
output/                 # Generated: checkpoints, logs, results, visualizations
```

## Configuration

All config lives in `src/3dpose_estimator/config.py`. No CLI flags for config — edit the file directly.

Key settings:
- `backbone`: LPSKI (skip concat) | LPRES (residual concat) | LPWO (no concat)
- `input_shape`: (256, 256), `output_shape`: (32, 32), `depth_dim`: 32
- `trainset_3d` / `trainset_2d` / `testset`: dataset selection
- `batch_size`: 32, `lr`: 1e-3, `lr_dec_epoch`: [5, 10], `end_epoch`: 12

## Common Commands

```bash
# Install
pip install -e .

# Generate dummy data for quick testing
python scripts/generate_dummy_data.py

# Train (single GPU)
python -m src.train

# Train (multi-GPU via torchrun)
bash runs/train.sh

# Resume training
python -m src.train --continue

# Test
python -m src.test --test_epoch 0

# Export to ONNX
python -m src.3dpose_estimator.export -f onnx -m output/model_dump/snapshot_0.pth.tar

# Export to CoreML
python -m src.3dpose_estimator.export -f coreml -m output/model_dump/snapshot_0.pth.tar

# Demo (single image)
python demo/demo.py -m demo/models/pose_3d.onnx -i image.jpg

# Demo (video)
python demo/video_inference.py -i video.mp4 -m output/model_dump/snapshot_0.pth.tar

# Web UI (Gradio)
python demo/web_video_inference.py
```

## Architecture Details

- **Backbone**: InvertedResidual blocks (MobileNet-style), embedding_size=2048, width_multiplier=1.0
- **Decoder**: Differentiable soft-argmax over (joint_num, depth_dim, height, width) heatmaps -> (x, y, z) per joint
- **Loss**: L1 on 3D coordinates with per-joint visibility masking; depth loss only when GT depth available
- **Joints**: 18 keypoints (Pelvis, R/L_Hip, R/L_Knee, R/L_Ankle, Torso, Neck, Nose, Head, L/R_Shoulder, L/R_Elbow, L/R_Wrist, Thorax)
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Datasets

| Dataset   | Type       | Use           |
|-----------|------------|---------------|
| Human36M  | 3D mocap   | Train + Test  |
| MuCo      | 3D synth   | Train         |
| MuPoTS    | 3D multi   | Test          |
| MSCOCO    | 2D annot   | Train + Test  |
| MPII      | 2D annot   | Train + Test  |
| Dummy     | Synthetic  | Quick PoC     |

Annotations follow COCO-style JSON format. Multi-dataset training uses balanced batch sampling.

## Evaluation Metrics

- **MPJPE** (Mean Per Joint Position Error) in mm
- **PA-MPJPE** (Procrustes-Aligned MPJPE)

## Dependencies

Core: numpy, torch, torchvision, opencv-python, matplotlib, pycocotools, scipy, tqdm, gradio
Optional: onnx, onnxruntime, coremltools, ultralytics, roboflow

## Notes

- Distributed training auto-detected via `torch.distributed` env vars (set by `torchrun`)
- RootNet falls back to heuristic depth estimation (1500mm * sqrt(image_area/bbox_area)) when ONNX model unavailable
- ONNX export uses opset 11 by default
- Output paths auto-derived from config file location
