# Getting Started: MobileHumanPose

전체 파이프라인: **데이터셋 생성 → 학습 → 가중치 변환 → 데모 실행**

---

## 1. 설치

```bash
pip install -e .
pip install onnx onnxruntime coremltools  # export용 (선택)
pip install ultralytics  # detection=ultralytics (선택)
```

- **데모 (demo.py, webcam_inference)**: CUDA 환경 권장 (GPU 없으면 CPU fallback 없을 수 있음)

---

## 2. 데이터셋 생성

### Dummy 데이터 (PoC, 다운로드 없음)

```bash
python data/Dummy/generate_dummy_data.py
# 또는
python scripts/generate_dummy_data.py
```

### 실제 데이터

| Dataset | 사용법 |
|---------|--------|
| **Human36M** | `cd data/Human36M && python download.py` |
| **MPII** | `cd data/MPII && python download.py` |
| **MuCo** | `cd data/MuCo && python download.py` |
| **MuPoTS** | `cd data/MuPoTS && python download.py` |
| **MSCOCO** | `cd data/MSCOCO && python download.py` |

설정은 `src/config.py`에서 `trainset_3d`, `trainset_2d`, `testset` 수정.

---

## 3. 학습 (Train)

설정: `src/config.py`  
- `backbone`: LPSKI | LPRES | LPWO  
- `trainset_3d`: Human36M | MuCo | Dummy  
- `trainset_2d`: MSCOCO | MPII  
- `testset`: Human36M | MuPoTS | MSCOCO | Dummy  

```bash
# Single GPU
python -m src.train

# Resume
python -m src.train --continue

# Multi-GPU
bash runs/train.sh
NPROC=8 bash runs/train.sh
```

체크포인트: `output/model_dump/snapshot_*.pth.tar`

---

## 4. 가중치 변환 (Export)

학습된 모델을 ONNX/CoreML로 변환.

```bash
# ONNX
python -m src.3dpose_estimator.export -f onnx -m output/model_dump/snapshot_0.pth.tar [--check]

# CoreML
python -m src.3dpose_estimator.export -f coreml -m output/model_dump/snapshot_0.pth.tar -o output/pose_3d.mlpackage
```

자세한 내용: [docs/COREML_ONNX_CONVERSION.md](COREML_ONNX_CONVERSION.md)

---

## 5. 데모 (Demo)

데모는 **ONNX**로 추론합니다. `demo/models/`에 ONNX 파일이 있어야 합니다.

### demo/models 준비 (최초 1회)

```bash
# RootNet ckpt 다운로드 + person_detector, rootnet, pose_3d ONNX 변환
python scripts/download_and_export_demo_models.py
```

- RootNet: [snapshot_18.pth.tar](https://drive.google.com/file/d/1ZHoXNFxBBsmis-5Xzu7dfXYGNxjpntgt/view) 다운로드 후 rootnet.onnx 변환
- person_detector.onnx: YOLOv8n → ONNX
- pose_3d.onnx: output/model_dump/snapshot_0.pth.tar → ONNX (학습 후)

### 이미지 추론

```bash
# ONNX 사용 (권장)
python demo/demo.py -m demo/models/pose_3d.onnx -i <image.jpg>
# PyTorch 사용
python demo/demo.py -m output/model_dump/snapshot_0.pth.tar -i <image.jpg> [--bbox x y w h]
```

### 비디오 추론

```bash
python demo/video_inference.py -i video.mp4 -m output/model_dump/snapshot_0.pth.tar
```

### 웹캠 스트리밍 (브라우저)

```bash
python demo/web_video_inference.py
# 브라우저에서 http://localhost:7860 접속
```

출력 영상: `assets/videos/` 또는 화면 출력

---

## 전체 흐름 요약

```
데이터셋 생성 → 학습 → (선택) 가중치 변환 → 데모
     │              │              │            │
data/Dummy/    runs/train.sh   export.py    demo/*.py
generate_*.py  src.train      ONNX/CoreML   webcam/image/video
```
