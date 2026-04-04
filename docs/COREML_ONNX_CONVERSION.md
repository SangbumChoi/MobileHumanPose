# CoreML and ONNX Conversion Guide

MobileHumanPose 모델을 CoreML 및 ONNX로 변환하여 iOS/Android에서 실행하는 방법입니다.

## Prerequisites

```bash
pip install -r requirements.txt
pip install onnx onnxruntime coremltools
```

## ONNX Conversion

### 1. Export PyTorch → ONNX

```bash
python -m src.3dpose_estimator.export --format onnx --modelpath output/model_dump/snapshot_0.pth.tar [--check]
```

- `--joint`: 18 (Dummy/H36M) or 21 (MuCo), default 18
- `--modelpath`: PyTorch 체크포인트 경로
- `--check`: PyTorch vs ONNX 출력 비교

### 2. ONNX 출력

- 기본 경로: `output/pose_3d.onnx` (cfg.output_dir 기준)
- 입력: `input`, `(1, 3, 256, 256)` float32, ImageNet 정규화 (mean/std)
- 출력: `output`, `(1, joint_num, 3)` 2D+Z 좌표

### 3. ONNX 검증

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("output/pose_3d.onnx")
x = np.random.randn(1, 3, 256, 256).astype(np.float32)
out = session.run(None, {"input": x})
print(out[0].shape)  # (1, 18, 3) or (1, 21, 3)
```

## CoreML Conversion

### 1. Export PyTorch → CoreML

```bash
python -m src.3dpose_estimator.export --format coreml --modelpath output/model_dump/snapshot_0.pth.tar [--output output/pose_3d.mlpackage]
```

- CoreML은 mlprogram 포맷으로 저장되며 `.mlpackage` 디렉터리로 저장됩니다.

### 2. CoreML 출력

- 기본 경로: `output/pose_3d.mlpackage`
- 입력: `input`, TensorType (1, 3, 256, 256) float32 (클라이언트에서 동일 mean/std로 정규화)
- 출력: mlprogram (iOS 15+)

### 3. iOS Swift 예시

```swift
import CoreML

let model = try! pose_3d(configuration: MLModelConfiguration())
let input = try! pose_3dInput(input: mlArray)  // (1, 3, 256, 256) normalized
let output = try! model.prediction(input: input)
// output.var_xxx: MLMultiArray (1, 18, 3) or (1, 21, 3)
```

## TFLite (Android)

ONNX/TFLite 파이프라인은 별도 스크립트로 분리되어 있습니다. `export.py`는 ONNX와 CoreML만 지원합니다.

- TFLite가 필요하면 ONNX export 후 `onnx-tf` 등으로 별도 변환: `pip install onnx-tf tensorflow`

## 데이터 정규화

입력 이미지는 다음으로 정규화되어야 합니다:

- mean: (0.485, 0.456, 0.406)
- std: (0.229, 0.224, 0.225)
- Crop & resize to 256×256

## 참고

- [PoseEstimation-TFLiteSwift](https://github.com/tucan9389/PoseEstimation-TFLiteSwift) - 공식 TFLite iOS 데모
