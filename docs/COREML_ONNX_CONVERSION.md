# CoreML and ONNX Conversion Guide

MobileHumanPose 모델을 CoreML 및 ONNX로 변환하여 iOS/Android에서 실행하는 방법입니다.

## Prerequisites

```bash
cd main
pip install -r requirements.txt
pip install onnx onnxruntime coremltools
```

## ONNX Conversion

### 1. Export PyTorch → ONNX

```bash
cd main
python -m src.pytorch2onnx --joint 18 --modelpath output/model_dump/snapshot_0.pth.tar
```

- `--joint`: 18 (Dummy/H36M) or 21 (MuCo)
- `--joint`: 18 (Dummy/H36M) or 21 (MuCo)
- `--modelpath`: PyTorch 체크포인트 경로

### 2. ONNX 출력

- 기본 경로: `../output/baseline.onnx`
- 입력: `(1, 3, 256, 256)` float32, ImageNet 정규화 (mean/std)
- 출력: `(1, joint_num, 3)` 2D+Z 좌표

### 3. ONNX 검증

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("output/baseline.onnx")
x = np.random.randn(1, 3, 256, 256).astype(np.float32)
out = session.run(None, {"input": x})
print(out[0].shape)  # (1, 18, 3) or (1, 21, 3)
```

## CoreML Conversion

### 1. Export PyTorch → CoreML

```bash
cd main
python -m src.pytorch2coreml --joint 18 --modelpath output/model_dump/snapshot_0.pth.tar
```

### 2. CoreML 출력

- 기본 경로: `src/test.mlmodel` (스크립트 실행 디렉토리)
- 입력: `input_1`, ImageType (1, 3, 256, 256)

### 3. iOS Swift 예시

```swift
import CoreML

let model = try! baseline()
let input = baselineInput(input_1: pixelBuffer)  // CVPixelBuffer 256x256
let output = try! model.prediction(input: input)
// output.var_xxx: MLMultiArray (1, 18, 3) or (1, 21, 3)
```

## TFLite (Android)

현재 `pytorch2onnx.py`는 ONNX → TensorFlow → TFLite 파이프라인을 포함합니다.

```bash
# pytorch2onnx.py 실행 시 자동으로 생성
# output/baseline.tflite (INT8 양자화)
```

- `onnx-tf` 패키지 필요: `pip install onnx-tf tensorflow`

## 데이터 정규화

입력 이미지는 다음으로 정규화되어야 합니다:

- mean: (0.485, 0.456, 0.406)
- std: (0.229, 0.224, 0.225)
- Crop & resize to 256×256

## 참고

- [PoseEstimation-TFLiteSwift](https://github.com/tucan9389/PoseEstimation-TFLiteSwift) - 공식 TFLite iOS 데모
