# WebGPU Browser Demo

- **왼쪽 패널**: 입력 스트림 (웹캠 영상만)
- **오른쪽 패널**: 출력 (2D 스켈레톤 오버레이 + 3D 스켈레톤)

## 1. 모델 준비

**권장**: `prepare_models.py` 실행 (pose는 demo/models에서 복사, person_detector는 onnxruntime-web 호환으로 별도 export):

```bash
cd demo/webgpu && python prepare_models.py
```

person_detector는 브라우저에서 `array length mismatch` 방지를 위해 `dynamic=False`로 export됨.

또는 프로젝트 루트에서:

```bash
python -c "
from pathlib import Path
p = Path('demo/webgpu/models')
p.mkdir(parents=True, exist_ok=True)
src = Path('demo/models')
if (src / 'pose_3d.onnx').exists():
    (src / 'pose_3d.onnx').hardlink_to(p / 'pose.onnx') if (p / 'pose.onnx').exists() else __import__('shutil').copy(src / 'pose_3d.onnx', p / 'pose.onnx')
if (src / 'person_detector.onnx').exists():
    (src / 'person_detector.onnx').hardlink_to(p / 'person_detector.onnx') if (p / 'person_detector.onnx').exists() else __import__('shutil').copy(src / 'person_detector.onnx', p / 'person_detector.onnx')
print('OK: demo/webgpu/models/')
"
```

## 2. 서버 실행

```bash
cd demo/webgpu && python -m http.server 8080
```

브라우저에서 `http://localhost:8080` 접속 후 **Start camera** 클릭.

## 3. 테스트

- 카메라 권한 허용 후 왼쪽에 입력 영상, 오른쪽에 2D/3D 스켈레톤이 나오면 정상 동작.
