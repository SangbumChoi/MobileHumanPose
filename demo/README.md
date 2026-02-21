# Demo

- **Webcam (local)**: `python demo/webcam_inference.py` — human detection → pose → 2D overlay + 3D (matplotlib).  
  Options: `--detection ultralytics|hf|onnx`, `--no-3d`, `--camera 0`, `--model_path ...`
- **Web (browser)**: WebGPU + ONNX in browser with 3D skeleton (Three.js).

## WebGPU browser demo

- **왼쪽 패널**: 입력 스트림 (웹캠만)
- **오른쪽 패널**: 출력 (2D 스켈레톤 + 3D 스켈레톤)

1. 모델 준비 (demo/models/에 ONNX가 있다면):
   ```bash
   python demo/webgpu/prepare_models.py
   ```
   또는 `demo/export_for_browser.py`로 직접 export (snapshot_0.pth.tar + ultralytics 필요).

2. 서버 실행:
   ```bash
   cd demo/webgpu && python -m http.server 8080
   ```
   브라우저에서 `http://localhost:8080` 접속.

3. **Start camera** 클릭 → 왼쪽에 입력 영상, 오른쪽에 2D/3D 결과 표시.

## Image demo (demo.py)

- **bbox**: Human detection (`--detection ultralytics|hf|onnx`) or manual `--bbox x y w h`
- **root depth**: RootNet ([3DMPPE_ROOTNET_RELEASE](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE), checkpoint [snapshot_18.pth.tar](https://drive.google.com/file/d/1ZHoXNFxBBsmis-5Xzu7dfXYGNxjpntgt/view))

```bash
# Export RootNet to ONNX (once)
pip install gdown && python demo/export_rootnet_onnx.py

# Run demo
python demo/demo.py -m output/model_dump/snapshot_0.pth.tar -i image.jpg
```

## Human detection (bbox input)

- `get_person_bboxes(frame_bgr, backend='ultralytics'|'hf'|'onnx', ...)` → list of `[x, y, w, h]`
