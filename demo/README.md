# Demo

- **Webcam (local)**: `python demo/webcam_inference.py` — human detection → pose → 2D overlay + 3D (matplotlib).  
  Options: `--detection ultralytics|hf|onnx`, `--no-3d`, `--camera 0`, `--model_path ...`
- **Web (browser)**: WebGPU + ONNX in browser with 3D skeleton (Three.js).

## WebGPU browser demo

1. Export ONNX models (once):
   ```bash
   python demo/export_for_browser.py
   ```
   Writes `demo/webgpu/models/pose.onnx` and `demo/webgpu/models/person_detector.onnx`.  
   Requires a trained snapshot at `output/model_dump/snapshot_0.pth.tar` and `ultralytics` for the detector.

2. Serve the `demo/webgpu` folder over HTTP (required for Web Workers / ONNX):
   ```bash
   cd demo/webgpu && python -m http.server 8080
   ```
   Then open `http://localhost:8080` (or `https://` if needed for camera).

3. In the page: click “Start camera” to run detection + pose and see 2D overlay and 3D skeleton.

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
