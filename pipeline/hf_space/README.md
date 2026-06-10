---
title: MobileHumanPose Demo
emoji: 🕺
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# MobileHumanPose — Hugging Face Space

2D human pose estimation: a torchvision Faster-RCNN detects people, then the
**MobileHumanPose (LpNet)** model — exported to ONNX and run with
**onnxruntime** — predicts a 19-joint skeleton per person.

The ONNX graph includes soft-argmax, so it outputs `(N, 19, 3)` coordinates in
the 32×32 output grid, which the app maps back to image space.

> The bundled `pose_model.onnx` comes from the pipeline's small, CPU-trained,
> auto-labeled demo run, so predictions are approximate. The point is the
> end-to-end pipeline (crawl → curate → annotate → embed → train → deploy).

## Run locally

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860
```

## Deploy to Hugging Face Spaces

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create mobilehumanpose-demo --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<user>/mobilehumanpose-demo
cp app.py requirements.txt README.md pose_model.onnx mobilehumanpose-demo/
cd mobilehumanpose-demo && git add . && git commit -m "MobileHumanPose demo" && git push
```

Source / full pipeline: https://github.com/SangbumChoi/MobileHumanPose (`pipeline/`).
