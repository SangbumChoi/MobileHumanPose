"""MobileHumanPose — Hugging Face Spaces demo (Gradio SDK).

Self-contained: detects people with a torchvision Faster-RCNN, runs the
MobileHumanPose (LpNet) model exported to ONNX via onnxruntime, and draws the
19-joint skeleton. The ONNX graph already includes soft-argmax, so it outputs
(N, 19, 3) coords in the 32x32 output grid.

Deploy: create a Space (SDK=Gradio) and push this folder (app.py, requirements.txt,
README.md, pose_model.onnx). Locally: `pip install -r requirements.txt && python app.py`.
"""

import os.path as osp

import numpy as np
import cv2
import gradio as gr
import onnxruntime as ort
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

INPUT = 256
OUTPUT = 32
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
COCO_PERSON = 1
MODEL_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "pose_model.onnx")

# 19-joint MSCOCO convention (17 COCO + Thorax + Pelvis).
SKELETON = [(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9),
            (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12),
            (17, 5), (17, 6), (18, 11), (18, 12)]

_SESS = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_DET = None


def _detector():
    global _DET
    if _DET is None:
        w = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        _DET = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=w).eval()
    return _DET


# --- bbox + affine patch helpers (self-contained copy of pipeline.common) ---
def process_bbox(bbox, w, h, expand=1.25):
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    if bw > bh:
        bh = bw
    else:
        bw = bh
    bw *= expand; bh *= expand
    return np.array([cx - bw / 2.0, cy - bh / 2.0, bw, bh], np.float32)


def _trans(c_x, c_y, sw, sh, dw, dh):
    src = np.array([[c_x, c_y], [c_x, c_y + sh * 0.5], [c_x + sw * 0.5, c_y]], np.float32)
    dst = np.array([[dw * 0.5, dh * 0.5], [dw * 0.5, dh], [dw, dh * 0.5]], np.float32)
    return cv2.getAffineTransform(src, dst)


def patch_of(img, bbox):
    cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
    t = _trans(cx, cy, bbox[2], bbox[3], INPUT, INPUT)
    p = cv2.warpAffine(img, t, (INPUT, INPUT), flags=cv2.INTER_LINEAR)
    return p[:, :, ::-1].astype(np.float32), t


@torch.no_grad()
def detect(img_bgr, score=0.7, max_people=8):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = _detector()([to_tensor(rgb)])[0]
    boxes = []
    for i in range(len(out["scores"])):
        if int(out["labels"][i]) == COCO_PERSON and float(out["scores"][i]) >= score:
            x1, y1, x2, y2 = out["boxes"][i].numpy()
            boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_people]


def pose_of(img_bgr, bbox):
    h, w = img_bgr.shape[:2]
    pb = process_bbox(bbox, w, h)
    patch, t = patch_of(img_bgr, pb)
    x = ((patch / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    coords = _SESS.run(None, {"input": x})[0][0]      # (19, 3), grid units
    xy = coords[:, :2] / OUTPUT * INPUT
    xy1 = np.concatenate([xy, np.ones((len(xy), 1))], 1)
    inv = np.linalg.inv(np.concatenate([t, [[0, 0, 1]]], 0))
    return (inv @ xy1.T).T[:, :2]


def draw(img_bgr, people_kps):
    vis = img_bgr.copy()
    for kps in people_kps:
        for a, b in SKELETON:
            cv2.line(vis, tuple(np.int32(kps[a])), tuple(np.int32(kps[b])), (0, 255, 0), 2, cv2.LINE_AA)
        for x, y in kps:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return vis


def predict(image_rgb, det_score):
    if image_rgb is None:
        return None, "Upload an image."
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    boxes = detect(bgr, score=det_score)
    kps = [pose_of(bgr, b) for b in boxes]
    out = cv2.cvtColor(draw(bgr, kps), cv2.COLOR_BGR2RGB)
    return out, ("Detected %d person(s)." % len(boxes)) if boxes else "No person detected."


with gr.Blocks(title="MobileHumanPose") as demo:
    gr.Markdown("# MobileHumanPose — 2D pose (ONNX Runtime)\n"
                "Detects people and runs the MobileHumanPose (LpNet) ONNX model. "
                "Weights are from a small auto-labeled demo run, so predictions are "
                "approximate — this showcases the end-to-end pipeline.")
    with gr.Row():
        inp = gr.Image(type="numpy", label="Input")
        outp = gr.Image(type="numpy", label="Pose")
    score = gr.Slider(0.3, 0.95, value=0.7, step=0.05, label="Detection threshold")
    status = gr.Textbox(label="Status", interactive=False)
    gr.Button("Estimate pose", variant="primary").click(predict, [inp, score], [outp, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
