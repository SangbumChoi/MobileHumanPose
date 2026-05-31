"""Stage 6a - Gradio web demo for MobileHumanPose.

Upload an image -> detect people -> run the trained LpNet -> draw 2D skeletons.
Run:  python stage06_demo.py            (then open the printed URL)
"""

import argparse

import numpy as np
import cv2
import gradio as gr

from inference import load_pose_model, run_image
from common import get_logger

log = get_logger("demo")
_MODEL = None


def _predict(image_rgb, det_score):
    if image_rgb is None:
        return None, "Upload an image."
    global _MODEL
    if _MODEL is None:
        _MODEL = load_pose_model()
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    vis_bgr, boxes, kps = run_image(_MODEL, bgr, det_score=det_score)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    msg = f"Detected {len(boxes)} person(s)." if boxes else "No person detected."
    return vis_rgb, msg


def build_demo():
    with gr.Blocks(title="MobileHumanPose Demo") as demo:
        gr.Markdown("# MobileHumanPose - 2D Pose Demo\n"
                    "End-to-end pipeline output: crawl -> curate -> annotate -> "
                    "embed-curate -> **train (this model)** -> deploy.")
        with gr.Row():
            inp = gr.Image(type="numpy", label="Input image")
            out = gr.Image(type="numpy", label="Predicted pose")
        score = gr.Slider(0.3, 0.95, value=0.7, step=0.05, label="Person detection threshold")
        status = gr.Textbox(label="Status", interactive=False)
        gr.Button("Estimate pose", variant="primary").click(
            _predict, inputs=[inp, score], outputs=[out, status])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_demo().launch(server_name=args.host, server_port=args.port, share=args.share)
