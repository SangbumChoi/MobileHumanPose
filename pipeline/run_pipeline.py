"""Run the full MobileHumanPose data-to-deploy pipeline end-to-end.

Stages:
  1. crawl          - acquire person images (Wikimedia / DuckDuckGo / fallback)
  2. curate         - dedup + quality + person-presence filtering
  3. annotate       - pretrained KeypointRCNN -> COCO-format 2D keypoints
  4. embed-curate   - ResNet-50 embeddings -> cluster -> rebalance distribution
  5. train          - train the LpNet (MobileHumanPose) backbone
  6. export         - ONNX export for the in-browser (github.io) demo

The Gradio web app is launched separately:  python stage06_demo.py

Example:
  python run_pipeline.py --query "person full body" --limit 60 --epochs 20
"""

import argparse
import os.path as osp

from common import PIPELINE_DIR, get_logger
import stage01_crawl
import stage02_curate
import stage03_annotate
import stage04_embed_curate
import stage05_train
import export_onnx

log = get_logger("pipeline")


def main():
    ap = argparse.ArgumentParser(description="MobileHumanPose end-to-end pipeline.")
    ap.add_argument("--query", default="person full body standing")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--skip_crawl", action="store_true",
                    help="reuse existing work/01_raw instead of crawling")
    args = ap.parse_args()

    if not args.skip_crawl:
        log.info("STAGE 1/6  crawl"); stage01_crawl.crawl(args.query, args.limit)
    log.info("STAGE 2/6  curate"); stage02_curate.curate()
    log.info("STAGE 3/6  annotate"); stage03_annotate.annotate()
    log.info("STAGE 4/6  embed-curate"); stage04_embed_curate.embed_curate()
    log.info("STAGE 5/6  train"); stage05_train.train(epochs=args.epochs,
                                                       batch_size=args.batch_size)
    log.info("STAGE 6/6  export ONNX")
    export_onnx.export(web_dir=osp.join(PIPELINE_DIR, "web"))
    log.info("Pipeline complete. Launch the demo with:  python stage06_demo.py")


if __name__ == "__main__":
    main()
