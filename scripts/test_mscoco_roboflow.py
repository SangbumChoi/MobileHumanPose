#!/usr/bin/env python3
"""
Roboflow -> MSCOCO 변환 및 학습 연동 검증.
1) generate_dummy_mscoco() 실행 (Train/Valid/Test -> train2017, val2017)
2) 이미지/어노테이션 개수 확인
3) MSCOCO 클래스로 train/test 로드 확인
4) (선택) 1 step 학습
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
MSCOCO_BASE = DATA / "MSCOCO"
TRAIN_IMG = MSCOCO_BASE / "images" / "train2017"
VAL_IMG = MSCOCO_BASE / "images" / "val2017"
TRAIN_ANN = MSCOCO_BASE / "annotations" / "person_keypoints_train2017.json"
VAL_ANN = MSCOCO_BASE / "annotations" / "person_keypoints_val2017.json"
BBOX_ROOT = MSCOCO_BASE / "bbox_root" / "bbox_root_coco_output.json"


def run_conversion(workspace="tennis-court-segmentation", project="tennis-player-pose", version=1):
    from data.Dummy.generate_dummy_data import generate_dummy_mscoco
    generate_dummy_mscoco(workspace=workspace, project_name=project, version_num=version)


def check_counts():
    import json
    n_train_img = len(list(TRAIN_IMG.glob("*.jpg")) + list(TRAIN_IMG.glob("*.jpeg")) + list(TRAIN_IMG.glob("*.png")))
    n_val_img = len(list(VAL_IMG.glob("*.jpg")) + list(VAL_IMG.glob("*.jpeg")) + list(VAL_IMG.glob("*.png")))
    print(f"  images: train2017={n_train_img}, val2017={n_val_img}")

    if not TRAIN_ANN.exists():
        print("  FAIL: person_keypoints_train2017.json not found")
        return False
    train_data = json.loads(TRAIN_ANN.read_text())
    train_images = train_data.get("images", [])
    train_anns = train_data.get("annotations", [])
    print(f"  train json: images={len(train_images)}, annotations={len(train_anns)}")

    if not VAL_ANN.exists():
        print("  FAIL: person_keypoints_val2017.json not found")
        return False
    val_data = json.loads(VAL_ANN.read_text())
    val_images = val_data.get("images", [])
    val_anns = val_data.get("annotations", [])
    print(f"  val json: images={len(val_images)}, annotations={len(val_anns)}")

    if BBOX_ROOT.exists():
        bbox_data = json.loads(BBOX_ROOT.read_text())
        n_bbox = len(bbox_data) if isinstance(bbox_data, list) else 0
        print(f"  bbox_root: {n_bbox} entries")

    if len(train_images) != n_train_img:
        print(f"  WARN: train image count mismatch (json={len(train_images)} vs dir={n_train_img})")
    if len(val_images) != n_val_img:
        print(f"  WARN: val image count mismatch (json={len(val_images)} vs dir={n_val_img})")
    return True


def check_mscoco_load():
    from common.base import get_ds
    ds_train = get_ds("MSCOCO", "train")
    ds_test = get_ds("MSCOCO", "test")
    n_train = len(ds_train.data)
    n_test = len(ds_test.data)
    print(f"  MSCOCO train samples: {n_train}")
    print(f"  MSCOCO test samples: {n_test}")
    if n_train == 0:
        print("  FAIL: no train samples (check annotations and num_keypoints>0)")
        return False
    return True


def run_one_train_step():
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from src.config import cfg
    from common.base import get_ds, DatasetLoader
    from data.multiple_datasets import MultipleDatasets
    t = T.Compose([T.ToTensor(), T.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    ref = None
    ds_3d = get_ds("Dummy", "train")
    ref = ds_3d.joints_name
    loader_3d = DatasetLoader(ds_3d, ref, True, t)
    ds_2d = get_ds("MSCOCO", "train")
    loader_2d = DatasetLoader(ds_2d, ref, True, t)
    combined = MultipleDatasets([loader_3d, loader_2d], make_same_len=True)
    loader = DataLoader(combined, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"  one batch: img {batch[0].shape}, coord {batch[1].shape}")
    print("  OK: data pipeline works")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Skip Roboflow download, only verify existing data")
    parser.add_argument("--workspace", default="tennis-court-segmentation")
    parser.add_argument("--project", default="tennis-player-pose")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--train-step", action="store_true", help="Run one training step")
    args = parser.parse_args()

    print("=== Roboflow -> MSCOCO 변환 검증 ===\n")

    if not args.skip_download:
        print("1. Download & convert (generate_dummy_mscoco)...")
        try:
            run_conversion(args.workspace, args.project, args.version)
        except Exception as e:
            print(f"  FAIL: {e}")
            return 1
        print()
    else:
        print("1. Skip download (--skip-download)\n")

    print("2. Counts (images, annotations)...")
    if not check_counts():
        return 1
    print()

    print("3. MSCOCO load (train/test)...")
    if not check_mscoco_load():
        return 1
    print()

    if args.train_step:
        print("4. One training step...")
        try:
            run_one_train_step()
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print()
    else:
        print("4. Skip train step (use --train-step to run one batch)\n")

    print("Done. Run: python -m src.3dpose_estimator.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
