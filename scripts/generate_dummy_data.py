#!/usr/bin/env python3
"""
Generate minimal dummy datasets for all MobileHumanPose data folders.
Enables PoC/testing without full dataset downloads.
Usage: python scripts/generate_dummy_data.py
"""
import os
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _make_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _write_img_placeholder(path: Path, w: int = 256, h: int = 256):
    """Write minimal valid image (1x1 RGB)."""
    try:
        import cv2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = 128
        cv2.imwrite(str(path), img)
    except ImportError:
        path.touch()


def generate_dummy_h36m():
    """Dummy Human36M: same structure as Dummy."""
    base = DATA / "Human36M"
    _make_dirs(base / "images" / "s_1_act_01_subact_01_ca_01",
               base / "annotations", base / "bbox_root")

    img_path = base / "images" / "s_1_act_01_subact_01_ca_01" / "s_1_act_01_subact_01_ca_01_000001.jpg"
    _write_img_placeholder(img_path, 1000, 1002)

    data_json = {
        "images": [{
            "id": 1000001, "file_name": "s_1_act_01_subact_01_ca_01/s_1_act_01_subact_01_ca_01_000001.jpg",
            "width": 1000, "height": 1002, "subject": 1, "action_name": "Directions",
            "action_idx": 1, "subaction_idx": 1, "cam_idx": 1, "frame_idx": 0
        }],
        "annotations": [{
            "id": 1000001, "image_id": 1000001,
            "keypoints_vis": [True] * 17, "bbox": [300, 200, 330, 420]
        }],
    }
    (base / "annotations" / "Human36M_subject1_data.json").write_text(
        json.dumps(data_json, indent=2))

    cam = {"1": {"R": np.eye(3).tolist(), "t": [0, 0, 5500], "f": [1145, 1144], "c": [512, 515]}}
    (base / "annotations" / "Human36M_subject1_camera.json").write_text(
        json.dumps(cam, indent=2))

    j3d = {"1": {"1": {"0": [[0.0] * 3 for _ in range(17)]}}}
    (base / "annotations" / "Human36M_subject1_joint_3d.json").write_text(
        json.dumps(j3d, indent=2))

    bbox_root = [{"image_id": 1000001, "bbox": [300, 200, 330, 420],
                  "root_cam": [0.0, 0.0, 1000.0]}]
    (base / "bbox_root" / "bbox_root_human36m_output.json").write_text(
        json.dumps(bbox_root))


def generate_dummy_mpii():
    """Dummy MPII: COCO-format train.json + images."""
    base = DATA / "MPII"
    _make_dirs(base / "images", base / "annotations")

    img_path = base / "images" / "000001.jpg"
    _write_img_placeholder(img_path, 256, 256)

    train = {
        "images": [{"id": 1, "file_name": "000001.jpg", "width": 256, "height": 256}],
        "annotations": [{
            "id": 1, "image_id": 1, "num_keypoints": 16,
            "bbox": [50, 50, 100, 150],
            "keypoints": [100.0, 100.0, 1.0] * 16
        }],
    }
    (base / "annotations" / "train.json").write_text(json.dumps(train, indent=2))


def generate_dummy_mscoco():
    """Dummy MSCOCO: train2017/val2017, annotations, bbox_root."""
    base = DATA / "MSCOCO"
    _make_dirs(base / "images" / "train2017", base / "images" / "val2017",
               base / "annotations", base / "bbox_root")

    for name in ("000000000001.jpg", "000000000001.jpg"):
        _write_img_placeholder(base / "images" / "train2017" / name)
        _write_img_placeholder(base / "images" / "val2017" / name)

    train = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 256, "height": 256}],
        "annotations": [{
            "id": 1, "image_id": 1, "num_keypoints": 17, "iscrowd": 0,
            "bbox": [50, 50, 100, 150],
            "keypoints": [100.0, 100.0, 1.0] * 17
        }],
    }
    (base / "annotations" / "person_keypoints_train2017.json").write_text(
        json.dumps(train, indent=2))

    val = {"images": [{"id": 1, "file_name": "000000000001.jpg", "width": 256, "height": 256}]}
    (base / "annotations" / "person_keypoints_val2017.json").write_text(
        json.dumps(val, indent=2))

    bbox_root = [{"image_id": 1, "bbox": [50, 50, 100, 150], "root_cam": [0.0, 0.0, 1000.0]}]
    (base / "bbox_root" / "bbox_root_coco_output.json").write_text(
        json.dumps(bbox_root))


def generate_dummy_muco():
    """Dummy MuCo: MuCo-3DHP.json + image dirs."""
    base = DATA / "MuCo" / "data"
    _make_dirs(base / "augmented_set", base / "unaugmented_set")

    img_name = "000000_0.jpg"
    _write_img_placeholder(base / "augmented_set" / img_name)
    _write_img_placeholder(base / "unaugmented_set" / img_name)

    coco = {
        "images": [{
            "id": 1, "file_name": "augmented_set/000000_0.jpg", "width": 256, "height": 256,
            "f": [1500, 1500], "c": [128, 128]
        }],
        "annotations": [{
            "id": 1, "image_id": 1, "bbox": [50, 50, 100, 150],
            "keypoints_img": [[100.0, 100.0] for _ in range(21)],
            "keypoints_cam": [[0.0, 0.0, 1000.0 + i * 10] for i in range(21)]
        }],
    }
    (base / "MuCo-3DHP.json").write_text(json.dumps(coco, indent=2))


def generate_dummy_mupots():
    """Dummy MuPoTS: MultiPersonTestSet, MuPoTS-3D.json, bbox_root."""
    base = DATA / "MuPoTS"
    _make_dirs(base / "data" / "MultiPersonTestSet" / "TS1", base / "bbox_root")

    img_path = base / "data" / "MultiPersonTestSet" / "TS1" / "img_0001.jpg"
    _write_img_placeholder(img_path)

    coco = {
        "images": [{
            "id": 1, "file_name": "TS1/img_0001.jpg", "width": 256, "height": 256,
            "intrinsic": [1500, 1500, 128, 128]
        }],
        "annotations": [{
            "id": 1, "image_id": 1, "is_valid": 1, "bbox": [50, 50, 100, 150],
            "keypoints_img": [[100.0, 100.0] for _ in range(17)],
            "keypoints_cam": [[0.0, 0.0, 1000.0 + i * 10] for i in range(17)]
        }],
    }
    (base / "data" / "MuPoTS-3D.json").write_text(json.dumps(coco, indent=2))

    bbox_root = [{"image_id": 1, "bbox": [50, 50, 100, 150], "root_cam": [0.0, 0.0, 1000.0]}]
    (base / "bbox_root" / "bbox_mupots_output.json").write_text(
        json.dumps(bbox_root))
    (base / "bbox_root" / "bbox_root_mupots_output.json").write_text(
        json.dumps(bbox_root))


def _extend_dummy_subject2():
    """Dummy test uses subject 2."""
    base = DATA / "Dummy"
    _make_dirs(base / "images" / "s_2_act_01_subact_01_ca_01", base / "annotations")
    _write_img_placeholder(base / "images" / "s_2_act_01_subact_01_ca_01" / "s_2_act_01_subact_01_ca_01_000001.jpg", 1000, 1002)
    data_json = {
        "images": [{
            "id": 2000001, "file_name": "s_2_act_01_subact_01_ca_01/s_2_act_01_subact_01_ca_01_000001.jpg",
            "width": 1000, "height": 1002, "subject": 2, "action_name": "Directions",
            "action_idx": 1, "subaction_idx": 1, "cam_idx": 1, "frame_idx": 0
        }],
        "annotations": [{"id": 2000001, "image_id": 2000001, "keypoints_vis": [True] * 17, "bbox": [300, 200, 330, 420]}],
    }
    (base / "annotations" / "Dummy_subject2_data.json").write_text(json.dumps(data_json, indent=2))
    cam = {"1": {"R": np.eye(3).tolist(), "t": [0, 0, 5500], "f": [1145, 1144], "c": [512, 515]}}
    (base / "annotations" / "Dummy_subject2_camera.json").write_text(json.dumps(cam, indent=2))
    j3d = {"1": {"1": {"0": [[0.0] * 3 for _ in range(17)]}}}
    (base / "annotations" / "Dummy_subject2_joint_3d.json").write_text(json.dumps(j3d, indent=2))


def _extend_h36m_subjects():
    """H36M protocol 2 train uses subjects 1,5,6,7,8."""
    base = DATA / "Human36M"
    for subj in [5, 6, 7, 8]:
        _make_dirs(base / "images" / f"s_{subj}_act_01_subact_01_ca_01")
        _write_img_placeholder(base / "images" / f"s_{subj}_act_01_subact_01_ca_01" / f"s_{subj}_act_01_subact_01_ca_01_000001.jpg", 1000, 1002)
        data_json = {
            "images": [{
                "id": subj * 1000001, "file_name": f"s_{subj}_act_01_subact_01_ca_01/s_{subj}_act_01_subact_01_ca_01_000001.jpg",
                "width": 1000, "height": 1002, "subject": subj, "action_name": "Directions",
                "action_idx": 1, "subaction_idx": 1, "cam_idx": 1, "frame_idx": 0
            }],
            "annotations": [{"id": subj * 1000001, "image_id": subj * 1000001, "keypoints_vis": [True] * 17, "bbox": [300, 200, 330, 420]}],
        }
        (base / "annotations" / f"Human36M_subject{subj}_data.json").write_text(json.dumps(data_json, indent=2))
        cam = {"1": {"R": np.eye(3).tolist(), "t": [0, 0, 5500], "f": [1145, 1144], "c": [512, 515]}}
        (base / "annotations" / f"Human36M_subject{subj}_camera.json").write_text(json.dumps(cam, indent=2))
        j3d = {"1": {"1": {"0": [[0.0] * 3 for _ in range(17)]}}}
        (base / "annotations" / f"Human36M_subject{subj}_joint_3d.json").write_text(json.dumps(j3d, indent=2))
    # H36M test uses subjects 9,11
    for subj in [9, 11]:
        _make_dirs(base / "images" / f"s_{subj}_act_01_subact_01_ca_01")
        _write_img_placeholder(base / "images" / f"s_{subj}_act_01_subact_01_ca_01" / f"s_{subj}_act_01_subact_01_ca_01_000001.jpg", 1000, 1002)
        data_json = {
            "images": [{
                "id": subj * 1000001, "file_name": f"s_{subj}_act_01_subact_01_ca_01/s_{subj}_act_01_subact_01_ca_01_000001.jpg",
                "width": 1000, "height": 1002, "subject": subj, "action_name": "Directions",
                "action_idx": 1, "subaction_idx": 1, "cam_idx": 1, "frame_idx": 0
            }],
            "annotations": [{"id": subj * 1000001, "image_id": subj * 1000001, "keypoints_vis": [True] * 17, "bbox": [300, 200, 330, 420]}],
        }
        (base / "annotations" / f"Human36M_subject{subj}_data.json").write_text(json.dumps(data_json, indent=2))
        cam = {"1": {"R": np.eye(3).tolist(), "t": [0, 0, 5500], "f": [1145, 1144], "c": [512, 515]}}
        (base / "annotations" / f"Human36M_subject{subj}_camera.json").write_text(json.dumps(cam, indent=2))
        j3d = {"1": {"1": {"0": [[0.0] * 3 for _ in range(17)]}}}
        (base / "annotations" / f"Human36M_subject{subj}_joint_3d.json").write_text(json.dumps(j3d, indent=2))
    # Update bbox_root for test image_ids
    bbox_root = [
        {"image_id": 9 * 1000001, "bbox": [300, 200, 330, 420], "root_cam": [0.0, 0.0, 1000.0]},
        {"image_id": 11 * 1000001, "bbox": [300, 200, 330, 420], "root_cam": [0.0, 0.0, 1000.0]},
    ]
    existing = json.loads((base / "bbox_root" / "bbox_root_human36m_output.json").read_text())
    existing = existing if isinstance(existing, list) else [existing]
    (base / "bbox_root" / "bbox_root_human36m_output.json").write_text(json.dumps(bbox_root + existing))


def fix_dummy_bbox_root():
    """Ensure Dummy has bbox_root_human36m_output.json with root_cam."""
    base = DATA / "Dummy" / "bbox_root"
    base.mkdir(parents=True, exist_ok=True)
    bbox_root = [{"image_id": 1877420, "bbox": [309.17, 252.84, 326.17, 368.19],
                 "root_cam": [0.0, 0.0, 1000.0]}]
    (base / "bbox_root_human36m_output.json").write_text(json.dumps(bbox_root))


def main():
    print("Generating dummy datasets...")
    generate_dummy_h36m()
    print("  Human36M")
    generate_dummy_mpii()
    print("  MPII")
    generate_dummy_mscoco()
    print("  MSCOCO")
    generate_dummy_muco()
    print("  MuCo")
    generate_dummy_mupots()
    print("  MuPoTS")
    fix_dummy_bbox_root()
    _extend_dummy_subject2()
    _extend_h36m_subjects()
    print("  Dummy (bbox_root + subject2)")
    print("  Human36M (subjects 5,6,7,8,9,11)")
    print("Done.")


if __name__ == "__main__":
    main()
