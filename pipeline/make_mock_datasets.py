"""Generate small but REAL mock datasets in every format the repo supports.

For each dataset (Human3.6M, MPII, MSCOCO, MuCo, MuPoTS) this writes the exact
on-disk structure the corresponding loader expects, populated with:
  * real person images (from the committed fallback assets), and
  * real 2D keypoints from a pretrained KeypointRCNN.

For the 3D datasets the 2D keypoints are lifted to camera-consistent 3D: with
focal f and principal point c, each joint gets a plausible root-relative depth
(anatomical template) and X,Y are back-projected so the 3D skeleton reprojects
exactly onto the real 2D detection. The result is a genuine "3D human in the
image with keypoints", not random noise.

Output: data/<Dataset>/...   (committed; small, images resized to <=640px)
Run:    python pipeline/make_mock_datasets.py
"""

import os
import os.path as osp
import json
import shutil

import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

from common import REPO_DIR, FALLBACK_DIR, ensure_dir, get_logger

log = get_logger("mockgen")
DATA_DIR = osp.join(REPO_DIR, "data")
COCO_PERSON = 1
MAX_SIDE = 640
ROOT_DEPTH = 4000.0  # mm
FOCAL = [1500.0, 1500.0]

COCO_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho",
              "l_elb", "r_elb", "l_wri", "r_wri", "l_hip", "r_hip", "l_kne",
              "r_kne", "l_ank", "r_ank"]

# root-relative depth offset (mm) per canonical joint -> plausible 3D structure
DEPTH_OFF = {"pelvis": 0, "spine": -20, "torso": -20, "thorax": -40, "neck": -40,
             "head": -60, "nose": -60, "head_top": -70, "l_sho": -40, "r_sho": -40,
             "l_elb": -10, "r_elb": -10, "l_wri": 40, "r_wri": 40, "l_hip": 0,
             "r_hip": 0, "l_kne": 30, "r_kne": 30, "l_ank": 60, "r_ank": 60}

# joint order per dataset, by canonical name (hands->wrists, toes->ankles)
MPII16 = ["r_ank", "r_kne", "r_hip", "l_hip", "l_kne", "l_ank", "pelvis", "thorax",
          "neck", "head_top", "r_wri", "r_elb", "r_sho", "l_sho", "l_elb", "l_wri"]
H36M17 = ["pelvis", "r_hip", "r_kne", "r_ank", "l_hip", "l_kne", "l_ank", "torso",
          "neck", "nose", "head_top", "l_sho", "l_elb", "l_wri", "r_sho", "r_elb", "r_wri"]
MUCO21 = ["head_top", "thorax", "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri",
          "r_hip", "r_kne", "r_ank", "l_hip", "l_kne", "l_ank", "pelvis", "spine",
          "head", "r_wri", "l_wri", "r_ank", "l_ank"]
MUPOTS17 = MUCO21[:17]


def _mid(a, b):
    return np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2])])


def named_points(kps):
    """kps: (17,3) [x,y,score] -> dict of canonical named points (with vis)."""
    p = {COCO_NAMES[i]: kps[i].astype(np.float64) for i in range(17)}
    p["pelvis"] = _mid(p["l_hip"], p["r_hip"])
    p["thorax"] = _mid(p["l_sho"], p["r_sho"])
    p["neck"] = p["thorax"].copy()
    p["spine"] = _mid(p["thorax"], p["pelvis"])
    p["torso"] = p["spine"].copy()
    p["head"] = p["nose"].copy()
    p["head_top"] = p["nose"] + 0.4 * (p["nose"] - p["thorax"])
    return p


def order_2d(p, order):
    """-> (N,3) [x,y,vis] in the dataset's joint order."""
    out = np.zeros((len(order), 3))
    for i, name in enumerate(order):
        x, y, s = p[name]
        out[i] = [x, y, 2.0 if s > 0.3 else (1.0 if s > 0 else 0.0)]
    return out


def lift_3d(p, order, f, c, root_depth=ROOT_DEPTH):
    """Camera-consistent 3D: returns (cam (N,3), img2d (N,2)) that reproject exactly."""
    cam = np.zeros((len(order), 3))
    img2d = np.zeros((len(order), 2))
    for i, name in enumerate(order):
        x, y = p[name][0], p[name][1]
        z = root_depth + DEPTH_OFF.get(name, 0)
        cam[i] = [(x - c[0]) / f[0] * z, (y - c[1]) / f[1] * z, z]
        img2d[i] = [x, y]
    return cam, img2d


def _load_models():
    kw = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    kp = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=kw).eval()
    return kp


@torch.no_grad()
def collect_people(kp_model, max_images=5, det_score=0.85):
    """Return list of (resized_bgr, [persons]) from the committed real images."""
    files = sorted(f for f in os.listdir(FALLBACK_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    out = []
    for fn in files:
        bgr = cv2.imread(osp.join(FALLBACK_DIR, fn))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = kp_model([to_tensor(rgb)])[0]
        people = []
        for i in range(len(pred["scores"])):
            if int(pred["labels"][i]) != COCO_PERSON or float(pred["scores"][i]) < det_score:
                continue
            kps = pred["keypoints"][i].numpy().copy()
            kps[:, 2] = pred["keypoints_scores"][i].numpy()
            x1, y1, x2, y2 = pred["boxes"][i].numpy()
            people.append({"kps": kps, "bbox": [float(x1), float(y1),
                                                float(x2 - x1), float(y2 - y1)]})
        if people:
            out.append((bgr, people))
        if len(out) >= max_images:
            break
    if not out:
        raise RuntimeError("No person images found in fallback assets.")
    log.info("Collected %d images with people for mock generation", len(out))
    return out


def _fresh(d):
    if osp.isdir(d):
        shutil.rmtree(d)
    return ensure_dir(d)


def gen_mscoco(samples):
    root = osp.join(DATA_DIR, "MSCOCO")
    img_dir = _fresh(osp.join(root, "images", "train2017"))
    ensure_dir(osp.join(root, "annotations"))
    images, anns = [], []
    aid = 0
    for iid, (bgr, people) in enumerate(samples):
        fn = "%012d.jpg" % iid
        cv2.imwrite(osp.join(img_dir, fn), bgr)
        h, w = bgr.shape[:2]
        images.append({"id": iid, "file_name": fn, "width": w, "height": h})
        for person in people:
            p = named_points(person["kps"])
            kp2d = order_2d(p, COCO_NAMES)
            anns.append({"id": aid, "image_id": iid, "category_id": 1,
                         "iscrowd": 0, "bbox": person["bbox"],
                         "area": person["bbox"][2] * person["bbox"][3],
                         "num_keypoints": int((kp2d[:, 2] > 0).sum()),
                         "keypoints": [round(v, 2) for v in kp2d.reshape(-1)]})
            aid += 1
    cats = [{"id": 1, "name": "person", "keypoints": COCO_NAMES, "skeleton": []}]
    with open(osp.join(root, "annotations", "person_keypoints_train2017.json"), "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    return len(anns)


def gen_mpii(samples):
    root = osp.join(DATA_DIR, "MPII")
    img_dir = _fresh(osp.join(root, "images"))
    ensure_dir(osp.join(root, "annotations"))
    images, anns = [], []
    aid = 0
    for iid, (bgr, people) in enumerate(samples):
        fn = osp.join("images", "img_%d.jpg" % iid)
        cv2.imwrite(osp.join(root, fn), bgr)
        h, w = bgr.shape[:2]
        images.append({"id": iid, "file_name": fn, "width": w, "height": h})
        for person in people:
            p = named_points(person["kps"])
            kp2d = order_2d(p, MPII16)
            anns.append({"id": aid, "image_id": iid, "category_id": 1, "iscrowd": 0,
                         "bbox": person["bbox"], "num_keypoints": int((kp2d[:, 2] > 0).sum()),
                         "keypoints": [round(v, 2) for v in kp2d.reshape(-1)]})
            aid += 1
    with open(osp.join(root, "annotations", "train.json"), "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "person"}]}, f)
    return len(anns)


def gen_human36m(samples):
    root = osp.join(DATA_DIR, "Human36M")
    annot_dir = ensure_dir(osp.join(root, "annotations"))
    _fresh(osp.join(root, "images"))
    subjects = [1, 5, 6, 7, 8]   # H36M train subjects (loader opens a file per subject)
    per = {s: {"data": {"images": [], "annotations": []},
               "cam": {"1": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "f": FOCAL, "c": [0, 0]}},
               "joints": {}} for s in subjects}
    iid = 0
    for i, (bgr, people) in enumerate(samples):
        s = subjects[i % len(subjects)]
        action_idx = 2 + i // len(subjects)   # unique per subject
        stem = "s_%02d_act_%02d_subact_01_ca_01" % (s, action_idx)
        ensure_dir(osp.join(root, "images", stem))
        fn = osp.join(stem, "%s_%06d.jpg" % (stem, 0))
        cv2.imwrite(osp.join(root, "images", fn), bgr)
        h, w = bgr.shape[:2]
        c = [w / 2.0, h / 2.0]
        per[s]["cam"]["1"]["c"] = c
        p = named_points(people[0]["kps"])
        cam3d, _ = lift_3d(p, H36M17, FOCAL, c)   # world == cam (R=I, t=0)
        per[s]["data"]["images"].append({"id": iid, "file_name": fn, "width": w, "height": h,
                                         "subject": s, "action_name": "Directions",
                                         "action_idx": action_idx, "subaction_idx": 1,
                                         "cam_idx": 1, "frame_idx": 0})
        per[s]["data"]["annotations"].append({"id": iid, "image_id": iid, "bbox": people[0]["bbox"]})
        per[s]["joints"].setdefault(str(action_idx), {})["1"] = {"0": cam3d.tolist()}
        iid += 1
    for s in subjects:
        with open(osp.join(annot_dir, "Human36M_subject%d_data.json" % s), "w") as f:
            json.dump(per[s]["data"], f)
        with open(osp.join(annot_dir, "Human36M_subject%d_camera.json" % s), "w") as f:
            json.dump(per[s]["cam"], f)
        with open(osp.join(annot_dir, "Human36M_subject%d_joint_3d.json" % s), "w") as f:
            json.dump(per[s]["joints"], f)
    return iid


def gen_muco(samples):
    root = osp.join(DATA_DIR, "MuCo")
    img_dir = _fresh(osp.join(root, "images"))
    images, anns = [], []
    aid = 0
    for iid, (bgr, people) in enumerate(samples):
        fn = osp.join("images", "img_%d.jpg" % iid)
        cv2.imwrite(osp.join(root, fn), bgr)
        h, w = bgr.shape[:2]
        c = [w / 2.0, h / 2.0]
        images.append({"id": iid, "file_name": fn, "width": w, "height": h, "f": FOCAL, "c": c})
        for k, person in enumerate(people):
            p = named_points(person["kps"])
            cam3d, img2d = lift_3d(p, MUCO21, FOCAL, c, ROOT_DEPTH + 500 * k)
            anns.append({"id": aid, "image_id": iid, "bbox": person["bbox"],
                         "keypoints_cam": cam3d.tolist(), "keypoints_img": img2d.tolist()})
            aid += 1
    with open(osp.join(root, "MuCo-3DHP.json"), "w") as f:
        json.dump({"images": images, "annotations": anns}, f)
    return aid


def gen_mupots(samples):
    root = osp.join(DATA_DIR, "MuPoTS")
    data_dir = osp.join(root, "data")
    img_root = _fresh(osp.join(data_dir, "MultiPersonTestSet"))
    images, anns = [], []
    aid = 0
    for iid, (bgr, people) in enumerate(samples):
        ts = ensure_dir(osp.join(img_root, "TS%d" % (iid + 1)))
        cv2.imwrite(osp.join(ts, "img_000001.jpg"), bgr)
        fn = osp.join("TS%d" % (iid + 1), "img_000001.jpg")
        h, w = bgr.shape[:2]
        c = [w / 2.0, h / 2.0]
        images.append({"id": iid, "file_name": fn, "width": w, "height": h,
                       "intrinsic": [FOCAL[0], FOCAL[1], c[0], c[1]]})
        for k, person in enumerate(people):
            p = named_points(person["kps"])
            cam3d, img2d = lift_3d(p, MUPOTS17, FOCAL, c, ROOT_DEPTH + 500 * k)
            anns.append({"id": aid, "image_id": iid, "is_valid": 1, "bbox": person["bbox"],
                         "keypoints_cam": cam3d.tolist(), "keypoints_img": img2d.tolist()})
            aid += 1
    ensure_dir(data_dir)
    with open(osp.join(data_dir, "MuPoTS-3D.json"), "w") as f:
        json.dump({"images": images, "annotations": anns}, f)
    return aid


def main():
    kp = _load_models()
    samples = collect_people(kp)
    summary = {
        "MSCOCO (2D)": gen_mscoco(samples),
        "MPII (2D)": gen_mpii(samples),
        "Human36M (3D)": gen_human36m(samples),
        "MuCo (3D)": gen_muco(samples),
        "MuPoTS (3D)": gen_mupots(samples),
    }
    with open(osp.join(DATA_DIR, "MOCK_DATASETS.json"), "w") as f:
        json.dump({"note": "Small real-image mock datasets for each format. "
                          "2D from KeypointRCNN; 3D camera-consistently lifted.",
                   "instances_per_dataset": summary}, f, indent=2)
    for k, v in summary.items():
        log.info("  %-16s %d person instances", k, v)
    log.info("Mock datasets written under %s", DATA_DIR)


if __name__ == "__main__":
    main()
