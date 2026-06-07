"""CrawlPipeline dataset — feeds the pipeline's crawled + auto-annotated images
into the *original* MobileHumanPose training machinery (base.Trainer).

It is a thin specialisation of the repo's ``MSCOCO`` train loader: the crawler
(``pipeline/stage03_annotate.py``) already emits standard COCO-17 person
keypoints, so here we just point the same 2D loading logic at the pipeline's
``work/`` outputs and add the Thorax/Pelvis joints exactly like MSCOCO, giving
the repo's canonical 19-joint, depth-less (``joints_have_depth=False``) training
target.

Optionally honours the Stage-4 embedding-rebalancing weights by oversampling
rare samples, so the original (shuffle-based) DataLoader trains on the
rebalanced distribution without any change to base.py.
"""

import os.path as osp
import json

import numpy as np
from pycocotools.coco import COCO

from config import cfg
from utils.pose_utils import process_bbox

# pipeline/work resolved relative to this file (repo/data/CrawlPipeline/..)
_REPO = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
_WORK = osp.join(_REPO, 'pipeline', 'work')


class CrawlPipeline:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join(_WORK, '02_curated')
        self.annot_path = osp.join(_WORK, '03_annotated', 'annotations.json')
        self.weights_path = osp.join(_WORK, '04_balanced', 'sample_weights.json')

        # Canonical repo 2D-COCO convention (identical to MSCOCO 'train').
        self.joint_num = 19  # 17 COCO + Thorax + Pelvis
        self.joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
                            'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                            'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee',
                            'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
        self.flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.skeleton = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
                         (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12))
        self.joints_have_depth = False
        self.eval_joint = tuple(range(17))
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')
        self.lhip_idx = self.joints_name.index('L_Hip')
        self.rhip_idx = self.joints_name.index('R_Hip')

        self.data = self.load_data()

    def _add_thorax_pelvis(self, joint_img):
        thorax = (joint_img[self.lshoulder_idx, :] + joint_img[self.rshoulder_idx, :]) * 0.5
        thorax[2] = joint_img[self.lshoulder_idx, 2] * joint_img[self.rshoulder_idx, 2]
        pelvis = (joint_img[self.lhip_idx, :] + joint_img[self.rhip_idx, :]) * 0.5
        pelvis[2] = joint_img[self.lhip_idx, 2] * joint_img[self.rhip_idx, 2]
        return np.concatenate((joint_img, thorax.reshape(1, 3), pelvis.reshape(1, 3)), axis=0)

    def load_data(self):
        assert osp.exists(self.annot_path), \
            "Missing %s -- run pipeline stages 1-3 first." % self.annot_path
        db = COCO(self.annot_path)
        id2name = {im['id']: im['file_name'] for im in db.dataset['images']}

        weights = {}
        if self.data_split == 'train' and osp.exists(self.weights_path):
            with open(self.weights_path) as f:
                weights = json.load(f)

        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            if ann.get('iscrowd', 0) or ann.get('num_keypoints', 0) == 0:
                continue
            img = db.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            bbox = process_bbox(np.array(ann['bbox']), width, height)
            if bbox is None:
                continue

            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            joint_img = self._add_thorax_pelvis(joint_img)
            joint_vis = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
            joint_img[:, 2] = 0  # 2D supervision only

            item = {
                'img_path': osp.join(self.img_dir, id2name[ann['image_id']]),
                'bbox': bbox,
                'joint_img': joint_img,        # [x, y, 0]
                'joint_vis': joint_vis,
                'joint_cam': np.zeros((self.joint_num, 3)),  # unused (2D)
                'root_cam': np.zeros((3,)),
                'f': np.array([1500, 1500]),
                'c': np.array([width / 2, height / 2]),
            }
            # Stage-4 rebalancing: oversample by (rounded) embedding weight.
            reps = max(1, int(round(weights.get(str(aid), 1.0)))) if self.data_split == 'train' else 1
            data.extend([item] * reps)

        print('CrawlPipeline (%s): %d samples (oversampled from %d annotations)'
              % (self.data_split, len(data), len(db.anns)))
        return data

    def evaluate(self, preds, result_dir):
        """No 3D ground truth for crawled data; save 2D predictions in output grid."""
        out = osp.join(result_dir, 'crawlpipeline_preds_2d.npy')
        np.save(out, np.asarray(preds))
        msg = 'CrawlPipeline: saved %d 2D predictions to %s (no quantitative GT).' % (len(preds), out)
        print(msg)
        return msg
