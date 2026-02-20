#!/usr/bin/env python3
"""
Export COCO val image names from annotations to a text file (one name per line).
Used by legacy MATLAB vis; Python 2D/3D vis use common.utils.vis.
Usage: python scripts/export_coco_img_names.py [--data_dir data/MSCOCO] [--output output/vis/coco_img_name.txt]
"""
import argparse
import os.path as osp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='MSCOCO data dir (default: ROOT/data/MSCOCO)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output text file (default: ROOT/output/vis/coco_img_name.txt)')
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / 'data' / 'MSCOCO'
    annot_path = data_dir / 'annotations' / 'person_keypoints_val2017.json'
    if not annot_path.is_file():
        print(f'Annotation not found: {annot_path}')
        return
    out_path = Path(args.output) if args.output else ROOT / 'output' / 'vis' / 'coco_img_name.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from pycocotools.coco import COCO
    db = COCO(str(annot_path))
    with open(out_path, 'w') as fp:
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            name = img['file_name'].split('.')[0]
            fp.write('coco_' + name + '\n')
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
