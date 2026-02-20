#!/usr/bin/env python3
"""
Export MuPoTS-3D image ids (folder_id frame_id) to a text file.
Used by legacy MATLAB vis; Python 2D/3D vis use common.utils.vis.
Usage: python scripts/export_mupots_img_names.py [--data_dir data/MuPoTS] [--output output/vis/mupots_img_name.txt]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='MuPoTS data dir (default: ROOT/data/MuPoTS)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output text file (default: ROOT/output/vis/mupots_img_name.txt)')
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / 'data' / 'MuPoTS'
    annot_path = data_dir / 'data' / 'MuPoTS-3D.json'
    if not annot_path.is_file():
        print(f'Annotation not found: {annot_path}')
        return
    out_path = Path(args.output) if args.output else ROOT / 'output' / 'vis' / 'mupots_img_name.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from pycocotools.coco import COCO
    db = COCO(str(annot_path))
    with open(out_path, 'w') as fp:
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            parts = img['file_name'].split('/')
            folder_id = int(parts[0][2:])
            frame_id = int(parts[1].split('.')[0][4:])
            fp.write(f'{folder_id} {frame_id}\n')
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
