#!/usr/bin/env python3
"""
Human3.6M dataset: official source requires login at https://vision.imar.ro/human3.6m/
This script tries a public mirror via gdown (if provided) or prints manual instructions.
Expected layout after setup:
  data/Human36M/
    images/           (per-subject frames)
    annotations/     (Human36M_subject*_data.json, _camera.json, _joint_3d.json)
    bbox_root/       (bbox_root_human36m_output.json)
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Optional: public mirror folder ID (e.g. from 3DMPPE or paper). Empty = manual only.
GDRIVE_MIRROR_ID = os.environ.get("H36M_GDRIVE_ID", "")


def main():
    if GDRIVE_MIRROR_ID:
        try:
            import gdown
            out = SCRIPT_DIR / "human36m_mirror.zip"
            url = f"https://drive.google.com/drive/folders/{GDRIVE_MIRROR_ID}"
            gdown.download_folder(url, output=str(SCRIPT_DIR), quiet=False)
            print("Extract the downloaded folder so annotations/ and images/ are under", SCRIPT_DIR)
            return
        except Exception as e:
            print("Mirror download failed:", e)
    print("Human3.6M requires agreement and login at the official site.")
    print("  1. Go to https://vision.imar.ro/human3.6m/")
    print("  2. Create an account and accept the license.")
    print("  3. Download the dataset (or use a mirror if you have one).")
    print("  4. Place annotations (e.g. Human36M_subject*_data.json, _camera.json, _joint_3d.json)")
    print("     under data/Human36M/annotations/ and images under data/Human36M/images/.")
    print("  5. Generate bbox_root (e.g. from RootNet or use provided bbox_root script).")
    print("To use a Google Drive mirror, set H36M_GDRIVE_ID=your_folder_id and run again.")
    sys.exit(0)


if __name__ == "__main__":
    main()
