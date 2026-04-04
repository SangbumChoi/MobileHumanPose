#!/usr/bin/env python3
"""
Download MuPoTS-3D evaluation set.
Official: https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# MuPoTS-3D eval package (annotations + possibly image list)
URL_MUPOTS_ZIP = "https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/content/mupots-3d-eval.zip"


def main():
    os.chdir(SCRIPT_DIR)
    zip_path = SCRIPT_DIR / "mupots-3d-eval.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("Downloading MuPoTS-3D evaluation package...")
        try:
            urllib.request.urlretrieve(URL_MUPOTS_ZIP, zip_path)
        except Exception as e:
            print("Download failed:", e)
            sys.exit(1)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(SCRIPT_DIR)
    # Expected: data/MuPoTS/data/MuPoTS-3D.json, and images in data/MuPoTS/data/MultiPersonTestSet/
    data_dir = SCRIPT_DIR / "data"
    if data_dir.exists():
        print("Extracted to", data_dir)
    else:
        print("Extracted to", SCRIPT_DIR)
    print("If images are not included, download MultiPersonTestSet from the project page.")
    print("See https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/")


if __name__ == "__main__":
    main()
