#!/usr/bin/env python3
"""
Download MuCo-3DHP (scripts + optional data).
Official: https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
URL_MUCO_ZIP = "https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/content/muco-3dhp.zip"


def main():
    os.chdir(SCRIPT_DIR)
    zip_path = SCRIPT_DIR / "muco-3dhp.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("Downloading MuCo-3DHP scripts...")
        try:
            urllib.request.urlretrieve(URL_MUCO_ZIP, zip_path)
        except Exception as e:
            print("Download failed:", e)
            sys.exit(1)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(SCRIPT_DIR)
    print("Done. MuCo-3DHP.zip contains scripts; full training data may require generating from 3DHP or another source.")
    print("See project README or https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/")


if __name__ == "__main__":
    main()
