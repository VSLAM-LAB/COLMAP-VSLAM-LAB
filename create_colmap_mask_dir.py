"""
Module: VSLAM-LAB - Baselines - colmap - create_colmap_mask_dir.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5)
- Version: 1.0
- Created: 2026-08-28
- Updated: 2026-08-28
- License: GPLv3 License

Turns the path_mask_<i> column of an rgb_exp.csv (written by the run pipeline for
'segmentation: mask2former', 'refraction: refrax', or datasets that ship masks; 1 = usable pixel,
0 = masked out, which is COLMAP's own convention: no features where the mask is 0) into what
colmap feature_extractor accepts, and prints one line the calling shell script evals:
  camera_mask:<png>   every frame shares one mask (refrax's mask.png) -> --ImageReader.camera_mask_path
  mask_dir:<dir>      per-frame masks -> --ImageReader.mask_path: <dir>/<image name>.png symlinks,
                      one per frame (COLMAP looks masks up by image name + '.png')
  none                the csv has no mask column for this camera (or a mask file is missing)
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def create_colmap_mask_dir(rgb_csv: str, camera_name: str, sequence_path: str, mask_dir: str) -> str:
    df = pd.read_csv(rgb_csv)
    cam_idx = camera_name.rsplit("_", 1)[-1]
    mask_col, path_col = f"path_mask_{cam_idx}", f"path_{camera_name}"
    if mask_col not in df.columns:
        print(f"    no '{mask_col}' column in {os.path.basename(rgb_csv)}; extracting features without masks")
        return "none"

    masks = [Path(sequence_path) / p for p in df[mask_col]]
    missing = [m for m in masks if not m.exists()]
    if missing:
        print(f"    {len(missing)}/{len(masks)} mask files missing (e.g. {missing[0]}); extracting features without masks")
        return "none"

    if len(set(masks)) == 1:
        return f"camera_mask:{masks[0].resolve()}"

    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for image, mask in zip(df[path_col], masks):
        link = mask_dir / f"{os.path.basename(image)}.png"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(mask.resolve(), link)
    return f"mask_dir:{mask_dir.resolve()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb_csv", help="Path to the experiment's rgb csv")
    parser.add_argument("camera_name", help="camera_name (e.g. rgb_0)")
    parser.add_argument("sequence_path", help="Sequence folder the csv paths are relative to")
    parser.add_argument("mask_dir", help="Where to build the per-frame mask directory if needed")
    args = parser.parse_args()
    print(create_colmap_mask_dir(args.rgb_csv, args.camera_name, args.sequence_path, args.mask_dir))
