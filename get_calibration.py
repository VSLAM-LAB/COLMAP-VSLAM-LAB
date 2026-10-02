"""
Module: VSLAM-LAB - Baselines - colmap - get_calibration.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 1.1
- Created: 2024-07-12
- Updated: 2026-10-03
- License: GPLv3 License

Reads one camera of a VSLAM-LAB calibration yaml as (model, [fx, fy, cx, cy, *distortion]).
The model is the distortion_type when the camera has distortion fields, else its cam_model
(so 'pinhole' and 'unknown' come through unchanged). Importable (colmap_utilities.colmap_camera)
and runnable: `python get_calibration.py <calibration_yaml> <camera_name>` prints the same
fields space-separated on one line.
"""

import argparse
from pathlib import Path

import yaml


def get_camera_intrinsics(calibration_yaml: str | Path, cam_name: str) -> tuple[str, list[float]]:
    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cam = next(c for c in data.get('cameras', []) if c['cam_name'] == cam_name)

    fx, fy = cam['focal_length'][0], cam['focal_length'][1]
    cx, cy = cam['principal_point'][0], cam['principal_point'][1]
    params = [float(fx), float(fy), float(cx), float(cy)]

    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    if has_dist:
        return cam['distortion_type'], params + [float(d) for d in cam['distortion_coefficients']]
    return cam['cam_model'], params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_yaml", help="Path to the calibration YAML")
    parser.add_argument("camera_name", help="camera_name")
    args = parser.parse_args()

    model, params = get_camera_intrinsics(args.calibration_yaml, args.camera_name)
    print(model, *params)
