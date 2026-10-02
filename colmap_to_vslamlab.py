"""
Module: VSLAM-LAB - Baselines - colmap - colmap_to_vslamlab.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 1.1
- Created: 2024-07-12
- Updated: 2026-10-03
- License: GPLv3 License

Writes <exp_folder>/<exp_id>_KeyFrameTrajectory.csv (VSLAM-LAB's TUM-style csv, camera-to-world
poses) from the TXT model <exp_folder>/colmap_<exp_id>/images.txt, taking each image's timestamp
from the experiment's rgb csv (COLMAP image ids are 1-based row indices of that csv, as the image
list was written in csv order). Importable (vslamlab_colmap.py) and runnable with the positional
arguments the old shell pipeline used.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def get_colmap_keyframes(images_file: str | Path, number_of_header_lines: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image ids, t_wc, q_wc xyzw) sorted by image id, quaternion signs made continuous."""
    print(f"get_colmap_keyframes: {images_file}")

    image_id, q_wc_xyzw, t_wc = [], [], []
    with open(images_file, 'r') as file:
        for _ in range(number_of_header_lines):
            file.readline()
        while True:
            line1 = file.readline()
            if not line1:
                break
            elements = line1.split()
            image_id.append(int(elements[0]))

            qw, qx, qy, qz = (float(e) for e in elements[1:5])
            t_cw_i = np.array([float(e) for e in elements[5:8]])
            q_wc_i = R.from_quat([qx, qy, qz, qw]).inv()
            q_wc_xyzw.append(q_wc_i.as_quat())
            t_wc.append(-q_wc_i.as_matrix() @ t_cw_i)

            file.readline()  # POINTS2D line

    image_id, q_wc_xyzw, t_wc = np.array(image_id), np.array(q_wc_xyzw), np.array(t_wc)
    order = image_id.argsort()
    image_id, q_wc_xyzw, t_wc = image_id[order], q_wc_xyzw[order], t_wc[order]

    for i in range(1, len(q_wc_xyzw)):
        if np.dot(q_wc_xyzw[i - 1], q_wc_xyzw[i]) < 0:
            q_wc_xyzw[i] = -q_wc_xyzw[i]
    return image_id, t_wc, q_wc_xyzw


def write_trajectory_tum_format(file_name: str | Path, image_ts: np.ndarray, t_wc: np.ndarray, q_wc_xyzw: np.ndarray) -> None:
    print(f"writeTrajectoryTUMformat: {file_name}")
    data = np.hstack((image_ts.reshape(-1, 1), t_wc, q_wc_xyzw))
    data = data[data[:, 0].argsort()]
    with open(file_name, 'w', newline='') as file:
        file.write('ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n')
        for row in data:
            file.write(','.join(f'{x:.15f}' for x in row) + '\n')


def get_timestamps(rgb_csv: str | Path, camera_name: str) -> list[float]:
    print(f"getTimestamps: {rgb_csv}")
    return pd.read_csv(rgb_csv)[f'ts_{camera_name} (ns)'].to_list()


def colmap_to_vslamlab(exp_folder: str | Path, exp_id: str, rgb_csv: str | Path, camera_name: str) -> Path:
    images_file = Path(exp_folder) / f'colmap_{exp_id}' / 'images.txt'
    image_id, t_wc, q_wc_xyzw = get_colmap_keyframes(images_file)

    image_ts = np.array(get_timestamps(rgb_csv, camera_name))
    timestamps = np.array([float(image_ts[i - 1]) for i in image_id])

    trajectory_csv = Path(exp_folder) / f'{exp_id}_KeyFrameTrajectory.csv'
    write_trajectory_tum_format(trajectory_csv, timestamps, t_wc, q_wc_xyzw)
    return trajectory_csv


if __name__ == "__main__":
    # positional: sequence_path exp_folder exp_id verbose rgb_csv camera_name (sequence_path/verbose unused)
    _, exp_folder_, exp_id_, _, rgb_csv_, camera_name_ = sys.argv[1:7]
    colmap_to_vslamlab(exp_folder_, exp_id_, rgb_csv_, camera_name_)
