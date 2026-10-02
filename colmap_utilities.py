"""
Module: VSLAM-LAB - Baselines - colmap - colmap_utilities.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 1.0
- Created: 2026-10-03
- Updated: 2026-10-03
- License: GPLv3 License

Helpers shared by the colmap pipeline stages (vslamlab_colmap.py, colmap_matcher.py,
colmap_mapper.py): running colmap commands, GPU / thread detection, and mapping a VSLAM-LAB
calibration onto a COLMAP camera model.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from get_calibration import get_camera_intrinsics

COLMAP_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], capture: bool = False) -> str:
    """Run a command, exiting with its return code if it fails. Returns the merged output when capture=True."""
    if capture:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"    ERROR: '{' '.join(cmd[:2])}' failed with return code {result.returncode}")
        if capture:
            print(result.stdout)
        sys.exit(result.returncode)
    return result.stdout if capture else ""


def shell_output(cmd: list[str]) -> str:
    """stdout+stderr of a command, '' if it cannot run (e.g. no nvidia-smi)."""
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    except FileNotFoundError as e:
        return str(e)


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.rstrip("\n").split("\n"))


def detect_gpus(use_gpu: int) -> tuple[str, int]:
    """GPU index list for COLMAP's gpu_index options and the CPU thread count.

    Extraction / matching run as a single job across all usable GPUs, since COLMAP spawns one
    worker per listed GPU index and splits the workload internally. When CUDA_VISIBLE_DEVICES is
    set (a scheduler restricted this job to a device set, often MIG slice UUIDs on HPC that
    nvidia-smi does not enumerate per job), CUDA remaps the listed devices to ordinals 0..N-1 inside
    the process and COLMAP's gpu_index is just cudaSetDevice(ordinal), so the ordinals are used
    rather than the real indices. The thread count comes from the scheduling affinity (what `nproc`
    reports) rather than COLMAP's num_threads=-1 auto-detection, which on cgroup-limited nodes often
    sees the whole node's cores.
    """
    print("    detecting GPUs ...")
    print("        nvidia-smi -L:")
    print(indent(shell_output(["nvidia-smi", "-L"]), " " * 12))
    print("        nvidia-smi --query-gpu=index,name,uuid --format=csv:")
    print(indent(shell_output(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv"]), " " * 12))
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"        CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    if cuda_visible_devices:
        num_gpus = len([d for d in cuda_visible_devices.split(",") if d.strip()])
        gpu_ids = list(range(num_gpus))
    else:
        out = shell_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        gpu_ids = [int(tok) for tok in out.split() if tok.isdigit()]
    if len(gpu_ids) < 1 or use_gpu == 0:
        gpu_ids = [0]
    gpu_index_list = ",".join(str(i) for i in gpu_ids)
    print(f"        use_gpu: {use_gpu}")
    print(f"        detected gpu_ids: {' '.join(str(i) for i in gpu_ids)}")
    print(f"        gpu_index_list passed to colmap: {gpu_index_list}")

    num_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    print(f"        num_threads: {num_threads}")
    return gpu_index_list, num_threads


def rgb_path_from_csv(sequence_path: Path, rgb_csv: Path, camera_name: str) -> Path:
    """Frame folder of camera_name as the csv names it: the run pipeline may point path_rgb_0 at a
    generated folder (refrax_0 for 'refraction: refrax') rather than rgb_0."""
    df = pd.read_csv(rgb_csv, nrows=1)
    rgb_dir = str(df[f"path_{camera_name}"].iloc[0]).split("/")[0]
    return sequence_path / rgb_dir


def colmap_camera(calibration_yaml: Path, camera_name: str) -> tuple[str, str, str]:
    """(vslamlab calibration model, COLMAP camera model, comma-separated COLMAP params or '')."""
    calibration_model, params = get_camera_intrinsics(calibration_yaml, camera_name)
    params_csv = ",".join(str(p) for p in params)
    if calibration_model == "unknown":
        return calibration_model, "OPENCV", ""
    if calibration_model == "pinhole":
        return calibration_model, "PINHOLE", params_csv
    if calibration_model == "radtan4":
        return calibration_model, "OPENCV", params_csv
    if calibration_model == "radtan5":
        return calibration_model, "FULL_OPENCV", params_csv + ",0,0,0"
    if calibration_model == "equid4":
        return calibration_model, "OPENCV_FISHEYE", params_csv
    print(f"Unknown calibration_model: {calibration_model}")
    sys.exit(1)
