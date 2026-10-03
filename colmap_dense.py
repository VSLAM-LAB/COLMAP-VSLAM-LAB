"""
Module: VSLAM-LAB - Baselines - colmap - colmap_dense.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2026-09-21
- Updated: 2026-10-03
- License: GPLv3 License

Optional dense stage of the colmap pipeline (dense=1), run after the sparse model and the
trajectory are written: COLMAP's image_undistorter -> patch_match_stereo -> stereo_fusion on the
best sub-model, into <exp_folder>/colmap_<exp_id>/dense/, with the fused cloud copied to
<exp_folder>/<exp_id>_dense.ply, then optionally one of COLMAP's meshers (delaunay, poisson,
advancing_front) copied to <exp_folder>/<exp_id>_mesh.ply. patch_match_stereo needs a CUDA build
of colmap and use_gpu=1; otherwise, or if any dense command fails, the stage prints a warning
and returns, leaving the sparse result and the run's success untouched.
"""

import shutil
import subprocess
from pathlib import Path

from colmap_utilities import colmap_has_cuda, detect_gpus, run

MESHERS = ('none', 'delaunay', 'poisson', 'advancing_front')


def ply_count(ply: Path, element: str) -> int:
    """Number of <element>s ('vertex' / 'face') declared in a PLY header, 0 if absent."""
    with open(ply, 'rb') as f:
        for raw in f:
            line = raw.decode('ascii', errors='replace').strip()
            if line.startswith(f"element {element} "):
                return int(line.split()[2])
            if line == "end_header":
                break
    return 0


def run_dense(exp_folder: Path, exp_id: str, exp_folder_colmap: Path, best_model: Path, rgb_path: Path,
              use_gpu: int, max_image_size: int, mesher: str) -> None:
    print("\nExecuting colmap_dense ...")
    if mesher not in MESHERS:
        print(f"    WARNING: unknown mesher '{mesher}' (expected one of {list(MESHERS)}); skipping the dense stage")
        return
    if use_gpu != 1 or not colmap_has_cuda():
        print(f"    WARNING: dense reconstruction needs the CUDA colmap build and use_gpu=1 "
              f"(use_gpu={use_gpu}, cuda build={colmap_has_cuda()}); skipping the dense stage")
        return

    gpu_index_list, num_threads = detect_gpus(use_gpu)
    dense_dir = exp_folder_colmap / "dense"
    fused_ply = dense_dir / "fused.ply"
    print(f"        sparse model        : {best_model}")
    print(f"        dense workspace     : {dense_dir}")
    print(f"        max image size      : {max_image_size} (-1: full resolution)")
    print(f"        mesher              : {mesher}")

    try:
        print("    colmap image_undistorter ...")
        run(["colmap", "image_undistorter",
             "--image_path", str(rgb_path),
             "--input_path", str(best_model),
             "--output_path", str(dense_dir),
             "--output_type", "COLMAP",
             "--max_image_size", str(max_image_size)], exit_on_error=False)

        print("    colmap patch_match_stereo ...")
        print(f"        gpu_index: {gpu_index_list}")
        run(["colmap", "patch_match_stereo",
             "--workspace_path", str(dense_dir),
             "--workspace_format", "COLMAP",
             "--PatchMatchStereo.max_image_size", str(max_image_size),
             "--PatchMatchStereo.geom_consistency", "1",
             "--PatchMatchStereo.gpu_index", gpu_index_list], exit_on_error=False)

        print("    colmap stereo_fusion ...")
        run(["colmap", "stereo_fusion",
             "--workspace_path", str(dense_dir),
             "--workspace_format", "COLMAP",
             "--input_type", "geometric",
             "--output_path", str(fused_ply),
             "--StereoFusion.max_image_size", str(max_image_size),
             "--StereoFusion.num_threads", str(num_threads)], exit_on_error=False)
        dense_copy = exp_folder / f"{exp_id}_dense.ply"
        shutil.copyfile(fused_ply, dense_copy)
        print(f"        fused point cloud: {fused_ply} ({ply_count(fused_ply, 'vertex')} points)")
        print(f"        copied to {dense_copy}")

        if mesher == 'none':
            return
        mesh_ply = dense_dir / f"meshed-{mesher}.ply"
        print(f"    colmap {mesher}_mesher ...")
        if mesher == 'poisson':
            run(["colmap", "poisson_mesher", "--input_path", str(fused_ply), "--output_path", str(mesh_ply),
                 "--PoissonMeshing.num_threads", str(num_threads)], exit_on_error=False)
        elif mesher == 'delaunay':
            run(["colmap", "delaunay_mesher", "--input_path", str(dense_dir), "--input_type", "dense",
                 "--output_path", str(mesh_ply)], exit_on_error=False)
        else:
            run(["colmap", "advancing_front_mesher", "--input_path", str(dense_dir), "--output_path", str(mesh_ply)],
                exit_on_error=False)
        mesh_copy = exp_folder / f"{exp_id}_mesh.ply"
        shutil.copyfile(mesh_ply, mesh_copy)
        print(f"        mesh: {mesh_ply} ({ply_count(mesh_ply, 'vertex')} vertices, {ply_count(mesh_ply, 'face')} faces)")
        print(f"        copied to {mesh_copy}")
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"    WARNING: dense stage aborted ({e}); the sparse reconstruction and trajectory are unaffected")
