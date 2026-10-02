"""
Module: VSLAM-LAB - Baselines - colmap - colmap_mapper.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2024-07-12
- Updated: 2026-10-03
- License: GPLv3 License

Mapping stage of the colmap pipeline (Python port of colmap_mapper.sh): runs COLMAP's incremental
mapper or GLOMAP's global mapper on the database, picks the sub-model with the most registered
images and exports it as TXT into the colmap folder. Called by vslamlab_colmap.py.
"""

import re
from pathlib import Path

from colmap_utilities import colmap_camera, run


def registered_images(model_dir: Path) -> int:
    out = run(["colmap", "model_analyzer", "--path", str(model_dir)], capture=True)
    match = re.search(r"Registered images: (\d+)", out)
    return int(match.group(1)) if match else 0


def select_best_model(exp_folder_colmap: Path) -> tuple[Path, int]:
    """COLMAP may split the scene into several sub-models (<exp_folder_colmap>/0, /1, ...), and the
    largest one is not necessarily /0. Returns the sub-model with the most registered images and
    records the choice in <exp_folder_colmap>/best_model for the rest of the pipeline (gui, ...)."""
    best_model, best_num_images = None, -1
    for model_dir in sorted(p for p in exp_folder_colmap.iterdir() if p.is_dir() and p.name.isdigit()):
        if not (model_dir / "images.bin").is_file():
            continue
        num_images = registered_images(model_dir)
        print(f"        sub-model {model_dir.name}: {num_images} registered images")
        if num_images > best_num_images:
            best_model, best_num_images = model_dir, num_images
    if best_model is None or best_num_images == 0:  # glomap writes an empty /0 when its pose graph is empty
        raise SystemExit(f"    no reconstruction produced (no sub-model with registered images in {exp_folder_colmap})")
    (exp_folder_colmap / "best_model").write_text(f"{best_model}\n")
    return best_model, best_num_images


def run_mapper(exp_folder_colmap: Path, rgb_path: Path, calibration_yaml: Path, camera_name: str,
               mapper_type: str, optimize_intrinsics: int) -> Path:
    """Reconstruct from <exp_folder_colmap>/colmap_database.db; returns the best sub-model folder."""
    print("Executing colmap_mapper ...")

    calibration_model, _, _ = colmap_camera(calibration_yaml, camera_name)
    print(f"        camera model : {calibration_model}")

    # optimize_intrinsics (0/1, default 1): whether bundle adjustment refines the camera intrinsics.
    # With 1, focal length and distortion (extra params) are refined and the principal point stays
    # fixed, matching COLMAP's own defaults; with 0 the intrinsics from the calibration yaml are kept
    # as given. An 'unknown' calibration model has no intrinsics to keep (the matcher started COLMAP
    # from a guess), so it always refines regardless of the flag.
    if calibration_model == "unknown" and optimize_intrinsics != 1:
        print(f"        WARNING: optimize_intrinsics={optimize_intrinsics} ignored: camera model is 'unknown' (no intrinsics to keep fixed), refining intrinsics")
        optimize_intrinsics = 1
    refine = "1" if optimize_intrinsics == 1 else "0"
    ba_refine_focal_length, ba_refine_principal_point, ba_refine_extra_params = refine, "0", refine
    print(f"        optimize_intrinsics: {optimize_intrinsics} (ba_refine_focal_length={ba_refine_focal_length}, "
          f"ba_refine_principal_point={ba_refine_principal_point}, ba_refine_extra_params={ba_refine_extra_params})")

    database = exp_folder_colmap / "colmap_database.db"
    if mapper_type == "glomap":
        print("    global mapper (GLOMAP) ...")
        command, prefix = "global_mapper", "GlobalMapper"
    else:
        print("    colmap mapper (COLMAP) ...")
        command, prefix = "mapper", "Mapper"
    run(["colmap", command,
         "--database_path", str(database),
         "--image_path", str(rgb_path),
         "--output_path", str(exp_folder_colmap),
         f"--{prefix}.ba_refine_focal_length", ba_refine_focal_length,
         f"--{prefix}.ba_refine_principal_point", ba_refine_principal_point,
         f"--{prefix}.ba_refine_extra_params", ba_refine_extra_params])

    best_model, best_num_images = select_best_model(exp_folder_colmap)

    print(f"    colmap model_converter (sub-model {best_model.name}, {best_num_images} images) ...")
    run(["colmap", "model_converter", "--input_path", str(best_model), "--output_path", str(exp_folder_colmap), "--output_type", "TXT"])
    return best_model
