"""
Module: VSLAM-LAB - Baselines - colmap - colmap_matcher.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 2.0
- Created: 2024-07-12
- Updated: 2026-10-03
- License: GPLv3 License

Feature extraction and matching stage of the colmap pipeline (Python port of colmap_matcher.sh):
creates the COLMAP database, extracts features for the frames listed in the experiment's rgb csv
(optionally masked) and matches them exhaustively or sequentially. Called by vslamlab_colmap.py.
"""

from pathlib import Path

from colmap_utilities import PROFILER, Settings, colmap_camera, detect_gpus, run, settings_args
from create_colmap_image_list import create_colmap_image_list
from create_colmap_mask_dir import create_colmap_mask_dir

# matching_type -> (FeatureExtraction.type, FeatureMatching.type), named <extractor>_<matcher>.
# ALIKED and LoMa are ONNX models that COLMAP downloads once into ~/.cache/colmap. LoMa (ECCV26,
# DeDoDe architecture): LOMA_B is a 256-dim descriptor with dedicated matchers B (LightGlue-sized),
# R (rotation-augmented), L and G (larger, slower, more accurate); LOMA_B128 is a lighter 128-dim
# descriptor with its own B128 matcher; LOMA_BRUTEFORCE (cosine similarity) takes either.
MATCHING_TYPES: dict[str, tuple[str, str]] = {
    'sift_bruteforce': ('SIFT', 'SIFT_BRUTEFORCE'),
    'sift_lightglue': ('SIFT', 'SIFT_LIGHTGLUE'),
    'aliked_bruteforce': ('ALIKED_N16ROT', 'ALIKED_BRUTEFORCE'),
    'aliked_lightglue': ('ALIKED_N16ROT', 'ALIKED_LIGHTGLUE'),
    'loma_b_bruteforce': ('LOMA_B', 'LOMA_BRUTEFORCE'),
    'loma_b_loma': ('LOMA_B', 'LOMA_B'),
    'loma_b_loma_r': ('LOMA_B', 'LOMA_R'),
    'loma_b_loma_l': ('LOMA_B', 'LOMA_L'),
    'loma_b_loma_g': ('LOMA_B', 'LOMA_G'),
    'loma_b128_bruteforce': ('LOMA_B128', 'LOMA_BRUTEFORCE'),
    'loma_b128_loma': ('LOMA_B128', 'LOMA_B128'),
}


def run_matcher(sequence_path: Path, exp_folder_colmap: Path, rgb_path: Path, rgb_csv: Path, calibration_yaml: Path,
                camera_name: str, matcher_type: str, matching_type: str, use_gpu: int, use_mask: int,
                settings: Settings | None = None) -> Path:
    """Build <exp_folder_colmap>/colmap_database.db with features and matches; returns its path.
    settings: the [feature_extractor] / [matcher] sections of the settings yaml (colmap_utilities.load_settings)."""
    settings = settings or {}
    print("\nExecuting colmap_matcher ...")
    PROFILER.set_stage("extraction")

    if matching_type not in MATCHING_TYPES:
        raise SystemExit(f"Unknown matching_type: {matching_type}")
    feature_extraction_type, feature_matching_type = MATCHING_TYPES[matching_type]

    gpu_index_list, num_threads = detect_gpus(use_gpu)

    calibration_model, colmap_camera_model, camera_params = colmap_camera(calibration_yaml, camera_name)

    colmap_image_list = exp_folder_colmap / "colmap_image_list.txt"
    create_colmap_image_list(str(rgb_csv), str(colmap_image_list), camera_name)

    database = exp_folder_colmap / "colmap_database.db"
    database.unlink(missing_ok=True)
    run(["colmap", "database_creator", "--database_path", str(database)])

    print(f"    colmap feature_extractor ({feature_extraction_type}) ...")
    print(f"        gpu_index: {gpu_index_list}")
    print(f"        camera model : {calibration_model} (colmap: {colmap_camera_model})")
    camera_params_args: list[str] = []
    if camera_params:
        print(f"        camera params: {camera_params}")
        camera_params_args = ["--ImageReader.camera_params", camera_params]

    # Masks (use_mask=1): the rgb csv's path_mask_<i> column (1 = usable pixel, 0 = masked out - no
    # features are extracted where the mask is 0). One shared mask (refrax) goes through
    # --ImageReader.camera_mask_path, per-frame masks (mask2former, datasets shipping masks) through
    # a directory of <image name>.png symlinks and --ImageReader.mask_path.
    mask_args: list[str] = []
    if use_mask == 1:
        mask_spec = create_colmap_mask_dir(str(rgb_csv), camera_name, str(sequence_path), str(exp_folder_colmap / "colmap_masks"))
        if mask_spec.startswith("camera_mask:"):
            mask_args = ["--ImageReader.camera_mask_path", mask_spec[len("camera_mask:"):]]
            print(f"        mask: shared camera mask {mask_spec[len('camera_mask:'):]}")
        elif mask_spec.startswith("mask_dir:"):
            mask_args = ["--ImageReader.mask_path", mask_spec[len("mask_dir:"):]]
            print(f"        mask: per-frame masks in {mask_spec[len('mask_dir:'):]}")
        else:
            print(f"        mask: none available for {camera_name} in {rgb_csv}")
    else:
        print(f"        mask: disabled (use_mask={use_mask})")

    extractor_explicit = ["ImageReader.camera_model", "ImageReader.single_camera", "ImageReader.single_camera_per_folder",
                          "ImageReader.camera_params", "ImageReader.camera_mask_path", "ImageReader.mask_path",
                          "FeatureExtraction.type", "FeatureExtraction.use_gpu", "FeatureExtraction.gpu_index",
                          "FeatureExtraction.num_threads"]
    run(["colmap", "feature_extractor",
         "--database_path", str(database),
         "--image_path", str(rgb_path),
         "--image_list_path", str(colmap_image_list),
         "--ImageReader.camera_model", colmap_camera_model,
         "--ImageReader.single_camera", "1",
         "--ImageReader.single_camera_per_folder", "1",
         "--FeatureExtraction.type", feature_extraction_type,
         "--FeatureExtraction.use_gpu", str(use_gpu),
         "--FeatureExtraction.gpu_index", gpu_index_list,
         "--FeatureExtraction.num_threads", str(num_threads),
         *camera_params_args, *mask_args,
         *settings_args(settings, "feature_extractor", "feature_extractor", extractor_explicit)])

    matching_args = ["--database_path", str(database),
                     "--FeatureMatching.type", feature_matching_type,
                     "--FeatureMatching.use_gpu", str(use_gpu),
                     "--FeatureMatching.gpu_index", gpu_index_list,
                     "--FeatureMatching.num_threads", str(num_threads)]
    matcher_explicit = ["FeatureMatching.type", "FeatureMatching.use_gpu", "FeatureMatching.gpu_index", "FeatureMatching.num_threads",
                        "SequentialMatching.loop_detection", "SequentialMatching.vocab_tree_path"]

    PROFILER.set_stage("matching")
    if matcher_type == "exhaustive":
        print(f"    colmap exhaustive_matcher ({feature_matching_type}) ...")
        print(f"        gpu_index: {gpu_index_list}")
        run(["colmap", "exhaustive_matcher", *matching_args,
             *settings_args(settings, "matcher", "exhaustive_matcher", matcher_explicit)])
    elif matcher_type == "sequential":
        # Loop detection uses COLMAP's default vocabulary tree for the feature type (a FAISS index
        # per SIFT / ALIKED / LoMa, auto-downloaded once into ~/.cache/colmap). The legacy
        # flickr100K FLANN trees are rejected by COLMAP >= 3.12 ("Failed to read faiss index").
        print(f"    colmap sequential_matcher ({feature_matching_type}) ...")
        print(f"        Vocabulary Tree: COLMAP default for {feature_extraction_type} (cached in ~/.cache/colmap)")
        print(f"        gpu_index: {gpu_index_list}")
        run(["colmap", "sequential_matcher", *matching_args,
             "--SequentialMatching.loop_detection", "1",
             *settings_args(settings, "matcher", "sequential_matcher", matcher_explicit)])
    else:
        raise SystemExit(f"Unknown matcher_type: {matcher_type}")

    return database
