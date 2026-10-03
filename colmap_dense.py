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
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from colmap_utilities import colmap_has_cuda, detect_gpus, run

MESHERS = ('none', 'delaunay', 'poisson', 'advancing_front')
GUI_MODEL_DIR = "sparse_fused"  # synthetic model (fused points as points3D) for `colmap gui`, see write_gui_model
GUI_TRACK_LENGTH = 3  # the gui hides points with shorter tracks (RenderOptions::min_track_len = 3)


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


_PLY_TYPES = {'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4',
              'float': 'f4', 'double': 'f8', 'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2',
              'int32': 'i4', 'uint32': 'u4', 'float32': 'f4', 'float64': 'f8'}


def read_ply_vertices(ply: Path) -> np.ndarray:
    """Vertex records of a binary little-endian PLY whose vertex element comes first with scalar
    properties only (COLMAP's fused.ply: x y z nx ny nz red green blue)."""
    fields, num_vertices, in_vertex = [], 0, False
    with open(ply, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{ply}: no end_header")
            tokens = line.decode('ascii', errors='replace').split()
            if tokens[:1] == ['format'] and tokens[1] != 'binary_little_endian':
                raise ValueError(f"{ply}: unsupported PLY format {tokens[1]}")
            if tokens[:1] == ['element']:
                in_vertex = tokens[1] == 'vertex'
                if in_vertex:
                    num_vertices = int(tokens[2])
            elif tokens[:1] == ['property'] and in_vertex:
                if tokens[1] == 'list':
                    raise ValueError(f"{ply}: list vertex properties are not supported")
                fields.append((tokens[2], '<' + _PLY_TYPES[tokens[1]]))
            elif tokens[:1] == ['end_header']:
                header_end = f.tell()
                break
    return np.fromfile(ply, dtype=np.dtype(fields), count=num_vertices, offset=header_end)


def write_gui_model(dense_dir: Path, fused_ply: Path) -> Path:
    """<dense_dir>/sparse_fused: the undistorted model's cameras and poses with the fused cloud as its 3D
    points, so `colmap gui --import_path` shows the dense result (the gui can only import a model folder).
    The viewer hides points with tracks shorter than 3 and has no CLI hook for fused.ply, so each point
    gets observations in the 3 nearest cameras that see it (projected pixel coordinates, valid indices)."""
    with tempfile.TemporaryDirectory(dir=dense_dir) as tmp:
        run(["colmap", "model_converter", "--input_path", str(dense_dir / "sparse"), "--output_path", tmp, "--output_type", "TXT"],
            exit_on_error=False)
        tmp = Path(tmp)
        cameras = {}
        for line in (tmp / "cameras.txt").read_text().splitlines():
            if line and not line.startswith('#'):
                tok = line.split()
                cameras[int(tok[0])] = (tok[1], int(tok[2]), int(tok[3]), [float(x) for x in tok[4:]])
        images = []  # (image_id, pose tokens, camera_id, name)
        lines = [l for l in (tmp / "images.txt").read_text().splitlines() if l and not l.startswith('#')]
        for pose_line in lines[0::2]:
            tok = pose_line.split()
            images.append((int(tok[0]), tok[1:8], int(tok[8]), tok[9]))
        extra = [p for p in ("rigs.txt", "frames.txt") if (tmp / p).is_file()]
        extra_text = {p: (tmp / p).read_text() for p in extra}

    verts = read_ply_vertices(fused_ply)
    xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=1).astype(np.float64)
    rgb = np.stack([verts['red'], verts['green'], verts['blue']], axis=1).astype(np.int64)
    num_points, num_images = len(xyz), len(images)

    # Camera poses: world -> camera, X_c = R X + t
    quats = np.array([[float(q) for q in pose[:4]] for _, pose, _, _ in images])  # qw qx qy qz
    rots = R.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()  # (M,3,3)
    trans = np.array([[float(t) for t in pose[4:7]] for _, pose, _, _ in images])  # (M,3)
    centers = -np.einsum('mij,mi->mj', rots, trans)  # -R^T t
    intr = np.array([cameras[cam_id][3][:4] for _, _, cam_id, _ in images])  # fx fy cx cy (PINHOLE after undistortion)
    sizes = np.array([cameras[cam_id][1:3] for _, _, cam_id, _ in images], dtype=np.float64)  # w h

    k = min(GUI_TRACK_LENGTH, num_images)
    obs_img = np.empty((num_points, k), dtype=np.int64)
    obs_uv = np.empty((num_points, k, 2))
    for start in range(0, num_points, 20000):
        P = xyz[start:start + 20000]  # (n,3)
        Xc = np.einsum('mij,nj->nmi', rots, P) + trans[None]  # (n,M,3)
        z = Xc[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = intr[None, :, 0] * Xc[..., 0] / z + intr[None, :, 2]
            v = intr[None, :, 1] * Xc[..., 1] / z + intr[None, :, 3]
        visible = (z > 0) & (u >= 0) & (u < sizes[None, :, 0]) & (v >= 0) & (v < sizes[None, :, 1])
        dist = np.linalg.norm(P[:, None, :] - centers[None], axis=2)
        dist = np.where(visible, dist, dist + 1e6)  # prefer cameras that see the point, fall back to nearest
        sel = np.argsort(dist, axis=1)[:, :k]
        obs_img[start:start + 20000] = sel
        rows = np.arange(len(P))[:, None]
        obs_uv[start:start + 20000, :, 0] = np.clip(np.nan_to_num(u[rows, sel]), 0, sizes[sel, 0] - 1)
        obs_uv[start:start + 20000, :, 1] = np.clip(np.nan_to_num(v[rows, sel]), 0, sizes[sel, 1] - 1)

    # Per-image 2D observation lists; a point's POINT2D_IDX in an image is its position in that list
    per_image_idx: list[list[np.ndarray]] = [[] for _ in range(num_images)]
    per_image_uv: list[list[np.ndarray]] = [[] for _ in range(num_images)]
    counts = np.zeros(num_images, dtype=np.int64)
    point_tracks = np.empty((num_points, k), dtype=np.int64)
    for j in range(k):
        for m in range(num_images):
            idx = np.nonzero(obs_img[:, j] == m)[0]
            point_tracks[idx, j] = counts[m] + np.arange(len(idx))
            counts[m] += len(idx)
            per_image_idx[m].append(idx)
            per_image_uv[m].append(obs_uv[idx, j])

    out = dense_dir / GUI_MODEL_DIR
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    with open(out / "cameras.txt", 'w') as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam_id, (model, w, h, params) in cameras.items():
            f.write(f"{cam_id} {model} {w} {h} {' '.join(repr(p) for p in params)}\n")
    image_ids = np.array([img_id for img_id, _, _, _ in images])
    with open(out / "images.txt", 'w') as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for m, (img_id, pose, cam_id, name) in enumerate(images):
            f.write(f"{img_id} {' '.join(pose)} {cam_id} {name}\n")
            idx, uv = np.concatenate(per_image_idx[m]), np.concatenate(per_image_uv[m])
            f.write(' '.join(f"{x:.3f} {y:.3f} {pid + 1}" for (x, y), pid in zip(uv, idx)) + "\n")
    with open(out / "points3D.txt", 'w') as f:
        f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(num_points):
            track = ' '.join(f"{image_ids[obs_img[i, j]]} {point_tracks[i, j]}" for j in range(k))
            f.write(f"{i + 1} {xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} {rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]} 0 {track}\n")
    for p, text in extra_text.items():
        (out / p).write_text(text)
    return out


def run_dense(exp_folder: Path, exp_id: str, exp_folder_colmap: Path, best_model: Path, rgb_path: Path,
              use_gpu: int, max_image_size: int, mesher: str, gui_model: bool = False) -> Path | None:
    """Dense pipeline on best_model; with gui_model=True also writes the synthetic model for `colmap gui`
    and returns its folder (None when the dense stage was skipped or failed)."""
    print("\nExecuting colmap_dense ...")
    if mesher not in MESHERS:
        print(f"    WARNING: unknown mesher '{mesher}' (expected one of {list(MESHERS)}); skipping the dense stage")
        return None
    if use_gpu != 1 or not colmap_has_cuda():
        print(f"    WARNING: dense reconstruction needs the CUDA colmap build and use_gpu=1 "
              f"(use_gpu={use_gpu}, cuda build={colmap_has_cuda()}); skipping the dense stage")
        return None

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

        gui_dir = None
        if gui_model:
            gui_dir = write_gui_model(dense_dir, fused_ply)
            print(f"        gui model (fused points as 3D points): {gui_dir}")

        if mesher == 'none':
            return gui_dir
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
        return gui_dir
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        print(f"    WARNING: dense stage aborted ({e}); the sparse reconstruction and trajectory are unaffected")
        return None
