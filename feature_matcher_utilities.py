import torch
import numpy as np
import cv2
from lightglue import LightGlue
from lightglue.utils import rbd
from lightglue import  SuperPoint, SIFT
from lightglue.utils import load_image


def unrotate_kps_W(kps_rot, k, H, W):
    # Ensure inputs are Numpy
    if hasattr(kps_rot, 'cpu'): kps_rot = kps_rot.cpu().numpy()
    if hasattr(k, 'cpu'): k = k.cpu().numpy()
    
    # Squeeze if necessary
    if k.ndim > 1: k = k.squeeze()
    if kps_rot.ndim > 2: kps_rot = kps_rot.squeeze()

    x_r = kps_rot[:, 0]
    y_r = kps_rot[:, 1]
    
    x = np.zeros_like(x_r)
    y = np.zeros_like(y_r)
    
    mask0 = (k == 0)
    x[mask0], y[mask0] = x_r[mask0], y_r[mask0]
    
    mask1 = (k == 1)
    x[mask1], y[mask1] = (W - 1) - y_r[mask1], x_r[mask1]
    
    mask2 = (k == 2)
    x[mask2], y[mask2] = (W - 1) - x_r[mask2], (H - 1) - y_r[mask2]
    
    mask3 = (k == 3)
    x[mask3], y[mask3] = y_r[mask3], (H - 1) - x_r[mask3]
    
    return np.stack([x, y], axis=-1)

def extract_keypoints(path_to_image0, features='superpoint', rotations = [0,1,2,3]):
    # --- Models on GPU ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Load images as Torch tensors (3,H,W) in [0,1] ---
    timg = load_image(path_to_image0).to(device)
    _, h, w = timg.shape

    if features == 'sift':
        extractor = SIFT(max_num_keypoints=2048).eval().to(device)
        feats = extractor.extract(timg)
        return feats , h, w
    
    if features == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # --- Extract local features ---
    feats = {}
    for k in (rotations):
        timg_rotated = torch.rot90(timg, k, dims=(1, 2))
        feats[k] = extractor.extract(timg_rotated)
        #print(f"Extracted {feats[k]['keypoints'].shape[1]} keypoints for rotation {k}")

    # --- Merge features back to original coordinate system ---
    all_keypoints = []
    all_scores = []
    all_descriptors = []
    all_rotations = []
    for k, feat in feats.items():
        kpts = feat['keypoints']  # Shape (1, N, 2)
        num_kpts = kpts.shape[1]
        # if k == 0:
        #     kpts_corrected = kpts
        # elif k == 1:
        #     kpts_corrected = torch.stack(
        #         [w - 1 - kpts[..., 1], kpts[..., 0]], dim=-1
        #     )
        # elif k == 2:
        #     kpts_corrected = torch.stack(
        #         [w - 1 - kpts[..., 0], h - 1 - kpts[..., 1]], dim=-1
        #     )
        # elif k == 3:
        #     kpts_corrected = torch.stack(
        #         [kpts[..., 1], h - 1 - kpts[..., 0]], dim=-1
        #     )

        rot_indices = torch.full((1, num_kpts), k, dtype=torch.long, device=device)
        all_keypoints.append(feat['keypoints'])
        all_scores.append(feat['keypoint_scores'])
        all_descriptors.append(feat['descriptors'])
        all_rotations.append(rot_indices)

    # Concatenate all features along the keypoint dimension (dim=1)
    feats_merged = {
        'keypoints': torch.cat(all_keypoints, dim=1),
        'keypoint_scores': torch.cat(all_scores, dim=1),
        'descriptors': torch.cat(all_descriptors, dim=1),
        'rotations': torch.cat(all_rotations, dim=1)
    }
    
    num_kpts = feats_merged['keypoints'].shape[1]
    # perm = torch.randperm(num_kpts, device=device)

    # feats_merged['keypoints'] = feats_merged['keypoints'][:, perm, :]
    # feats_merged['keypoint_scores'] = feats_merged['keypoint_scores'][:, perm]
    # feats_merged['descriptors'] = feats_merged['descriptors'][:, perm, :]

    # Optional: If you want to retain other keys like 'shape' or 'image_size'
    #feats_merged['image_size'] = torch.tensor([w, h], device=device).unsqueeze(0)
    #feats_merged['scales'] = torch.tensor([w, h], device=device).unsqueeze(0)

    # for f in feats_merged:
    #     if 'scales' not in f:
    #         f['scales'] = torch.ones(all_keypoints.shape[:-1], device=device)
    #     if 'oris' not in f:
    #         f['oris'] = torch.zeros(all_keypoints.shape[:-1], device=device)

    return feats_merged , h, w

def feature_matching(feats0, feats1, matcher = None, features='superpoint', matcher_type='lightglue'):
    #print(f"Matching features using {matcher_type} matcher with {features} features.")
    if matcher_type == "lightglue":
        if matcher is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            matcher = LightGlue(features=features).eval().to(device)
    
        out_k = matcher({'image0': feats0, 'image1': feats1})
        _, _, out_k = [rbd(x) for x in [feats0, feats1, out_k]]   # remove batch dim
        return out_k['matches'] 

    if matcher_type == "brute_force":
        desc0 = feats0['descriptors'].squeeze(0) 
        desc1 = feats1['descriptors'].squeeze(0) 
        dists = torch.cdist(desc0, desc1, p=2)
        min_dist0, idx0 = dists.min(dim=1)
        min_dist1, idx1 = dists.min(dim=0)
        indices0 = torch.arange(len(idx0), device='cuda')
        mutual_mask = (idx1[idx0] == indices0) 
        matches0 = indices0[mutual_mask]
        matches1 = idx0[mutual_mask]
            
        return torch.stack([matches0, matches1], dim=-1) # [N, 2]