import torch
import numpy as np
import cv2
from lightglue import LightGlue
from lightglue.utils import rbd

def unrotate_kps_W(kps_rot, k, H, W):
    import numpy as np
    
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

def unrotate_kps(kps_rot, k, H, W):
    import torch
    # k is how many times you rotated CCW by 90° to create the rotated image
    x_r, y_r = kps_rot[:, 0].clone(), kps_rot[:, 1].clone()
    if k == 0:
        x, y = x_r, y_r
    elif k == 1:   # 90° CCW
        x = (W - 1) - y_r
        y = x_r
    elif k == 2:   # 180°
        x = (W - 1) - x_r
        y = (H - 1) - y_r
    elif k == 3:   # 270° CCW
        x = y_r
        y = (H - 1) - x_r
    else:
        raise ValueError("k must be 0..3")
    return torch.stack([x, y], dim=-1)

# def lightglue_matching(path_to_image0, path_to_image1, plot=False, features='superpoint'):
#     from lightglue import LightGlue, SuperPoint, SIFT
#     from lightglue.utils import load_image, rbd
#     from lightglue import viz2d
#     import torch

#     # --- Models on GPU ---
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     if features == 'superpoint':
#         extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
#     if features == 'sift':
#         extractor = SIFT(max_num_keypoints=2048).eval().to(device)

#     matcher = LightGlue(features=features).eval().to(device)

#     # --- Load images as Torch tensors (3,H,W) in [0,1] ---
#     timg0 = load_image(path_to_image0).to(device)
#     timg1 = load_image(path_to_image1).to(device)

#     # --- Extract local features ---
#     feats0 = extractor.extract(timg0)   # auto-resize inside

#     max_num_matches = -1
#     best_k = 0
#     best_feats0 = None
#     best_feats1 = None
#     for k in range(4):
#         timg1_rotated = torch.rot90(timg1, k, dims=(1, 2))
#         feats1_k = extractor.extract(timg1_rotated)
#         out_k = matcher({'image0': feats0, 'image1': feats1_k})
#         feats0_k, feats1_k, out_k = [rbd(x) for x in [feats0, feats1_k, out_k]]   # remove batch dim
#         matches_k = out_k['matches']   # (K,2) long
#         num_k = len(matches_k)
#         if num_k > max_num_matches:
#             max_num_matches = num_k
#             matches = matches_k
#             best_feats0 = feats0_k
#             best_feats1 = feats1_k
#             best_k = k
    
#     # --- Keypoints in matched order (Torch tensors on CPU) ---
#     H1, W1 = timg1.shape[-2], timg1.shape[-1]

#     kpts0 = best_feats0['keypoints'][matches[:, 0]]
#     kpts1 = best_feats1['keypoints'][matches[:, 1]]
#     kpts1 = unrotate_kps(kpts1, best_k, H1, W1)             # (K,2) mapped to original image1 coords

#     desc0 = best_feats0['descriptors'][matches[:, 0]]
#     desc1 = best_feats1['descriptors'][matches[:, 1]]

#     if plot:
#         if len(kpts0) == 0 or len(kpts1) == 0:
#             print("No matches found.")
#             return None, None
#         ax = viz2d.plot_images([timg0.cpu(), timg1.cpu()])
#         viz2d.plot_matches(kpts0.cpu(), kpts1.cpu(), color=None, lw=0.8, axes=ax)
#         #ax0 = ax[0] if isinstance(ax, (list, tuple, np.ndarray)) else ax
#         #fig = ax0.figure

#         #return kpts0, kpts1 #, fig, ax
        

#     return kpts0, kpts1, desc0, desc1

def lightglue_keypoints(path_to_image0, features='superpoint', rotations = [0,1,2,3]):
    from lightglue import LightGlue, SuperPoint, SIFT
    from lightglue.utils import load_image, rbd
    from lightglue import viz2d
    import torch

    # --- Models on GPU ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if features == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    if features == 'sift':
        extractor = SIFT(max_num_keypoints=2048).eval().to(device)

    # --- Load images as Torch tensors (3,H,W) in [0,1] ---
    timg = load_image(path_to_image0).to(device)
    _, h, w = timg.shape

    
    # --- Extract local features ---
    feats = {}
    for k in (rotations):
        timg_rotated = torch.rot90(timg, k, dims=(1, 2))
        feats[k] = extractor.extract(timg_rotated)
        print(f"Extracted {feats[k]['keypoints'].shape[1]} keypoints for rotation {k}")

    # --- Merge features back to original coordinate system ---
    all_keypoints = []
    all_scores = []
    all_descriptors = []
    all_rotations = []
    for k, feat in feats.items():
        kpts = feat['keypoints']  # Shape (1, N, 2)
        num_kpts = kpts.shape[1]
        if k == 0:
            kpts_corrected = kpts
        elif k == 1:
            kpts_corrected = torch.stack(
                [w - 1 - kpts[..., 1], kpts[..., 0]], dim=-1
            )
        elif k == 2:
            kpts_corrected = torch.stack(
                [w - 1 - kpts[..., 0], h - 1 - kpts[..., 1]], dim=-1
            )
        elif k == 3:
            kpts_corrected = torch.stack(
                [kpts[..., 1], h - 1 - kpts[..., 0]], dim=-1
            )

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
    feats_merged['image_size'] = torch.tensor([w, h], device=device).unsqueeze(0)
    return feats_merged

def lightglue_matching(feats0, feats1, plot=False, features='superpoint', path_to_image0=None, path_to_image1=None):
    from lightglue import LightGlue, SuperPoint, SIFT
    from lightglue.utils import load_image, rbd
    from lightglue import viz2d
    import torch

    # --- Models on GPU ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    matcher = LightGlue(features=features).eval().to(device)

    # --- Load images as Torch tensors (3,H,W) in [0,1] ---
    if plot:
        timg0 = load_image(path_to_image0).to(device)
        timg1 = load_image(path_to_image1).to(device)

    # --- Extract local features ---

    max_num_matches = -1
    best_k = 0
    best_feats0 = None
    best_feats1 = None
    for k in range(1):
        #timg1_rotated = torch.rot90(timg1, k, dims=(1, 2))
        feats1_k = feats1 #extractor.extract(timg1_rotated)
        out_k = matcher({'image0': feats0, 'image1': feats1_k})
        feats0_k, feats1_k, out_k = [rbd(x) for x in [feats0, feats1_k, out_k]]   # remove batch dim
        matches_k = out_k['matches']   # (K,2) long
        num_k = len(matches_k)
        if num_k > max_num_matches:
            max_num_matches = num_k
            matches = matches_k
            best_feats0 = feats0_k
            best_feats1 = feats1_k
            best_k = k
    print(f"LightGlue found {len(matches)} matches.")
    # --- Keypoints in matched order (Torch tensors on CPU) ---
    #H1, W1 = timg1.shape[-2], timg1.shape[-1]

    # kpts0 = best_feats0['keypoints'][matches[:, 0]]
    # kpts1 = best_feats1['keypoints'][matches[:, 1]]
    # #kpts1 = unrotate_kps(kpts1, best_k, H1, W1)             # (K,2) mapped to original image1 coords

    # desc0 = best_feats0['descriptors'][matches[:, 0]]
    # desc1 = best_feats1['descriptors'][matches[:, 1]]

    # pts0 = kpts0.detach().cpu().numpy().astype(np.float32)  # (K,2)
    # pts1 = kpts1.detach().cpu().numpy().astype(np.float32)  # (K,2)
    # H, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)

    # if inliers is not None:
    #     mask = inliers.ravel() == 1
    #     mask_tensor = torch.from_numpy(mask).to(matches.device)
    #     matches = matches[mask_tensor]
    # else:
    #     # If geometry check failed completely, return no matches
    #     return None
    
    # if plot:
    #     if len(kpts0) == 0 or len(kpts1) == 0:
    #         print("No matches found.")
    #         return None, None
    #     ax = viz2d.plot_images([timg0.cpu(), timg1.cpu()])
    #     viz2d.plot_matches(kpts0.cpu(), kpts1.cpu(), color=None, lw=0.8, axes=ax)
    #     #ax0 = ax[0] if isinstance(ax, (list, tuple, np.ndarray)) else ax
    #     #fig = ax0.figure

    #     #return kpts0, kpts1 #, fig, ax
        

    return matches