import sqlite3
from feature_matcher_utilities import extract_keypoints, feature_matching, unrotate_kps_W
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import argparse
from pathlib import Path

from lightglue import LightGlue

# ==========================================
# ==========================================
# DATABASE UTILITIES
# ==========================================
def load_colmap_db(db_path): 
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor

def create_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * 2147483647 + image_id2

def clean_database(cursor):
    """Removes existing features and matches to ensure a clean overwrite."""
    tables = ["keypoints", "descriptors"]#, "matches"], "two_view_geometry"]
    for table in tables:
        cursor.execute(f"DELETE FROM {table};")
    print("Database cleaned (keypoints, descriptors, matches removed).")

def insert_keypoints(cursor, image_id, keypoints, descriptors):
    """
    keypoints: (N, 2) numpy array, float32
    descriptors: (N, D) numpy array, float32
    """
    keypoints_blob = keypoints.tobytes()
    descriptors_blob = descriptors.tobytes()
    
    # Keypoints
    cursor.execute(
        "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?)",
        (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_blob)
    )
    
    # Descriptors (Optional but good practice)
    cursor.execute(
        "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?)",
        (image_id, descriptors.shape[0], descriptors.shape[1], descriptors_blob)
    )

def insert_matches(cursor, image_id1, image_id2, matches):
    """
    matches: (K, 2) numpy array, uint32. 
             Col 0 is index in image1, Col 1 is index in image2
    """
    pair_id = create_pair_id(image_id1, image_id2)
    matches_blob = matches.tobytes()
    
    cursor.execute(
        "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?)",
        (pair_id, matches.shape[0], matches.shape[1], matches_blob)
    )

def verify_matches_visual(cursor, image_id1, image_id2, image_dir):
    """
    Reads matches and keypoints from the COLMAP db and plots them.
    
    Args:
        cursor: SQLite cursor connected to the database.
        image_id1: ID of the first image.
        image_id2: ID of the second image.
        image_dir: Path to the directory containing the images.
    """
    
    # 1. Helper to ensure image_id1 < image_id2 for pair_id calculation
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
        swapped = True
    else:
        swapped = False
        
    pair_id = image_id1 * 2147483647 + image_id2

    # 2. Fetch Matches
    cursor.execute("SELECT data FROM matches WHERE pair_id = ?", (pair_id,))
    match_row = cursor.fetchone()
    
    if match_row is None:
        print(f"No matches found in DB for pair {image_id1}-{image_id2}")
        return

    # Decode Matches: UINT32 (N, 2)
    matches = np.frombuffer(match_row[0], dtype=np.uint32).reshape(-1, 2)

    # If we swapped inputs to generate pair_id, we must swap columns in matches
    # so matches[:,0] corresponds to the requested image_id1
    if swapped:
        matches = matches[:, [1, 0]]

    # 3. Fetch Keypoints for both images
    def get_keypoints_and_name(img_id):
        # Get Name
        cursor.execute("SELECT name FROM images WHERE image_id = ?", (img_id,))
        name = cursor.fetchone()[0]
        
        # Get Keypoints
        cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (img_id,))
        kp_row = cursor.fetchone()
        # Decode Keypoints: FLOAT32 (N, 2)
        kpts = np.frombuffer(kp_row[0], dtype=np.float32).reshape(-1, 2)
        return name, kpts

    name1, kpts1 = get_keypoints_and_name(image_id1)
    name2, kpts2 = get_keypoints_and_name(image_id2)

    # 4. Filter Keypoints using the Matches indices
    # matches[:, 0] are indices into kpts1
    # matches[:, 1] are indices into kpts2
    valid_kpts1 = kpts1[matches[:, 0]]
    valid_kpts2 = kpts2[matches[:, 1]]

    # 5. Load Images
    path1 = os.path.join(image_dir, name1)
    path2 = os.path.join(image_dir, name2)
    
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 6. Plotting
    # Concatenate images side-by-side
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    
    # Create a canvas large enough for both
    height = max(h1, h2)
    width = w1 + w2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    canvas[:h1, :w1, :] = img1
    canvas[:h2, w1:w1+w2, :] = img2

    plt.figure(figsize=(15, 10))
    plt.imshow(canvas)
    
    # Plot lines
    # Shift x-coordinates of image2 by w1
    for (x1, y1), (x2, y2) in zip(valid_kpts1, valid_kpts2):
        plt.plot([x1, x2 + w1], [y1, y2], 'c-', alpha=0.6, linewidth=0.5)
        plt.plot(x1, y1, 'r.', markersize=2)
        plt.plot(x2 + w1, y2, 'r.', markersize=2)

    plt.title(f"DB Verification: {name1} (ID:{image_id1}) <-> {name2} (ID:{image_id2}) | Matches: {len(matches)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sqlite3

def plot_matches_from_db(cursor, image_id1, image_id2, image_dir):
    """
    Reads matches and keypoints for a specific pair from the COLMAP DB and plots them.
    
    Args:
        cursor: SQLite cursor.
        image_id1, image_id2: The IDs of the two images to plot.
        image_dir: Path to the directory containing the actual image files.
    """
    
    # 1. Resolve Pair ID (Colmap requires id1 < id2 for unique pair_id)
    if image_id1 > image_id2:
        id_a, id_b = image_id2, image_id1
        swapped = True
    else:
        id_a, id_b = image_id1, image_id2
        swapped = False
        
    pair_id = id_a * 2147483647 + id_b

    # 2. Fetch Matches
    print(f"Fetching matches for pair {image_id1}-{image_id2} (PairID: {pair_id})...")
    cursor.execute("SELECT data, rows, cols FROM matches WHERE pair_id = ?", (pair_id,))
    match_row = cursor.fetchone()
    
    if match_row is None:
        print(f"No matches found in database for Pair {image_id1}-{image_id2}")
        return

    # Decode Matches (UINT32)
    # Blob is match_row[0], rows is [1], cols is [2]
    matches_blob = match_row[0]
    matches = np.frombuffer(matches_blob, dtype=np.uint32).reshape(-1, 2)
    
    # If inputs were swapped relative to how COLMAP stores them, swap the columns
    # so matches[:,0] refers to image_id1 and matches[:,1] refers to image_id2
    if swapped:
        matches = matches[:, [1, 0]]

    # 3. Fetch Keypoints & Image Names
    def get_image_data(img_id):
        cursor.execute("SELECT name FROM images WHERE image_id = ?", (img_id,))
        res = cursor.fetchone()
        if not res:
            raise ValueError(f"Image ID {img_id} not found in 'images' table.")
        name = res[0]
        
        cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (img_id,))
        kp_res = cursor.fetchone()
        if not kp_res:
             raise ValueError(f"No keypoints found for Image ID {img_id}.")
        
        # Decode Keypoints (FLOAT32)
        kpts = np.frombuffer(kp_res[0], dtype=np.float32).reshape(-1, 2)
        return name, kpts

    name1, kpts1 = get_image_data(image_id1)
    name2, kpts2 = get_image_data(image_id2)

    # 4. Filter Keypoints using Match Indices
    valid_kpts1 = kpts1[matches[:, 0]]
    valid_kpts2 = kpts2[matches[:, 1]]

    # 5. Visualization
    path1 = os.path.join(image_dir, name1)
    path2 = os.path.join(image_dir, name2)
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Error: Could not find image files at \n{path1}\n{path2}")
        return

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Create canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2

    plt.figure(figsize=(20, 10))
    plt.imshow(canvas)
    
    # Plot matches
    # x2 coordinates need to be shifted by w1
    for (x1, y1), (x2, y2) in zip(valid_kpts1, valid_kpts2):
        plt.plot([x1, x2 + w1], [y1, y2], 'g-', alpha=0.5, linewidth=1.5)
        plt.plot(x1, y1, 'r.', markersize=4)
        plt.plot(x2 + w1, y2, 'r.', markersize=4)

    plt.title(f"{name1} <-> {name2} | Total Matches: {len(matches)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--rgb_path", type=Path, required=True)
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--matcher", type=str, required=True)

    args, _ = parser.parse_known_args()

    DB_PATH = args.database
    IMAGE_DIR = args.rgb_path
    FEATURE_TYPE = args.feature
    MATCHER_TYPE = args.matcher
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    matches_file_path = os.path.join(os.path.dirname(DB_PATH), "matches.txt")

    conn, cursor = load_colmap_db(DB_PATH)
    cursor.execute("SELECT image_id, name FROM images")
    images_info = {row[0]: row[1] for row in cursor.fetchall()}
    image_ids = sorted(images_info.keys())

    clean_database(cursor)
    conn.commit() 

    # Keypoint Extraction
    fts = {}
    for i in tqdm(range(len(image_ids)), desc="Feature Extraction"):
        id = image_ids[i]
        fname = images_info[id]
        path = os.path.join(IMAGE_DIR, fname)

        feats_dict, h, w = extract_keypoints(path, features=FEATURE_TYPE)
        
        fts[id] = feats_dict

        kpts = feats_dict['keypoints'].squeeze(0).cpu().numpy().astype(np.float32)
        descs = feats_dict['descriptors'].squeeze(0).cpu().numpy().astype(np.float32)

        if FEATURE_TYPE == 'superpoint':
            kpts_rot = unrotate_kps_W(kpts, feats_dict['rotations'].squeeze(0).cpu().numpy().astype(np.float32), h, w)
        else:
            kpts_rot = kpts
        insert_keypoints(cursor, id, kpts_rot, descs)

    conn.commit() 

    # Feature Matching
    if MATCHER_TYPE == 'lightglue':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1, flash=True).eval().to(device)
    else:
        matcher = None

    with open(matches_file_path, "w") as f_match:
        for i in tqdm(range(len(image_ids)), desc="Feature Extraction"):
            id1 = image_ids[i]
            fname1 = images_info[id1]
            path1 = os.path.join(IMAGE_DIR, fname1)

            for j in range(i + 1, len(image_ids)):
                if j == i:
                    continue
                id2 = image_ids[j]

                fname2 = images_info[id2]
                path2 = os.path.join(IMAGE_DIR, fname2)
                matches_tensor = feature_matching(fts[id1], fts[id2], matcher=matcher, features=FEATURE_TYPE, matcher_type=MATCHER_TYPE) 

                if matches_tensor is not None and len(matches_tensor) > 0:
                    matches_np = matches_tensor.cpu().numpy().astype(np.uint32)
                    #insert_matches(cursor, id1, id2, matches_np)
                    
                    f_match.write(f"{fname1} {fname2}\n")
                    np.savetxt(f_match, matches_np, fmt="%d")
                    f_match.write("\n")
                
                    #verify_matches_visual(cursor, image_ids[i], image_ids[j], IMAGE_DIR)                
            #plt.show()
    
    conn.commit()

    #plot_matches_from_db(cursor, image_ids[0], image_ids[1], IMAGE_DIR)

    conn.close()
    print("Database overwrite complete.")    
