import argparse
import pandas as pd
import os
def create_colmap_image_list(rgb_csv, colmap_image_list_txt, cam_name):

    df = pd.read_csv(rgb_csv)       
    image_list = df[f'path_{cam_name}'].to_list()

    with open(colmap_image_list_txt, 'w') as f:
        for name in image_list:
            file_name = os.path.basename(name)
            f.write(f"{file_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb_csv", help="Path to the rgb_csv")
    parser.add_argument("colmap_image_list", help="Path to the colmap_image_list")
    parser.add_argument("camera_name", help="camera_name")

    args = parser.parse_args()
    create_colmap_image_list(args.rgb_csv, args.colmap_image_list, args.camera_name)