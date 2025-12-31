import yaml
import sys
import argparse
import numpy as np

def get_camera_intrinsics(calibration_yaml, cam_name):
    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_name:
            cam = cam_;
            break;
  
    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    K = np.array([[cam['focal_length'][0], 0,  cam['principal_point'][0]],
                  [0,  cam['focal_length'][1], cam['principal_point'][1]],
                  [0,  0,   1]], dtype=np.float32)
    
    if has_dist:
        dist= " ".join(map(str, cam['distortion_coefficients']))
        print(f"{cam['distortion_type']} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]} {dist}")
    else:
        print(f"{cam['cam_model']} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_yaml", help="Path to the calibration YAML")
    parser.add_argument("camera_name", help="camera_name")
    args = parser.parse_args()
    
    get_camera_intrinsics(args.calibration_yaml, args.camera_name)