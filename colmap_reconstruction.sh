#!/bin/bash

# Default values
matcher_type="exhaustive"
matching_type="sift_bruteforce"
mapper_type="colmap"
use_gpu="1"
use_mask="0"
verbose="0"
settings_yaml=""
sequence_path=""
exp_folder=""
exp_id=""
calibration_yaml=""
rgb_csv=""
camera_name="rgb_0"

# Function to split key-value pairs and assign them to variables
split_and_assign() {
  local input=$1
  local key=$(echo $input | cut -d':' -f1)
  local value=$(echo $input | cut -d':' -f2-)
  eval $key=$value
}

# Read Inputs
for ((i=1; i<=$#; i++)); do
    split_and_assign "${!i}"
done

exp_id=$(printf "%05d" ${exp_id})

echo -e "\n================= Experiment Configuration ================="
echo "  Sequence Path     : $sequence_path"
echo "  Experiment Folder : $exp_folder"
echo "  Experiment ID     : $exp_id"
echo "  Verbose           : $verbose"
echo "  Matcher Type      : $matcher_type"
echo "  Matching Type     : $matching_type"
echo "  Mapper Type       : $mapper_type"
echo "  Use GPU           : $use_gpu"
echo "  Use Mask          : $use_mask"
echo "  Settings YAML     : $settings_yaml"
echo "  Calibration YAML  : $calibration_yaml"
echo "  RGB CSV           : $rgb_csv"
echo "  Camera Name       : $camera_name"
echo "============================================================"

# Create folder to save colmap files
exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
rm -rf "$exp_folder_colmap"
mkdir "$exp_folder_colmap"

# Run COLMAP scripts for matching and mapping
export QT_QPA_PLATFORM_PLUGIN_PATH="$CONDA_PREFIX/plugins/platforms"
colmap_args="$sequence_path $exp_folder $exp_id $settings_yaml $calibration_yaml $rgb_csv"
./Baselines/colmap/colmap_matcher.sh $colmap_args $matcher_type $use_gpu $camera_name $matching_type $use_mask
./Baselines/colmap/colmap_mapper.sh $colmap_args $camera_name $mapper_type

# Convert COLMAP outputs to a format suitable for VSLAM-LAB
python Baselines/colmap/colmap_to_vslamlab.py $sequence_path $exp_folder $exp_id $verbose $rgb_csv $camera_name

# Visualization with colmap gui
if [ "$verbose" -eq 1 ]; then
  exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
  rgb_dir=$(awk -F, 'NR==2 { split($2,a,"/"); print a[1]; exit }' "$rgb_csv")
  rgb_path="${sequence_path}/${rgb_dir}"
  database="${exp_folder_colmap}/colmap_database.db"
  # The largest sub-model, as picked by colmap_mapper.sh (falls back to /0).
  best_model=$(cat "${exp_folder_colmap}/best_model" 2>/dev/null || echo "${exp_folder_colmap}/0")
  colmap gui --import_path "${best_model}" --database_path ${database} --image_path ${rgb_path}
fi

# # Remove colmap data
# rm -rf ${exp_folder_colmap}