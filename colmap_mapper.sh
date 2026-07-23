#!/bin/bash
echo "Executing colmap_mapper.sh ..."

sequence_path="$1" 
exp_folder="$2" 
exp_id="$3" 
settings_yaml="$4"
calibration_yaml="$5"
rgb_csv="$6"
camera_name="$7"
mapper_type="$8"

exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
rgb_dir="${camera_name}"
rgb_path="${sequence_path}/${rgb_dir}"

read -r calibration_model more_ <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")
echo "        camera model : $calibration_model"
ba_refine_focal_length="0"
ba_refine_principal_point="0"
ba_refine_extra_params="0"
if [ "${calibration_model}" == "unknown" ]
then
  ba_refine_focal_length="1"
  ba_refine_principal_point="1"
  ba_refine_extra_params="1"
fi

database="${exp_folder_colmap}/colmap_database.db"

if [ "${mapper_type}" == "glomap" ]
then
    echo "    global mapper (GLOMAP) ..."
    colmap global_mapper \
        --database_path ${database} \
        --image_path ${rgb_path} \
        --output_path ${exp_folder_colmap} \
        --GlobalMapper.ba_refine_focal_length ${ba_refine_focal_length} \
        --GlobalMapper.ba_refine_principal_point ${ba_refine_principal_point} \
        --GlobalMapper.ba_refine_extra_params ${ba_refine_extra_params}
else
    echo "    colmap mapper (COLMAP) ..."
    colmap mapper \
        --database_path ${database} \
        --image_path ${rgb_path} \
        --output_path ${exp_folder_colmap} \
        --Mapper.ba_refine_focal_length ${ba_refine_focal_length} \
        --Mapper.ba_refine_principal_point ${ba_refine_principal_point} \
        --Mapper.ba_refine_extra_params ${ba_refine_extra_params}
fi

echo "    colmap model_converter ..."
colmap model_converter \
	--input_path ${exp_folder_colmap}/0 --output_path ${exp_folder_colmap} --output_type TXT