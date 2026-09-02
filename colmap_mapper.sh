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
# The frame folder comes from the csv, as in colmap_matcher.sh: the run pipeline may point
# path_rgb_0 at a generated folder (refrax_0 for 'refraction: refrax') rather than rgb_0, and
# the mapper reads the images from image_path to colour the points.
rgb_dir=$(awk -F, 'NR==2 { split($2,a,"/"); print a[1]; exit }' "$rgb_csv")
rgb_path="${sequence_path}/${rgb_dir}"

read -r calibration_model more_ <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")
echo "        camera model : $calibration_model"
ba_refine_focal_length="1"
ba_refine_principal_point="0"
ba_refine_extra_params="1"
if [ "${calibration_model}" == "unknown" ]
then
  ba_refine_focal_length="1"
  ba_refine_principal_point="0"
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

# COLMAP may split the scene into several sub-models (${exp_folder_colmap}/0, /1, ...), and the
# largest one is not necessarily /0. Pick the sub-model with the most registered images and
# record the choice in ${exp_folder_colmap}/best_model for the rest of the pipeline (gui, ...).
best_model=""
best_num_images=-1
for model_dir in "${exp_folder_colmap}"/[0-9]*; do
  [ -f "${model_dir}/images.bin" ] || continue
  num_images=$(colmap model_analyzer --path "${model_dir}" 2>&1 | grep -oE "Registered images: [0-9]+" | grep -oE "[0-9]+$")
  num_images=${num_images:-0}
  echo "        sub-model $(basename "${model_dir}"): ${num_images} registered images"
  if [ "${num_images}" -gt "${best_num_images}" ]; then
    best_num_images=${num_images}
    best_model=${model_dir}
  fi
done
if [ -z "${best_model}" ]; then
  echo "    no reconstruction produced (no sub-model found in ${exp_folder_colmap})"
  exit 1
fi
echo "${best_model}" > "${exp_folder_colmap}/best_model"

echo "    colmap model_converter (sub-model $(basename "${best_model}"), ${best_num_images} images) ..."
colmap model_converter \
	--input_path "${best_model}" --output_path ${exp_folder_colmap} --output_type TXT