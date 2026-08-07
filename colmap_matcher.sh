#!/bin/bash
echo ""
echo "Executing colmap_matcher.sh ..."

sequence_path="$1"
exp_folder="$2"
exp_id="$3"
settings_yaml="$4"
calibration_yaml="$5"
rgb_csv="$6"
matcher_type="$7"
use_gpu="$8"
camera_name="$9"
matching_type="${10}"

exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
rgb_dir=$(awk -F, 'NR==2 { split($2,a,"/"); print a[1]; exit }' "$rgb_csv")
rgb_path="${sequence_path}/${rgb_dir}"

# matching_type: FeatureExtraction.type + FeatureMatching.type
feature_matching_type=""
feature_extraction_type="SIFT"

case "${matching_type}" in
    sift_bruteforce)
        feature_extraction_type="SIFT"
        feature_matching_type="SIFT_BRUTEFORCE"
        ;;
    sift_lightglue)
        feature_extraction_type="SIFT"
        feature_matching_type="SIFT_LIGHTGLUE"
        ;;
    aliked_bruteforce)
        feature_extraction_type="ALIKED_N16ROT"
        feature_matching_type="ALIKED_BRUTEFORCE"
        ;;
    aliked_lightglue)
        feature_extraction_type="ALIKED_N16ROT"
        feature_matching_type="ALIKED_LIGHTGLUE"
        ;;
    *)
        echo "Unknown matching_type: ${matching_type}"
        exit 1
        ;;
esac

# Detect usable GPUs: extraction/matching run as a single job across all of
# them, since COLMAP spawns one worker per listed GPU index and splits the
# per-image / per-block workload internally.
echo "    detecting GPUs ..."
echo "        nvidia-smi -L:"
nvidia-smi -L 2>&1 | sed 's/^/            /'
echo "        nvidia-smi --query-gpu=index,name,uuid --format=csv:"
nvidia-smi --query-gpu=index,name,uuid --format=csv 2>&1 | sed 's/^/            /'
echo "        CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
  # The scheduler already restricted this job to a specific device set (often
  # MIG slice UUIDs on HPC, which `nvidia-smi --query-gpu=index` does not
  # enumerate - it lists the whole node's GPUs, not what's gated to this job).
  # CUDA remaps whatever's listed here to ordinals 0..N-1 inside the process,
  # and COLMAP's gpu_index just calls cudaSetDevice(ordinal), so address by
  # ordinal count rather than trying to resolve real indices/UUIDs ourselves.
  num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
  gpu_ids=($(seq 0 $((num_gpus - 1))))
else
  gpu_ids=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null))
fi
if [ "${#gpu_ids[@]}" -lt 1 ] || [ "${use_gpu}" == "0" ]; then
  gpu_ids=(0)
fi
gpu_index_list=$(IFS=,; echo "${gpu_ids[*]}")
echo "        use_gpu: ${use_gpu}"
echo "        detected gpu_ids: ${gpu_ids[*]}"
echo "        gpu_index_list passed to colmap: ${gpu_index_list}"

# Bound CPU thread count too: num_threads=-1 (COLMAP's default) auto-detects
# via APIs that on cgroup-limited HPC nodes often report the physical node's
# total core count rather than what's actually allocated to this job. `nproc`
# (not `nproc --all`) reports the affinity-restricted count instead, so it
# respects whatever the scheduler's cgroup actually granted.
num_threads=$(nproc)
echo "        num_threads: ${num_threads}"

# Get calibration model and parameters
read -r calibration_model params <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")

case "${calibration_model}" in
    unknown)
        colmap_camera_model="OPENCV"
        camera_params=""
        ;;
    pinhole)
        colmap_camera_model="PINHOLE"
        camera_params="${params// /,}"
        ;;
    radtan4)
        colmap_camera_model="OPENCV"
        camera_params="${params// /,}"
        ;;
    radtan5)
        colmap_camera_model="FULL_OPENCV"
        camera_params="${params// /,},0,0,0"
        ;;
    equid4)
        colmap_camera_model="OPENCV_FISHEYE"
        camera_params="${params// /,}"
        ;;
    *)
        echo "Unknown calibration_model: ${calibration_model}"
        exit 1
        ;;
esac

# Create colmap image list
colmap_image_list="${exp_folder_colmap}/colmap_image_list.txt"
python3 Baselines/colmap/create_colmap_image_list.py "$rgb_csv" "$colmap_image_list" "$camera_name"

# Create Colmap Database
database="${exp_folder_colmap}/colmap_database.db"
rm -rf ${database}
colmap database_creator --database_path ${database}

# Feature extractor
echo "    colmap feature_extractor (${feature_extraction_type}) ..."
echo "        gpu_index: ${gpu_index_list}"
echo "        camera model : ${calibration_model} (colmap: ${colmap_camera_model})"
[ -n "${camera_params}" ] && echo "        camera params: ${camera_params}"

camera_params_args=()
[ -n "${camera_params}" ] && camera_params_args=(--ImageReader.camera_params "${camera_params}")

colmap feature_extractor \
    --database_path ${database} \
    --image_path ${rgb_path} \
    --image_list_path ${colmap_image_list} \
    --ImageReader.camera_model "${colmap_camera_model}" \
    --ImageReader.single_camera 1 \
    --ImageReader.single_camera_per_folder 1 \
    --FeatureExtraction.type ${feature_extraction_type} \
    --FeatureExtraction.use_gpu ${use_gpu} \
    --FeatureExtraction.gpu_index "${gpu_index_list}" \
    --FeatureExtraction.num_threads "${num_threads}" \
    "${camera_params_args[@]}"

# Exhaustive Feature Matcher
if [ "${matcher_type}" == "exhaustive" ];
then
  echo "    colmap exhaustive_matcher (${feature_matching_type}) ..."
  echo "        gpu_index: ${gpu_index_list}"
  colmap exhaustive_matcher \
    --database_path "${database}" \
    --FeatureMatching.type "${feature_matching_type}" \
    --FeatureMatching.use_gpu "${use_gpu}" \
    --FeatureMatching.gpu_index "${gpu_index_list}" \
    --FeatureMatching.num_threads "${num_threads}"
fi

# Sequential Feature Matcher
if [ "${matcher_type}" == "sequential" ]
then
  num_rgb=$(( $(wc -l < "$rgb_csv") - 1 ))

  # Pick vocabulary tree based on the number of images
  vocabulary_tree="Baselines/colmap/vocab_tree_flickr100K_words32K.bin"
  if [ "$num_rgb" -gt 1000 ]; then
    vocabulary_tree="Baselines/colmap/vocab_tree_flickr100K_words256K.bin"
  fi
  if [ "$num_rgb" -gt 10000 ]; then
    vocabulary_tree="Baselines/colmap/vocab_tree_flickr100K_words1M.bin"
  fi

  echo "    colmap sequential_matcher (${feature_matching_type}) ..."
  echo "        Vocabulary Tree: $vocabulary_tree"
  echo "        gpu_index: ${gpu_index_list}"
  colmap sequential_matcher \
    --database_path "${database}" \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.vocab_tree_path ${vocabulary_tree} \
    --FeatureMatching.type "${feature_matching_type}" \
    --FeatureMatching.use_gpu "${use_gpu}" \
    --FeatureMatching.gpu_index "${gpu_index_list}" \
    --FeatureMatching.num_threads "${num_threads}"
fi