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

exp_folder_colmap="${exp_folder}/colmap_${exp_id}"
rgb_dir=$(awk -F, 'NR==2 { split($2,a,"/"); print a[1]; exit }' "$rgb_csv")
rgb_path="${sequence_path}/${rgb_dir}"

# Get calibration model
read -r calibration_model more_ <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")

# Create colmap image list
colmap_image_list="${exp_folder_colmap}/colmap_image_list.txt"
python3 Baselines/colmap/create_colmap_image_list.py "$rgb_csv" "$colmap_image_list" "$camera_name"

# Create Colmap Database
database="${exp_folder_colmap}/colmap_database.db"
rm -rf ${database}
colmap database_creator --database_path ${database}

# Feature extractor
echo "    colmap feature_extractor ..."

if [ "${calibration_model}" == "unknown" ]
then
echo "        camera model : $calibration_model"
colmap feature_extractor \
   --database_path ${database} \
   --image_path ${rgb_path} \
   --image_list_path ${colmap_image_list} \
   --ImageReader.camera_model SIMPLE_PINHOLE \
   --ImageReader.single_camera 1 \
   --ImageReader.single_camera_per_folder 1 \
   --FeatureExtraction.use_gpu ${use_gpu}
fi

if [ "${calibration_model}" == "pinhole" ]
then
  read -r calibration_model fx fy cx cy <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")
  echo "        camera model : $calibration_model"
  echo "            fx: $fx , fy: $fy , cx: $cx , cy: $cy"
  colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model PINHOLE \
	--ImageReader.single_camera 1 \
	--ImageReader.single_camera_per_folder 1 \
	--FeatureExtraction.use_gpu ${use_gpu} \
	--ImageReader.camera_params "${fx},${fy},${cx},${cy}"
fi

if [ "${calibration_model}" == "radtan4" ]
then
  read -r calibration_model fx fy cx cy k1 k2 p1 p2 <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")
  echo "        camera model : $calibration_model"
  echo "            fx: $fx , fy: $fy , cx: $cx , cy: $cy"
  echo "            k1: $k1 , k2: $k2 , p1: $p1 , p2: $p2"
  colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model "OPENCV" \
	--ImageReader.single_camera 1 \
  --ImageReader.single_camera_per_folder 1 \
	--FeatureExtraction.use_gpu ${use_gpu} \
  --ImageReader.camera_params "${fx},${fy},${cx},${cy},${k1},${k2},${p1},${p2}"
fi

if [ "${calibration_model}" == "radtan5" ]
then
  read -r calibration_model fx fy cx cy k1 k2 p1 p2 k3 <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")
  echo "        camera model : $calibration_model"
  echo "            fx: $fx , fy: $fy , cx: $cx , cy: $cy"
  echo "            k1: $k1 , k2: $k2 , p1: $p1 , p2: $p2, k3: $k3"
  colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model "FULL_OPENCV" \
	--ImageReader.single_camera 1 \
  --ImageReader.single_camera_per_folder 1 \
	--FeatureExtraction.use_gpu ${use_gpu} \
  --ImageReader.camera_params "${fx},${fy},${cx},${cy},${k1},${k2},${p1},${p2},${k3},0,0,0"
fi

if [ "${calibration_model}" == "equid4" ]
then
  read -r calibration_model fx fy cx cy k1 k2 k3 k4 <<< $(python3 Baselines/colmap/get_calibration.py "$calibration_yaml" "$camera_name")
  echo "        camera model : $calibration_model"
  echo "            fx: $fx , fy: $fy , cx: $cx , cy: $cy"
  echo "            k1: $k1 , k2: $k2 , k3: $k3 , k4: $k4"
	colmap feature_extractor \
	--database_path ${database} \
	--image_path ${rgb_path} \
	--image_list_path ${colmap_image_list} \
	--ImageReader.camera_model "OPENCV_FISHEYE"\
	--ImageReader.single_camera 1 \
	--ImageReader.single_camera_per_folder 1 \
	--FeatureExtraction.use_gpu ${use_gpu} \
	--ImageReader.camera_params "${fx},${fy},${cx},${cy},${k1},${k2},${k3},${k4}"
fi

# Exhaustive Feature Matcher
if [ "${matcher_type}" == "exhaustive" ]
then
	echo "    colmap exhaustive_matcher ..."
  colmap exhaustive_matcher \
     --database_path ${database} \
     --FeatureMatching.use_gpu ${use_gpu}
fi

# Sequential Feature Matcher
if [ "${matcher_type}" == "sequential" ]
then
  num_rgb=$(( $(wc -l < "$rgb_csv") - 1 ))

  # Pick vocabulary tree based on the number of images
  vocabulary_tree="Baselines/colmap/vocab_tree_faiss_flickr100K_words32K.bin"
  if [ "$num_rgb" -gt 1000 ]; then
    vocabulary_tree="Baselines/colmap/vocab_tree_faiss_flickr100K_words256K.bin"
  fi
  if [ "$num_rgb" -gt 10000 ]; then
    vocabulary_tree="Baselines/colmap/vocab_tree_faiss_flickr100K_words1M.bin"
  fi

  echo "    colmap sequential_matcher ..."
  echo "        Vocabulary Tree: $vocabulary_tree"
  colmap sequential_matcher \
    --database_path "${database}" \
    --SequentialMatching.loop_detection 1 \
    --SequentialMatching.vocab_tree_path ${vocabulary_tree} \
    --FeatureMatching.use_gpu "${use_gpu}"
fi

# LightGlue Feature Matcher
if [ "${matcher_type}" == "custom" ]
then
  colmap exhaustive_matcher \
      --database_path ${database} \
      --FeatureMatching.use_gpu ${use_gpu}

  pixi run -e lightglue python3 Baselines/colmap/feature_matcher.py --database ${database} --rgb_path ${rgb_path} --rgb_csv ${rgb_csv}
fi

