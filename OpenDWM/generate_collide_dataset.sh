#!/bin/bash
trajectory_file_path=$1
output_path=$2
dataset_type=$3
# 
echo ">>> ..."

#  conda 
source /miniconda3/etc/profile.d/conda.sh
echo ">>>  conda "

# 
set -e
echo ">>> "

# 
echo ">>> ..."
rm -rf /OpenDWM/text_description/collide
rm -rf /nuscenes/interp_12Hz_trainval_collide
rm -rf /nuscenes/v1.0-collision
rm -f /nuscenes/v1.0-trainval-zip/interp_12Hz_trainval_collide.zip
rm -rf /ASAP/out
rm -rf OpenDWM_output/preview_dataset_collision
rm -rf /nuscenes/new_samples
rm -rf /nuscenes/new_sweeps
rm -rf /nuscenes/v1.0-trainval-zip/new_sweeps.zip
rm -rf /nuscenes/v1.0-trainval-zip/new_samples.zip
echo ">>> "

#  OpenDWM  openDWM 
cd /OpenDWM/tools
echo ">>>  OpenDWM/tools "
conda activate openDWM
echo ">>>  openDWM "

#  nusc_annotation_changer.py 
echo ">>>  nusc_annotation_changer.py ..."
python nusc_annotation_changer_new.py --trajectory-file=${trajectory_file_path} --dataset_type=${dataset_type} 
echo ">>> nusc_annotation_changer.py "

#  ASAP  ASAP 
cd /ASAP
echo ">>>  ASAP "
conda activate ASAP
echo ">>>  ASAP "

#  ann_generator.sh 
echo ">>>  ann_generator.sh ..."
bash scripts/ann_generator.sh 12 --ann_strategy 'interp'
echo ">>> ann_generator.sh "

#  nuscenes 
cd /nuscenes
echo ">>>  nuscenes "

#  interp_12Hz_trainval_collide 
echo ">>>  interp_12Hz_trainval_collide ..."
zip -r interp_12Hz_trainval_collide.zip interp_12Hz_trainval_collide/
echo ">>> :interp_12Hz_trainval_collide.zip"

# 
echo ">>>  v1.0-trainval-zip ..."
mv interp_12Hz_trainval_collide.zip /nuscenes/v1.0-trainval-zip
echo ">>> "


#  new_samples 
echo ">>>  new_samples ..."
zip -r new_samples.zip new_samples/
echo ">>> :new_samples"

# 
echo ">>>  v1.0-trainval-zip ..."
mv new_samples.zip /nuscenes/v1.0-trainval-zip
echo ">>> "

#  new_sweeps 
echo ">>>  new_sweeps ..."
zip -r new_sweeps.zip new_sweeps/
echo ">>> :new_sweeps"

# 
echo ">>>  v1.0-trainval-zip ..."
mv new_sweeps.zip /nuscenes/v1.0-trainval-zip
echo ">>> "

#  OpenDWM/src  openDWM 
cd /OpenDWM/src
echo ">>>  OpenDWM/src "
conda activate openDWM
echo ">>>  openDWM "

#  generate_preview.py 
echo ">>>  generate_preview.py ..."
# python generate_preview.py -o=${output_path}
echo ">>> generate_preview.py "

# 
echo ">>> "