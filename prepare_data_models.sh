#!/bin/bash
# This is for evaluator
cd data
python -m gdown 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n
unzip glove.zip
rm glove.zip

cd ../third_packages/TMR
echo -e "Download part-level annotation to TMR dataset/annotation/humanml3d"
python -m gdown "1r17J_5tQipkEmunfZvpA1u47alGU3Zt6" -O datasets/annotations/humanml3d/
python prepare/body_part_annotation_augmentation.py --only_split=True

echo -e "Prepare TMR data"
ln -s ../HumanML3D/pose_data datasets/motions/pose_data

ehco -e "Compute motion features"
python -m prepare.compute_guoh3dfeats
python -m prepare.compute_body_part_guoh3dfeats part_name=head
python -m prepare.compute_body_part_guoh3dfeats part_name=left_arm
python -m prepare.compute_body_part_guoh3dfeats part_name=right_arm
python -m prepare.compute_body_part_guoh3dfeats part_name=torso
python -m prepare.compute_body_part_guoh3dfeats part_name=left_leg
python -m prepare.compute_body_part_guoh3dfeats part_name=right_leg

ehco -e "Compute text embeddings"
python -m prepare.text_embeddings data=humanml3d    
python -m prepare.text_embeddings data=humanml3d_head
python -m prepare.text_embeddings data=humanml3d_left_arm
python -m prepare.text_embeddings data=humanml3d_right_arm
python -m prepare.text_embeddings data=humanml3d_torso
python -m prepare.text_embeddings data=humanml3d_left_leg
python -m prepare.text_embeddings data=humanml3d_right_leg

echo -e "Compute mean and std"
python -m prepare.motion_stats data=humanml3d
python -m prepare.motion_stats data=humanml3d_head
python -m prepare.motion_stats data=humanml3d_left_arm
python -m prepare.motion_stats data=humanml3d_right_arm
python -m prepare.motion_stats data=humanml3d_torso
python -m prepare.motion_stats data=humanml3d_left_leg
python -m prepare.motion_stats data=humanml3d_right_leg

echo -e "Prepare TMR models"
bash prepare/download_pretrain_models.sh       
echo -e "Download pretrained part-level models"
python -m gdown "1Qh7uhtEhPKMcztDufgIz7WLaeeBiTPx-"
unzip body_part_tmr.zip -d models/body_part_tmr
rm body_part_tmr.zip


echo -e "Prepare LGTM model"
cd ../../
python -m gdown "1r_baVr24sf1jnz1KPHFaPB-jfkkzS-Iy"
unzip lgtm.zip -d checkpoints
rm lgtm.zip

echo -e "Finished!"
