#!/bin/bash

# cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/
# date_name_list=(bicyclist front_face side_face person)
# # date_name_list=(car bus truck motorcyclist license)
# # date_name_list=(car bus truck motorcyclist license bicyclist front_face side_face person)
# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
#     epoch=epoch_20
#     model_root=/yuanhuan/model/image/mm_grounding_dino/zbw/mm_grounding_dino_l_capture_face
#     config=$model_root/grounding_dino_swin-l_finetune_8xb4_20e_capture_face.py
#     checkpoint=$model_root/$epoch.pth
#     xml_output_dir=$model_root/dataset_test_select_very_choosy_2_0/$epoch/$date_name/Annotations/
#     image=/yuanhuan/data/image/RM_Capture/dataset_test_select_very_choosy_2_0/$date_name/JPEGImages/
#     python ./demo/image_demo_xml.py \
#         $image \
#         $config \
#         --weights $checkpoint \
#         --out-dir $xml_output_dir \
#         --save-xml \
#         --batch-size 1 \
#         --texts 'bicyclist . front_face . side_face . person .'
# done

#!/bin/bash
cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/
img_dir=/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/face/jpg/
for date_name in $img_dir*; do 
    date_name=$(basename $date_name)
    echo $date_name
    epoch=epoch_20
    model_root=/yuanhuan/model/image/mm_grounding_dino/zbw/mm_grounding_dino_l_capture_faceocclusion
    config=$model_root/grounding_dino_swin-l_finetune_8xb4_20e_capture_faceocclusion.py
    checkpoint=$model_root/$epoch.pth
    xml_output_dir=$model_root/dataset_RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240702_侧向镜头_人车漏检/face/$date_name/Annotations/
    image=$img_dir/$date_name/
    python ./demo/image_demo_xml.py \
        $image \
        $config \
        --weights $checkpoint \
        --out-dir $xml_output_dir \
        --save-xml \
        --batch-size 1 \
        --texts 'front_face . side_face . face_occlusion . person . bicyclist . motorcyclist .'
done