#!/opt/conda/bin bash
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet

cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/
date_name_list=(car bus truck motorcyclist license bicyclist front_face side_face person)
for date_name in ${date_name_list[@]}; do 
    echo $date_name
    epoch=epoch_20
    model_root=/yuanhuan/model/image/mm_grounding_dino/zbw/mm_grounding_dino_l_capture_2w
    config=$model_root/grounding_dino_swin-l_finetune_8xb4_20e_capture_2w.py
    checkpoint=$model_root/$epoch.pth
    xml_output_dir=$model_root/dataset_test_select_very_choosy_2_0/$epoch/$date_name/Annotations/
    image=/yuanhuan/data/image/RM_Capture/dataset_test_select_very_choosy_2_0/$date_name/JPEGImages/
    python ./demo/image_demo_xml.py \
        $image \
        $config \
        --weights $checkpoint \
        --out-dir $xml_output_dir \
        --save-xml \
        --batch-size 1 \
        --texts 'car . bus . truck . bicyclist . motorcyclist . license . front_face . side_face . person .'
done