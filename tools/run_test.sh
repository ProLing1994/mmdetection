cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/

date_name_list=(car bus truck motorcyclist license)

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    
    epoch=epoch_20
    model_root=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_balanced_1w_capture
    config=$model_root/grounding_dino_swin-l_finetune_capture.py
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
        --texts 'car . bus . truck . motorcyclist . license .'

done