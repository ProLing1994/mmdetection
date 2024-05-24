cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/

date_name_list=(car bus truck motorcyclist license)

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    
    epoch=epoch_20
    model_root=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_balanced_1w_capture
    config=/yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/configs/mm_grounding_dino/rm/grounding_dino_swin-l_finetune_capture.py
    xml_output_dir=$model_root/test/$epoch/$date_name/Annotations/
    image=/yuanhuan/data/image/RM_Capture/dataset_test_select_very_choosy_2_0/$date_name/JPEGImages/

    python ./demo/image_demo_xml.py \
        $image \
        $config \
        --weights "/yuanhuan/model/image/mm_grounding_dino/grounding_dino_swin-l_pretrain_all-56d69e78.pth" \
        --out-dir $xml_output_dir \
        --save-xml \
        --batch-size 1 \
        --texts 'car . bus . truck . motorcyclist . license .'

done