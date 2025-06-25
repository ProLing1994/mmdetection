#!/opt/conda/bin bash
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet
export PYTHONPATH=/yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/:$PYTHONPATH

# epoch=grounding_dino_swin-l_finetune_adas_10_percent_epoch_20_v1.1.2
# model_root=/yuanhuan/model/image/mm_grounding_dino/ADAS
# config=$model_root/grounding_dino_swin-l_finetune_adas_10_percent_epoch_20_v1.1.2.py
# checkpoint=$model_root/$epoch.pth

epoch=gdino_pretrain_1.7w_aebs_zero_shot_59.4
model_root=/yuanhuan/model/image/mm_grounding_dino/ADAS/v1.1/
config=$model_root/grounding_dino_swin-l_pretrain_obj365_goldg.py
checkpoint=$model_root/$epoch.pth

# # file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_BSD_R151.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_C28.txt
# for line in $(cat ${file_path})

image_paths=(
    "/yuanhuan/data/image/RM_Capture/training/Capture_Plate_Balanced_selection_c27/1w"
    "/yuanhuan/data/image/RM_Capture/training/Capture_Plate_1w_dupes_0_1"
)
for line in "${image_paths[@]}"
do
    echo "$line"

    # image="${line}/JPEGImages_dupes_0_95/"
    # xml="${line}/Annotations_dupes_0_95_ADAS_MMGroundingDINO/"
    # xml_nms="${line}/Annotations_dupes_0_95_ADAS_MMGroundingDINO_NMS/"
    image="${line}/JPEGImages/"
    xml="${line}/Annotations_ADAS_MMGroundingDINO/"
    xml_nms="${line}/Annotations_ADAS_MMGroundingDINO_NMS/"

    # if [[ -d $xml_nms ]]; then
    #     echo "XML NMS directory is Done: $xml"
    #     continue  # 跳过这个路径
    # fi
    
    # if [[ ! -d $image ]]; then
    #     echo "Image directory not found: $image"
    #     continue  # 跳过这个路径
    # fi

    if [[ ! -d $xml ]]; then
        echo "XML output directory not found, creating: $xml"
        mkdir -p "$xml"
    fi

    # 运行 Python 脚本并记录日志
    cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/
    python ./demo/image_demo_xml.py \
        "$image" \
        "$config" \
        --weights "$checkpoint" \
        --out-dir "$xml" \
        --save-xml \
        --batch-size 1 \
        --texts 'car . bus . truck . tricycle . bicycle . motorcycle . car_reg . car_big_reg . car_front . car_big_front . tricycle_reg . person . bicyclist . motorcyclist . tricyclist . license .' 

    if [[ ! -d $xml ]]; then
        echo "xml directory not found: $xml"
        continue  # 跳过这个路径
    fi

    # 运行 Python 脚本并记录日志
    python /yuanhuan/code/demo/Image/Basic/script/xml/xml_nms_dupes.py \
        --jpg_dir "$image" \
        --input_xml_dir "$xml" \
        --output_xml_dir "$xml_nms"

done