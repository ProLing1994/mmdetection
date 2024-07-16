#!/opt/conda/bin bash
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet

epoch=epoch_20
model_root=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture
config=$model_root/grounding_dino_swin-l_finetune_capture_rm.py
checkpoint=$model_root/$epoch.pth
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_total.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_add_202407.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_add_202407_1.txt
file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_add_202407_2.txt
for line in $(cat ${file_path})
do
    echo "$line"

    # image="${line}/JPEGImages/"
    # xml="${line}/Annotations_Captue_MMGroundingDINO/"
    # xml_nms="${line}/Annotations_Captue_MMGroundingDINO_NMS/"
    image="${line}/JPEGImages_dupes_0_95/"
    xml="${line}/Annotations_dupes_0_95_Captue_MMGroundingDINO/"
    xml_nms="${line}/Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS/"

    if [[ -d $xml_nms ]]; then
        echo "XML NMS directory is Done: $xml"
        continue  # 跳过这个路径
    fi
    
    if [[ ! -d $image ]]; then
        echo "Image directory not found: $image"
        continue  # 跳过这个路径
    fi

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
        --batch-size 4 \
        --texts 'car . bus . truck . motorcyclist . license .' 

    if [[ ! -d $xml ]]; then
        echo "xml directory not found: $xml"
        continue  # 跳过这个路径
    fi

    # 运行 Python 脚本并记录日志
    python /yuanhuan/code/demo/Image/Basic/script/xml/xml_nms.py \
        --jpg_dir "$image" \
        --xml_dir "$xml" \
        --out_xml_dir "$xml_nms"

done