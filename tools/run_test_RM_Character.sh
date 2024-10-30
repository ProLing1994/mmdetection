#!/opt/conda/bin bash
# pytorch-mmdet-mmseg-yuanhuan
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet
export http_proxy=http://192.168.151.254:7890 ; export https_proxy=http://192.168.151.254:7890

epoch=epoch_20
model_root=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_character
config=$model_root/grounding_dino_swin-l_finetune_character_rm.py
checkpoint=$model_root/$epoch.pth

# file_path=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/DIFFSTE/original_test_scale_padding/dataset_list_total.txt
# for line in $(cat ${file_path})

image_paths=(
    # "/yuanhuan/data/image/RM_ANPR/original/RM_Character/"
    # "/yuanhuan/data/image/RM_HUANWEI/original/Argentina/DIFFSTE/original_test_scale_padding/"
    "/yuanhuan/data/image/RM_HUANWEI/original/Argentina/DIFFSTE/original_220240923_scale_padding/"
)
for line in "${image_paths[@]}"
do
    echo "$line"

    image="${line}/JPEGImages/"
    xml="${line}/Annotations_Captue_MMGroundingDINO/"
    xml_nms="${line}/Annotations_Captue_MMGroundingDINO_NMS/"
    # image="${line}/JPEGImages_test/"
    # xml="${line}/Annotations_test_Captue_MMGroundingDINO/"
    # xml_nms="${line}/Annotations_test_Captue_MMGroundingDINO_NMS/"


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
        --texts 'char .'

    if [[ ! -d $xml ]]; then
        echo "xml directory not found: $xml"
        continue  # 跳过这个路径
    fi

    # 运行 Python 脚本并记录日志
    python /yuanhuan/code/demo/Image/Basic/script/xml/xml_nms_dupes.py \
        --jpg_dir "$image" \
        --xml_dir "$xml" \
        --out_xml_dir "$xml_nms"

done