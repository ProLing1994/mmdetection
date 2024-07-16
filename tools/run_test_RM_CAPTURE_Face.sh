#!/opt/conda/bin bash
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet

epoch=epoch_20
model_root=/yuanhuan/model/image/mm_grounding_dino/zbw/mm_grounding_dino_l_capture_faceocclusion
config=$model_root/grounding_dino_swin-l_finetune_8xb4_20e_capture_faceocclusion.py
checkpoint=$model_root/$epoch.pth

image_paths=(
    # "/yuanhuan/data/image/RM_Face/original/capture/capture_wheel_2024-07-11"
    # "/yuanhuan/data/image/RM_Face/original/temp/capture_2024-07-02"
    # "/yuanhuan/data/image/RM_Face/original/temp/capture_2024-07-04"
    "/yuanhuan/data/image/RM_Face/original/temp/face_occlusion/000"
    "/yuanhuan/data/image/RM_Face/original/temp/face_occlusion/001"
    "/yuanhuan/data/image/RM_Face/original/temp/face_occlusion/002"
    "/yuanhuan/data/image/RM_Face/original/temp/face_occlusion/003"
    "/yuanhuan/data/image/RM_Face/original/temp/face_occlusion/004"
    "/yuanhuan/data/image/RM_Face/original/temp/face_occlusion/005"
)
for line in "${image_paths[@]}"
do
    echo "$line"

    image="${line}/JPEGImages/"
    xml="${line}/Annotations_Face_wZhedang_MMGroundingDINO/"
    xml_nms="${line}/Annotations_Face_wZhedang_MMGroundingDINO_NMS/"

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
        --texts 'front_face . side_face . face_occlusion . person . bicyclist . motorcyclist .' 

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