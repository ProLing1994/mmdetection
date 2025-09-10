#!/opt/conda/bin bash
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet
export PYTHONPATH=/yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/:$PYTHONPATH

epoch=gdino_tiny_avm_gj_ep3
model_root=/yuanhuan/model/image/mm_grounding_dino/avm
config=$model_root/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_avm.py
checkpoint=$model_root/$epoch.pth

# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_BSD_R151.txt
file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_C28.txt

for line in $(cat ${file_path})
do
    echo "$line"

    # image="${line}/JPEGImages_dupes_0_95/"
    # xml="${line}/Annotations_dupes_0_95_AVM_MMGroundingDINO/"
    # xml_nms="${line}/Annotations_dupes_0_95_AVM_MMGroundingDINO_NMS/"
    image="${line}/JPEGImages_dupes_0_97/"
    xml="${line}/Annotations_dupes_0_97_AVM_MMGroundingDINO/"
    xml_nms="${line}/Annotations_dupes_0_97_AVM_MMGroundingDINO_NMS/"

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
    export PYTHONPATH=/yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/:$PYTHONPATH
    cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/
    python ./demo/image_demo_xml.py \
        "$image" \
        "$config" \
        --weights "$checkpoint" \
        --out-dir "$xml" \
        --save-xml \
        --batch-size 1 \
        --texts 'person . cyclist . car .' 

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