#!/opt/conda/bin bash
# pytorch-mmdet-mmseg-yuanhuan
#export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet


epoch=epoch_20
model_root=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture
config=$model_root/grounding_dino_swin-l_finetune_capture_rm.py
checkpoint=$model_root/$epoch.pth

# file_path=/yuanhuan/data/image/RM_AVM/AVM/dataset_list_AVM.txt   
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_BSD_R151.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_ZG.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_C28.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_ANPR.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_LicensePlate_detection.txt

# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_total.txt
# file_path=/yuanhuan/data/image/RM_Capture/analysis/dataset_list_add_202407.txt

file_path=/yuanhuan/data/image/ZG_test/test.txt
for line in $(cat ${file_path})

# image_paths=(
#     "/yuanhuan/data/image/RM_Capture/training/Capture_Plate_w_mutildupes_FilterSmallBox_OnlyLicense"
# )
# for line in "${image_paths[@]}"
do
    echo "$line"

    image="${line}/JPEGImages/"
    xml="${line}/Annotations_Captue_MMGroundingDINO/"
    xml_nms="${line}/Annotations_Captue_MMGroundingDINO_NMS/"
    # image="${line}/JPEGImages_dupes_0_93/"
    # xml="${line}/Annotations_dupes_0_93_Captue_MMGroundingDINO/"
    # xml_nms="${line}/Annotations_dupes_0_93_Captue_MMGroundingDINO_NMS/"
    # image="${line}/JPEGImages_dupes_0_95/"
    # xml="${line}/Annotations_dupes_0_95_Captue_MMGroundingDINO/"
    # xml_nms="${line}/Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS/"
    # image="${line}/JPEGImages_dupes_0_97/"
    # xml="${line}/Annotations_dupes_0_97_Captue_MMGroundingDINO/"
    # xml_nms="${line}/Annotations_dupes_0_97_Captue_MMGroundingDINO_NMS/"
    # image="${line}/wrong_JPEGImages/"
    # xml="${line}/Annotations_wrong/"
    # xml_nms="${line}/Annotations_wrong_NMS/"

    if [[ ! -d $image ]]; then
        echo "Image directory not found: $image"
        continue  # 跳过这个路径
    fi

    # 删除 $image 文件夹中不是 .jpg 的数据
    # find "$image" -type f ! -name "*.jpg" -delete
    find "$image" -type f ! -name "*.jpg" -print0 | xargs -0 rm -f

    if [[ -d $xml_nms ]]; then

        # 统计输入文件夹下的文件数量
        input_file_count=$(find "$image" -type f | wc -l)
        # 统计输出文件夹下的文件数量
        output_file_count=$(find "$xml_nms" -type f | wc -l)

        echo "input_file_count: $input_file_count"
        echo "output_file_count: $output_file_count"

        if [[ $(($input_file_count > $output_file_count ? $input_file_count - $output_file_count : $output_file_count - $input_file_count)) -lt 100 ]]; then
            echo "XML NMS directory is Done: $xml_nms"
            continue  # 跳过这个路径
        else
            echo "输入和输出文件夹下的文件数量不一致"
        fi
    fi

    if [[ ! -d $xml ]]; then
        echo "XML output directory not found, creating: $xml"
        mkdir -p "$xml"
    fi

    # 运行 Python 脚本并记录日志
    cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/
    export PYTHONPATH=/yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection:$PYTHONPATH
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
    python /yuanhuan/code/demo/Image/Basic/script/xml/xml_nms_dupes.py \
        --jpg_dir "$image" \
        --input_xml_dir "$xml" \
        --output_xml_dir "$xml_nms"

done