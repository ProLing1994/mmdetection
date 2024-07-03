#!/bin/bash

cd /yuanhuan/code/demo/Image/Basic/script/xml/
dataset_root=/yuanhuan/data/image/RM_Face/original

# 定义包含所有路径的数组
relative_image_paths=( 
    "aebs/ADAS_AllInOne"
    "aebs/aebs_ganglong_shengang_all_126w_frames_5f"
    "aebs/C40A_videos_231128_frames_5f"
    "aebs/C40A_videos_231220_frames_2s"
    "aebs/C40A_videos_231226_frames_5f"
    "avm/avm"
    "bsd/Ahead"
    "bsd/LongFocus"
    "bsd/WideAngle"
    "capture/capture_00C800041A_20240408_frames_5f"
    "capture/capture_00C800041A_20240409_frames_5f"
    "capture/capture_00C800041A_20240410_frames_5f"
    "capture/capture_00C800064_face_20240410_0415_frames_5f"
    "capture/capture_00C800064E_20240402_frames_5f"
    "capture/capture_00C800064E_20240403_frames_5f"
    "capture/capture_00C800064E_20240407_frames_5f"
    "capture/capture_00C800064E_20240408_frames_5f"
    "capture/capture_00C800064E_20240416_frames_5f"
    "capture/capture_00C800064E_20240417_frames_5f"
    "capture/capture_003F000BC8_20240409_frames_5f"
    "capture/capture_003F000BC8_20240410_frames_5f"
    "capture/capture_003F000BC8_20240411_frames_5f"
    "capture/capture_003F000BC8_20240413_0415_frames_5f"
    "capture/capture_003F000BC8_20240415_frames_5f"
    "capture/capture_003F000BC8_20240416_frames_5f"
    "capture/capture_003F000BC8_20240417_frames_5f"
    "capture/capture_0025000439_20220818_frames_5f"
    "capture/capture_0025000446_20220406_frames_5f"
    "capture/capture_0025000446_20220407_frames_5f"
    "capture/capture_0025000446_20220408_frames_5f"
    "capture/capture_0025000462_20220519_frames_5f"
    "capture/capture_0025000462_20220524_frames_5f"
    "capture/capture_WANFN7916_20240411_0413_frames_5f"
    "capture/capture_WANFN7916_20240414_0416_frames_5f"
)

# 遍历每个路径并运行命令
for relative_path in "${relative_image_paths[@]}"; do
    image="${dataset_root}/${relative_path}/JPEGImages/"
    xml="${dataset_root}/${relative_path}/Annotations_Face_wZhedang_MMGroundingDINO/"
    nmsxml="${dataset_root}/${relative_path}/Annotations_Face_wZhedang_MMGroundingDINO_NMS/"

    if [[ ! -d $image ]]; then
        echo "Image directory not found: $image"
    fi

    if [[ ! -d $xml ]]; then
        echo "xml directory not found: $xml"
    fi
    
    # 运行 Python 脚本并记录日志
    python /yuanhuan/code/demo/Image/Basic/script/xml/xml_nms.py \
        --jpg_dir "$image" \
        --xml_dir "$xml" \
        --out_xml_dir "$nmsxml" \

done
