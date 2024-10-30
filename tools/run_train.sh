#!/opt/conda/bin bash
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/bin/activate base
conda init bash
conda activate mmdet

export http_proxy=http://192.168.151.254:7890 ; export https_proxy=http://192.168.151.254:7890
cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/

# ./tools/dist_train.sh configs/mm_grounding_dino/rm/grounding_dino_swin-t_finetune_capture.py 2 --work-dir /yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_t_balanced_1w_capture
# ./tools/dist_train.sh configs/mm_grounding_dino/rm/grounding_dino_swin-l_finetune_capture.py 2 --work-dir /yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_balanced_1w_capture
# ./tools/dist_train.sh configs/mm_grounding_dino/rm/grounding_dino_swin-l_finetune_capture_rm.py 2 --work-dir /yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture
# ./tools/dist_train.sh configs/mm_grounding_dino/rm/grounding_dino_swin-l_finetune_capture_face.py 2 --work-dir /yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_1w_3k_capture_face_occlusion
# ./tools/dist_train.sh configs/mm_grounding_dino/rm/grounding_dino_swin-l_finetune_huanwei_rm.py 2 --work-dir /yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_balanced_3k_huanwei
./tools/dist_train.sh configs/mm_grounding_dino/rm/grounding_dino_swin-l_finetune_character_rm.py 2 --work-dir /yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_character_add_huanwei