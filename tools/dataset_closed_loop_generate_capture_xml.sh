export HF_ENDPOINT=https://hf-mirror.com

# user config info
IMAGE_PATH=$1
RESULTS_PATH=$2

# local reps config
ROOT_PATH=/yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection

CONFIG_PATH=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture/grounding_dino_swin-l_finetune_capture_rm.py
WEIGHT_PATH=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_2w_capture/epoch_20.pth

# cd reps root path
pushd ${ROOT_PATH}

# generate xml dt results
CUDA_VISIBLE_DEVICES=0 python demo/image_demo_xml.py \
    ${IMAGE_PATH} \
    ${CONFIG_PATH} \
    --weights ${WEIGHT_PATH} \
    --out-dir  ${RESULTS_PATH} \
    --save-xml \
    --batch-size 4 \
    --texts 'car . bus . truck . motorcyclist . license .'

popd