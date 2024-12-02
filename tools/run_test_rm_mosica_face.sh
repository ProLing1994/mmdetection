#!/home/ubuntu/anaconda3/bin bash
source /home/ubuntu/anaconda3/bin/activate mmdet3
conda activate mmdet3

export PYTHONPATH=/home/ubuntu/code/demo/Image/detection2d/ori_mmdetection/mmdetection/:$PYTHONPATH
cd /home/ubuntu/code/demo/Image/detection2d/ori_mmdetection/mmdetection/

epoch=epoch_20
model_root=/data_test/models/mm_grounding_dino/mm_grounding_dino_l_mosica_face_2024_11/
config=$model_root/grounding_dino_swin-l_finetune_mosica_face_rm.py
checkpoint=$model_root/$epoch.pth

file_path=/data_test/images/RM_Face_Mosaic/testing/test_data_size_up_708_480_20240918/analysis/dataset_list_mosica_face_total_balanced_merge_testing.txt
# file_path=/data_test/images/RM_Face_Mosaic/testing/test_data_size_up_708_480_20240918/analysis/dataset_list_mosica_face_total_balanced_testing.txt
JPEGImages_folder=JPEGImages_test
Annotations_folder=Annotations_test_res/mm_grounding_dino_l_mosica_face_2024_11
for line in $(cat ${file_path})
do
    echo "$line"
    image=$line/$JPEGImages_folder/
    xml_output_dir=$line/$Annotations_folder/
    python ./demo/image_demo_xml.py \
        $image \
        $config \
        --weights $checkpoint \
        --out-dir $xml_output_dir \
        --save-xml \
        --batch-size 1 \
        --texts 'head .' \
        --device 'cuda:0'
done