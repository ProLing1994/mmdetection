_base_ = '../coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py'

load_from = "/yuanhuan/model/image/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
data_root = '/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/'
class_name = ('car', 'bus', 'truck', 'motorcyclist', 'license',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255), (0, 60, 100)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/1w/capture_train_1w.json',
        data_prefix=dict(img='1w/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/1w/capture_train_1w.json',
        data_prefix=dict(img='1w/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file='/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/1w/capture_train_1w.json')
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[13,15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)

work_dir = 'mm_grounding_dino_t_capture'