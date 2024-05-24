_base_ = '../coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py'

load_from = "/yuanhuan/model/image/mm_grounding_dino/grounding_dino_swin-l_pretrain_all-56d69e78.pth"
# load_from = "/yuanhuan/model/image/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
data_root = '/yuanhuan/data/image/RM_Capture/original_c27_dupes_0_95/Balanced_selection/'
class_name = ('car', 'bus', 'truck', 'motorcyclist', 'license',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255), (0, 60, 100)])

num_levels = 5
model = dict(
    use_autocast=True,
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=None),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))),
    bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
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

work_dir = 'mm_grounding_dino_l_capture'