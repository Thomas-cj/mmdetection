_base_ = [
    '../../custom_configs/faster_rcnn/faster-rcnn_r50_fpn_custom.py',
    '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/schedules/schedule_1x.py', 
    '../../configs/_base_/default_runtime.py'
]

checkpoint = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)))

# `lr` and `weight_decay` have been searched to be optimal.
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.1),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))


