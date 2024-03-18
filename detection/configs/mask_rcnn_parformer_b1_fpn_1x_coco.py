_base_ = [
    '../configs/_base_/models/mask_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
model = dict(
    # pretrained='pretrained/pvt_tiny.pth',
    pretrained='/home/ndr/Documents/VisionWorkspace/checkpoints/finetune/ParFormer_B1/model.pth',
    backbone=dict(
        type='ParFormer_B1',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00005, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
