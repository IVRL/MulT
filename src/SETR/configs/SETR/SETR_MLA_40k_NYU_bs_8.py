_base_ = [
    '../_base_/models/setr_mla.py',
    '../_base_/datasets/nyu_seg.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
find_unused_parameters = True

model = dict(
    backbone=dict(
        img_size=(464, 464),
        num_classes=41
    ),
    decode_head=dict(
        ignore_index=0,
        img_size=(464, 464),
        num_classes=41
    )
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

data = dict(
    train=dict(args=dict(use_dense_depth=False)),
    val=dict(args=dict(use_dense_depth=False)),
    test=dict(args=dict(use_dense_depth=False)))
args = dict(use_dense_depth=False)
