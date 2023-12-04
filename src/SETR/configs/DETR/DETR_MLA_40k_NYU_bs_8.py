_base_ = [
    '../_base_/models/detr_mla.py',
    '../_base_/datasets/nyu.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
find_unused_parameters = True
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
