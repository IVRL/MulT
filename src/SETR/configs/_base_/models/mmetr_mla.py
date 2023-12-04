# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderBoth',
    backbone=dict(
        type='MMETR',
        model_name='vit_large_patch16_384',
        img_size=(464, 464),
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=41,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        mla_channels=256,
        mla_index=(5, 11, 17, 23)
    ),
    decode_head=dict(
        type='METR_Head',
        in_channels=1024,
        channels=512,
        img_size=(464, 464),
        mla_channels=256,
        mlahead_channels=128,
        num_classes=41,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
