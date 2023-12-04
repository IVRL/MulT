dataset_type = 'DepthDataset'
args = dict(
    dataset='NYU',
    max_depth=10.0,
    height=464,
    width=464,
    data_path='/sinergia/deblina/datasets/NYU_Depth_V2/official_splits',
    trainfile_nyu='/workspace/project/LapDepth-release/datasets'
                  '/nyudepthv2_labeled_train_files_with_gt_dense_contours.txt',
    testfile_nyu='/workspace/project/LapDepth-release/datasets'
                 '/nyudepthv2_test_files_with_gt_dense_contours.txt',
    use_dense_depth=True,
    use_sparse=False,
    use_seg=False
)
data = dict(
    samples_per_gpu=8,  # TODO: 8?
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        args=args,
        train=True,
        return_filename=False
    ),
    val=dict(
        type=dataset_type,
        args=args,
        train=False,
        return_filename=False
    ),
    test=dict(
        type=dataset_type,
        args=args,
        train=False,
        return_filename=False
    )
)
