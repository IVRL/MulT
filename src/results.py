import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm

left = 43 + 50
right = left + 464
top = 8
bottom = top + 464

depth_cmap = plt.cm.turbo_r
diff_cmap = sns.color_palette("vlag", as_cmap=True)

# folder with models directories
if len(sys.argv) > 1:
    main_dir = sys.argv[1] or '.'
else:
    main_dir = '.'

dirs = next(os.walk(main_dir))[1]  # models directories
source_dirs = [os.path.join(main_dir, dr) for dr in dirs]  # models directories
dest_dirs = [os.path.join(main_dir, 'clean', 'clean_' + dr) for dr in dirs]
if not os.path.exists(os.path.join(main_dir, 'clean')):
    os.makedirs(os.path.join(main_dir, 'clean'))

for source_dir, dest_dir in zip(source_dirs, dest_dirs):  # model directory
    print(source_dir, '->', dest_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dirs = next(os.walk(source_dir))[1]  # results directories

    if 'input_rgb' in dirs:
        print('cleaning rgb . . .')
        if not os.path.exists(os.path.join(dest_dir, 'clean_input_rgb')):
            os.makedirs(os.path.join(dest_dir, 'clean_input_rgb'))
        rgb_dir = os.path.join(source_dir, 'input_rgb')
        files = next(os.walk(rgb_dir))[2]
        for file in tqdm(files):
            rgb = Image.open(os.path.join(rgb_dir, file))
            rgb = np.asarray(rgb)
            if rgb.shape[0] > 464 or rgb.shape[1] > 464:
                assert rgb.shape[0] == 480 and rgb.shape[1] == 640, 'rgb.shape: ' + str(rgb.shape)
                rgb = rgb[top:bottom, left:right]
            Image.fromarray(rgb).save(os.path.join(dest_dir, 'clean_input_rgb', 'clean_' + file))

    # if 'dense_gt' in dirs and 'output_depth' in dirs:
    #     print('cleaning depth . . .')
    #     if not os.path.exists(os.path.join(dest_dir, 'clean_dense_gt')):
    #         os.makedirs(os.path.join(dest_dir, 'clean_dense_gt'))
    #     if not os.path.exists(os.path.join(dest_dir, 'clean_output_depth')):
    #         os.makedirs(os.path.join(dest_dir, 'clean_output_depth'))
    #     if not os.path.exists(os.path.join(dest_dir, 'cdiff_output_depth')):
    #         os.makedirs(os.path.join(dest_dir, 'cdiff_output_depth'))
    #     if not os.path.exists(os.path.join(dest_dir, 'diff_output_depth')):
    #         os.makedirs(os.path.join(dest_dir, 'diff_output_depth'))
    #     dense_gt = os.path.join(source_dir, 'dense_gt')
    #     output_depth = os.path.join(source_dir, 'output_depth')
    #     gt_files = next(os.walk(dense_gt))[2]
    #     pred_files = next(os.walk(output_depth))[2]
    #     for gt_file, pred_file in tqdm(list(zip(gt_files, pred_files))):
    #         gt = Image.open(os.path.join(dense_gt, gt_file))
    #         pred = Image.open(os.path.join(output_depth, pred_file))
    #         gt = np.asarray(gt)
    #         pred = np.asarray(pred)
    #         if gt.shape[0] > 464 or gt.shape[1] > 464 or pred.shape[0] > 464 or pred.shape[1] > 464:
    #             assert gt.shape[0] == 480 and gt.shape[1] == 640, 'gt.shape: ' + str(gt.shape)
    #             assert pred.shape[0] == 480 and pred.shape[1] == 640, 'pred.shape: ' + str(pred.shape)
    #             gt = gt[top:bottom, left:right]
    #             pred = pred[top:bottom, left:right]
    #         mn = gt.min()
    #         mx = gt.max()
    #         gt = (gt - mn) * 1. / (mx - mn)
    #         pred = (pred - mn) * 1. / (mx - mn)
    #         diff = pred - gt
    #         cdiff = diff / 2 + 0.5
    #         gt = depth_cmap(gt)
    #         pred = depth_cmap(pred)
    #         cdiff = diff_cmap(cdiff)
    #         diff = plt.cm.Greys(np.abs(diff))
    #         Image.fromarray(np.uint8(gt * 255)).save(
    #             os.path.join(dest_dir, 'clean_dense_gt', 'clean_' + gt_file)[:-4] + '.png')
    #         Image.fromarray(np.uint8(pred * 255)).save(
    #             os.path.join(dest_dir, 'clean_output_depth', 'clean_' + pred_file)[:-4] + '.png')
    #         Image.fromarray(np.uint8(cdiff * 255)).save(
    #             os.path.join(dest_dir, 'cdiff_output_depth', 'cdiff_' + pred_file)[:-4] + '.png')
    #         Image.fromarray(np.uint8(diff * 255)).save(
    #             os.path.join(dest_dir, 'diff_output_depth', 'diff_' + pred_file)[:-4] + '.png')

    if 'attn' in dirs:
        print('cleaning attention . . .')
        if not os.path.exists(os.path.join(dest_dir, 'clean_attn')):
            os.makedirs(os.path.join(dest_dir, 'clean_attn'))
        attn_dir = os.path.join(source_dir, 'attn')
        files = next(os.walk(attn_dir))[2]
        with open(
                '/workspace/project/LapDepth-release/datasets/nyudepthv2_test_files_with_gt_dense_contours.txt',
                'r') as f:
            fileset = f.readlines()
        fileset = sorted(fileset)
        for file in tqdm(files):
            if 'f_blk_' not in file:
                divided_file = fileset[int(file.split('_')[1])].split()
                divided_file_ = divided_file[0].split('/')
                filename = divided_file_[1] + '_' + divided_file_[2][-9:]

                attn = Image.open(os.path.join(attn_dir, file))
                attn = np.asarray(attn)
                attnf = Image.open(os.path.join(attn_dir, file.replace('_blk_', 'f_blk_')))
                attnf = np.asarray(attnf)[:, ::-1]
                attnm = attn * 0.5 + attnf * 0.5
                Image.fromarray(np.uint8(plt.cm.Greys_r(attnm / 255) * 255)).save(
                    os.path.join(dest_dir, 'clean_attn', 'clean_attn_blk_' + file.split('_')[-1][:-4] + '_' + filename)[
                    :-4] + '.png')
