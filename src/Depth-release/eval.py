# -*- coding: utf-8 -*-
import argparse
import csv

import seaborn as sns
import torch.backends.cudnn as cudnn
import torch.utils.data
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from datasets.datasets_list import MyDataset
from logger import TermLogger
from model_merged import *
from trainer import validate
from utils import *

parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting
parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--trainfile_kitti', type=str, default="./datasets/eigen_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_kitti', type=str, default="./datasets/eigen_test_files_with_gt_dense.txt")
parser.add_argument('--trainfile_nyu', type=str, default="./datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str, default="./datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str, default="./datasets/KITTI")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')
parser.add_argument('--use_contours', action='store_true')

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epochs', default=0, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=24, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default="KITTI")

# Logging setting
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-metric', default='_LRDN_evaluation.csv', metavar='PATH',
                    help='csv where to save validation metric value')
parser.add_argument('--val_in_train', action='store_true', help='validation process in training')

# Model setting
parser.add_argument('--encoder', type=str, default="ResNext101")
parser.add_argument('--norm', type=str, default="BN")
parser.add_argument('--act', type=str, default="ReLU")
parser.add_argument('--height', type=int, default=352)
parser.add_argument('--width', type=int, default=704)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# Evaluation setting
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--multi_test', action='store_true', help='test all of model in the dir')
parser.add_argument('--img_save', action='store_true', help='will save test set image')
parser.add_argument('--cap', default=80.0, type=float, metavar='MaxVal', help='cap setting for kitti eval')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default="0,1,2,3", help='force available gpu index')
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


def main():
    args = parser.parse_args()
    print("=> No Distributed Training")
    print('=> Index of using GPU: ', args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    torch.manual_seed(args.seed)

    if args.evaluate is True:
        save_path = save_path_formatter(args, parser)
        args.save_path = 'checkpoints' / save_path
        print("=> information will be saved in {}".format(args.save_path))
        args.save_path.makedirs_p()
        training_writer = SummaryWriter(args.save_path)

        ######################   Data loading part    ##########################
    if args.dataset == 'KITTI':
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0

    if args.result_dir == '':
        args.result_dir = './' + args.dataset + '_Eval_results'
    args.log_metric = args.dataset + '_' + args.encoder + args.log_metric

    test_set = MyDataset(args, train=False, return_top=True)
    print("=> Dataset: ", args.dataset)
    print("=> Data height: {}, width: {} ".format(args.height, args.width))
    print('=> test  samples_num: {}  '.format(len(test_set)))

    test_sampler = None

    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    cudnn.benchmark = True
    ###########################################################################

    ###################### setting model list #################################
    if args.multi_test is True:
        print("=> all of model tested")
        models_list_dir = Path(args.models_list_dir)
        models_list = sorted(models_list_dir.files('*.pkl'))
    else:
        print("=> just one model tested")
        models_list = [args.model_dir]

    ###################### setting Network part ###################
    print("=> creating model")
    Model = LDRN(args)

    num_params_encoder = 0
    num_params_decoder = 0
    for p in Model.encoder.parameters():
        num_params_encoder += p.numel()
    for p in Model.decoder.parameters():
        num_params_decoder += p.numel()
    print("===============================================")
    print("model encoder parameters: ", num_params_encoder)
    print("model decoder parameters: ", num_params_decoder)
    print("Total parameters: {}".format(num_params_encoder + num_params_decoder))
    print("===============================================")
    Model = Model.cuda()
    Model = torch.nn.DataParallel(Model)

    if args.evaluate is True:
        ############################ data log #######################################
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(val_loader), args.epoch_size),
                            valid_size=len(val_loader))
        with open(args.save_path / args.log_metric, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            if args.dataset == 'KITTI':
                writer.writerow(['Filename', 'Abs_diff', 'Abs_rel', 'Sq_rel', 'a1', 'a2', 'a3', 'RMSE', 'RMSE_log'])
            elif args.dataset == 'Make3D':
                writer.writerow(['Filename', 'Abs_diff', 'Abs_rel', 'log10', 'rmse'])
            elif args.dataset == 'NYU':
                writer.writerow(['Filename', 'Abs_diff', 'Abs_rel', 'log10', 'a1', 'a2', 'a3', 'RMSE', 'RMSE_log'])
        ########################### Evaluating part #################################
        test_model = Model

        print("Model Initialized")

        test_len = len(models_list)
        print("=> Length of model list: ", test_len)

        for i in range(test_len):
            filename = models_list[i].split('/')[-1]
            logger.reset_valid_bar()
            test_model.load_state_dict(torch.load(models_list[i], map_location='cuda:0'))
            # test_model.load_state_dict(torch.load(models_list[i]))
            test_model.eval()
            if args.dataset == 'KITTI':
                errors, error_names = validate(args, val_loader, test_model, logger, 'KITTI')
            elif args.dataset == 'NYU':
                errors, error_names = validate(args, val_loader, test_model, logger, 'NYU')
            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, 0)
            logger.valid_writer.write(' * model: {}'.format(models_list[i]))
            print("")
            error_string = ', '.join('{} : {:.4f}'.format(name, error) for name, error in
                                     zip(error_names[0:len(error_names)], errors[0:len(errors)]))
            logger.valid_writer.write(' * Avg {}'.format(error_string))
            print("")
            logger.valid_bar.finish()
            with open(args.save_path / args.log_metric, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow(['%s' % filename] + ['%.4f' % (errors[k]) for k in range(len(errors))])

        print(args.dataset, " valdiation finish")
        ##  Test

        if args.img_save is False:
            print("--only Test mode finish--")
            return
    else:
        test_model = Model
        test_model.load_state_dict(torch.load(models_list[0], map_location='cuda:0'))
        # test_model.load_state_dict(torch.load(models_list[0]))
        test_model.eval()
        print("=> No validation")

    test_set = MyDataset(args, train=False, return_filename=True)
    test_sampler = None
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    if args.img_save is True:
        cmap = plt.cm.turbo_r
        diff_cmap = sns.color_palette("vlag", as_cmap=True)
        left = 43 + 50
        right = left + 464
        top = 8
        bottom = top + 464
        if not os.path.exists(os.path.join(args.result_dir, 'output_depth_cmap_turbor')):
            os.makedirs(os.path.join(args.result_dir, 'output_depth_cmap_turbor'))
        if not os.path.exists(os.path.join(args.result_dir, 'dense_gt_cmap_turbor')):
            os.makedirs(os.path.join(args.result_dir, 'dense_gt_cmap_turbor'))
        if not os.path.exists(os.path.join(args.result_dir, 'cdiff_output_depth')):
            os.makedirs(os.path.join(args.result_dir, 'cdiff_output_depth'))
        if not os.path.exists(os.path.join(args.result_dir, 'diff_output_depth')):
            os.makedirs(os.path.join(args.result_dir, 'diff_output_depth'))

        for rgb_data, gt_data, gt_dense, c0, c1, c2, c3, filename in tqdm(val_loader):
            if gt_data.ndim != 4 and gt_data[0] == False:
                continue
            img_H = gt_data.shape[2]
            img_W = gt_data.shape[3]
            gt_data = Variable(gt_data.cuda())
            input_img = Variable(rgb_data.cuda())
            gt_data = gt_data.clamp(0, args.max_depth)
            if args.use_dense_depth is True:
                gt_dense = Variable(gt_dense.cuda())
                gt_dense = gt_dense.clamp(0, args.max_depth)

            input_img_flip = torch.flip(input_img, [3])

            if args.use_contours:
                contours0, contours1, contours2, contours3 = c0, c1, c2, c3
                contours0, contours1, contours2, contours3 = contours0.cuda(), contours1.cuda(), contours2.cuda(), contours3.cuda()
                contours0, contours1, contours2, contours3 = Variable(contours0), Variable(contours1), Variable(
                    contours2), Variable(contours3)
                contours0_flip, contours1_flip, contours2_flip, contours3_flip = torch.flip(contours0, [3]), torch.flip(
                    contours1, [3]), torch.flip(contours2, [3]), torch.flip(contours3, [3])
                input_img = (input_img, contours0, contours1, contours2, contours3)
                input_img_flip = (input_img_flip, contours0_flip, contours1_flip, contours2_flip, contours3_flip)

            with torch.no_grad():
                _, final_depth = test_model(input_img)
                _, final_depth_flip = test_model(input_img_flip)
            final_depth_flip = torch.flip(final_depth_flip, [3])
            final_depth = 0.5 * (final_depth + final_depth_flip)

            final_depth = final_depth.clamp(0, args.max_depth)

            gt_dense = gt_dense[:, :, top:bottom, left:right]
            final_depth = final_depth[:, :, top:bottom, left:right]

            d_min = gt_dense.min()
            d_max = gt_dense.max()

            d_min = d_min.cpu().detach().numpy().astype(np.float32)
            d_max = d_max.cpu().detach().numpy().astype(np.float32)
            # TODO: save contours
            filename = filename[0]

            gt_ = np.squeeze(gt_dense.cpu().numpy().astype(np.float32))
            out_ = np.squeeze(final_depth.cpu().numpy().astype(np.float32))
            gt_ = ((gt_ - d_min) / (d_max - d_min))
            out_ = ((out_ - d_min) / (d_max - d_min))
            diff = out_ - gt_
            cdiff = diff / 2 + 0.5
            gt_ = cmap(gt_) * 255
            out_ = cmap(out_) * 255
            diff_ = plt.cm.Greys(np.abs(diff)) * 255
            cdiff_ = diff_cmap(cdiff) * 255
            Image.fromarray(gt_.astype('uint8')).save(
                os.path.join(args.result_dir, 'dense_gt_cmap_turbor', 'gt_dense_cmap_turbor_' + filename)[:-4] + '.png')
            Image.fromarray(out_.astype('uint8')).save(
                os.path.join(args.result_dir, 'output_depth_cmap_turbor', 'cmap_turbor_' + filename)[:-4] + '.png')
            Image.fromarray(diff_.astype('uint8')).save(
                os.path.join(args.result_dir, 'diff_output_depth', 'diff_' + filename)[:-4] + '.png')
            Image.fromarray(cdiff_.astype('uint8')).save(
                os.path.join(args.result_dir, 'cdiff_output_depth', 'cdiff_' + filename)[:-4] + '.png')

    return


if __name__ == "__main__":
    main()
