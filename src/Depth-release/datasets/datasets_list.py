import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torch.nn.functional import interpolate
from torchvision import transforms
from transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensorNumpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


def ntrplt(mask, scale_or_size, times=1, isscale=True, mode='nearest'):
    if isscale:
        if isinstance(mask, torch.Tensor):
            tensor = torch.unsqueeze(mask, 0)
            for t in range(times):
                tensor = interpolate(tensor, scale_factor=scale_or_size, recompute_scale_factor=False, mode=mode)
            return tensor[0]
        else:
            tensor = torch.Tensor([np.moveaxis(mask, -1, 0)])
            for t in range(times):
                tensor = interpolate(tensor, scale_factor=scale_or_size, recompute_scale_factor=False, mode=mode)
            return np.moveaxis(np.array(tensor[0], dtype=mask.dtype), 0, -1)
    else:
        tensor = torch.Tensor([np.moveaxis(mask, -1, 0)])
        tensor = interpolate(tensor, size=scale_or_size, mode=mode)
        return np.moveaxis(np.array(tensor[0], dtype=mask.dtype), 0, -1)


def _is_pil_image(img):
    return isinstance(img, Image.Image)


class MyDataset(data.Dataset):
    def __init__(self, args, train=True, return_filename=False, return_top=False):
        self.use_dense_depth = args.use_dense_depth
        if train is True:
            if args.dataset == 'KITTI':
                self.datafile = args.trainfile_kitti
                self.angle_range = (-1, 1)
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.trainfile_nyu
                self.angle_range = (-2.5, 2.5)
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        else:
            if args.dataset == 'KITTI':
                self.datafile = args.testfile_kitti
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.testfile_nyu
                self.depth_scale = 1000.0
                args.height = 416
                args.width = 544
        self.train = train
        self.transform = Transformer(args)
        self.args = args
        self.return_filename = return_filename
        self.return_top = return_top
        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()
        self.fileset = sorted(self.fileset)

    def __getitem__(self, index):
        divided_file = self.fileset[index].split()
        if self.args.dataset == 'KITTI':
            date = divided_file[0].split('/')[0] + '/'

        # Opening image files.   rgb: input color image, gt: sparse depth map
        rgb_file = self.args.data_path + '/' + divided_file[0]
        rgb = Image.open(rgb_file)
        gt = False
        gt_dense = False
        c0 = False
        c1 = False
        c2 = False
        c3 = False
        if (self.train is False):
            divided_file_ = divided_file[0].split('/')
            if self.args.dataset == 'KITTI':
                filename = divided_file_[1] + '_' + divided_file_[4]
            else:
                filename = divided_file_[1] + '_' + divided_file_[2][-9:]

            if self.args.dataset == 'KITTI':
                # Considering missing gt in Eigen split
                if divided_file[1] != 'None':
                    gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                    gt = Image.open(gt_file)
                    if self.use_dense_depth is True:
                        gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
                        gt_dense = Image.open(gt_dense_file)
                else:
                    pass
            elif self.args.dataset == 'NYU':
                gt_file = self.args.data_path + '/' + divided_file[1]
                gt = Image.open(gt_file)
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
                    gt_dense = Image.open(gt_dense_file)
                if self.args.use_contours:
                    df = divided_file[3].split('_')
                    jf = '_'.join(df[:-1])
                    c0_file = self.args.data_path + '/' + jf + '0_' + df[-1]
                    c1_file = self.args.data_path + '/' + jf + '1_' + df[-1]
                    c2_file = self.args.data_path + '/' + jf + '2_' + df[-1]
                    c3_file = self.args.data_path + '/' + jf + '3_' + df[-1]
                    c0 = Image.open(c0_file)
                    c1 = Image.open(c1_file)
                    c2 = Image.open(c2_file)
                    c3 = Image.open(c3_file)
        else:
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            if self.args.dataset == 'KITTI':
                gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
            elif self.args.dataset == 'NYU':
                gt_file = self.args.data_path + '/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
                if self.args.use_contours:
                    df = divided_file[3].split('_')
                    jf = '_'.join(df[:-1])
                    c0_file = self.args.data_path + '/' + jf + '0_' + df[-1]
                    c1_file = self.args.data_path + '/' + jf + '1_' + df[-1]
                    c2_file = self.args.data_path + '/' + jf + '2_' + df[-1]
                    c3_file = self.args.data_path + '/' + jf + '3_' + df[-1]

            gt = Image.open(gt_file)
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)
            if self.use_dense_depth is True:
                gt_dense = Image.open(gt_dense_file)
                gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)
            if self.args.use_contours:
                c0 = Image.open(c0_file)
                c1 = Image.open(c1_file)
                c2 = Image.open(c2_file)
                c3 = Image.open(c3_file)
                c0 = c0.rotate(angle, resample=Image.NEAREST)
                c1 = c1.rotate(angle, resample=Image.NEAREST)
                c2 = c2.rotate(angle, resample=Image.NEAREST)
                c3 = c3.rotate(angle, resample=Image.NEAREST)

        # cropping in size that can be divided by 16
        if self.args.dataset == 'KITTI':
            h = rgb.height
            w = rgb.width
            bound_left = (w - 1216) // 2
            bound_right = bound_left + 1216
            bound_top = h - 352
            bound_bottom = bound_top + 352
        elif self.args.dataset == 'NYU':
            if self.train is True:
                bound_left = 43
                bound_right = 608
                bound_top = 45
                bound_bottom = 472
            else:
                bound_left = 0
                bound_right = 640
                bound_top = 0
                bound_bottom = 480
        # crop and normalize 0 to 1 ==>  rgb range:(0,1),  depth range: (0, max_depth)

        if (self.args.dataset == 'NYU' and (self.train is False) and (self.return_filename is False)):
            rgb = rgb.crop((40, 42, 616, 474))
            if self.args.use_contours:  # TODO: crop LR contours with rational black boundary?
                c0 = c0.crop((40, 42, 616, 474))
                c1 = c1.crop((20, 21, 308, 237))
                c2 = c2.crop((10, 10, 154, 118))
                c3 = c3.crop((5, 5, 77, 59))
                c0 = np.expand_dims(c0, axis=2)
                c1 = np.expand_dims(c1, axis=2)
                c2 = np.expand_dims(c2, axis=2)
                c3 = np.expand_dims(c3, axis=2)
        else:
            rgb = rgb.crop((bound_left, bound_top, bound_right, bound_bottom))
            if self.args.use_contours:
                c0 = c0.crop((bound_left, bound_top, bound_right, bound_bottom))
                c1 = c1.crop((int(bound_left / 2), int(bound_top / 2),
                              int(bound_left / 2) + (bound_right - bound_left) // 2,
                              int(bound_top / 2) + (bound_bottom - bound_top) // 2))
                c2 = c2.crop((int(bound_left / 4), int(bound_top / 4),
                              int(bound_left / 4) + (bound_right - bound_left) // 4,
                              int(bound_top / 4) + (bound_bottom - bound_top) // 4))
                c3 = c3.crop((int(bound_left / 8), int(bound_top / 8),
                              int(bound_left / 8) + (bound_right - bound_left) // 8,
                              int(bound_top / 8) + (bound_bottom - bound_top) // 8))
                c0 = np.expand_dims(c0, axis=2)
                c1 = np.expand_dims(c1, axis=2)
                c2 = np.expand_dims(c2, axis=2)
                c3 = np.expand_dims(c3, axis=2)

        rgb = np.asarray(rgb, dtype=np.float32) / 255.0

        if _is_pil_image(gt):
            gt = gt.crop((bound_left, bound_top, bound_right, bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32)) / self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.args.max_depth)
        if self.use_dense_depth is True:
            if _is_pil_image(gt_dense):
                gt_dense = gt_dense.crop((bound_left, bound_top, bound_right, bound_bottom))
                gt_dense = (np.asarray(gt_dense, dtype=np.float32)) / self.depth_scale
                gt_dense = np.expand_dims(gt_dense, axis=2)
                gt_dense = np.clip(gt_dense, 0, self.args.max_depth)
                gt_dense = gt_dense * (gt.max() / gt_dense.max())

        if self.args.use_contours:
            c0, c1, c2, c3 = c0, ntrplt(c1, 2), ntrplt(c2, 2, 2), ntrplt(c3, 2, 3)
            rgb = ntrplt(rgb, c3.shape[:-1], isscale=False, mode='bilinear')
            if self.train:
                gt = ntrplt(gt, c3.shape[:-1], isscale=False)
                if self.use_dense_depth is True:
                    gt_dense = ntrplt(gt_dense, c3.shape[:-1], isscale=False)
            c0 = ntrplt(c0, c3.shape[:-1], isscale=False)
            c1 = ntrplt(c1, c3.shape[:-1], isscale=False)
            c2 = ntrplt(c2, c3.shape[:-1], isscale=False)
            rgb, gt, gt_dense, c0, c1, c2, c3 = self.transform([rgb] + [gt] + [gt_dense] + [c0] + [c1] + [c2] + [c3],
                                                               self.train)
            c0, c1, c2, c3 = c0, ntrplt(c1, 0.5), ntrplt(c2, 0.5, 2), ntrplt(c3, 0.5, 3)
            # c0, c1, c2, c3 = c0.expand((3, -1, -1)), c1.expand((3, -1, -1)), c2.expand((3, -1, -1)), c3.expand(
            #     (3, -1, -1))  # TODO: change model channels instead?
            c0, c1, c2, c3 = c0.float(), c1.float(), c2.float(), c3.float()
        else:
            rgb, gt, gt_dense, c0, c1, c2, c3 = self.transform([rgb] + [gt] + [gt_dense] + [c0] + [c1] + [c2] + [c3],
                                                               self.train)

        if self.return_filename is True or self.return_top:
            return rgb, gt, gt_dense, c0, c1, c2, c3, filename
        else:
            return rgb, gt, gt_dense, c0, c1, c2, c3

    def __len__(self):
        return len(self.fileset)


class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25)), None, None, None,
                 None, None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None, None,
                 None, None, None]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None, None,
                 None, None, None]
            ])

    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
