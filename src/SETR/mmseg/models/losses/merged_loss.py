import numpy as np
import torch
import torch.nn as nn

from ..builder import LOSSES


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()

    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x


def imgrad_loss(pred, gt, mask=None):
    N, C, _, _ = pred.size()
    grad_y, grad_x = imgrad(pred)
    grad_y_gt, grad_x_gt = imgrad(gt)
    grad_y_diff = torch.abs(grad_y - grad_y_gt)
    grad_x_diff = torch.abs(grad_x - grad_x_gt)
    if mask is not None:
        grad_y_diff[~mask] = 0.1 * grad_y_diff[~mask]
        grad_x_diff[~mask] = 0.1 * grad_x_diff[~mask]
    return torch.mean(grad_y_diff) + torch.mean(grad_x_diff)


def scale_invariant_loss(valid_out, valid_gt):
    logdiff = torch.log(valid_out) - torch.log(valid_gt)
    scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - 0.85 * (logdiff.mean() ** 2)) * 10.0
    return scale_inv_loss


def make_mask(depths, crop_mask, dataset):
    # masking valid area
    if dataset == 'KITTI':
        valid_mask = depths > 0.001
    else:
        valid_mask = depths > 0.001

    if dataset == "KITTI":
        if crop_mask.size(0) != valid_mask.size(0):
            crop_mask = crop_mask[0:valid_mask.size(0), :, :, :]
        final_mask = crop_mask | valid_mask
    else:
        final_mask = valid_mask

    return valid_mask, final_mask


@LOSSES.register_module()
class MergedLoss(nn.Module):
    """MergedLoss.

    Args:
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 dataset,
                 height,
                 max_depth,
                 loss_weight=1.0):
        super(MergedLoss, self).__init__()
        self.dataset = dataset
        self.height = height
        self.max_depth = max_depth
        self.loss_weight = loss_weight
        self.crop_mask = None

    def forward(self,
                outputs,
                dense_depths,
                weight=None,
                ignore_index=None):
        """Forward function."""

        if self.crop_mask is None and self.dataset == "KITTI":
            # create mask for gradient loss
            H = self.height
            y1, y2 = int(0.3324324 * H), int(0.99189189 * H)
            crop_mask = dense_depths != dense_depths
            crop_mask[:, :, y1:y2, :] = 1
            self.crop_mask = crop_mask

        outputs = torch.sigmoid(outputs) * self.max_depth

        valid_mask, final_mask = make_mask(dense_depths, self.crop_mask, self.dataset)
        valid_out = outputs[valid_mask]
        valid_gt = dense_depths[valid_mask]
        scale_inv_loss = scale_invariant_loss(valid_out, valid_gt)
        gradient_loss = imgrad_loss(outputs, dense_depths, final_mask)
        gradient_loss = 0.1 * gradient_loss

        loss = scale_inv_loss + gradient_loss
        return self.loss_weight * loss
