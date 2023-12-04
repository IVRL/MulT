from collections import OrderedDict
from functools import partial

import timm
import torch
from mmseg.datasets import DepthDataset
from mmseg.models.backbones.vit import VisionTransformer
from torch import nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

backbone = dict(
    model_name='vit_large_patch16_384',
    img_size=(464, 464),
    patch_size=16,
    in_chans=3,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    num_classes=1,
    drop_rate=0.1,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    pos_embed_interp=True,
    align_corners=False
)


class VITClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            init_weights: bool = True
    ) -> None:
        super(VITClassifier, self).__init__()
        self.features = VisionTransformer(**backbone)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(backbone['embed_dim'])
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(backbone['embed_dim'], backbone['embed_dim'])),
            ('act', nn.Tanh())
        ]))
        self.head = nn.Linear(backbone['embed_dim'], num_classes)
        if init_weights:
            self._initialize_weights()
        self.features.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)[-1]
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def train_loop(dataloader, model, loss_fn, optimizer):
    gt_model = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True).cuda()
    gt_model.eval()
    size = len(dataloader.dataset)
    for batch, dct in enumerate(dataloader):
        X = dct['img'].cuda()
        with torch.no_grad():
            y = gt_model(interpolate(X, size=384, mode='bilinear')).argmax(1)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 13 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, epoch, scheduler=None):
    gt_model = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True).cuda()
    gt_model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for dct in dataloader:
            X = dct['img'][0].cuda()
            with torch.no_grad():
                y = gt_model(interpolate(X, size=384, mode='bilinear')).argmax(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    if scheduler is not None:
        scheduler.step(test_loss)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    if (epoch + 1) % 10 == 0:
        print("Saving . . .")
        torch.save(model.state_dict(),
                   'finetuned_vit_epoch_%d_acc_%.4f_loss_%.4f.pth' % (epoch + 1, correct, test_loss))
    print()


def main():
    args = dict(
        dataset='NYU',
        max_depth=10.0,
        height=464,
        width=464,
        data_path='/sinergia/cappella/datasets/NYU_Depth_V2/official_splits',
        trainfile_nyu='/workspace/project/LapDepth-release/datasets'
                      '/nyudepthv2_labeled_train_files_with_gt_dense_contours.txt',
        testfile_nyu='/workspace/project/LapDepth-release/datasets'
                     '/nyudepthv2_test_files_with_gt_dense_contours.txt',
        use_dense_depth=True,
        use_sparse=False
    )

    learning_rate = 1e-5
    momentum = 0  # 0.9
    batch_size = 4
    epochs = 100

    training_data = DepthDataset(args=args, train=True, return_filename=False)
    test_data = DepthDataset(args=args, train=False, return_filename=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    model = VITClassifier().cuda()
    # model.load_state_dict(torch.load('.pth'))
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    scheduler = None  # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn, t, scheduler)
    print("Done!")


if __name__ == '__main__':
    main()
