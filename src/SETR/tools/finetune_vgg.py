import timm
import torch
from mmseg.datasets import DepthDataset
from mmseg.models.backbones.vgg import vgg19_bn
from torch import nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader


class VGGClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            init_weights: bool = True
    ) -> None:
        super(VGGClassifier, self).__init__()
        self.features = vgg19_bn()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.features.init_weights('vgg19_bn')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)[-1]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
    gt_model = timm.create_model('vit_large_patch16_384', pretrained=True).cuda()
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
    gt_model = timm.create_model('vit_large_patch16_384', pretrained=True).cuda()
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
                   'finetuned_vgg19bn_epoch_%d_acc_%.4f_loss_%.4f.pth' % (epoch + 1, correct, test_loss))
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

    learning_rate = 1e-3
    momentum = 0  # 0.9
    batch_size = 32
    epochs = 100

    training_data = DepthDataset(args=args, train=True, return_filename=False)
    test_data = DepthDataset(args=args, train=False, return_filename=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)

    model = VGGClassifier().cuda()
    # model.load_state_dict(torch.load('.pth'))
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = None  # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn, t, scheduler)
    print("Done!")


if __name__ == '__main__':
    main()
