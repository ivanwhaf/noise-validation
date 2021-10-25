"""
2021/10/24
train fixmatch
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, utils

from models import MNISTNet, CNN9Layer, CIFAR10Net
from randaugment import RandAugmentMC

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='fixmatch')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='../dataset')
parser.add_argument('-dataset', type=str, help='dataset type', default='cifar10')
parser.add_argument('-num_classes', type=int, help='number of classes', default=10)
parser.add_argument('-epochs', type=int, help='training epochs', default=600)
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-l2_reg', type=float, help='l2 regularization', default=1e-4)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
parser.add_argument('-num_labeled', type=int, help='number of labeled samples', default=4000)
parser.add_argument('-num_unlabeled', type=int, help='number of unlabeled samples', default=46000)
parser.add_argument('-T', default=0.5, type=float)
parser.add_argument('-alpha', type=float, help='beta distribution param alpha', default=0.75)
parser.add_argument('-lam_u', type=float, help='lambda u', default=75)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadAndCrop(object):
    """Crop randomly the image.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        return self.weak_transform(x), self.strong_transform(x)


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch=0):
        loss_x = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        loss_u = torch.mean((torch.softmax(outputs_u, dim=1) - targets_u) ** 2)

        return loss_x, loss_u, args.lam_u * linear_rampup(epoch)


def create_dataloader(dataset_type, root):
    if dataset_type == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # load dataset
        train_set = datasets.MNIST(root, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(root, train=False, transform=transform, download=False)
        val_set = test_set

        indices = np.arange(len(train_set))
        np.random.shuffle(indices)
        labeled_set = Subset(train_set, indices=indices[:args.num_labeled])
        train_set = datasets.MNIST(root, train=True, transform=TransformFixMatch(mean, std), download=False)
        unlabeled_set = Subset(train_set, indices=indices[:args.num_unlabeled])

    elif dataset_type == 'cifar10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # load dataset
        train_set = datasets.CIFAR10(root, train=True, transform=transform, download=True)
        test_set = datasets.CIFAR10(root, train=False, transform=test_transform, download=False)
        val_set = test_set

        labeled_set = Subset(train_set, indices=np.random.permutation(len(train_set))[:args.num_labeled])
        train_set = datasets.CIFAR10(root, train=True, transform=TransformFixMatch(mean, std), download=False)
        unlabeled_set = Subset(train_set, indices=np.random.permutation(len(train_set))[:args.num_unlabeled])

    elif dataset_type == 'cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        # load dataset
        train_set = datasets.CIFAR100(root, train=True, transform=transform, download=True)
        test_set = datasets.CIFAR100(root, train=False, transform=test_transform, download=False)
        val_set = test_set

        indices = np.arange(len(train_set))
        np.random.shuffle(indices)
        labeled_set = Subset(train_set, indices=np.random.permutation(len(train_set))[:args.num_labeled])
        train_set = datasets.MNIST(root, train=True, transform=TransformFixMatch(mean, std), download=False)
        unlabeled_set = Subset(train_set, indices=np.random.permutation(len(train_set))[:args.num_unlabeled])

    # generate DataLoader
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print('Labeled data:', len(labeled_set), 'Unlabeled data:', len(unlabeled_set))
    return labeled_loader, unlabeled_loader, val_loader, test_loader


def train(model, labeled_loader, unlabeled_loader, optimizer, epoch, device, train_loss_lst,
          train_acc_lst):
    model.train()
    correct = 0
    train_loss = 0

    unlabeled_iter = iter(unlabeled_loader)
    for batch_idx, (inputs_x, labels_x) in enumerate(labeled_loader):
        try:
            (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
        except Exception as e:
            unlabeled_iter = iter(unlabeled_loader)
            (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

        loss_s = F.cross_entropy(inputs_x, labels_x)

        inputs_u_w, inputs_u_s = inputs_u_w.to(device), inputs_u_s.to(device)

        # Transform label to one-hot
        # targets_x = torch.zeros(inputs_x.size(0), args.num_classes).scatter_(1, labels_x.view(-1, 1).long(), 1)
        targets_x = F.one_hot(labels_x, args.num_classes).float().to(device)
        inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

        # calculate training accuracy
        with torch.no_grad():
            outputs = model(inputs_x)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels_x.view_as(pred)).sum().item()

        # compute guessed labels of unlabeled samples
        # with torch.no_grad():
        #     outputs_u1 = model(inputs_u1)
        #     outputs_u2 = model(inputs_u2)
        #     p = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        #     pt = p ** (1 / args.T)
        #     targets_u = pt / pt.sum(dim=1, keepdim=True)
        #     targets_u = targets_u.detach()

        # # MixUp(X,W) + MixUp(U,W)
        # inputs_w = torch.cat([inputs_x, inputs_u1, inputs_u2], dim=0)  # W
        # targets_w = torch.cat([targets_x, targets_u, targets_u], dim=0)
        # inputs_u = torch.cat([inputs_u1, inputs_u2], dim=0)  # U
        # targets_u = torch.cat([targets_u, targets_u], dim=0)
        #
        # # ----------------------------MixUp(X,W)----------------------------------
        # indices = torch.randperm(len(inputs_w))[:len(inputs_x)]
        # lam = np.random.beta(args.alpha, args.alpha)
        # inputs_x = lam * inputs_x + (1 - lam) * inputs_w[indices]
        # targets_x = lam * targets_x + (1 - lam) * targets_w[indices]
        # outputs_x = model(inputs_x)
        #
        # # ----------------------------MixUp(U,W)---------------------------------
        # indices = torch.randperm(len(inputs_w))[:len(inputs_u)]
        # inputs_u = lam * inputs_u + (1 - lam) * inputs_w[indices]
        # targets_u = lam * targets_u + (1 - lam) * targets_w[indices]
        # outputs_u = model(inputs_u)
        #
        # # MixMatch loss
        # criterion = SemiLoss()
        # loss_x, loss_u, lam_u = criterion(outputs_x, targets_x, outputs_u, targets_u, epoch)
        # loss = loss_x + lam_u * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs_x.detach().cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            plt.close(fig)

        # print train loss and accuracy
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs_x), len(labeled_loader.dataset),
                          100. * batch_idx / len(labeled_loader), loss.item()))

    # record loss and accuracy
    train_loss /= len(labeled_loader)
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(labeled_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()
    val_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            val_loss += criterion(output, target).item()
            # val_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print val loss and accuracy
    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

    # record loss and accuracy
    val_loss_lst.append(val_loss)
    val_acc_lst.append(correct / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print test loss and accuracy
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.project_name + ' ' + now)
    os.makedirs(output_path)

    labeled_loader, unlabeled_loader, val_loader, test_loader = create_dataloader(args.dataset, args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'mnist':
        model = MNISTNet().to(device)
    elif args.dataset == 'cifar10':
        model = CIFAR10Net().to(device)
        # model = CNN9Layer(num_classes=10, input_shape=3).to(device)
    elif args.dataset == 'cifar100':
        model = CNN9Layer(num_classes=100, input_shape=3).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_reg)

    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    # main loop(train,val,test)
    for epoch in range(args.epochs):
        train_loss_lst, train_acc_lst = train(model, labeled_loader, unlabeled_loader, optimizer, epoch, device,
                                              train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(model, val_loader, device, val_loss_lst, val_acc_lst)

        # modify learning rate
        # if epoch in [40, 80]:
        #     args.lr *= 0.1
        #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_red)

    test(model, test_loader, device)

    # plot loss and accuracy curve
    fig = plt.figure('Loss and acc', dpi=200)
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.plot(range(args.epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(args.epochs), val_acc_lst, 'b', label='val acc')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_acc.png'))
    plt.show()
    plt.close(fig)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, args.project_name + ".pth"))
