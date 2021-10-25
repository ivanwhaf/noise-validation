"""
2021/5/8
train mnist baseline noisy
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from loss import NCEandRCE
from models import MNISTNet, CNN9Layer, CIFAR10Net
from utils.dataset import MNISTNoisy, CIFAR10Noisy, CIFAR100Noisy

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='mnist_noisy_apl_s0.2')
parser.add_argument('-noise_type', type=str, help='noise type', default='symmetric')
parser.add_argument('-noise_rate', type=float, help='noise rate', default=0.2)
parser.add_argument('-dataset', type=str, help='dataset type', default='mnist')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='../dataset')
parser.add_argument('-num_classes', type=int, help='number of classes', default=10)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
parser.add_argument('-batch_size', type=int, help='batch size', default=128)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-l2_reg', type=float, help='l2 regularization', default=1e-4)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
parser.add_argument('-alpha', type=float, help='alpha', default=1.0)
parser.add_argument('-beta', type=float, help='beta', default=0.1)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def create_dataloader(dataset_type, root, noise_type, noise_rate):
    if dataset_type == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        # load noisy dataset
        train_set = MNISTNoisy(transform=transform, root=root, train=True, download=True,
                               noise_type=noise_type, noise_rate=noise_rate)

        test_set = datasets.MNIST(root, train=False, transform=test_transform, download=False)
        val_set = test_set

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

        # load noisy dataset
        train_set = CIFAR10Noisy(transform=transform, root=root, train=True, download=True,
                                 noise_type=noise_type, noise_rate=noise_rate)

        test_set = datasets.CIFAR10(root, train=False, transform=test_transform, download=False)
        val_set = test_set

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

        # load noisy dataset
        train_set = CIFAR100Noisy(transform=transform, root=root, train=True, download=True,
                                  noise_type=noise_type, noise_rate=noise_rate)

        test_set = datasets.CIFAR100(root, train=False, transform=test_transform, download=False)
        val_set = test_set

    # generate DataLoader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_set.clean_sample_idx, train_set.noisy_sample_idx, len(train_set)


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst, sample_loss):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # record loss
        criterion = NCEandRCE(alpha=args.alpha, beta=args.beta, num_classes=outputs.shape[1], reduction='none')
        loss = criterion(outputs, labels)
        n_loss = loss.detach().cpu().numpy()
        sample_loss[batch_idx * args.batch_size:batch_idx * args.batch_size + inputs.size(0)] += n_loss
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.detach().cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            plt.close(fig)

        # print train loss and accuracy
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    # record loss and accuracy
    train_loss /= len(train_loader)  # must divide iter num
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst, sample_loss


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

    # get data loader
    train_loader, val_loader, test_loader, clean_sample_idx, noisy_sample_idx, dataset_len = create_dataloader(
        args.dataset,
        args.dataset_path,
        args.noise_type,
        args.noise_rate)

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

    clean_mean_loss_lst, noisy_mean_loss_lst = [], []
    clean_min_loss_lst, noisy_min_loss_lst = [], []
    clean_max_loss_lst, noisy_max_loss_lst = [], []

    sample_loss = np.zeros(dataset_len)

    # main loop(train,val,test)
    for epoch in range(args.epochs):
        train_loss_lst, train_acc_lst, sample_loss = train(model, train_loader, optimizer,
                                                           epoch, device, train_loss_lst, train_acc_lst, sample_loss)

        # calculate mean, min, max loss of clean and noisy samples
        clean_sample_loss = sample_loss[clean_sample_idx]
        noisy_sample_loss = sample_loss[noisy_sample_idx]
        clean_mean_loss = np.mean(clean_sample_loss)
        noisy_mean_loss = np.mean(noisy_sample_loss)
        clean_min_loss = np.min(clean_sample_loss)
        noisy_min_loss = np.min(noisy_sample_loss)
        clean_max_loss = np.max(clean_sample_loss)
        noisy_max_loss = np.max(noisy_sample_loss)
        clean_mean_loss_lst.append(clean_mean_loss)
        noisy_mean_loss_lst.append(noisy_mean_loss)
        clean_min_loss_lst.append(clean_min_loss)
        noisy_min_loss_lst.append(noisy_min_loss)
        clean_max_loss_lst.append(clean_max_loss)
        noisy_max_loss_lst.append(noisy_max_loss)
        print('Clean sample mean loss:', clean_mean_loss, 'Noisy sample mean loss:', noisy_mean_loss)
        sample_loss = np.zeros(dataset_len)

        val_loss_lst, val_acc_lst = validate(
            model, val_loader, device, val_loss_lst, val_acc_lst)

        # modify learning rate
        if epoch in [40, 80]:
            args.lr *= 0.1
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_reg)

    test(model, test_loader, device)

    # plot train/val/test loss and accuracy curve
    fig = plt.figure('Loss and acc', dpi=200)
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.plot(range(args.epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(args.epochs), val_acc_lst, 'b', label='val acc')

    # plot clean and noisy sample loss
    plt.plot(range(args.epochs), clean_mean_loss_lst, 'springgreen', label='clean loss')
    plt.plot(range(args.epochs), noisy_mean_loss_lst, 'deepskyblue', label='noisy loss')
    plt.fill_between(range(args.epochs), clean_min_loss_lst, clean_max_loss_lst, color='aquamarine', alpha=0.3)
    plt.fill_between(range(args.epochs), noisy_min_loss_lst, noisy_max_loss_lst, color='paleturquoise', alpha=0.3)

    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_acc.png'))
    plt.show()
    plt.close(fig)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, args.project_name + ".pth"))
