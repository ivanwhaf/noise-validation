"""
2021/3/9
train mnist decoupling
"""
import argparse
import os
import time

# from torchvision.models import resnet18, resnet34, resnet50
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from models.models import *

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, help='project name', default='mnist_decoupling')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='../dataset')
parser.add_argument('-batch_size', type=int, help='batch size', default=64)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def create_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # load dataset
    train_set = datasets.MNIST(
        args.dataset_path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(
        args.dataset_path, train=False, transform=transform, download=False)

    # split train set into train-val set
    train_set, val_set = torch.utils.data.random_split(train_set, [
        50000, 10000])

    # generate DataLoader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model1, model2, train_loader, optimizer1, optimizer2, epoch, device, train_loss_lst, train_acc_lst):
    model1.train()  # Set the module in training mode
    model2.train()
    correct1 = 0
    correct2 = 0
    train_loss1 = 0
    train_loss2 = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        pred1 = outputs1.max(1, keepdim=True)[1]
        pred2 = outputs2.max(1, keepdim=True)[1]
        correct1 += pred1.eq(labels.view_as(pred1)).sum().item()
        correct2 += pred2.eq(labels.view_as(pred2)).sum().item()

        index = []
        for i in range(inputs.size(0)):
            if pred1[i] != pred2[i]:
                index.append(i)

        criterion = nn.CrossEntropyLoss(reduction='none')
        loss1 = criterion(outputs1, labels)
        loss2 = criterion(outputs2, labels)

        loss1 = torch.index_select(loss1, dim=-1, index=torch.tensor(index, dtype=torch.int64).to(device)).mean()
        loss2 = torch.index_select(loss2, dim=-1, index=torch.tensor(index, dtype=torch.int64).to(device)).mean()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()

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
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss1: {:.6f},  Loss2: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss1.item(), loss2.item()))

    # record loss and accuracy
    train_loss1 /= len(train_loader)  # must divide iter num
    train_loss_lst.append(train_loss1)
    train_acc_lst.append(correct1 / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model1, model2, val_loader, device, val_loss_lst, val_acc_lst):
    model1.eval()  # Set the module in evaluation mode
    model2.eval()
    val_loss1 = 0
    val_loss2 = 0
    correct1 = 0
    correct2 = 0

    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output1 = model1(data)
            output2 = model2(data)

            criterion = nn.CrossEntropyLoss()
            val_loss1 += criterion(output1, target).item()
            val_loss2 += criterion(output2, target).item()

            # find index of max prob
            pred1 = output1.max(1, keepdim=True)[1]
            pred2 = output2.max(1, keepdim=True)[1]
            correct1 += pred1.eq(target.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target.view_as(pred2)).sum().item()

    # print val loss and accuracy
    val_loss1 /= len(val_loader)
    print('\nVal set: Average loss1: {:.6f}, Accuracy1: {}/{} ({:.2f}%)'
          .format(val_loss1, correct1, len(val_loader.dataset),
                  100. * correct1 / len(val_loader.dataset)))
    val_loss2 /= len(val_loader)
    print('Val set: Average loss2: {:.6f}, Accuracy2: {}/{} ({:.2f}%)\n'
          .format(val_loss2, correct2, len(val_loader.dataset),
                  100. * correct2 / len(val_loader.dataset)))

    # record loss and accuracy
    val_loss_lst.append(val_loss1)
    val_acc_lst.append(correct1 / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model1, model2, test_loader, device):
    model1.eval()  # Set the module in evaluation mode
    model2.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output1 = model1(data)
            output2 = model2(data)

            criterion = nn.CrossEntropyLoss()
            test_loss1 += criterion(output1, target).item()
            test_loss2 += criterion(output2, target).item()

            # find index of max prob
            pred1 = output1.max(1, keepdim=True)[1]
            pred2 = output2.max(1, keepdim=True)[1]
            correct1 += pred1.eq(target.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target.view_as(pred2)).sum().item()

    # print test loss and accuracy
    test_loss1 /= len(test_loader.dataset)
    print('Test set: Average loss1: {:.6f}, Accuracy1: {}/{} ({:.2f}%)'
          .format(test_loss1, correct1, len(test_loader.dataset),
                  100. * correct1 / len(test_loader.dataset)))
    test_loss2 /= len(test_loader.dataset)
    print('Test set: Average loss2: {:.6f}, Accuracy2: {}/{} ({:.2f}%)\n'
          .format(test_loss2, correct2, len(test_loader.dataset),
                  100. * correct2 / len(test_loader.dataset)))


if __name__ == "__main__":
    torch.manual_seed(0)
    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.name + now)
    os.makedirs(output_path)

    train_loader, val_loader, test_loader = create_dataloader()  # get data loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = MNISTNet().to(device)
    model2 = MNISTNet().to(device)
    # model = resnet18(num_classes=32).to(device)

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9)

    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    # main loop(train,val,test)
    for epoch in range(args.epochs):
        train_loss_lst, train_acc_lst = train(model1, model2, train_loader, optimizer1, optimizer2,
                                              epoch, device, train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(
            model1, model2, val_loader, device, val_loss_lst, val_acc_lst)

        # modify learning rate
        if epoch in [40, 60, 80]:
            args.lr *= 0.1
            optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9)
            optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9)

    test(model1, model2, test_loader, device)

    # plot loss and accuracy curve
    fig = plt.figure('Loss and acc')
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
    torch.save(model1.state_dict(), os.path.join(output_path, args.name + "_model1.pth"))
    torch.save(model2.state_dict(), os.path.join(output_path, args.name + "_model2.pth"))
