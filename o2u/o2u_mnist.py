import os
import time
import logging
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
# from torchvision.models import resnet18, resnet34, resnet50
import matplotlib.pyplot as plt
from models import *

pretrain_batch_size = 128
cycle_train_batch_size = 16
retrain_batch_size = 128

pretrain_epochs = 60
# cycle_train_epochs = 200
retrain_epochs = 60
# train_epochs = 100

pretrain_lr = 0.001
retrain_lr = 0.005


def create_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(
        './dataset', train=True, transform=transform, download=True)
    test_set = datasets.MNIST(
        './dataset', train=False, transform=transform, download=False)

    # split trainset into train-val set
    train_set, val_set = torch.utils.data.random_split(train_set, [
        50000, 10000])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_set


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Set the module in training mode
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show batch0 dataset
        # if batch_idx == 0 and epoch == 0:
        #     fig = plt.figure()
        #     inputs = inputs.detach().cpu()  # convert to cpu
        #     grid = utils.make_grid(inputs)
        #     print(labels)
        #     plt.imshow(grid.numpy().transpose((1, 2, 0)))
        #     plt.show()

        # print loss and accuracy
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)  # must divide
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def validate(model, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()  # Set the module in evaluation mode
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

    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))
    logging.info('Val set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'
                 .format(val_loss, correct, len(val_loader.dataset),
                         100. * correct / len(val_loader.dataset)))

    val_loss_lst.append(val_loss)
    val_acc_lst.append(correct / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model, test_loader, device):
    model.eval()  # Set the module in evaluation mode
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

    # record loss and acc
    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    logging.info('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'
                 .format(test_loss, correct, len(test_loader.dataset),
                         100. * correct / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)


def cyclical_train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst, sample_loss):
    model.train()  # Set the module in training mode
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(outputs, labels)
        loss = loss.detach().cpu().numpy()
        sample_loss[batch_idx * cycle_train_batch_size:batch_idx * cycle_train_batch_size + inputs.size(0)] += loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # print loss and accuracy
        if (batch_idx + 1) % 250 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)  # must divide
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst, sample_loss


def plot_loss_acc(train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, epochs, fig_path):
    # plot loss and accuracy
    fig = plt.figure('Loss and acc')
    plt.plot(range(epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(epochs), val_loss_lst, 'k', label='val loss')
    plt.plot(range(epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(epochs), val_acc_lst, 'b', label='val acc')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(fig_path)
    plt.close()


def o2u():
    torch.manual_seed(0)
    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join('output', now)
    os.makedirs(output_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(output_path, 'run.log'),
                        filemode='a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================Step1: Pre-training=============================================
    logging.info('Step1: Pre-training')
    # prepare dataset model and optimizer
    train_loader, val_loader, test_loader, train_set = create_dataloader(batch_size=pretrain_batch_size)
    model = MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=pretrain_lr, momentum=0.9, weight_decay=5e-4)

    # train validate and test
    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []
    for epoch in range(pretrain_epochs):
        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer,
                                              epoch, device, train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(
            model, val_loader, device, val_loss_lst, val_acc_lst)
    test(model, test_loader, device)

    # plot loss and accuracy
    plot_loss_acc(train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, pretrain_epochs,
                  os.path.join(output_path, 'o2u_mnist_pretrain.png'))
    # =======================================================================================================

    # =====================================Step2: Cyclical Training==========================================
    logging.info('Step2: Cyclical Training')
    c = 10  # epochs per cycle
    cycle_rounds = 4
    r1, r2 = 0.01, 0.001
    t = 0  # total epochs idx
    train_loader, val_loader, test_loader, train_set = create_dataloader(batch_size=cycle_train_batch_size)
    sample_loss = np.zeros(len(train_set))

    # cycle train
    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []
    for cycle_round in range(cycle_rounds):
        for epoch in range(c):
            st = (1 + ((t - 1) % c)) / c
            rt = (1 - st) * r1 + st * r2
            optimizer = optim.SGD(model.parameters(), lr=rt, momentum=0.9, weight_decay=5e-4)
            train_loss_lst, train_acc_lst, sample_loss = cyclical_train(model, train_loader, optimizer,
                                                                        t, device, train_loss_lst, train_acc_lst,
                                                                        sample_loss)
            val_loss_lst, val_acc_lst = validate(
                model, val_loader, device, val_loss_lst, val_acc_lst)
            t += 1
        test(model, test_loader, device)

    # plot loss and accuracy
    plot_loss_acc(train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, t,
                  os.path.join(output_path, 'o2u_mnist_cyclicaltrain.png'))

    # rank loss of each sample
    ranks = [(idx, loss) for idx, loss in enumerate(sample_loss.tolist())]
    ranks.sort(key=lambda x: x[1], reverse=True)
    print(ranks[:50])
    # sub noisy set and clean set
    noisy_indices = [item[0] for item in ranks[:100]]  # subtract top 100 noisy data from dataset
    noisy_set = Subset(train_set, noisy_indices)  # noisy set
    clean_indices = [
        i for i in list(range(len(train_set))) if i not in noisy_indices]  # diff indices
    clean_set = Subset(train_set, clean_indices)  # clean set
    # ======================================================================================================

    # =====================================Step3: Training on clean data====================================
    logging.info('Step3: Training on clean data')
    # prepare dataset model and optimizer
    train_loader = DataLoader(
        clean_set, batch_size=retrain_batch_size, shuffle=True)
    model = MNISTNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=retrain_lr, momentum=0.9, weight_decay=5e-4)

    # save noisy data
    noisy_loader = DataLoader(
        noisy_set, batch_size=len(noisy_set), shuffle=False)
    for batch_idx, (inputs, labels) in enumerate(noisy_loader):
        fig = plt.figure()
        inputs = inputs[:104].detach().cpu()  # convert to cpu
        grid = utils.make_grid(inputs)
        print('Noisy labels:', labels)
        logging.info('Noisy labels:' + str(labels[:104].detach().cpu().numpy().tolist()))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig(os.path.join(output_path, 'mnist_noisy.png'))
        plt.close()

    # train validate and test
    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []
    for epoch in range(retrain_epochs):
        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer,
                                              epoch, device, train_loss_lst, train_acc_lst)
        val_loss_lst, val_acc_lst = validate(
            model, val_loader, device, val_loss_lst, val_acc_lst)
    test(model, test_loader, device)

    # plot loss and accuracy
    plot_loss_acc(train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, retrain_epochs,
                  os.path.join(output_path, 'o2u_mnist_retrain.png'))
    # save model
    torch.save(model, os.path.join(output_path, "o2u_mnist.pth"))
    # ======================================================================================================


if __name__ == "__main__":
    o2u()
