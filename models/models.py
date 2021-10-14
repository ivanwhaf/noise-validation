"""
2021/3/6 wyf
Some pytorch nn models
"""

import torch.nn.functional as F
from torch import nn


class LeNet(nn.Module):
    """
    LeNet5 model
    input: 32*32
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 32*32
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """
    AlexNet,5conv,3fc
    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, num_classes)
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


class VGGNet(nn.Module):
    """
    VGG16 net,input shape 227*227
    """

    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


# -----------------Customized Neural Network--------------------#
class CNN9Layer(nn.Module):
    """
    9 Layer CNN
    """

    def __init__(self, num_classes, input_shape):
        super(CNN9Layer, self).__init__()
        self.conv1a = nn.Conv2d(input_shape, 128, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv1c = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2c = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3a = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        self.conv3b = nn.Conv2d(512, 256, kernel_size=3, padding=0)
        self.conv3c = nn.Conv2d(256, 128, kernel_size=3, padding=0)

        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1a(x)
        x = self.relu(x)
        x = self.conv1b(x)
        x = self.relu(x)
        x = self.conv1c(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2a(x)
        x = self.relu(x)
        x = self.conv2b(x)
        x = self.relu(x)
        x = self.conv2c(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.relu(x)
        x = self.conv3c(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[2])

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MNISTNet(nn.Module):
    """
    customized neural network
    """

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # block1
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.relu(out)

        # block2
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.relu(out)

        # block3
        out = self.conv3(out)
        out = self.relu(out)

        # block4
        out = out.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)

        return out


class MNIST2ConvNet(nn.Module):
    """
    customized neural network
    """

    def __init__(self):
        super(MNIST2ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # out = self.softmax(out)

        return x


class CIFAR10Net(nn.Module):
    """
    customized neural network
    """

    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(480, 84)
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # block1
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.relu(out)

        # block2
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.relu(out)

        # block3
        out = self.conv3(out)
        out = self.relu(out)

        # block4
        out = out.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)

        return out


class CIFAR102ConvNet(nn.Module):
    """
    customized neural network
    """

    def __init__(self):
        super(CIFAR102ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # out = self.softmax(out)

        return x


class CIFAR10LeNet(nn.Module):
    """
    LeNet5 model
    input: 32*32
    """

    def __init__(self):
        super(CIFAR10LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)  # 32*32
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)  # 64 576
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.softmax(x)

        return x
