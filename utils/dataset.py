"""
2021/5/8
Noisy dataset: MNIST, CIFAR10
"""

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils


class MNISTNoisy(Dataset):
    def __init__(self, root, train, transform, download):
        self.mnist = datasets.MNIST(root, train, transform, download=download)
        self.num_noise = 0

    def uniform_mix(self, noise_rate, mixing_ratio, num_classes):
        # symmetric noise
        ntm = mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
              (1 - mixing_ratio) * np.eye(num_classes)

        num_noise = int(len(self.mnist) * noise_rate)
        self.num_noise = num_noise

        indices = np.arange(len(self.mnist))
        np.random.shuffle(indices)

        for i in indices[:num_noise]:
            self.mnist.targets[i] = np.random.choice(num_classes, p=ntm[self.mnist.targets[i]])

    def flip(self, noise_rate, corruption_prob, num_classes):
        # asymmetric noise
        ntm = np.eye(num_classes) * (1 - corruption_prob)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            ntm[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob

        num_noise = int(len(self.mnist) * noise_rate)
        self.num_noise = num_noise

        indices = np.arange(len(self.mnist))
        np.random.shuffle(indices)

        for i in indices[:num_noise]:
            self.mnist.targets[i] = np.random.choice(num_classes, p=ntm[self.mnist.targets[i]])

    def __len__(self):
        return self.mnist.__len__()

    def __getitem__(self, index):
        return self.mnist.__getitem__(index)


class CIFAR10Noisy(Dataset):
    def __init__(self, root, train, transform, download):
        self.cifar10 = datasets.CIFAR10(root, train, transform, download=download)
        self.num_noise = 0

    def uniform_mix(self, noise_rate, mixing_ratio, num_classes):
        # symmetric noise
        ntm = mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
              (1 - mixing_ratio) * np.eye(num_classes)

        num_noise = int(len(self.cifar10) * noise_rate)
        self.num_noise = num_noise

        indices = np.arange(len(self.cifar10))
        np.random.shuffle(indices)

        for i in indices[:num_noise]:
            self.cifar10.targets[i] = np.random.choice(num_classes, p=ntm[self.cifar10.targets[i]])

    def flip(self, noise_rate, corruption_prob, num_classes):
        # asymmetric noise
        ntm = np.eye(num_classes) * (1 - corruption_prob)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            ntm[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob

        num_noise = int(len(self.cifar10) * noise_rate)
        self.num_noise = num_noise

        indices = np.arange(len(self.cifar10))
        np.random.shuffle(indices)

        for i in indices[:num_noise]:
            self.cifar10.targets[i] = np.random.choice(num_classes, p=ntm[self.cifar10.targets[i]])

    def __len__(self):
        return self.cifar10.__len__()

    def __getitem__(self, index):
        return self.cifar10.__getitem__(index)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    mnist_noisy = MNISTNoisy(transform=transform, root='../dataset', train=True, download=True)
    mnist_noisy.uniform_mix(noise_rate=1.0, mixing_ratio=0.5, num_classes=10)
    # mnist_noisy.flip(noise_rate=1.0, corruption_prob=0.5, num_classes=10)

    noisy_loader = DataLoader(
        mnist_noisy, batch_size=64, shuffle=False)

    # save noisy figure
    inputs, labels = next(iter(noisy_loader))
    fig = plt.figure()
    inputs = inputs.detach().cpu()  # convert to cpu
    grid = utils.make_grid(inputs)
    print('Noisy labels:', labels)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig('./mnist_noisy.png')
    plt.close(fig)
