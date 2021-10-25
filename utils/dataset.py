"""
2021/5/8
Noisy dataset: MNIST, CIFAR10
"""

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils


# np.random.seed(0)

class MNISTNoisy(Dataset):
    def __init__(self, root, train, transform, download, noise_type='symmetric', noise_rate=0.2):
        self.mnist = datasets.MNIST(root, train, transform, download=download)
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.clean_sample_idx = []
        self.noisy_sample_idx = []

        # add label noise
        if self.noise_type == 'symmetric':
            self.uniform(noise_rate, 10)
        elif self.noise_type == 'asymmetric':
            self.flip(noise_rate, 10)

    def uniform(self, noise_rate: float, num_classes: int):
        """Add symmetric noise"""

        # noise transition matrix
        ntm = noise_rate * np.full((num_classes, num_classes), 1 / (num_classes - 1))
        np.fill_diagonal(ntm, 1 - noise_rate)

        sample_indices = np.arange(len(self.mnist))
        # np.random.shuffle(indices)

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.mnist.targets[i]])  # new label
            if label != self.mnist.targets[i]:
                self.noisy_sample_idx.append(i)
            self.mnist.targets[i] = label

        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Symmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def flip(self, noise_rate: float, num_classes: int):
        """Add asymmetric noise"""

        # noise transition matrix
        ntm = np.eye(num_classes) * (1 - noise_rate)

        d = {7: 1, 2: 7, 5: 6, 6: 5, 3: 8}  # 7->1, 2->7, 5->6, 6->5, 3->8
        for raw_class, new_class in d.items():
            ntm[raw_class][new_class] = noise_rate

        for i in [0, 1, 4, 8, 9]:
            ntm[i][i] = 1

        sample_indices = np.arange(len(self.mnist))

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.mnist.targets[i]])
            if label != self.mnist.targets[i]:
                self.noisy_sample_idx.append(i)
            self.mnist.targets[i] = label

        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Asymmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def __len__(self):
        return self.mnist.__len__()

    def __getitem__(self, index):
        return self.mnist.__getitem__(index)


class CIFAR10Noisy(Dataset):
    def __init__(self, root, train, transform, download, noise_type='symmetric', noise_rate=0.2):
        self.cifar10 = datasets.CIFAR10(root, train, transform, download=download)
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.noisy_sample_idx = []
        self.clean_sample_idx = []
        # add label noise
        if self.noise_type == 'symmetric':
            self.uniform(noise_rate, 10)
        elif self.noise_type == 'asymmetric':
            self.flip(noise_rate, 10)

    def uniform(self, noise_rate: float, num_classes: int):
        """Add symmetric noise"""

        # noise transition matrix
        ntm = noise_rate * np.full((num_classes, num_classes), 1 / (num_classes - 1))
        np.fill_diagonal(ntm, 1 - noise_rate)

        sample_indices = np.arange(len(self.cifar10))

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.cifar10.targets[i]])
            if label != self.cifar10.targets[i]:
                self.noisy_sample_idx.append(i)
            self.cifar10.targets[i] = label

        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Symmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def flip(self, noise_rate: float, num_classes: int):
        """Add asymmetric noise"""

        # noise transition matrix
        ntm = np.eye(num_classes) * (1 - noise_rate)

        d = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}  # truck->automobile, bird->airplane, cat->dog, dog->cat, deer->horse
        for raw_class, new_class in d.items():
            ntm[raw_class][new_class] = noise_rate

        for i in [0, 1, 6, 7, 8]:
            ntm[i][i] = 1

        sample_indices = np.arange(len(self.cifar10))

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.cifar10.targets[i]])
            if label != self.cifar10.targets[i]:
                self.noisy_sample_idx.append(i)
            self.cifar10.targets[i] = label

        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Asymmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def __len__(self):
        return self.cifar10.__len__()

    def __getitem__(self, index):
        return self.cifar10.__getitem__(index)


class CIFAR100Noisy(Dataset):
    def __init__(self, root, train, transform, download, noise_type='symmetric', noise_rate=0.2):
        self.cifar100 = datasets.CIFAR100(root, train, transform, download=download)
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.noisy_sample_idx = []
        self.clean_sample_idx = []
        # add label noise
        if self.noise_type == 'symmetric':
            self.uniform(noise_rate, 100)
        elif self.noise_type == 'asymmetric':
            self.flip(noise_rate, 100)

    def uniform(self, noise_rate: float, num_classes: int):
        """Add symmetric noise"""

        # noise transition matrix
        ntm = noise_rate * np.full((num_classes, num_classes), 1 / (num_classes - 1))
        np.fill_diagonal(ntm, 1 - noise_rate)

        sample_indices = np.arange(len(self.cifar100))

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.cifar100.targets[i]])
            if label != self.cifar100.targets[i]:
                self.noisy_sample_idx.append(i)
            self.cifar100.targets[i] = label

        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Symmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def flip(self, noise_rate: float, num_classes: int):
        """Add asymmetric noise"""

        ntm = np.eye(num_classes) * (1 - noise_rate)

        # row_indices = np.arange(num_classes)
        # for i in range(num_classes):
        #     ntm[i][np.random.choice(row_indices[row_indices != i])] = noise_rate

        for i in range(num_classes):
            ntm[i][i + 1 if i + 1 < num_classes else 0] = noise_rate

        sample_indices = np.arange(len(self.cifar100))

        # generate noisy label by noise transition matrix
        for i in sample_indices:
            label = np.random.choice(num_classes, p=ntm[self.cifar100.targets[i]])
            if label != self.cifar100.targets[i]:
                self.noisy_sample_idx.append(i)
            self.cifar100.targets[i] = label

        self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)

        print('Noise type: Asymmetric')
        print('Noise rate:', noise_rate)
        print('Noise transition matrix:\n', ntm)
        print('Clean samples:', len(self.clean_sample_idx), 'Noisy samples:', len(self.noisy_sample_idx))

    def __len__(self):
        return self.cifar100.__len__()

    def __getitem__(self, index):
        return self.cifar100.__getitem__(index)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    mnist_noisy = MNISTNoisy(transform=transform, root='../dataset', train=True, download=True, noise_rate=0.5)
    noisy_loader = DataLoader(mnist_noisy, batch_size=64, shuffle=False)

    # save noisy figure
    inputs, labels = next(iter(noisy_loader))
    fig = plt.figure()
    inputs = inputs.detach().cpu()  # convert to cpu
    grid = utils.make_grid(inputs)
    print('Noisy labels:', labels)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig('./mnist_noisy.png')
    plt.close(fig)
