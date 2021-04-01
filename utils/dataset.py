"""
2021/3/9
create noise dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class MNISTNoisy(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        # self.uniform_mix(self.noise_rate, mixing_ratio=0.5, num_classes=self.num_classes)
        # self.flip(self.noise_rate, corruption_prob=0.5, num_classes=self.num_classes)

    # def create_symmetric_label_noise(self):
    #     for i in range(len(self.targets)):
    #         if random.random() < self.noise_rate:
    #             random_target = random.randint(0, 10)
    #             self.targets[i] = random_target

    def uniform_mix(self, noise_rate, mixing_ratio, num_classes):
        ntm = mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
              (1 - mixing_ratio) * np.eye(num_classes)

        # indices = np.arange(len(self.data))
        # np.random.shuffle(indices)

        num_noise = int(len(self.data) * noise_rate)

        for i in range(num_noise):
            self.targets[i] = np.random.choice(num_classes, p=ntm[self.targets[i]])

    def flip(self, noise_rate, corruption_prob, num_classes):
        ntm = np.eye(num_classes) * (1 - corruption_prob)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            ntm[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob

        # indices = np.arange(len(self.data))
        # np.random.shuffle(indices)

        num_noise = int(len(self.data) * noise_rate)

        for i in range(num_noise):
            self.targets[i] = np.random.choice(num_classes, p=ntm[self.targets[i]])

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)


class MNISTNoisy2:
    def __init__(self, root, train, transform, download):
        self.mnist = datasets.MNIST(root, train, transform, download=download)

    def uniform_mix(self, noise_rate, mixing_ratio, num_classes):
        ntm = mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
              (1 - mixing_ratio) * np.eye(num_classes)

        # indices = np.arange(len(self.data))
        # np.random.shuffle(indices)

        num_noise = int(len(self.mnist.data) * noise_rate)

        for i in range(num_noise):
            self.mnist.targets[i] = np.random.choice(num_classes, p=ntm[self.mnist.targets[i]])

    def flip(self, noise_rate, corruption_prob, num_classes):
        ntm = np.eye(num_classes) * (1 - corruption_prob)
        row_indices = np.arange(num_classes)
        for i in range(num_classes):
            ntm[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob

        # indices = np.arange(len(self.data))
        # np.random.shuffle(indices)

        num_noise = int(len(self.mnist.data) * noise_rate)

        for i in range(num_noise):
            self.mnist.targets[i] = np.random.choice(num_classes, p=ntm[self.mnist.targets[i]])

    def __len__(self):
        return self.mnist.__len__()

    def __getitem__(self, index):
        return self.mnist.__getitem__(index)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # mnist_noisy = MNISTNoisy(transform=transform, root='../dataset', train=True, download=True)
    # mnist_noisy.uniform_mix(noise_rate=1.0, mixing_ratio=0.5, num_classes=10)
    # mnist_noisy.flip(noise_rate=1.0, corruption_prob=0.5, num_classes=10)

    mnist_noisy = MNISTNoisy2(transform=transform, root='../dataset', train=True, download=True)
    mnist_noisy.uniform_mix(noise_rate=1.0, mixing_ratio=0.5, num_classes=10)

    noisy_loader = DataLoader(
        mnist_noisy, batch_size=64, shuffle=False)

    for batch_idx, (inputs, labels) in enumerate(noisy_loader):
        fig = plt.figure()
        inputs = inputs.detach().cpu()  # convert to cpu
        grid = utils.make_grid(inputs)
        print('Noisy labels:', labels)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig('./mnist_noisy.png')
        plt.close(fig)
        break
