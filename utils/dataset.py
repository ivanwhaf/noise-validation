"""
2021/3/9
create noise dataset
"""
import random

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class MNISTNoisy(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_rate=0.0):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_targets = self.targets
        self.noise_rate = noise_rate
        self.create_symmetric_label_noise()

    def create_symmetric_label_noise(self):
        for i in range(len(self.noise_targets)):
            if random.random() < self.noise_rate:
                random_target = random.randint(0, 10)
                self.noise_targets[i] = random_target

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    mnist_noisy = MNISTNoisy(noise_rate=0.5, transform=transform, root='../dataset', train=True, download=True)

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
