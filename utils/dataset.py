"""
2021/3/9
create noise dataset
"""
import random

from torchvision import datasets


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
        return super().__getitem__()


if __name__ == '__main__':
    mnist_noisy = MNISTNoisy(noise_rate=0.5, root='./dataset', train=True, download=True)
