"""
train mnist by inference loss
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# from torchvision.models import resnet18, resnet34, resnet50
from models.models import *

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, help='project name', default='mnist_loss')
parser.add_argument('-dataset_path', type=str, help='relative path of dataset', default='../dataset')
parser.add_argument('-inference_batch_size', type=int, help='inference batch size', default=64)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
args = parser.parse_args()


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(
        args.dataset_path, train=True, transform=transform, download=False)
    test_set = datasets.MNIST(
        args.dataset_path, train=False, transform=transform, download=False)
    return train_set, test_set


def train_loss():
    torch.manual_seed(0)
    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.name + now)
    os.makedirs(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)
    model = model.load_state_dict(torch.load('model.pth'))

    # =====================================Inference ============================================
    ranks = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    train_set, test_set = load_dataset()
    inference_loader = DataLoader(
        train_set, batch_size=args.inference_batch_size, shuffle=False)

    for batch_idx, (inputs, labels) in enumerate(inference_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss.detach().cpu().numpy().tolist()
        # accumulate loss
        for i in range(inputs.size(0)):
            ranks.append((batch_idx * args.inference_batch_size + i, loss[i]))
        print('inference batch done', batch_idx)
    print('inference done!')

    # sort by loss
    ranks.sort(key=lambda x: x[1], reverse=True)  # [(0,3),...]
    print(ranks[:100])

    selected_indices = [item[0] for item in ranks[:5000]]
    retrain_set = Subset(train_set, selected_indices)
    print('selected indices!')
    print('retrain set length:', len(retrain_set))
    # ================================================================================================


if __name__ == "__main__":
    train_loss()
