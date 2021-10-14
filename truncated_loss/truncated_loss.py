import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TruncatedLoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * \
               self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class QLoss(nn.Module):
    def __init__(self, q=0.7, reduction='mean'):
        super(QLoss, self).__init__()
        self.q = q
        self.reduction = reduction

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg ** self.q)) / self.q)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'none':
            loss = torch.squeeze(loss, 1)

        return loss
