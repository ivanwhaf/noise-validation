import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedCrossEntropy(nn.Module):
    """
        NCE
    """

    def __init__(self, num_classes, scale=1.0, reduction='mean'):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        if self.reduction == 'mean':
            nce = nce.mean()
        elif self.reduction == 'none':
            nce = nce
        return self.scale * nce


class ReverseCrossEntropy(torch.nn.Module):
    """
        RCE
    """

    def __init__(self, num_classes, scale=1.0, reduction='mean'):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        if self.reduction == 'mean':
            rce = rce.mean()
        elif self.reduction == 'none':
            rce = rce
        return self.scale * rce


class GeneralizedCrossEntropy(torch.nn.Module):
    """
        GCE
    """

    def __init__(self, num_classes, q=0.7, reduction='mean'):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q

        if self.reduction == 'mean':
            gce = gce.mean()
        elif self.reduction == 'none':
            gce = gce
        return gce


class NCEandRCE(torch.nn.Module):
    """
        NCE+RCE
    """

    def __init__(self, alpha, beta, num_classes, reduction='mean'):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes, reduction=reduction)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes, reduction=reduction)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)
