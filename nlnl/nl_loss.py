import torch
import torch.nn.functional as F


class NLLoss(torch.nn.Module):
    def __init__(self, num_classes, device, reduction='mean'):
        super(NLLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        # pred = F.log_softmax(pred, dim=1)

        # pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        # label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        if self.reduction == 'mean':
            loss = (-1 * torch.sum(label_one_hot * torch.log(1 - pred), dim=1)).mean()
        else:
            loss = (-1 * torch.sum(label_one_hot * torch.log(1 - pred), dim=1))
        # loss = F.nll_loss(1 - pred, labels)

        return loss
