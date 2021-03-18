import torch


def smooth_labels(labels, num_classes, smoothing=0.0):
    assert 0 < smoothing < 1
    conf = 1.0 - smoothing
    label_shape = torch.Size((labels.size(0), num_classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=labels.device)
        true_dist.fill_(smoothing / (num_classes - 1))
        # _, index = torch.max(labels, 0)
        true_dist.scatter_(1, torch.LongTensor(labels.data.unsqueeze(1), device=labels.device), conf)
    return true_dist
