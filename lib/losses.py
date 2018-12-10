import torch.nn as nn
import torch.nn.functional as F
import torch


class BceWithLogit(object):

    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, preds, targets):

        loss = self.loss(preds, targets)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)


def topk_categorical_accuracy(pred, truth, k=3):
    sorted_index = torch.topk(pred, k=k, dim=1)[1]
    index_true = torch.topk(truth, k=1, dim=1)[1]
    match_each_sample = torch.sum(torch.eq(sorted_index, index_true), dim=1)
    num_correct = torch.sum(match_each_sample).item()
    return num_correct/pred.shape[0]

