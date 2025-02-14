import torch
from torch import autograd
from torch import nn


class ClassBasedCrossEntropyLossVector(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax(dim=1)

    def __init__(self, class_weights, super_classes):
        super().__init__()
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())
        self.super_classes = torch.FloatTensor(super_classes).cuda()

    def forward(self, logits, target, prev_passed = None):
        log_probabilities = self.log_softmax(logits)
        truetarget = self.super_classes.index_select(0, target).long()
        
        indexedweights = self.class_weights.index_select(0, truetarget)
        l2 = -indexedweights * log_probabilities.index_select(-1, truetarget).diag()
        l3 = torch.sum(1/torch.sum(indexedweights) * l2)
        
        # nll = nn.NLLLoss(weight=self.class_weights)
        # loss = nll(log_probabilities, truetarget)

        return l3, prev_passed