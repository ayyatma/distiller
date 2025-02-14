import torch
from torch import autograd
from torch import nn


class CostAdjustedCrossEntropyLoss(nn.Module):
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
        predtarget = torch.argmax(log_probabilities, dim=1)
        
        w1 = self.class_weights.index_select(0, truetarget)
        w2 = w1.gather(1, predtarget.view(-1, 1))
        indexedweights = torch.reshape(w2, (-1,))
        l2 = -indexedweights * log_probabilities.index_select(-1, truetarget).diag()
        #mean reduction
        l3 = torch.sum(1/torch.sum(indexedweights) * l2)  
        # l3 = torch.sum(l2)  
        
        # l3 = torch.mean(l2)     
        #sum reduction
        # l3 = torch.sum(l2)
        # NLLLoss(x, class) = -weights[class] * x[class]
        # print(target)
        # print(truetarget)
        # loss = -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
        return l3, prev_passed