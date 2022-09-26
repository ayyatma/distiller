import torch
from torch import autograd, tensor
from torch import nn


class CostAdjustedCrossEntropyLossSamples(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    def __init__(self, class_weights, super_classes, threshold = None):
        super().__init__()
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())
        self.super_classes = torch.FloatTensor(super_classes).cuda()
        self.threshold = threshold

    def forward(self, logits, target, prev_passed):
        # log_probabilities = self.log_softmax(logits)
        # nll = nn.NLLLoss(weight=self.class_weights)
        # # truetarget = torch.tensor(self.super_classes.index_select(0, target), dtype=torch.long, device='cuda')
        # truetarget = self.super_classes.index_select(0, target).long()
        # loss = nll(log_probabilities, truetarget)
        
        # log = self.softmax(logits)
        
        e=torch.distributions.Categorical(logits=logits)
        output=e.entropy()

        passed = output < self.threshold
        curr_check = (~prev_passed).type(torch.cuda.FloatTensor)
        next_passed = torch.logical_or(passed, prev_passed)
        if (torch.sum(curr_check) == 0):
             return torch.tensor(0).cuda(), next_passed

        log_probabilities = self.log_softmax(logits)
        truetarget = self.super_classes.index_select(0, target).long()
        predtarget = torch.argmax(log_probabilities, dim=1)
        
        w1 = self.class_weights.index_select(0, truetarget)
        w2 = w1.gather(1, predtarget.view(-1, 1))
        indexedweights = torch.reshape(w2, (-1,))

        indexedweights = indexedweights * curr_check 
        # if (torch.sum(indexedweights) == 0):
        #     return l3, next_passed
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
        
        return l3, next_passed