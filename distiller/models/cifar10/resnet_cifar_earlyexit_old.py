from .resnet_cifar import BasicBlock
from .resnet_cifar import ResNetCifar
import torch.nn as nn
import distiller


__all__ = ['resnet20_cifar_earlyexit', 'resnet32_cifar_earlyexit', 'resnet44_cifar_earlyexit',
           'resnet56_cifar_earlyexit', 'resnet110_cifar_earlyexit', 'resnet1202_cifar_earlyexit']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_exits_def():
    # exits_def = [('layer1.2.relu2', nn.Sequential(nn.AvgPool2d(3),
    #                                               nn.Flatten(),
    #                                               nn.Linear(1600, NUM_CLASSES))),
    #              ('layer2.4.relu2', nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
    #                                               nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
    #                                               nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
    #                                               nn.AvgPool2d(3),
    #                                             nn.Flatten(),
    #                                             nn.Linear(200, NUM_CLASSES)))]

    exits_def = [('layer1', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(1600, NUM_CLASSES))),
                 ('layer2', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(800, NUM_CLASSES)))]

    return exits_def



class ResNetCifarEarlyExit(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def())

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs


def resnet20_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [5, 5, 5], **kwargs)
    print(model)
    return model

def resnet44_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet110_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet1202_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [200, 200, 200], **kwargs)
    return model