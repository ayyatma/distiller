#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Resnet for CIFAR10 with Early Exit branches

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.

@inproceedings{DBLP:conf/cvpr/HeZRS16,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  pages     = {770--778},
  publisher = {{IEEE} Computer Society},
  year      = {2016}
}

"""
from .resnet_cifar import BasicBlock
from .resnet_cifar import ResNetCifar
import torch.nn as nn
import distiller


__all__ = [ 'resnet20_cifar_earlyexit_classbased', 'resnet32_cifar_earlyexit_classbased', 'resnet44_cifar_earlyexit_classbased',
            'resnet56_cifar_earlyexit_classbased', 'resnet110_cifar_earlyexit_classbased', 'resnet1202_cifar_earlyexit_classbased',
            'resnet20_cifar_earlyexit_classbased_base', 'resnet32_cifar_earlyexit_classbased_base', 'resnet44_cifar_earlyexit_classbased_base',
            'resnet56_cifar_earlyexit_classbased_base', 'resnet110_cifar_earlyexit_classbased_base', 'resnet1202_cifar_earlyexit_classbased_base',
            'resnet56_cifar_earlyexit_classbased_cifar100', 'resnet110_cifar_earlyexit_classbased_cifar100', 'resnet1202_cifar_earlyexit_classbased_cifar100',
            'resnet56_cifar_earlyexit_classbased_multiple']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_exits_def_base():
    exits_def = [('layer1.1.conv1', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(1600, 10))),
                 ('layer2', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(800, 10)))]
    return exits_def

def get_exits_def_base_multiple():
    # exits_def = [('layer1.1.conv1', nn.Sequential(nn.AvgPool2d(12),
    #                                             nn.Flatten(),
    #                                             nn.Linear(64, 2))),
    #             ('layer1.3.conv2', nn.Sequential(nn.AvgPool2d(12),
    #                                             nn.Flatten(),
    #                                             nn.Linear(64, 3))),
    #             ('layer1.8.conv2', nn.Sequential(nn.AvgPool2d(10),
    #                                             nn.Flatten(),
    #                                             nn.Linear(144, 4))),
    #             ('layer2.4.conv1', nn.Sequential(nn.AvgPool2d(8),
    #                                             nn.Flatten(),
    #                                             nn.Linear(128, 5))),
    #             ('layer2.8.conv2', nn.Sequential(nn.AvgPool2d(4),
    #                                             nn.Flatten(),
    #                                             nn.Linear(512, 6))),
    #              ('layer3.4.conv1', nn.Sequential(nn.AvgPool2d(3),
    #                                             nn.Flatten(),
    #                                             nn.Linear(256, 7)))]
    exits_def = [('layer1', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(1600, 4))),
                 ('layer2', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(800, 7)))]

    return exits_def


def get_exits_def():
    exits_def = [('layer1', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(1600, 4))),
                 ('layer2', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(800, 7)))]
    return exits_def


def get_exits_def_imagenet():
    exits_def = [('layer1', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(44944, 10))),
                 ('layer2', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(21632, 10)))]
    return exits_def

    # exits_def = [('layer1.2.relu2', nn.Sequential(nn.AvgPool2d(3),
    #                                               nn.Flatten(),
    #                                               nn.Linear(1600, NUM_CLASSES)))]

def get_exits_def_cifar100():
    exits_def = [('layer1', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(1600, 100))),
                 ('layer3', nn.Sequential(nn.AvgPool2d(3),
                                                  nn.Flatten(),
                                                  nn.Linear(256, 100)))]
    return exits_def


class ResNetCifarEarlyExitClassBasedBase(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def_base())

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs

class ResNetCifarEarlyExitClassBasedMultiple(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def_base_multiple())

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs


class ResNetCifarEarlyExitClassBased(ResNetCifar):
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

class ResNetCifarEarlyExitClassBasedImageNette(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def_imagenet())

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs

class ResNetCifarEarlyExitClassBasedCifar100(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()
        self.ee_mgr.attach_exits(self, get_exits_def_cifar100())

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs

def resnet20_cifar_earlyexit_classbased(**kwargs):
    model = ResNetCifarEarlyExitClassBased(BasicBlock, [3, 3, 3], **kwargs)
    return model
def resnet20_cifar_earlyexit_classbased_base(**kwargs):
    model = ResNetCifarEarlyExitClassBasedBase(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar_earlyexit_classbased(**kwargs):
    model = ResNetCifarEarlyExitClassBased(BasicBlock, [5, 5, 5], **kwargs)
    return model
def resnet32_cifar_earlyexit_classbased_base(**kwargs):
    model = ResNetCifarEarlyExitClassBasedBase(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar_earlyexit_classbased(**kwargs):
    model = ResNetCifarEarlyExitClassBased(BasicBlock, [7, 7, 7], **kwargs)
    return model
def resnet44_cifar_earlyexit_classbased_base(**kwargs):
    model = ResNetCifarEarlyExitClassBasedBase(BasicBlock, [7, 7, 7], **kwargs)
    return model
# =========================================
def resnet56_cifar_earlyexit_classbased(**kwargs):
    model = ResNetCifarEarlyExitClassBased(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet56_cifar_earlyexit_classbased_multiple(**kwargs):
    model = ResNetCifarEarlyExitClassBasedMultiple(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet56_cifar_earlyexit_classbased_base(**kwargs):
    model = ResNetCifarEarlyExitClassBasedBase(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet56_cifar_earlyexit_classbased_imagenet(**kwargs):
    model = ResNetCifarEarlyExitClassBasedBase(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet56_cifar_earlyexit_classbased_cifar100(**kwargs):
    model = ResNetCifarEarlyExitClassBasedCifar100(BasicBlock, [9, 9, 9], **kwargs)
    return model
# =========================================
def resnet110_cifar_earlyexit_classbased(**kwargs):
    model = ResNetCifarEarlyExitClassBased(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet110_cifar_earlyexit_classbased_base(**kwargs):
    model = ResNetCifarEarlyExitClassBasedBase(BasicBlock, [18, 18, 18], **kwargs)
    return model
    
def resnet110_cifar_earlyexit_classbased_cifar100(**kwargs):
    model = ResNetCifarEarlyExitClassBasedCifar100(BasicBlock, [18, 18, 18], **kwargs)
    return model
# =========================================
def resnet1202_cifar_earlyexit_classbased(**kwargs):
    model = ResNetCifarEarlyExitClassBased(BasicBlock, [200, 200, 200], **kwargs)
    return model

def resnet1202_cifar_earlyexit_classbased_base(**kwargs):
    model = ResNetCifarEarlyExitClassBasedBase(BasicBlock, [200, 200, 200], **kwargs)
    return model
    
def resnet1202_cifar_earlyexit_classbased_cifar100(**kwargs):
    model = ResNetCifarEarlyExitClassBasedCifar100(BasicBlock, [200, 200, 200], **kwargs)
    return model