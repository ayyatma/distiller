#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""An implementation of a trivial MNIST model.
 
The original network definition is sourced here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch.nn as nn
import torch.nn.functional as F
import distiller

from .simplenet_mnist import Simplenet
from .simplenet_mnist import Simplenet_v2

__all__ = [
    'simplenet_mnist_earlyexit_classbased', 'simplenet_mnist_earlyexit_classbased_v2',
    'simplenet_mnist_earlyexit_classbased_base', 'simplenet_mnist_earlyexit_classbased_v2_base'
]

def get_exits_def():
    exits_def = [('pool1', nn.Sequential(nn.Flatten(),
                                                  nn.Linear(2880, 3))),
                 ('pool2', nn.Sequential(nn.Flatten(),
                                                  nn.Linear(800, 5)))]
    return exits_def

def get_exits_def_base():
    exits_def = [('pool1', nn.Sequential(nn.Flatten(),
                                                  nn.Linear(2880, 10))),
                 ('pool2', nn.Sequential(nn.Flatten(),
                                                  nn.Linear(800, 10)))]
    return exits_def

class SimpleNetMNISTEarlyExitClassBased(Simplenet):
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

class SimpleNetMNISTEarlyExitClassBasedBase(Simplenet):
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


class SimpleNetMNISTEarlyExitClassBasedV2(Simplenet_v2):
    """
    This is Simplenet but with only one small Linear layer, instead of two Linear layers,
    one of which is large.
    26K parameters.
    python compress_classifier.py ${MNIST_PATH} --arch=simplenet_mnist --vs=0 --lr=0.01

    ==> Best [Top1: 98.970   Top5: 99.970   Sparsity:0.00   Params: 26000 on epoch: 54]
    """
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

class SimpleNetMNISTEarlyExitClassBasedV2Base(Simplenet_v2):
    """
    This is Simplenet but with only one small Linear layer, instead of two Linear layers,
    one of which is large.
    26K parameters.
    python compress_classifier.py ${MNIST_PATH} --arch=simplenet_mnist --vs=0 --lr=0.01

    ==> Best [Top1: 98.970   Top5: 99.970   Sparsity:0.00   Params: 26000 on epoch: 54]
    """
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

def simplenet_mnist_earlyexit_classbased():
    model = SimpleNetMNISTEarlyExitClassBased()
    return model
def simplenet_mnist_earlyexit_classbased_base():
    model = SimpleNetMNISTEarlyExitClassBasedBase()
    return model

def simplenet_mnist_earlyexit_classbased_v2():
    model = SimpleNetMNISTEarlyExitClassBasedV2()
    return model
def simplenet_mnist_earlyexit_classbased_v2_base():
    model = SimpleNetMNISTEarlyExitClassBasedV2Base()
    return model
# def simplenet_v2_mnist():
#     model = Simplenet_v2()
#     return model