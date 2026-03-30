import torch
import torch.nn as nn
from modules_ME_2 import *

# chin_out = 5 # N of minerals + 1 channel for the mixed. Here we have pore + quartz + feltspar + clay + mixed -> 5
# factors = [8] # here we have 1 because we start from the initial size
# steps = [8, 4, 2] # these steps are the distance between the nodes for different levels of upsampling. For x8 we have 3 levels of upsampling
# sf = 8
# channels = [32, 64, 128, 256, 512]

class SupRES(torch.nn.Module):
    
    def __init__(self, chin_out, factors, steps, sf, channels):
        super(SupRES, self).__init__()

        self.chin_out = chin_out
        self.factors = factors
        self.steps = steps
        self.sf = sf
        self.channels = channels

        self.block_standard = nn.Sequential(
            nn.Conv3d(self.chin_out, self.channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.channels[4]),
            nn.ELU(),)
        
        self.classification_0 = ME.MinkowskiConvolution(self.channels[4], self.chin_out, kernel_size=1, bias=True, dimension=3)

        self.block1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(self.channels[4], self.channels[3], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(self.channels[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(self.channels[3], self.channels[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(self.channels[3]),
            ME.MinkowskiELU(),)

        self.classification_1 = ME.MinkowskiConvolution(self.channels[3], self.chin_out, kernel_size=1, bias=True, dimension=3)

        self.block2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(self.channels[3], self.channels[2], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(self.channels[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(self.channels[2], self.channels[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(self.channels[2]),
            ME.MinkowskiELU(),)
        
        self.classification_2 = ME.MinkowskiConvolution(self.channels[2], self.chin_out, kernel_size=1, bias=True, dimension=3)

        self.block3 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(self.channels[2], self.channels[1], kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(self.channels[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(self.channels[1], self.channels[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(self.channels[1]),
            ME.MinkowskiELU(),)

        self.classification_3 = ME.MinkowskiConvolution(self.channels[1], self.chin_out, kernel_size=1, bias=True, dimension=3)

        self.softmax = ME.MinkowskiSoftmax(dim=1) 
        self.pruning = ME.MinkowskiPruning()
        
    def forward(self, x):

        bs, ch, X, Y, Z = x.size()
        
        x = self.block_standard(x)
        # print('1', x.size())

        x = separate_coordinates_and_features_dense(x, factor = self.factors[0])
        # print('2', x.C.size())
        # print('2', x.F.size())

        sftmx_0 = self.softmax(self.classification_0(x))
        mask = torch.argmax(sftmx_0.F, dim=1) == (sftmx_0.F.size(1) - 1)
        x = self.pruning(x, mask)
        # print('3', x.C.size())
        # print('3', x.F.size())
        memorized_0 = self.pruning(sftmx_0, ~mask)
        # print('4', memorized_0.C.size())
        # print('4', memorized_0.F.size())

        x = self.block1(x)
        # print('5', x.C.size())
        # print('5', x.F.size())

        sftmx_1 = self.softmax(self.classification_1(x))
        mask = torch.argmax(sftmx_1.F, dim=1) == (sftmx_1.F.size(1) - 1)
        x = self.pruning(x, mask)
        # print('6', x.C.size())
        # print('6', x.F.size())
        memorized_1 = self.pruning(sftmx_1, ~mask)
        # print('7', memorized_1.C.size())
        # print('7', memorized_1.F.size())

        x = self.block2(x)
        # print('8', x.C.size())
        # print('8', x.F.size())

        sftmx_2 = self.softmax(self.classification_2(x))
        mask = torch.argmax(sftmx_2.F, dim=1) == (sftmx_2.F.size(1) - 1)
        x = self.pruning(x, mask)
        # print('9', x.C.size())
        # print('9', x.F.size())
        memorized_2 = self.pruning(sftmx_2, ~mask)
        # print('10', memorized_2.C.size())
        # print('10', memorized_2.F.size())

        x = self.block3(x)

        sftmx_3 = self.softmax(self.classification_3(x))
        # print('11', sftmx_3.C.size())
        # print('11', sftmx_3.F.size())

        dense_tensor = torch.zeros(bs, ch, X*self.sf, Y*self.sf, Z*self.sf).cuda()
        dense_tensor = update_dense_tensor(dense_tensor, sftmx_3.C, sftmx_3.F)
        for tensor, step in zip([memorized_0, memorized_1, memorized_2], self.steps):
            dense_tensor = expand_coordinates_and_properties(tensor, step, dense_tensor)
        # print('done9')
        
        return dense_tensor
        
        # print("Memory allocated after operation_1: ", torch.cuda.memory_allocated()/10**9, "GB")
        # print("Max memory allocated after operation_1: ", torch.cuda.max_memory_allocated()/10**9, "GB")
