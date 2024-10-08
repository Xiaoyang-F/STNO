# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:28:33 2023

@author: Yoren Mo
"""

from scipy.io import savemat

import torch
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial


#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class XMTransform(nn.Module):
    def __init__ (self, nodes1, nodes2, width):
        super(XMTransform, self).__init__()

        '''
        在原始空间上对空间域维度进行参数化
        '''
        self.in_channels = nodes1 
        self.out_channels = nodes2
        self.modes1 =  width 
        
        self.scale = (1 / (self.in_channels*self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes1, self.in_channels, self.out_channels, dtype=torch.float)) 
        
    def forward(self, x):  
        
        x = x.permute(0,3,1,2) 
        x = torch.einsum("txbi,xio->txbo", x, self.weights1) 
        
        #(Nt*width*bs*nodes1),(width*nodes1*nodes2)->(Nt*width*bs*nodes2)
        return x     

#LBO Network
#########################
class SpectralConv1d(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, Fmodes):
        super(SpectralConv1d, self).__init__()
        '''
        LBO and SVD
        '''
        self.in_channels = in_channels #32
        self.out_channels = out_channels #32
        self.modes1 = modes #128
        self.modes2 = Fmodes #16
        # self.LBO_MATRIX = LBO_MATRIX #201*128 
        # self.LBO_INVERSE = LBO_INVERSE #128*201
        
        '''
        使该尺度参数成为可训练的参数
        '''  
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes1, self.modes2, in_channels, out_channels, dtype=torch.cfloat)) # 32*32*128*16
        
        
    def compl_mul1d(self, input, weights):
        # 张量乘积，右边的x是L氏域中的k个模态,t为F域K个模态
        return torch.einsum("xtbi,xtio->xtbo", input, weights)
                      
    def forward(self, x): #(128,112,10,32)
    
        out = torch.zeros(self.modes1, x.size(1), x.size(2), self.in_channels, device=x.device, dtype=torch.cfloat)  
        out[:, :self.modes2, :, :] = self.compl_mul1d(x[:, :self.modes2, :, :], self.weights1) #128*112*10*32
        
        return out
    
        
class FMeshNO(nn.Module):
    def __init__(self, modes, t1dmodes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input, E, E_inverse, Nt): # km变为128 dv=64 L基矩阵201*128 L基伪逆128*201
        super(FMeshNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.modes2 = t1dmodes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        self.MATRIX_Output = MATRIX_Output #201*128
        self.INVERSE_Output = INVERSE_Output #128*201
        self.MATRIX_Input = MATRIX_Input
        self.INVERSE_Input = INVERSE_Input    
        self.E = E #50*16
        self.E_inverse = E_inverse #16*50
        self.Nt = Nt     
        # self.convt = XMTransform(self.nodes, self.nodes, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1, self.modes2 )# 32 32 空间场基 时间场基
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1, self.modes2 )
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1, self.modes2 )
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1, self.modes2 )
        # self.conv4 = SpectralConv1d(self.width, self.width, self.modes1, self.modes2 )
        
        
        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        # self.w4 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        
        
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x): #x[10,201,223]
    
    
        grid = self.get_grid(x.shape, x.device)# x[10,201,223] grid[223,10,201,1]                
        x = x.permute(2, 0, 1) # x[223,10,201]
        
        x = x.unsqueeze(-1) # x[223,10,201,1]   
        
        x = torch.cat((x, grid), dim=-1) # x[223,10,201,2]
        
        
        '''
        提升通道数
        '''
       
        x = self.fc0(x) # x[223,10,201,32]
        x = x.permute(0, 1, 3, 2) # x[223,10,32,201]
        
        # '''
        # 层1：输入数据的编码与解码
        # '''
        # x1 = self.Lmapping(x, self.INVERSE_Input) #几何域投影到L域，几何离散维度降低(223,10,32,128)       
        # x1 = self.Fmapping(x1, self.modes2) #时域投影到F域，时间离散维度降低(128,112,10,32)     
        # x1 = self.conv0(x1) # x1(128*16*10*32) 
        # x1 = self.iFmapping(x1, self.Nt)# 10*32*128*223      
        # x1 = self.iLmapping(x1, self.MATRIX_Input) # x1 223*10*32*201
        
        # x2 = x.permute(1, 2, 0, 3) # 10*32*223*201
        # x2 = self.w0(x2) # x2 10*32*223*201 
        # x2 = x2.permute(2, 0, 1, 3) # x2 223*10*32*201
        
        # x = x1 + x2
        # x = F.gelu(x) #x 223*10*32*201 激活
                
        
                
        '''
        层3: 输入数据域到输出数据域映射
        '''
        x1 = self.Lmapping(x, self.INVERSE_Input)      
        x1 = self.Fmapping(x1, self.modes2) #时域投影到F域，时间离散维度降低(128,112,10,32)     
        x1 = self.conv0(x1) # x1(128*16*10*32) 
        x1 = self.iFmapping(x1, self.Nt)# 10*32*128*223      
        x1 = self.iLmapping(x1, self.MATRIX_Output) 
        
        # x2 = x.permute(1, 2, 0, 3)
        # x2 = self.w0(x2) 
        # x2 = x2.permute(2, 0, 1, 3) 
        
        # x = x1 + x2
        x = F.gelu(x1) 
        
        '''
        层2: 输入数据的编码与解码
        '''
        x1 = self.Lmapping(x, self.INVERSE_Input)    
        x1 = self.Fmapping(x1, self.modes2) #时域投影到F域，时间离散维度降低(128,112,10,32)     
        x1 = self.conv1(x1) # x1(128*16*10*32) 
        x1 = self.iFmapping(x1, self.Nt)# 10*32*128*223      
        x1 = self.iLmapping(x1, self.MATRIX_Input) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w1(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = F.gelu(x) 
        
        '''
        层4
        '''
        x1 = self.Lmapping(x, self.INVERSE_Output)      
        x1 = self.Fmapping(x1, self.modes2) #时域投影到F域，时间离散维度降低(128,112,10,32)     
        x1 = self.conv2(x1) # x1(128*16*10*32) 
        x1 = self.iFmapping(x1, self.Nt)# 10*32*128*223      
        x1 = self.iLmapping(x1, self.MATRIX_Output) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w2(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2  # x 223*10*32*201
        x = F.gelu(x) 
        '''
        层5
        '''
        x1 = self.Lmapping(x, self.INVERSE_Output)      
        x1 = self.Fmapping(x1, self.modes2) #时域投影到F域，时间离散维度降低(128,112,10,32)     
        x1 = self.conv3(x1) # x1(128*16*10*32) 
        x1 = self.iFmapping(x1, self.Nt)# 10*32*128*223      
        x1 = self.iLmapping(x1, self.MATRIX_Output) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w3(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2  # x 223*10*32*201
        # x = F.gelu(x) 
        
        '''
        全连接层
        '''
        x = x.permute(0, 1, 3, 2) #223*10*201*32
        x = self.fc1(x) # 223*10*201*128
        x = F.gelu(x)
        x = self.fc2(x) # 223*10*201*1
        
        # x = x.squeeze(-1) # x[223,10,201] 
        x = x.permute(1, 2, 0, 3)
        
        
       
        
        return x  #[10,201,223]

    def get_grid(self, shape, device):
        timenodes, batchsize, size_x = shape[2], shape[0], shape[1] #223,10,201
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([timenodes, batchsize, 1, 1])
        return gridx.to(device)     #10*8576*1 从0到1
    
    def Lmapping(self, x, LBO_INVERSE): ## x[223,10,32,201]  201*128  128*201
        '''
        先将各个时刻扩充后的通道投影到L域上
        '''
        # LBO domain
        x = x = x.permute(0,1,3,2) #(223*10*201*32)
        x = LBO_INVERSE @ x  # LBO_MATRIX mesh_number*modes      LBO_inverse(128*201)*(223*10*201*32)
        x = x.permute(0,1,3,2) #(223,10,128,32)->(223,10,32,128)
        return x
    
    
    def Fmapping(self, x, modes2):
        '''
        时域投影到F域上
        '''
        x = x.permute(3,1,2,0) # (128*10*32*223)      
        x_ft = torch.fft.rfft(x).cfloat()   # (128*10*32*112)     
        x_ft = x_ft.permute(0,3,1,2)  # (128*112*10*32) 
        return x_ft
       
    def iFmapping(self, x, Nt):
        '''
        F域投回到时域
        '''
        x = x.permute(2,3,0,1)  # (*10*32*128*112) 
        x_rft = torch.fft.irfft(x, Nt)  # (*10*32*128*223) 
        
        return x_rft
        
        
    def iLmapping(self, x, LBO_MATRIX): # 10*32*128*223 128*201
        '''
        L域投回到时域
        '''
        x = x.permute(3,0,1,2) #223*10*32*128
        x = x @ LBO_MATRIX.T #223*10*32*201
        
        return x
        
    
