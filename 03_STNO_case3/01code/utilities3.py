# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:36:29 2023

@author: Yoren Mo
"""

import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0) #沿0维方向求平均
        self.std = torch.std(x, 0) #沿0维方向求方差
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer2(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer2, self).__init__()
        self.mymin = torch.min(x.contiguous().view(x.size()[0], -1), 0)[0].view(-1)
        self.mymax = torch.max(x.contiguous().view(x.size()[0], -1), 0)[0].view(-1)

        # self.a = (high - low)/(mymax - mymin)
        # self.b = -self.a*mymax + high

    def encode(self, x):
        
        s = x.size()
        x = x.contiguous().view(s[0], -1)
        
        x = (x-self.mymin)/(self.mymax-self.mymin + 0.01)        
        x = x.view(s)
        
        return x

    def decode(self, x):
        s = x.size()
        x = x.contiguous().view(s[0], -1)
        x = x*(self.mymax-self.mymin)+self.mymin
        x = x.view(s)
        return x

class RangeNormalizer(object):   
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        
        self.s = x.size()
        x = x.reshape(self.s[0], -1)
        
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low)/(mymax - mymin + 0.001)
        self.b = -self.a*mymax + high

    def encode(self, x):  
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.reshape(s[0], s[1], s[2])
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.reshape(s[0], s[1], s[2])
        return x
    
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)      
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
                # return torch.sum(diff_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)




# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


class datatransfer(object):
    def __init__(self, ntrain, Nt, nodes):
        
        super(datatransfer, self).__init__()
        self.ntrain = ntrain
        self.Nt = Nt
        self.nodes = nodes
          
        
    def dim_reduction(self, x):
        y_sec = torch.zeros((self.ntrain*self.Nt,self.nodes))
        for i in range(self.ntrain):
            for j in range(self.Nt):
                y_sec[(i)*self.Nt+j] = x.permute(0,2,1)[i,j,:]
        return y_sec
    
    def dim_ascending(self, x):
        y = torch.zeros((self.ntrain, self.Nt, self.nodes))
        for i in range(self.ntrain):
            for j in range(self.Nt):
                y[i,j,:] = x[(i)*self.Nt+j]
        return y.permute(0,2,1)
            

# get L discrete Laplace operator and spectral transform coefficient             
class GetDLO(object):
    def __init__(self, k=32):
        super(GetDLO, self).__init__()
        
        # self.x = x
        self.k = k
        
    def Get1dLaplace(self, Nt):
        
        L = np.zeros((Nt, Nt))
        
        for i in range(Nt):
            if i!=0 and i!=Nt-1:
                L[i][i] = 1 
                L[i][i-1] = -1/2
                L[i][i+1] = -1/2
                
            if i == 0:
                L[i][i] = 1 
                L[i][Nt-1] = -1/2
                L[i][i+1] = -1/2
                
            if i == Nt-1:
                L[i][i] = 1 
                L[i][i-1] = -1/2
                L[i][0] = -1/2
                
        return L
    
    def Get1dLaplace_right(self, Nt, E):
        
        L = np.zeros((Nt, Nt))
        
        L[:,1:] = E[:,:-1]
        L[:,0] = E[:,-1]
                
        return L
    
    def Get1dLaplace_left(self, Nt, E):
        
        L = np.zeros((Nt, Nt))
        
        L[:,:-1] = E[:,1:]
        L[:,-1] = E[:,0]
                
        return L
    
    def Get1dLaplace_slide(self, Nt):
        
        L = np.zeros((Nt, Nt))
        
        for i in range(Nt):
            if i!=Nt-2 and i!=Nt-1 and i!=0 and i!=1 :
                L[i][i] = 1 
                L[i][i+1] = -1/2
                L[i][i-1] = -1/2
                L[i][i+2] = -1/4
                L[i][i-2] = -1/4
                               
            if i == Nt-1:
                L[i][Nt-1] = 1 
                L[i][Nt-2] = -1/2
                L[i][Nt-3] = -1/4
                L[i][1] = -1/4
                L[i][0] = -1/2
                
            if i == Nt-2:
                L[i][Nt-1] = -1/2
                L[i][Nt-2] = 1
                L[i][Nt-3] = -1/2
                L[i][Nt-4] = -1/4
                L[i][0] = -1/4
                
            if i == 0:
                L[i][Nt-1] = -1/2
                L[i][Nt-2] = -1/4
                L[i][2] = -1/4
                L[i][1] = -1/2
                L[i][0] = 1  
                
            if i == 1:
                L[i][Nt-1] = -1/4
                L[i][3] = -1/4
                L[i][2] = -1/2
                L[i][1] = 1
                L[i][0] = -1/2
            
                
        return L
    
    def GetE(self, L_matrix):
        
        eigenvalue, featurevector = np.linalg.eig(L_matrix)
        featurevector = featurevector[:,:self.k]      
        
        return eigenvalue, featurevector

        
        
        
