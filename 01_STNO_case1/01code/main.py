# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:27:23 2023

@author: Yoren Mo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lapy import TriaMesh, Solver, TetMesh
from timeit import default_timer
import scipy.io as sio
import time
import pandas as pd
from utilities3 import RangeNormalizer, count_params,LpLoss,UnitGaussianNormalizer, GaussianNormalizer, GetDLO
# from utilsLNO import MeshNO
from utilsFNO import FMeshNO
import os
from Adam import Adam
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main(args):  
    
    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")
    print('M',args.modes,' W',args.width,' B',args.batch_size,' LR', args.lr,' BS-', args.basis)
    
    ################################################################
    # configs
    ################################################################
    
    temporal_basis = args.temporal_basis
    PATH = args.data_dir
    ntrain = args.num_train  # 250
    ntest = args.num_test  # 100 
    batch_size = args.batch_size # 10
    learning_rate = args.lr # 0.001    
    epochs = args.epochs # 10    
    modes = args.modes # 截取K个模态 KL=64
    t1dmodes = args.t1dmodes #F域模态 Kf=16
    width = args.width # V=32
    
    step_size = 100
    gamma = 0.5
    
    s = args.size_of_nodes #316

    BASIS = args.basis #LBO
    
    ################################################################
    # reading data and normalization
    ################################################################   
    
    data = sio.loadmat(PATH) 
    
    ## 得到L氏基
    k = 128
    Points = np.vstack((data['nodes'].T, np.zeros(s).reshape(1,-1))) 
    mesh = TriaMesh(Points.T,data['elements']-1)# 坐标 和 网格数的索引
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k) #得到L基 201*128
    ## 得到L氏基
    
   
    y_dataIn = torch.Tensor(data['a_field'])  # (400, 272, 32)
    x_dataIn = torch.Tensor(data['T_field']) # (400, 272, 32)
    
    
    x_data = x_dataIn 
    y_data = y_dataIn #(y_dataIn - torch.mean(y_dataIn))/torch.std(y_dataIn)
    
    x_train = x_data[:ntrain,:,:]   
    y_train = y_data[:ntrain,:,:]

    
    x_test = x_data[ntrain:ntrain+ntest,:,:]    
    y_test = y_data[ntrain:ntrain+ntest,:,:]

    
    # x1 = x_train.numpy()
    # x2 = y_train.numpy()
    # x3 = x_test.numpy()
    # x4 = y_test.numpy()
            
    ##数据维度处理
    nodes = y_train.shape[1]
    Nt = y_train.shape[2]
      
    ##数据标准化
    norm_x  = UnitGaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train)  # (400, 272, 32)
    x_test  = norm_x.encode(x_test)  # (400, 272, 32)
    
    norm_y  = UnitGaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train) # (400, 272, 32)
    y_test  = norm_y.encode(y_test) # (400, 272, 32)
    
                
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True, drop_last=True) #30*(10, 272, 32)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False, drop_last=True) #10*(10, 272, 32)
    
    ## 得到时间域一维离散L矩阵以及L基
    DLO = GetDLO(t1dmodes)
    L_matrix = DLO.Get1dLaplace_slide(Nt) 
    L_right = DLO.Get1dLaplace_right(Nt, L_matrix)
    L_left = DLO.Get1dLaplace_left(Nt, L_matrix)   
    L_mid = L_left + L_right
       
    eigenvalue, E = DLO.GetE(L_mid) # 此时截断到k个特征向量
    E_inverse = E.T
       
    ### 画二维图像查看一下
    # A = np.zeros((Nt, 1))
    # A = E[:,3]
    # x_axis_data = np.arange(0,Nt,1)
    # y_axis_data = A.T
    # plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='featurevector')
    # plt.show()
       
    LBO_Output = LBO_MATRIX
    
    BASE_Output = LBO_Output[:,:2*modes]
    MATRIX_Output = torch.Tensor(BASE_Output).cuda()
    INVERSE_Output = (MATRIX_Output.T @ MATRIX_Output).inverse() @ MATRIX_Output.T
    
    print('BASE_MATRIX_output:', MATRIX_Output.shape, 'BASE_INVERSE_output:', INVERSE_Output.shape)
    
    LBO_Input = LBO_MATRIX
    
    BASE_Input = LBO_Input[:,:2*modes]
    MATRIX_Input = torch.Tensor(BASE_Input).cuda()
    INVERSE_Input = (MATRIX_Input.T @ MATRIX_Input).inverse() @ MATRIX_Input.T
    
    print('BASE_MATRIX_input:', MATRIX_Input.shape, 'BASE_INVERSE_input:', INVERSE_Input.shape)
    
    BASE_MATRIX = LBO_MATRIX[:,:2*modes]
    BASE_MATRIX = torch.Tensor(BASE_MATRIX).cuda() #几何域的L基
    BASE_INVERSE = (BASE_MATRIX.T@BASE_MATRIX).inverse()@BASE_MATRIX.T #L基矩阵求逆
    
    E = torch.Tensor(E).cuda() 
    E_inverse = torch.Tensor(E_inverse).cuda()
    
    
    # model = MeshNO(2*modes, t1dmodes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input, E, E_inverse).cuda() # km变为128 dv=64 L基矩阵201*128 L基伪逆128*201      
    model = FMeshNO(BASE_MATRIX.shape[1], t1dmodes, width, BASE_MATRIX, BASE_INVERSE, x_train.size(2)).cuda() # km变为128 dv=64 L基矩阵201*128 L基伪逆128*201    
    ################################################################
    # training and evaluation
    ################################################################
    """
    选择优化器
    """
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # 效果还不错
    # optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #收敛较快 但最大误差较大
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #定义迭代优化器
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) #设置学习率衰减，每过step_size个epoch，做一次更新
    
    myloss = LpLoss(d=3, p=2, size_average  = False) #自定义损失函数
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))
    
    for ep in range(epochs):
        model.train() #启用训练模式
        loss_max_train = 0
        train_l2 = 0
        for x, y in train_loader:# x,y分别为一个batch_size 待验 x,y:[10,201,223]
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()#梯度先归零
            out = model(x)#调用forward函数 out:[10,201,223]
            
            """
            显示更新梯度前网络参数和梯度
            """
            # for name, parms in model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->para:', parms.size())
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("===")
                      
            # mse = F.mse_loss(out.contiguous().view(batch_size,-1), y.view(batch_size, -1), reduction='mean')#计算误差，维度缩减求均值
            
            l2 = myloss(out, y)            
            l2.backward() # use the l2 relative loss 反向传播计算得到每个参数的梯度值
            
            """
            显示更新梯度后网络参数和梯度
            """
            # print("=============更新之后===========")
            # for name, parms in model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->para:', parms.size())
            #     print('-->grad_requirs:',parms.requires_grad)
            #     if parms.grad != None:
            #         print('-->grad_value:',(parms.grad.data).size())
            #     print("===")
            
                               
            out_real = norm_y.decode(out.cpu()).contiguous().view(batch_size, -1) #[10,201*223]
            y_real = norm_y.decode(y.cpu()).view(batch_size, -1) #[10,201*223]
            
            # out_real = out_real
            # y_real = y_real
                                            
            train_l2 += myloss(out_real, y_real).item()   #去标准化数据l2误差          
            loss_max_train += (abs(out_real- y_real)).max(axis=1).values.mean() #去标准化数据最大误差
                   
            optimizer.step() #梯度下降更新参数
            # train_mse += mse.item()
            
        scheduler.step()#更新学习率
        model.eval() #启用测试模式
        test_l2 = 0.0
        loss_max_test = 0.0
        
        
        with torch.no_grad(): #只是想看一下训练的效果，并不是想通过验证集来更新网络
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
    
                out = model(x) # out:[10,201,223]
                out_real = norm_y.decode(out.cpu()).contiguous().view(batch_size, -1)
                y_real = norm_y.decode(y.cpu()).view(batch_size, -1)
                
                # out_real = out_real
                # y_real = y_real
                
                test_l2 += myloss(out_real, y_real).item()  #当前所有bs的误差和 10*error    
                
                loss_max_test += (abs(out_real- y_real)).max(axis=1).values.mean() #每行计算最大向量
                
                # loss_max_test = loss_max_test * torch.std(y_dataIn) + torch.mean(y_dataIn)
    
       
        train_l2 /= ntrain # 当前ep的平均L2误差
        test_l2 /= ntest
        loss_max_train /= len(train_loader)
        loss_max_test /= len(test_loader)
        
        train_error[ep] = train_l2
        test_error[ep] = test_l2        
        ET_list[ep] = loss_max_test
        
        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Epoch: %d, Train L2: %.5f, Test L2: %.5f, Emax_tr: %.5f, Emax_te: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, loss_max_train, loss_max_test, T))
        time_step = time.perf_counter()
          
    print("\n=============================")
    print("Training done...")
    print("=============================\n")
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)      # (100,201,223)
    y_test   = torch.zeros(y_test.shape)      # (100,201,223)
    x_test   = torch.zeros(x_test.shape)      # (100,201,223)
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            
            out_real = norm_y.decode(out.cpu())
            y_real   = norm_y.decode(y.cpu())
            x_real   = norm_x.decode(x.cpu())
            
            pre_test[index,:] = out_real
            y_test[index,:] = y_real
            x_test[index,:] = x_real
            
            index = index + 1
            
    # ================ Save Data ====================
    
    current_directory = os.getcwd()
    sava_path = current_directory + "/logs/" + args.CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)
        
    Test_error = np.abs(y_test.numpy()-pre_test.numpy())

    E_max = np.max(Test_error[:,:,:],axis=(1,2))
    E_mean = np.mean(Test_error[:,:,:],axis=(1,2))

    print('\n平均最大误差（mean):',np.round(np.mean(E_max),3),'方差:',np.round(np.std(E_max),3))
    print('平均误差(mean):',np.round(np.mean(E_mean),3),'方差:',np.round(np.std(E_mean),3))
    
    dataframe = pd.DataFrame({'mean error': [np.round(np.mean(E_mean),3)],
                              'mean std': [np.round(np.std(E_mean),3)],
                              'mean max error': [np.round(np.mean(E_max),3)],
                              'mean max std': [np.round(np.std(E_max),3)],
                              'loss_max_test': [loss_max_test],
                              'L2 error': [np.round(test_l2,5)],
                              'num_paras': [count_params(model)],
                              'train_time':[time_step_end - time_start]})
    
    dataframe.to_csv(sava_path + 'log.csv', index = False, sep = ',')
    
    
    loss_dict = {'train_error' :train_error,
                 'test_error'  :test_error,
                 'ET_list'  :ET_list}
    
    pred_dict = {'pre_test' : pre_test.cpu().detach().numpy(),
                    # 'pre_train': pre_train.cpu().detach().numpy(),
                    'x_test'   : x_test.cpu().detach().numpy(),
                    # 'x_train'  : x_train.cpu().detach().numpy(),
                    'y_test'   : y_test.cpu().detach().numpy()
                    # 'y_train'  : y_train.cpu().detach().numpy(),
                    }
    
    # torch.save(model_output, 'logs/Node316_testLNO_2023_06/net_params.pkl') ## 待修改参数
    sio.savemat(sava_path +'MeshNO_loss_' + args.CaseName + '.mat', mdict = loss_dict)                                                     
    sio.savemat(sava_path +'MeshNO_pre_'  + args.CaseName + '.mat', mdict = pred_dict)
    
    print('\nTesting error: %.3e'%(test_l2))
    print('training error: %.3e'%(train_l2))
    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))
    
    
    


if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            
    '''
    待修改参数
    '''
    for i in range(3):
        print('====================================')
        print('第' + str(i + 1) + '次训练......')
        print('====================================')
        for args in [
            {'temporal_basis':'FNO', #LNO or FNO
              'modes': 32,  
              't1dmodes': 32,
              'width': 32, 
              'batch_size': 50,     
              'epochs': 500,   
              'data_dir': '../../Data/case1.mat',
              'size_of_nodes': 450,
              'num_train': 1000, 
              'num_test': 200,
              'CaseName': 'case1_'+str(i+1), #此处保留测试记录用
              'basis':'LBO',
              'lr' : 0.001},
        ]:
            args = objectview(args)
    
        main(args)

