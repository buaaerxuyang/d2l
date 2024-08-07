'''这是学习多通道卷积核的练习代码'''
import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X,K):
    '''实现二维交叉相关，X是输入矩阵，K是卷积核'''
    h,w = K.shape
    a,b = X.shape
    Y=torch.zeros((a-h+1,b-w+1))
    for i in range(a-h+1):
        for j in range(b-w+1 ):
            Y[i][j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

'''corr2d(x,k) for x,k in zip(X,K) 似乎返回了一个数组/迭代器，可以理解为2*2*2'''

def corr2d_multi_in(X,K):
    '''
    第一维度相加
    X是3维的，K是3维的
    '''
    return sum(corr2d(x,k) for x,k in zip(X,K))

#print(corr2d_multi_in(X,K))


def corr2d_multi_in_out(X,K):
    '''
    K是4维的，X是3维的
    '''
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
    '''在第一个维度就进行堆叠'''