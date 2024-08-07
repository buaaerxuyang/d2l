'''
这是从零开始实现多层感知机的练习代码
'''

import torch
from torch import nn
from d2l import torch as d2l

from mytrainer import train_ch3

'''给出数据迭代器'''
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hidden=784,10,256

W1=nn.Parameter(torch.randn(num_inputs,num_hidden,requires_grad=True))
'''torch.randn生成标准正态分布的随机数，转为Parameter类型'''
b1=nn.Parameter(torch.zeros(num_hidden,requires_grad=True))
W2=nn.Parameter(torch.randn(num_hidden,num_outputs,requires_grad=True))
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params=[W1,b1,W2,b2]

'''定义激活函数'''
def reLu(X):
    a=torch.zeros_like(X)
    return torch.max(a,X)

def net(X):
    X=X.reshape((-1,num_inputs))
    H=reLu(X @ W1 + b1)
    return (H @ W2 + b2)
'''@ 可以表示矩阵乘法'''

loss=nn.CrossEntropyLoss()

num_epochs,lr=10,0.1
updater=torch.optim.SGD(params,lr=lr)

if __name__=="__main__":
    train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)


