'''这是利用pytorch框架实现多层感知机的练习代码'''

import torch
from torch import nn
from d2l import torch as d2l

from mytrainer import train_ch3

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hidden=784,10,256

net=nn.Sequential(nn.Flatten(),nn.Linear(num_inputs,num_hidden),
                  nn.ReLU(),nn.Linear(num_hidden,num_outputs))

def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.1)

net.apply(init_weight)

num_epochs,lr=10,0.1
updater=torch.optim.SGD(net.parameters(),lr=lr)
loss=nn.CrossEntropyLoss()

if __name__=="__main__":
    train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
