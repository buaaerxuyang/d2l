'''这是实现lenet的练习代码'''
import torch
from torch import nn
from d2l import torch as d2l

from mytrainer import train_ch6

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

class Reshape(nn.Module):
    def forward(self,X):
        return X.reshape((-1,1,28,28))
    
net=nn.Sequential(
    Reshape(),
    nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=(2,2),stride=2),
    nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5)),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=(2,2),stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.Sigmoid(),
    nn.Linear(84,10)
    )

if __name__=="__main__":
    print(d2l.try_gpu())
    
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

