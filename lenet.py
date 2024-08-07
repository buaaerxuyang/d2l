'''这是练习经典的卷积神经网络————Lenet的练习代码'''
import torch
from torch import nn
from d2l import torch as d2l

from mytrainer import train_ch3

'''
func=nn.Conv2d(1,6,kernel_size=(3,3))
X=torch.arange(75,dtype=torch.float32).reshape((3,1,5,5))
Z=torch.arange(25,dtype=torch.float32).reshape((1,5,5))
print(func(X).shape)
print(func(Z).shape)
注意，nn.Conv2d只接受3维或4维张量
'''

'''
X=torch.arange(75,dtype=torch.float32).reshape((3,1,5,5))
func=nn.Flatten()
Y=func(X)
print(Y.shape)
Flatten有参数，start_dim和end_dim，代表哪些维度需要展平
默认只有第一维度不被展平，留下一个矩阵
'''

'''
func=nn.AvgPool2d(3)
X=torch.arange(108).reshape((1,3,6,6))
Z=torch.arange(108).reshape((3,6,6))
print(func(X).shape)
print(func(Z).shape)
nn.AvgPool2d也只接受3维度或4维度张量
'''

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

'''
测试用
X=torch.arange(28*28,dtype=torch.float32)
for layer in net:
    print(X.shape)
    X=layer(X)
print(X.shape)
'''

num_epochs,lr=10,0.9
updater=torch.optim.SGD(net.parameters(),lr=lr)
loss=nn.CrossEntropyLoss()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


if __name__=="__main__":
    for layer in net:
        init_weights(layer)
    train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)



