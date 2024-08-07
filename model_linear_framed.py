'''
这是调用pytorch已有线性回归模型的练习代码
'''

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

from torch import nn
#引入神经网络

'''
确定标准参数，并调用d2l库自带的数据集生成函数来生成特征矩阵和标签向量
特征矩阵是由1000个行向量组成的，每个行向量的长度是n。
标签向量是一个一维向量，长度为1000
'''
true_w=torch.tensor([2,-3.4,5.0])
true_b=4.2
feathers,labels=d2l.synthetic_data(true_w,true_b,1000)

'''
构造一个数据迭代器，事实上是继承data库中的迭代器
'''
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    '''
    需要使用data库定义的类型，否则无法正常工作
    使用*可以实现数据的解包
    '''
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
    '''
    返回data库自带的迭代器，有自动打乱的功能
    '''

'''
获取数据的准备
'''
batch_size=10
data_iter=load_array((feathers,labels),batch_size)
'''
feathers和labels被作为元组传入后解包
'''

'''
for single in data_iter:
    print(single)
'''
'''
发现每次迭代出的结果是一个列表，且这种迭代器在遍历一遍后会自动停止！
'''



'''
使用事先预定好的框架
'''
net=nn.Sequential(nn.Linear(3,1))
'''
对于线性神经网络，3代表着权重向量的长度，1代表输出值的长度
Sequential可以理解为算式模板类
'''

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
'''
将网络第0层的权重和偏重进行初始值的设定
'''

loss=nn.MSELoss()
'''
平方损失函数已经被封装在MSELoss类中，直接调用
其计算的是均方误差！！！
'''

trainer = torch.optim.SGD(net.parameters(),lr=0.03)
'''
实例化梯度下降算法类，参数为net参数迭代器和学习率
'''

num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        #注意MSELoss的调用方式
        '''
        这里的y是真实的y 而net(X)可以理解为用当前的X数据和网络中的w b得到的y
        '''
        trainer.zero_grad()
        #梯度清零
        l.backward()
        #计算梯度
        trainer.step()
        #梯度下降
        '''
        trainer作为算法类，已经携带了w b等参数的引用，此处进行w b参数的更新
        '''
        
    l=loss(net(feathers),labels)
    '''
    用标准的数据检验拟合得到的参数是否靠谱
    '''
    print(l)

print(net[0].weight.data)

        
