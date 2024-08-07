'''
这是一个无框架实现线性模型的练习代码
'''

import random
import torch
from d2l import torch as d2l
import os

'''
需要使用torch模块和d2l模块为torch设计的组件
'''

'''
实现生成数据集的函数
'''
def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))
    '''
    normal函数用于从正态分布中生成随机数
    此处，均值为0，方差为1，生成的类型是二维的张量
    '''
    Y=torch.matmul(X,w)+b
    '''
    可以用于矩阵乘法的函数
    注意，每一组Xi为行向量，w为列向量
    '''
    Y+=torch.normal(0,0.01,Y.shape)
    '''
    Y.shape返回一个tuple 用于表示形状
    '''
    return X,Y.reshape((-1,1))
    '''
    注意，-1是reshape的形状缺省，可以用于代替任何值
    因此，最后返回一个Y的列向量
    '''
'''
实现一个迭代器函数，用于返回小批量数据，输入量为批量长度，特征矩阵和标签向量
注意，这是区别于定义迭代器类的另一种更为方便的迭代器定义方法。
只要用return代替yield！
'''
def data_iter(batch_size,features,labels):
    print("iterator started")
    #显然，这一部分只会运行一次！
    num_example = len(features)
    #len返回矩阵第一维的长度，即行数！
    indices=list(range(num_example))
    #range返回可迭代对象，可以转化为list
    random.shuffle(indices)
    #random模块的shuffle函数可以打乱作为参数传入的数组
    for i in range(0,num_example,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_example)])
        yield features[batch_indices],labels[batch_indices]
    '''
    这种迭代器在运行到结尾的时候会自动结束！
    '''

'''
定义线性回归模型
'''
def linreg(X,w,b):
    return torch.matmul(X,w)+b

'''
定义损失函数
'''
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2 / 2

'''
定义优化算法
lr为学习率
'''
def sgd(params,lr,batch_size):
    #这是一个上下文管理器，禁用了以下的梯度计算
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()


'''
生成数据集
'''
true_w=torch.tensor([2,-3.4,5.0])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

'''
打印数据集
'''

'''
d2l.set_figsize()
d2l.plt.scatter(features[:,0].detach().numpy(),labels.detach().numpy())
d2l.plt.show()
os.system("pause")
'''

batch_size=10

'''
测试数据迭代器
'''
'''
for x,y in data_iter(batch_size,features,labels):
    print(x,'\n',y)
    break
'''

'''
定义模型的初始化参数w b
'''
w=torch.normal(0,0.01,size=(3,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
#这里创建了一个大小为1的零张量

'''
设置学习的超参数
'''
lr=0.03
num_epochs=3 
#整个数据会被扫描3遍
net=linreg
loss=squared_loss
#python的函数也可以被赋值，类似C语言的函数指针

for epochs in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l= loss(net(features,w,b),labels)
        print(train_l.mean())

print(f'w:\n{w}')
print(f'b:\n{b}')





