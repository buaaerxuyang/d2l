'''这是学习池化层的练习代码'''
import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X,pool_size,mode='max'):
    p_h,p_w=pool_size
    Y=torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mode=='max':
                Y[i][j]=X[i:i+p_h,j:j+p_w].max()
            else:
                Y[i][j]=X[i:i+p_h,j:j+p_w].mean()
    '''tensor有max和mean函数，可以返回一个数'''

'''使用torch自带的池化层模型'''
X=torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
pool=nn.MaxPool2d(3)
'''这里确定池化层大小，池化层步幅默认为其大小，即默认扫过的两区域不重叠'''
Y=pool(X)
#print(Y)
'''所以只输入了10'''

pool=nn.MaxPool2d((2,3),stride=(2,3),padding=(1,1))
'''当然也可以手动设定哈'''

X=torch.cat((X,X+1),1)
#print(X.shape)
'''cat拼接函数不改变维度'''
pool=nn.MaxPool2d(3,padding=1,stride=2)
print(pool(X))
'''nn自带的池化函数在每一个输入渠道里单独计算'''