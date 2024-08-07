'''这是学习卷积核概念的练习代码'''

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
            '''
            X[1:3,2:4]为切片操作
            '''
    return Y

'''corr2d的测试'''
# X=torch.arange(9.0).reshape((3,3))
# K=torch.arange(4.0).reshape((2,2))
# print(corr2d(X,K))

class Conv2d(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return corr2d(x,self.weight)+self.bias

X = torch.ones((6,8))
X[:,2:6]=0
#print(X)

'''边缘检测'''  
K = torch.tensor([[1.0,-1.0]])
Y=corr2d(X,K)
#print(Y)

conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
'''参数一和二分别是输入通道数量和输出通道数量'''
'''卷积核默认对参数W和b开启梯度运算'''

X=X.reshape((1,1,6,8))
Y=Y.reshape((1,1,6,7))
'''第一维：通道数量    第二维度：批量大小维度'''

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat-Y)**2
    conv2d.zero_grad()
    '''可以直接把nn.Module的所有参数的梯度置为0'''
    l.sum().backward()
    '''梯度下降函数'''
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    '''grad属性即梯度'''
    print(f'batch:{i+1} loss:{l.sum():.3f}')

print(conv2d(X).data.reshape((6,7)))
print(conv2d.weight.data.reshape((1,2)))


