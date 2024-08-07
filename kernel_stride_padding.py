'''这是学习卷积层填充和步幅的练习代码'''
import torch
from torch import nn
from d2l import torch as d2l

'''
1-卷积层会减小张量的规模，在多卷积层的运算中，
为了防止张量卷着卷着卷没了，引入填充概念，
其实质就是在输入中添加额外的行和列
'''

'''
2-大尺度图片（如1000*1000） 在小的卷积核的作用（如5*5）
下需要经过很多轮的迭代运算才能获得较小的输出

卷积核在图片上可以一次移动较大距离，这样可以显著减少计算量
'''

def comp_conv2d(conv2d,X):
    '''输入量为2维的卷积计算'''
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1)
'''
kernel_size被设置为3时，意味着卷积核的规模为3*3
padding意味着填充大小，两边同时填充，实际上padding=1时会把输入矩阵大小+2
stride参数可以设置步幅
'''
X=torch.rand(8,8)
Y=comp_conv2d(conv2d,X)
#print(Y.shape)

conv2d=nn.Conv2d(1,1,kernel_size=(5,3),padding=(2,1))
'''填充参数也可以是一个元组'''
'''(5-1)/2=2 (3-1)/2=1'''
Y=comp_conv2d(conv2d,X)
#print(Y.shape)

