'''
这是自动求导函数的练习
'''

import torch

x=torch.arange(4.0,dtype=torch.float32)
x.requires_grad_(True)
'''
torch.tensor作为一个类型，默认不开辟空间来存储梯度值
而requires_grad_函数可以设置
'''

y=2*torch.dot(x,x)
y.backward()
'''
调用反向传播函数来求得导数，结果会自动保存到自变量x上
'''

print(x.grad)

x.grad.zero_()
'''
pytorch默认把梯度叠加，所以要调用zero_方法将数值清零，再进行下一次计算
'''

'''
反向传播函数只能对标量输出计算梯度
'''

y=x**2
y.sum().backward()
'''
可以把x向量的各个元素理解为不同的参数。如x1 x2 x3 x4等
分别求各个元素的偏导数，这样结果不变！
'''

print(x.grad)


x.grad.zero_()
y=x*x
u=y.detach()
'''
u将不再对x求导，也就是说du/dx被固定下来了！
'''
z=x*u
z.sum().backward()

print(x.grad)