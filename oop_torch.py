import torch
from torch import nn
from torch.nn import functional as func

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __call__(self, idx):
        return self.data[idx]
    
test=Accumulator(3)
test.add(0,1,2)
#print(test.__call__(2))    #打印2.0
#print(test(2))             #打印2.0


class MulLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.output=nn.Linear(256,10)

    def forward(self,X):
        return self.output(func.relu(self.hidden(X)))
    
# net=MulLayerPerceptron()
# X=torch.arange(20.0)
# print(net(X))

# net=nn.Sequential(nn.Linear(20,256),nn.Linear(256,10))
# print(net[0].state_dict())
# print(net[0].weight.data[0])
# net[0].weight.requires_grad=True


class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        self.layers=[]
        for layer in args:
            self.layers.append(layer)
    
    def forward(self,X):
        for layer in self.layers:
            X=layer(X)
        return X

# net=MySequential(nn.Linear(20,256),nn.Linear(256,10))
# X=torch.arange(20.0)
# print(net(X))

block=nn.Sequential(nn.Linear(20,256),nn.Linear(256,10))
net=nn.Sequential(block,nn.Linear(10,5))
X=torch.arange(20.0)

def init_normal(m):
    #print(m)
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,mean=0.0,std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)


#print(*[(name,param.shape) for name,param in net.named_parameters()])
'''
net.named_parameters()返回一个迭代器，一次返回一个元组
元组第一个元素是参数的名称，第二个元素是参数的张量本身

*可以将元组/列表解包
'''
'''
print([(name,param.shape) for name,param in net.named_parameters()])会打印一个列表
'''