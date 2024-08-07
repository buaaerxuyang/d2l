'''
这是自己利用torch和d2l手写的softmax模型
'''

import torch
from d2l import torch as d2l

'''
数据迭代器
'''
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
'''
训练迭代器一次迭代给出两个结果
256张图片 256个下标（独热向量1的下标）
'''

'''
给出待训练的权重数组W和偏置数组b
'''
num_inputs=784
num_outputs=10

W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)


'''
给出神经网络的计算方法
'''
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition
'''
X m*n
m为图片数量，n为类别数量
'''

def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

'''给出损失函数'''
def loss(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

'''
注意，y并不是独热向量，而是独热向量的下标
'''

'''优化函数'''
lr=0.1
def updater(batch_size):
    return d2l.sgd([W,b],lr,batch_size)

def main():
    '''训练'''
    num_epoch=10
    for epoch in range(num_epoch):
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y)
            l.sum().backward()
            updater(X.shape[0])
    
    '''测试'''
    X,y=next(iter(test_iter))
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    for i,(true,pred) in enumerate(zip(trues,preds)):
        print(f'{i}:'+true+' '+pred)


if __name__=="__main__":
    main()

        
    
    
