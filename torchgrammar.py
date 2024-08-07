import torch
from torch import nn
'''
1-broadcasting机制
'''

arr1=torch.tensor([[1,2,3]])
arr2=torch.tensor([2,4,6,8]).reshape((4,-1))
#这里需要把arr2改成4*1的二维矩阵，可以用后面的规则理解

'''
arr1:
1
2
3

arr2:
2 4 6 8

arr1-broadcast
1 1 1 1
2 2 2 2
3 3 3 3

arr2-broadcast
2 4 6 8
2 4 6 8
2 4 6 8

the plus the two matrix

'''

result=arr1+arr2
#print(result.shape)
#print(result)
'''得到一个3*4的矩阵'''


'''
理解三维矩阵
'''
arr1=torch.tensor([[1,2,3],[4,5,6]])
#print(arr1.shape)
'''
可以写为 
1 2 3
4 5 6
'''

arr2=torch.tensor([[[0,1,2,3],[4,5,6,7],[8,9,10,11]]
                   ,[[12,13,14,15],[16,17,18,19],[20,21,22,23]]])
#print(arr2.shape)
'''
最里面的是第三维度，最外面的是第一维度
矩阵有四层
每一层都是x2y3的形状
如第一层
0 4 8
12 16 20
'''
#print(arr2[1][2][3])
# x=1 y=2 z=3 对应23

'''
下面给出broadcast机制得以进行的两个必要条件
'''

'''
情况一：张量末尾维度的长度相等
'''

arr1=torch.tensor([[[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]],[[7,8,9],[7,8,9]]])
arr2=torch.tensor([1,2,3])

result=arr1+arr2
#print(result.shape)
#print(result)

'''
arr1可以理解为3层的矩阵，每一层为3*2 x3y2 注意x沿着上下衍生，y按着左右衍生

arr1-0
1 1
4 4
7 7

arr1-1
2 2
5 5
8 8

arr1-2
3 3 
6 6
9 9

对arr2进行广播
先将其变为2*3的矩阵
1 2 3
1 2 3

再将其变为3*2*3的张量
原先的第二维度变为第三维度，原先的第一维度变为第二维度
对于第一维度，直接复制！

arr2-broadcast-0
1 1
1 1
1 1

arr2-broadcast-1
2 2
2 2
2 2

arr2-broadcast-2
3 3
3 3
3 3

然后，对应相加即可
'''


'''
第二种：末尾维度长度为1
'''

'''
如果末尾维度为1，直接对矩阵进行复制即可
'''


'''
如，两个张量形状分别为3*2*4和1*4
怎么判断两者在广播机制下能否进行运算呢？
1、写成3*2*4和0*1*4
2、从末尾起开始比较各个维度
第一维：4和4 允许
第二维：1和2 允许
3、直至某个张量比较完毕，如比较到倒数第三维度，后面一个张量的维度已经比较完了
现在可以判断两者可以被广播
'''






'''
2-sum函数的性质和keepdim参数的作用
'''

arr1=torch.tensor([[1,2,3],[4,5,6]])
arr2=arr1.sum(1)
#print(arr2.shape)
#print(arr2)

'''
沿着第1维度求和，直接从2*3被降维为2长度的一维向量
'''

arr2=arr1.sum(1,keepdim=True)
#print(arr2.shape)
#print(arr2)

'''
沿着第1维度求和，从2*3变为2*1
虽然形状改变但是维度不变
'''


'''
3-argmax函数
'''

'''
返回最大值索引
'''

arr=torch.tensor([[1,5,3],[4,2,6]])
#print(arr.argmax(axis=0))
#print(arr.argmax(dim=1))
'''
axis或者dim设置为0 返回每一列最大值的行索引
设置为1，返回每一列最大值的列索引（第二维度索引）
'''


'''
4-可变参数
'''

def myfunction(*args):
    sum=0.0
    for arg in args:
        sum+=float(arg)
    return sum

#print(myfunction(3,5,6))

'''
*会把所有输入的参数打包为一个元组
'''


'''
5-assert语句
'''

def assertfunction(num:int):
    assert num>10
    '''
    断言语句
    意思是对num>10进行判断，否则抛出异常
    '''
    assert num<=10,num+10
    '''
    后面可以给定异常值
    '''

#print(assertfunction(15))  


'''
6-参数绑定
'''
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),shared, nn.ReLU(),
                    shared, nn.ReLU(),nn.Linear(8, 1))

'''
第二层和第四层的参数始终一致
'''


'''
7-tensor的拓展
'''
X=torch.arange(9.0).reshape((3,3))
Y=X.reshape((1,1)+X.shape)
#print(Y.shape)
'''变为了1*1*3*3的矩阵'''

'''
8-stack和cat操作
'''
X=torch.arange(4.0).reshape(2,2)
Y=torch.stack((X,X),0)
# print(Y)
# print(Y[0])
# print(Y.shape)
Y=torch.stack((X,X),2)
# print(Y)
# print(Y[0])
# print(Y.shape)
'''
进行堆叠，第一个参数是待堆叠的元组，第二个参数是在哪个维度进行堆叠
具体用树状图理解

注意stack会增加维度
'''

'''
cat函数的调用方式与stack一模一样，但cat不会增加维度
'''




'''
9-  max和mean
'''
X=torch.arange(8.0).reshape((2,4))
print(X.max())
X.reshape((2,2,2))
print(X.mean())
print(X.max(dim=0))
'''不加参数 就返回一个长度为1的张量'''