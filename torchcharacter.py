import torch 

x=torch.arange(12)
y=x.reshape((3,4))
#print(x)
#print(y)
y[0][0]=-1
#print(x)
#print(y)

'''
x和y是共享一片内存的
它们更加像不同类型的指针，用不同方式解析同一块内存下存着的数据
'''


'''
tensor的索引方法--索引里是一个列表
'''
x[0]=0
indexs=[2,3,5]
#print(x[indexs])

'''
x是一个tensor向量，如果[]中的值是一个list，则返回tensor张量中对应位置的数据
'''

indexs_tensor=torch.tensor(indexs)
#print(x[indexs_tensor])

'''
x是tensor类型的，也有类似的操作
'''

'''
tensor的索引方法--索引为[list,z]
'''
z=torch.tensor([0,2])
z_hat=torch.tensor([[1,2,3],[4,5,6]])
# print(z_hat[[0,1],z])
# print(z_hat[[0,0],[1,2]])

'''
从z_hat中拿取下标为[0][0] [1][2]
'''
'''
实际上是多个数组构成下标
'''

arr=torch.tensor([[[0,1,2,3],[4,5,6,7],[8,9,10,11]]
                ,[[12,13,14,15],[16,17,18,19],[20,21,22,23]]])
# print(arr[[0,0],[1,1],[2,1]])
'''
依次拿出两个元素作为向量
arr[0][1][2] arr[0][1][1]
'''


'''
这是一种python的逆天语法，需要注意！
'''
for i in x:
    i=i+3
#print(x)

for i in x:
    i+=3
#print(x)

def trans_one(x):
    x-=3

def trans_two(x):
    x=x-3

trans_one(x)
#print(x)

trans_two(x)
#print(x)


'''
索引方式
'''
a=torch.tensor([1,2,3,4])
a[:]+=1
print(a)

'''对tensor张量使用[:] 意味着所有元素执行同样的操作'''
