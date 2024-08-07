'''
这是使用图像分类测试集的练习代码
'''

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
#使用svg来显示图片

trans=transforms.ToTensor()
#准备一个函数，引用transforms库中的ToTensor函数

mnist_train = torchvision.datasets.FashionMNIST(
    root="../data",train=True,transform=trans,download=True
)
'''
创建FashionMINST数据集类，作为训练数据集的句柄，指定转化函数，需要下载资源
'''

mnist_test = torchvision.datasets.FashionMNIST(
    root="../data",train=False,transform=trans,download=True
)
'''
创建FashionMINST数据集类，作为测试数据集的句柄，指定转化函数，需要下载资源
'''

'''
print(len(mnist_train))
print(len(mnist_test))
'''

print(mnist_train[0][0].shape)
#训练数据集第0个样例第0个图片

def get_fashion_mnist_labels(lables):
    text_labels=[
        'T-shirt','trouser','pullover','dress','coat','sandal','shirt',
        'sneaker','bag','ankle boot'
    ]
    return [text_labels[int(i)] for i in lables]

def show_image(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    '''
    创建了一个有多个子图的图像，返回_容器和axes子图数组，这个数组是二维数组，每个元素是一张图片
    '''
    axes=axes.flatten()
    '''
    展平子图数组
    '''
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.set_title(titles[i])

    '''
    zip()将多个可迭代对象打包为元组
    '''

    '''
    enumerate()可以为可迭代对象加上索引
    本来，每一迭代的是(ax,img)这一个元组，
    现在，第一个元素是index，第二个元素才是元组
    '''

    d2l.plt.show()
    return axes


X,y=next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_image(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))

batch_size = 256

def get_dataloader_workers(): 
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

#整合版本
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    '''
    trans作为一个数组/列表，储存所有对图片处理的操作
    '''
    if resize:
        trans.insert(0, transforms.Resize(resize))
    '''
    如果需要重新设置大小，则把该操作插入到列表第一个
    '''
    trans = transforms.Compose(trans)
    '''
    新的trans可以完成所有操作
    '''
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))