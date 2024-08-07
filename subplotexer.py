'''这是subplot的练习代码'''

import matplotlib.pyplot as plt
import numpy as np

def subplotexer0():
    # 创建一个简单的数据集
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 第一个子图
    plt.subplot(2, 1, 1)  # 2行1列，当前是第1个子图
    plt.plot(x, y1)
    plt.title('sin')

    # 第二个子图
    plt.subplot(2, 1, 2)  # 仍然是2行1列，但现在是第2个子图
    plt.plot(x, y2)
    plt.title('cos')

    # 显示图表
    plt.tight_layout()  # 自动调整子图布局，避免重叠
    plt.show()

def subplotexer1():
    # 创建一个简单的数据集
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, axs = plt.subplots(2, 1, sharex=True)
    '''
    sharex参数设置为True可以共享x轴坐标
    返回axs 为子图数组 便于后续操作
    '''
    axs[0].plot(x,y1)
    axs[1].plot(x,y2)

    axs[0].set_xlabel('aixs-x')
    axs[1].set_xlabel('aixs-x')
    '''set_xlabel set_ylabel 方法可以给子图设置标签'''

    axs[0].set_title('sin')
    axs[1].set_title('cos')
    '''set_title方法设置子图标题'''

    plt.tight_layout()
    plt.show()

def subplotexer2():
    # 创建一个简单的数据集
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(x,y1,color='r',label='sin')
    axs[0].legend()
    axs[1].plot(x,y2,color='b',label='cos')
    axs[1].legend()
    '''传入label参数可以规定线的名称，在图例中得以显示'''
    '''对于子图，需要对子图句柄对象单独使用legend方法'''
    #plt.legend(loc='lower left')
    '''使用图例，loc参数规定图例的位置'''


    plt.tight_layout()
    plt.show()


def main():
    subplotexer2()

if __name__=='__main__':
    main()