'''这是plot的练习代码'''

import matplotlib.pyplot as plt
import numpy as np

'''
plot函数可以绘制散点图
'''

def plotexer0(type):
    y_points=np.arange(10)
    if type==0:
        plt.plot(y_points,marker='o')
    else:
        plt.plot(y_points,marker='*')
    plt.show()
    '''marker参数可以定义标记形状'''

def plotexer1():
    '''实用了fmt参数'''
    y_points=np.array([1,4,2,8,5,7])
    plt.plot(y_points,'o:r')
    '''
    o 代表标记
    : 代表虚线
    r 代表颜色
    '''
    plt.show()

def plotexer2():
    y_points=np.array([1,4,2,8,5,7])
    plt.plot(y_points,marker='o',ms=10,mfc='r',mec='b')
    '''
    ms 标记大小
    mfc 标记内部颜色
    mec 标记边框颜色i
    '''
    plt.show()

def plotexer3():
    y_points=np.array([1,4,2,8,5,7])
    plt.plot(y_points,marker='o',ls='--',color='g',lw=1)
    '''
    ls参数规定直线类型
    - 直线
    : 虚线
    -- 破折线

    color参数用于表示直线的颜色

    lw参数表示宽度
    '''
    plt.show()

def plotexer4():
    '''多条线的绘制方法'''
    y1_points=np.array([1,4,2,8,5,7])
    y2_points=np.array([2,8,5,7,1,4])
    plt.plot(y1_points,marker='o',ls='-',color='g',lw=1,ms=1)
    plt.plot(y2_points,marker='o',ls='-',color='r',lw=1,ms=1)
    plt.show()

def plotexer5():
    '''另一种绘制多条线的方法'''
    x1=np.array([0.0,1.0,2.0,4.0])
    x2=np.array([1.0,2.0,3.0,5.0])
    y1=np.array([1,4,2,8])
    y2=np.array([5,7,3,6])    
    plt.plot(x1,y1,x2,y2)
    '''注意x y的排列方式'''
    plt.show()

def plotexer6():
    y_points=np.array([1,4,2,8,5,7])
    plt.plot(y_points,marker='o',ls='--',color='g',lw=1)

    plt.xlabel('x-label')
    plt.ylabel('y-label')
    plt.title('test')

    plt.show()

if __name__=="__main__":
    #plotexer0(1)
    #plotexer1()
    #plotexer2()
    #plotexer3()
    #plotexer4()
    #plotexer5()
    plotexer6()