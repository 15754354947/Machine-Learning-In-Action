from numpy import *
from time import sleep
import json
import urllib.request as request
import matplotlib.pyplot as plt

def loaddata(filename):
    numfeat = len(open(filename).readline().strip().split("\t"))-1
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = []
        allfeat = line.strip().split("\t")
        for i in range(numfeat):
            linearr.append(float(allfeat[i]))
        datamat.append(linearr)
        labelmat.append(float(allfeat[-1]))
    return datamat,labelmat

'''
数据标准化，特征数据操作为减去均值，除以方差，标签数据只减去均值
'''
def regularize(xarr,yarr):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    xmean = mean(xmat,0)
    xvar = var(xmat,0)
    xmat = (xmat-xmean)/xvar
    ymean = mean(ymat,0)
    ymat = ymat-ymean
    return xmat,ymat

'''
对数据集求解ws
输入：数据集的特征数据集xarr，标签数据集yarr
输出：经过计算得出的ws
'''


def standregres(xarr,yarr):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    xTx = xmat.T*xmat
    if linalg.det(xTx)==0:         #det求矩阵的行列式，如果为零，矩阵不存在逆矩阵，也就不能计算对应的ws
        print("NO FUNCTION!")
        return
    ws = xTx.I * (xmat.T * ymat)               #根据计算公式，计算出ws
    return ws

# datamat,labelmat = loaddata("D://MLInAction//ch8//ex0.txt")
# ws = standregres(datamat,labelmat)
# xlist = mat(datamat)
# ylist = mat(labelmat)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xlist[:,1],ylist.T[:,0])
# sortedxlist = xlist.copy()
# sortedxlist.sort(0)
# y_predict = sortedxlist * ws
# ax.scatter(sortedxlist[:,1],y_predict)
# plt.show()
# print('相关系数矩阵：\n',corrcoef(y_predict.T,ylist))
# #相关系数矩阵一共行数为所求两个矩阵行数之和，将两个矩阵上下放置组成一个大矩阵，一起计算行数，相关系数矩阵中第m行第n列的数据表示大矩阵的第m行和第n行数据的相关性

'''
局部权重线性回归，对于每一个测试数据，使用核方法来对附近的点赋予更高的权重
输入：testpoint赋权重的样本数据,xarr整体特征数据集,yarr整体标签数据集,k = 1.0核方法参数
输出：对于testpoint样本的预测值
'''

def LWLR(testpoint,xarr,yarr,k = 1.0):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    m = shape(xmat)[0]
    weight = mat(eye((m)))                        #权重矩阵，m为数据集数
    for j in range(m):                             #每一样本都赋权重
        diffmat = testpoint - xmat[j,:]
        weight[j,j] = exp(diffmat * diffmat.T/(-2*k**2))             #计算权重的公式
    xtx = xmat.T * (weight * xmat)
    if linalg.det(xtx)==0:
        print("NO FUNCTION!")
        return
    ws = xtx.I * (xmat.T * (weight * ymat))                 #利用权重计算回归系数
    return testpoint * ws

'''
对每一个样本数据使用上一个函数进行预测
'''

def lwlrtest(testarr,xarr,yarr,k=1.0):
    m = shape(testarr)[0]
    y_predict = zeros(m)
    for i in range(m):
        y_predict[i] = LWLR(testarr[i],xarr,yarr,k)
    return y_predict

# datamat,labelmat = loaddata("D://MLInAction//ch8//ex0.txt")
# xmat = mat(datamat)
# ymat = mat(labelmat)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xmat[:, 1], ymat.T[:, 0])
# sortedxlist = xmat.copy()
# sortedxlist.sort(0)
# yhat = lwlrtest(sortedxlist,datamat,labelmat,0.003)
# ax.scatter(sortedxlist[:,1],yhat)
# plt.show()

# 计算平方误差
def rsserror(yarr,yhat):
    return ((yarr-yhat)**2).sum()

# abx,aby = loaddata("D://MLInAction//ch8//abalone.txt")
#利用局部加权线性回归的平方误差
# yhat01 = lwlrtest(abx[0:99],abx[0:99],aby[0:99],0.1)
# error = rsserror(aby[0:99],yhat01.T)
#简单的线性回归误差
# ws = standregres(abx[0:99],aby[0:99])
# y_predict = abx[100:199] * ws
# error = rsserror(aby[100:199],y_predict.T.A)

'''
岭回归计算回归系数，lam为岭回归计算参数
'''

def ridgeregres(xmat,ymat,lam = 0.2):
    xtx = xmat.T * xmat
    denom = xtx + eye(shape(xmat)[1])*lam
    if linalg.det(denom) ==0:
        print("NO FUNCTION!")
        return
    ws = denom.I * (xmat.T * ymat)
    return ws


'''
多次循环计算得出多组回归系数，每一次的岭回归计算参数不同
'''
def ridgetest(xarr,yarr):
    xmat,ymat = regularize(xarr,yarr)
    numtestpts = 30
    wmat = zeros((numtestpts,shape(xmat)[1]))
    for i in range(numtestpts):
        ws = ridgeregres(xmat,ymat,exp(i-10))
        wmat[i,:] = ws.T
    return wmat

# abx,aby = loaddata("D://MLInAction//ch8//abalone.txt")
# wmat = ridgetest(abx,aby)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(wmat)
# plt.xlim(-10,30)
# plt.show()

'''
前向逐步回归，eps每一步的步长,numit迭代次数
'''

def stagewise(xarr,yarr,eps=0.01,numit = 100):
    xmat,ymat = regularize(xarr,yarr)
    m,n = shape(xmat)
    returnmat = zeros((numit,n))
    ws = zeros((n,1))
    wsmax = ws.copy()
    for i in range(numit):             #循环迭代次数
        lowerror = inf                          #初始最小错误率设置为无穷大
        for j in range(n):                      #对每一个特征值都循环
            for sign in [-1,1]:                      #对特征可增加可减小一个步长
                wstest = ws.copy()
                wstest[j] += eps*sign
                ytest = xmat * wstest
                rsse = rsserror(ymat.A,ytest.A)            #计算此次的平方误差
                if rsse <lowerror:                 #如果为目前最小：保存平方误差
                    lowerror = rsse
                    wsmax = wstest                    #保存此次回归系数
        ws = wsmax.copy()
        returnmat[i,:] = ws.T                #将每一次迭代所得的回归系数添加矩阵中
    return returnmat

# abx,aby = loaddata("D://MLInAction//ch8//abalone.txt")
# xmat,ymat = regularize(abx,aby)
# wmat = stagewise(abx,aby,0.001,5000)
# print(wmat)


