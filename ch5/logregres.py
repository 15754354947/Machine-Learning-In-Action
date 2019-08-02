import math
from numpy import *


'''
读取100个点的数据
返回点坐标和类
'''
def loaddataset():
    datamat = []
    labelmat = []
    fr = open("D://MLInAction//ch5//testSet.txt")
    for line in fr.readlines():
        linearr = line.strip().split()
        datamat.append([1.0,float(linearr[0]),float(linearr[1])])
        labelmat.append(int(linearr[2]))
    return datamat,labelmat

#sigmoid函数
def sigmoid(inx):
    return 1.0/(1.0+exp(-inx))

'''
三种计算w的方法：
假设迭代次数round，数据集大小size
一、自定迭代次数，每一次遍历整个数据集后更新w，迭代所有次数后，返回w，一共更新round次
二、每一次根据数据集中的一组数据进行更新w，遍历完整个数据集一次，返回w，一共更新size次
三、自定迭代次数，每次迭代，遍历整个数据集，但每一次是随机的根据一组数据进行更新w，直到数据集全部访问后，才进行下一次迭代，一共更新round*size次
'''
def gradascent(datasetin,classlabels):
    datamatrix = mat(datasetin)
    classmat = mat(classlabels).transpose()
    m,n = shape(datamatrix)
    alpha = 0.001
    maxcycles = 500
    weights = ones((n,1))
    for k in range(maxcycles):
        h = sigmoid(datamatrix*weights)
        error = classmat - h
        weights = weights + alpha * datamatrix.transpose() * error
    return weights

def stocgradascent(datamatrix,classlabels):
    m,n = shape(datamatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(datamatrix[i]*weights))
        error = classlabels[i] - h
        weights = weights + error * alpha * datamatrix[i]
    return weights


def stocgradascent1(datamatrix,classlabels,numitem = 150):
    m,n = shape(datamatrix)
    weights = ones(n)
    for j in range(numitem):
        dataindex = list(range(m))
        for i in range(m):
            alpha = 4/(1+i+j)+0.1
            randindex = int(random.uniform(0,len(dataindex)))
            h = sigmoid(sum(datamatrix[randindex]*weights))
            error = classlabels[randindex] - h
            weights = weights + error * alpha * datamatrix[randindex]
            del(dataindex[randindex])
    return weights

'''
根据计算的w，在图像上表示出决策边界
'''

def plotbestfit(weights):
    import matplotlib.pyplot as plt
    datamat,lablemat = loaddataset()
    dataarr = array(datamat)
    n = shape(dataarr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(lablemat[i])==1:
            xcord1.append(dataarr[i,1])
            ycord1.append(dataarr[i,2])
        else:
            xcord2.append(dataarr[i, 1])
            ycord2.append(dataarr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = 'red',marker = 's')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()




def classifyvector(inx,weight):
    prob = sigmoid(sum(inx*weight))
    if prob>0.5:
        return 1
    else:
        return 0

def colictest():
    frtrain = open("D://MLInAction//ch5//horseColicTraining.txt")
    frtest = open("D://MLInAction//ch5//horseColicTest.txt")
    trainingset = []
    traininglabels = []
    for line in frtrain.readlines():
        currline = line.strip().split("\t")
        linearr = []
        for i in range(21):
            linearr.append(float(currline[i]))
        trainingset.append(linearr)
        traininglabels.append(float(currline[21]))
    trainweights = stocgradascent1(array(trainingset),traininglabels,500)
    errorcount = 0.0
    numtestvec = 0
    for line in frtest.readlines():
        numtestvec +=1
        currline = line.strip().split("\t")
        linearr = []
        for i in range(21):
            linearr.append(float(currline[i]))
        if int(classifyvector(array(linearr),trainweights))!=int(currline[21]):
            errorcount+=1
    errorrate = (float(errorcount)/numtestvec)
    print("此次测试错误率",errorrate)
    return errorrate

def multitest(numtests):
    errorsum = 0
    for i in range(numtests):
        errorsum +=colictest()
    errorsum = errorsum/numtests
    print(numtests,"次测试平均错误率",errorsum)

multitest(10)