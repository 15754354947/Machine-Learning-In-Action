from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir
import time

start = time.clock()

#当中是你的程序



def createDataSet():
    group = array([
        [1,1.1],
        [1,1],
        [0,0],
        [0,0.1]
    ])
    labels = ['A','B','C','D']
    return group,labels

"""
初阶KNN算法
输入：输入向量inx，样本数据dataset，样本类别labels，使用的K值k
输出：KNN算法计算预测输入向量的类别
"""
def classify0(inx,dataset,labels,k):
    datasetsize = dataset.shape[0]                               # shape返回数组或者矩阵的行，列等各维上的数字
    diffmat = tile(inx, (datasetsize, 1)) - dataset               # 将输入向量按列数不变，行数复制到与样本总数一样后，之后能计算每一个样本与输入向量的差
    spdiffmat = diffmat ** 2                                      #差值进行平方
    spdistences = spdiffmat.sum(axis=1)                          #将上一步的数值按照行，进行求和，
    distences = spdistences ** 0.5                              #数值开方即每个样本与输入向量的距离
    sorteddistences = distences.argsort()                        #将距离进行排序，返回的数组是索引值
    classcount = {}                                              #类别统计的字典
    for i in range(k):                                           #只统计k个最近的样本点
        voteilabel = labels[sorteddistences[i]]                  #从距离最短开始进行统计
        classcount[voteilabel] = classcount.get(voteilabel, 0) + 1             #在类别字典中查找是否有这一次类别的value，没有默认为0，有则+1
    sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)     #将字典按照各类别的value进行降序排序，得到的是数字
    return sortedclasscount[0][0]                                #返回数字中第一个子项的类


def file2matrix(filename):
    fr = open(filename)          #打开文件
    arrayolines = fr.readlines()             #逐行读取，返回数组
    numberoflines = len(arrayolines)                 #数据长度
    returnmat = zeros((numberoflines,3))                #生成对应大小的全零数组，存放属性值
    classlabelvector = []                                #定义存放所有数据类别的数组
    index = 0                                            #从第一个开始索引
    for line in arrayolines:                             #每一行数据
        line = line.strip()                               #移除头尾的空格、换行符
        listfromline = line.split('\t')                   #按照空格将数据分割
        returnmat[index,:]= listfromline[0:3]             #数据前三项为属性值，保存在属性数组中
        classlabelvector.append(listfromline[-1])         #最后一项是类别，保存中类别数组中
        index+=1                                          #索引后移一位

    '''以下代码为把字母型特征转成int型'''
    global set_class
    set_class = set(classlabelvector)
    all_class = []
    for oneclass in set_class:
        all_class.append(oneclass)
    for index in range(len(classlabelvector)):
        temp = 0
        for j in all_class:
            temp += 1
            if classlabelvector[index] == j:
                classlabelvector[index] = temp
    return returnmat,classlabelvector
'''
datingDataMat, datingLabels = file2matrix('D:\MLInAction\ch2\datingTestSet.txt')
fig = plt.figure('figure1')
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))  # 绘制散点图，注意，是第2列，第3列数据
plt.show()
'''


'''
将属性值中所有数值转成0-1中数字
转换公式:(当前属性值-当前属性所有样本的最小值)/(当前属性所有样本的最大值-当前属性所有样本的最小值)
'''
def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals-minvals
    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset - tile(minvals,(m,1))
    normdataset = normdataset/tile(ranges,(m,1))
    return normdataset,ranges,minvals

"""
可以根据交叉验证法进行测试,测试集占比horatio,从数据集头的horatio数据用来测试,其他数据用于训练
"""
def datingclasstest():
    horatio = 0.1
    datingDataMat, datingLabels = file2matrix('D:\MLInAction\ch2\datingTestSet.txt')
    normmat,ranges,minvals = autonorm(datingDataMat)
    m = normmat.shape[0]
    numtestvecs = int(horatio*m)
    errorcount = 0.0
    for i in range(numtestvecs):
        # 前numTestVecs条作为测试集（一个一个测试），后面的数据作为训练样本，训练样本的标签，3个近邻
        classifierResult = classify0(normmat[i, :], normmat[numtestvecs:m, :], datingLabels[numtestvecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorcount += 1.0
    print("The number of errr is: %d" % int(errorcount))
    print("The total error rate is: %f" % (errorcount / float(numtestvecs)))

# datingclasstest()

#将TXT中的32*32数据转成1*1024的向量,便于KNN计算
def img2vector(filename):
    returnvect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j] = int(linestr[j])
    return returnvect


def handwritingcalsstest():
    hwlabels = []
    trainingfilelist = listdir('D://MLInAction//ch2//trainingDigits')
    m = len(trainingfilelist)
    trainingmat = zeros((m,1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i,:] = img2vector('D://MLInAction//ch2//trainingDigits//%s'%filenamestr)
    #测试集数据
    testfilelist = listdir('D://MLInAction//ch2//testDigits')
    errorcount = 0
    mtest = len(testfilelist)
    for i in range(mtest):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        vectorundertest = img2vector('D://MLInAction//ch2//testDigits//%s'%filenamestr)
        classifierResult = classify0(vectorundertest, trainingmat, hwlabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classnumstr))
        if (classifierResult != classnumstr): errorcount += 1.0
    print("\nthe total number of errors is: %d" % errorcount)
    print("the total error rate is: %f" % (errorcount / float(mtest)))


handwritingcalsstest()
elapsed = (time.clock() - start)
print("Time used:",elapsed)