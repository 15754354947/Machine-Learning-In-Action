from math import log
import operator
from ch3 import treePlotter

'''将非list转为list'''
def tolist(notlist):
    islist = []
    for i in notlist:
        islist.append(i)
    return islist

'''
计算样本集的熵，prob各个类别所占比例，shannonent等于每个类别的prob*log（prob，2）之和的负数
'''
def calcshannonent(dataset):
    numentries = len(dataset)   #样本集数量
    labelcounts = {}           #类别数统计
    for featvec in dataset:        #每一个数据集取最后一项，即类别，进行数量统计
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] +=1
        shannonent = 0
        for key in labelcounts:
            prob = float(labelcounts[key])/numentries             #计算每个类别占比
            shannonent -=prob * log(prob,2)             #将所有类别的计算值进行相减，免去取负操作
    return shannonent

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# dataSet, labels = createDataSet()
# # dataSet[0][-1] = 'maybe'   # 把第一行的最后一个分类类别换为'maybe'，则熵会变大。熵越大，混合的数据也越多，数据越无序
# print(dataSet)
# shannonEnt = calcshannonent(dataSet)
# print(shannonEnt)

'''
划分数据集，dataset需要划分的数据集,axis依照第几个属性进行划分,取出属性为value的样本集
'''
def splitdataset(dataset,axis,value):
    retdataset = []
    for featvec in dataset:
        if featvec[axis]  ==value:
            reducedfeatvec = featvec[:axis]          #新样本集的每一组数据划分属性前的其他属性值不变
            reducedfeatvec.extend(featvec[axis+1:])      #之后的属性值也不变
            retdataset.append(reducedfeatvec)
    return retdataset

'''
选择一个数据集中划分的信息增益最大的属性
'''
def choosebestfeaturetosplit(dataset):
    numfeatures = len(dataset[0])-1          #一个样本的属性个数
    baseentropy = calcshannonent(dataset)
    bestinfogain = 0
    bestfeature = -1
    for i in range(numfeatures):
        featlist = [example[i] for example in dataset]
        uniquevals = set(featlist)
        newmentropy = 0
        for value in uniquevals:
            subdataset = splitdataset(dataset,i,value)
            prob = len(subdataset)/float(len(dataset))
            newmentropy +=prob * calcshannonent(subdataset)
        infogain = baseentropy - newmentropy
        if infogain>bestinfogain:
            bestinfogain = infogain
            bestfeature =i
    return bestfeature

'''
多数投票法决定分类
'''
def majoritycnt(classlist):
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount[vote]+=1
    sortedclasscount = sorted(classcount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

'''
创建数结构
'''
def createtree(dataset,labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) ==1:
        return majoritycnt(classlist)
    bestfeat = choosebestfeaturetosplit(dataset)
    bestfeatlabel = labels[bestfeat]
    mytree = {bestfeatlabel:{}}
    del(labels[bestfeat])
    featvalues = [example[bestfeat] for example in dataset]
    uniquevals = set(featvalues)
    for value in uniquevals:
        sublabels = labels[:]
        mytree[bestfeatlabel][value] = createtree(splitdataset(dataset,bestfeat,value),sublabels)
    return mytree
'''
依照树结构对未知类的数据进行分类
'''
def classify(inputtree,featlabels,testvec):
    firststr = tolist(inputtree.keys())[0]
    seconddict = inputtree[firststr]
    featindex = featlabels.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex] == key:
            if type(seconddict[key]).__name__ =='dict':
                classlabel = classify(seconddict[key],featlabels,testvec)
            else:
                classlabel = seconddict[key]
    return classlabel


def storetree(inputtree,filename):
    import pickle               #pickle模块必须使用二进制
    fw = open(filename,'wb')    #二进制方式写
    pickle.dump(inputtree,fw)
    fw.close()

def grabtree(filename):
    import pickle
    fr = open(filename,'rb')     #二进制方式读
    return pickle.load(fr)


fr = open('D:\MLInAction\ch3\lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenseslabels = ['age','prescript','astigmatic','tearRate']
lensestree = createtree(lenses,lenseslabels)

treePlotter.createPlot(lensestree)
