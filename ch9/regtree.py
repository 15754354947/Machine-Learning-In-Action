from numpy import *

'''
读取文件中的数据，返回的内容包括特征数据集和标签数据集 
'''
def loaddataset(filename):
    fr = open(filename)
    datamat = []
    for line in fr.readlines():
        str = line.strip().split("\t")
        litline = list(map(float,str))
        datamat.append(litline)
    return datamat

'''
对特征数据集进行二分
输入：dataset待划分数据集，含标签,feature划分特征的索引序号,value划分特征值
输出：mat0特征值大的数据集,mat1特征值小的数据集
'''

def binsplitdataset(dataset,feature,value):
    mat0 = dataset[nonzero(dataset[:,feature]>value)[0],:]
    mat1 = dataset[nonzero(dataset[:,feature]<=value)[0],:]
    return mat0,mat1

'''
求线性回归的回归系数
'''

def linearsolve(dataset):
    m,n =shape(dataset)
    X = mat(ones((m,n)))            #特征数据集
    Y = mat(ones((m,1)))             #标签数据集
    X[:,1:n] = dataset[:,0:n-1]
    Y = dataset[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0:  # det求矩阵的行列式，如果为零，矩阵不存在逆矩阵，也就不能计算对应的ws
        print("NO FUNCTION!")
        return
    ws = xTx.I * (X.T * Y)  # 根据计算公式，计算出ws
    return ws,X,Y            #将回归系数、特征数据集和标签数据集一并返回

'''
在数据集上使用线性回归进行拟合
输入：待拟合的数据集，含标签
输出：利用线性回归求得的回归系数ws
'''

def modelleaf(dataset):
    ws, X, Y = linearsolve(dataset)
    return ws

'''
计算线性回归拟合的平方误差
'''
def modelerr(dataset):
    ws, X, Y = linearsolve(dataset)
    yhat = X * ws               #利用回归系数拟合的预测值
    return sum(power(yhat-Y,2))

'''
计算叶节点数值，该叶节点上所有标签的平均数
'''
def regleaf(dataset):
    return mean(dataset[:,-1])

'''
计算叶节点上的平方误差，可用均方差乘以样本点数得到
'''
def regerr(dataset):
    return var(dataset[:,-1])*shape(dataset)[0]

'''
对数据集选择最好的二分属性
输入：dataset待分数据集,leaftype= regleaf叶节点生成模型选择，默认为regleaf,errtype=regerr误差计算方式，默认为regerr,前两个参数应对应使用，
ops=(1,4)，第一个参数是容许误差的下降值，第二个为切分的最少样本数
输出：最优切分的特征索引序号和特征值
'''
def choosebestsplit(dataset,leaftype= regleaf,errtype=regerr,ops=(1,4)):
    tolS = ops[0]                     #容许误差的下降值
    tolN = ops[1]                     #切分的最少样本数
    if len(set(dataset[:,-1].T.tolist()[0])) ==1:                 #如果当前数据集中为同一类，不需要进行切分，结算叶节点数据
        return None,leaftype(dataset)
    m,n = shape(dataset)
    S = errtype(dataset)                           #待切分数据集的总误差
    bestS = inf                           #初始化最小误差
    bestindex = 0                          #初始化最优划分特征索引序号
    bestval = 0                            #初始化最优划分特征值
    for featindex in range(n-1):                  #对每一个特征进行循环
        for splitval in set(dataset[:,featindex].T.tolist()[0]):                #对每一个特征值进行循环
            mat0,mat1 = binsplitdataset(dataset,featindex,splitval)             #对当前数据集依据本次的参数进行切分
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):                   #如果切分后两个数据集小于设定的最小样本数，直接进行下一个循环
                continue
            newS = errtype(mat0) + errtype(mat1)                        #计算切分后两个数据集的误差之和
            if newS<bestS:                              #如果本次切分后误差小于当前最小误差：记录本次切分的相关参数
                bestS = newS
                bestindex = featindex
                bestval = splitval
    if (S - bestS)<tolS:                               #如果切分前后的误差下降值小于设定的值，没有必要进行切分，结算叶节点数据
        return None,leaftype(dataset)
    mat0, mat1 = binsplitdataset(dataset, bestindex, bestval)
    if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):             #如果最优切分后两个数据集小于设定的最小样本数，没有必要进行切分，结算叶节点数据
        return None,leaftype(dataset)
    return bestindex,bestval

'''
对一个数据集递归使用最优切分方法进行生成树结构
输入：dataset待分数据集,leaftype= regleaf叶节点生成模型选择，默认为regleaf,errtype=regerr误差计算方式，默认为regerr,前两个参数应对应使用，
ops=(1,4)，第一个参数是容许误差的下降值，第二个为切分的最少样本数
输出：生成树结构
'''

def creattree(dataset,leaftype= regleaf,errtype=regerr,ops=(1,4)):
    feat , val = choosebestsplit(dataset,leaftype,errtype,ops)          #当前数据集的最优切分特征索引序号和特征值
    if feat ==None:                                          #如果是叶节点，直接返回数据
        return val
    rettree = {}                    #树结构用字典存
    rettree['spind']=feat               #当前数据集的最优切分特征索引序号
    rettree['spval']=val                    #当前数据集的最优切分特征值
    lset ,rset = binsplitdataset(dataset,feat,val)               #切分之后的两个子树数据集
    rettree['left']=creattree(lset,leaftype,errtype,ops)             #左子树数据集上递归生成树结构
    rettree['right']=creattree(rset,leaftype,errtype,ops)            #右子树数据集上递归生成树结构
    return rettree
'''
测试代码
datamat = loaddataset("D:\MLInAction\ch9\ex0.txt")
datamat = mat(datamat)
tree = creattree(mat(datamat))
print(tree)
'''

'''
判断对象类型是否为树结构
'''
def istree(obj):
    return (type(obj).__name__=='dict')

'''
对当前树结构计算平均数值
'''
def getmean(tree):
    if istree(tree['left']):                      #如果左边是树结构，递归计算平均值
        tree['left'] = getmean(tree['left'])
    if istree(tree['right']):
        tree['right'] = getmean(tree['right'])
    return (tree['left']+tree['right'])/2             #返回左右值的平均值


'''
修剪树结构
输入：tree待剪枝的树结构，在testdata数据上进行剪枝
输出：返回剪枝后的树结构
'''
def proune(tree,testdata):
    if shape(testdata)[0]==0:                      #首先检查testdata是否合格
        return getmean(tree)
    if (istree(tree['left'])) or (istree(tree['right'])):
        lset,rset = binsplitdataset(testdata,tree['spind'],tree['spval'])              #将测试数据集利用当前树结构进行切分
    if istree(tree['left']):                   #如果左边是树结构，递归剪枝操作，使用测试数据集的左边数据
        proune(tree['left'],lset)
    if istree(tree['right']):                  #判断右边
        proune(tree['right'],rset)
    if not istree(tree['left']) and not istree(tree['right']):                 #如果两边都不是树结构，进行剪枝判断操作
        lset, rset = binsplitdataset(testdata, tree['spind'], tree['spval'])            #将测试数据集利用当前树结构进行切分
        errornomerge = sum(power(lset[:,-1]-tree['left'],2)) + sum(power(rset[:,-1]-tree['right'],2))             #剪枝前的叶节点误差
        treemean = (tree['left']+ tree['right'])/2
        errormerge = sum(power(testdata[:,-1]-treemean,2))                                  #剪枝后的叶节点误差
        if errormerge<errornomerge:                #如果剪枝后误差小，执行剪枝操作
            print("merge!")
            return treemean
        else:                                      #否则，不执行剪枝操作
            return tree
    else:
        return tree


'''
测试代码
datamat = loaddataset("D:\MLInAction\ch9\ex2.txt")
datamat = mat(datamat)
tree = creattree(datamat,ops=(0,1))
print(tree)
testdata = loaddataset("D:\MLInAction\ch9\ex2test.txt")
testdata = mat(testdata)
newtree = proune(tree,testdata)
print(newtree)
'''

'''
测试代码
mymat = loaddataset("D:\MLInAction\ch9\exp2.txt")
mymat = mat(mymat)
tree = creattree(mymat,modelleaf,modelerr,(1,10))
print(tree)
'''

'''
根据生成叶节点的两种方法，直接用平均值和线性回归系数，预测叶节点也有两种方法
'''

def regtreeeval(model,indat):
    return float(model)

def modeltreeeval(model,indat):
    n= shape(indat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = indat
    return float(X * model)


'''
tree预测利用的树结构,indata单组预测数据,modeleval叶节点计算模型，默认为regtreeeval
'''
def treeforecast(tree,indata,modeleval=regtreeeval):
    if not istree(tree):                                    #如果是叶切分节点，直接利用叶节点计算模型返回数值
        return modeleval(tree,indata)
    if indata[tree['spind']]>tree['spval']:                    #根据树结构里的最优切分特征索引序号和特征值进行比较，大于该值，进入左边子树递归预测
        if istree(tree['left']):                                       #如果左边是树结构，递归使用本函数预测
            return treeforecast(tree['left'],indata,modeleval)
        else:                                                          #如果是叶切分节点，直接利用叶节点计算模型返回数值
            return modeleval(tree['left'],indata)
    else:                                                             #小于该值，进入右边子树递归预测
        if istree(tree['right']):
            return treeforecast(tree['right'],indata,modeleval)
        else:
            return modeleval(tree['right'],indata)

'''
在testdata数据上利用tree树结构进行预测，modeleval叶节点利用的模型，默认为regtreeeval
'''
def creatforecast(tree,testdata,modeleval=regtreeeval):
    m = len(testdata)                                    #测试数据的个数
    yhat = mat(zeros((m,1)))                             #初始化预测数据集
    for i in range(m):
        yhat[i,0]= treeforecast(tree,mat(testdata[i]),modeleval)                 #每个testdata进行预测
    return yhat

trainmat = mat(loaddataset("D://MLInAction//ch9//bikeSpeedVsIq_train.txt"))
testmat = mat(loaddataset("D://MLInAction//ch9//bikeSpeedVsIq_test.txt"))

# 第一种叶节点计算方法
# mytree = creattree(trainmat,ops=(1,20))
# yhat = creatforecast(mytree,testmat[:,0])
# print(corrcoef(yhat,testmat[:,1],rowvar=0)[0,1])
# 第二种叶节点结算方法
# mytree2 = creattree(trainmat,modelleaf,modelerr,ops=(1,20))
# yhat = creatforecast(mytree2,testmat[:,0],modeltreeeval)
# print(corrcoef(yhat,testmat[:,1],rowvar=0)[0,1])
# 纯用线性回归计算
# ws,X,Y, = linearsolve(trainmat)
# yhat = mat(zeros((shape(testmat)[0],1)))
# for i in range(shape(testmat)[0]):
#     yhat[i] =  testmat[i,0]*ws[1,0] + ws[0,0]
# print(corrcoef(yhat,testmat[:,1],rowvar=0)[0,1])