from numpy import *

def loadsimpdata():
    datamat = matrix([
        [1,2.1],
        [2,1.1],
        [1,1],
        [2,1],
        [1.3,1]])
    classlabel = [1,1,-1,-1,1]
    return datamat,classlabel


'''
输入：datamatrix数据特征集 , dimen需要比较的特征索引 , threshval需要比较的阈值 , threshineq需要比较的不等式规则
输出：返回一个依据阈值划分的数组
'''

def stumpclassify(datamatrix , dimen , threshval , threshineq):
    retarray = ones((shape(datamatrix)[0],1))
    if threshineq == "lt":
        retarray[datamatrix[:,dimen]<=threshval] = -1
    else:
        retarray[datamatrix[:, dimen] > threshval] = -1
    return retarray

'''
选出最优单层决策树，包括按照哪个特征，特征阈值，大于还是小于
输入：dataarr 特征数据集, classlabels标签数据集,D权重
输出：beststump最优决策树桩,minerror最小错误率,bestclasest依据最优决策树桩分类后的标签数组
'''

def buildstump(dataarr,classlabels,D):
    datamatrix = mat(dataarr)
    labelmat = mat(classlabels).T
    m,n = shape(datamatrix)
    numsteps = 10                #每个特征的步数10步
    beststump = {}                  #用来保存最优单层决策树的参数，包括按照哪个特征，特征阈值，大于还是小于
    bestclasest = mat(zeros((m,1)))             #用来保存决策树分类后的数组
    minerror = inf                      #初始化错误率设为无穷大
    for i in range(n):                        #每个特征进行循环计算
        rangemin = datamatrix[:,i].min()            #该特征的最小值
        rangemax = datamatrix[:,i].max()            #该特征的最大值
        stepssize = (rangemax-rangemin)/numsteps                   #用特征取值范围除以步数，得到每一步的步长
        for j in range(-1,int(numsteps)+1):                        #j循环来实现改版每一次阈值
            for inequal in  ['lt','gt']:                           #不等式在大于或小于中循环
                threshval = rangemin + float(j) * stepssize                  #每一次的阈值都在本次最小特征值上增加一个步长
                predictvalue = stumpclassify(datamatrix,i,threshval,inequal)   #根据本次决策树参数生产划分数组
                errarr = mat(ones((m,1)))                                      #用来计算错误率的矩阵
                errarr[predictvalue == labelmat] =0                            #如果划分正确，那么对应的位置改为0，即不加入错误率计算
                weightederror = D.T * errarr                                 #利用对用权重进行错误率的计算
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                # i, threshval, inequal, weightederror))
                if weightederror<minerror:                               #每一次决策树划分后，比较错误率是否变小了，变小的话，需要改变用来保存最优决策树参数的变量
                    minerror = weightederror
                    bestclasest = predictvalue.copy()
                    beststump["dim"] = i
                    beststump["thresh"] = threshval
                    beststump["ineq"] = inequal
    return beststump,minerror,bestclasest

'''
输入：datasrr特征数据集,classlabels特征标签集,numit最大迭代次数
输出：每一次迭代计算的最优决策树桩，即弱分类器
'''

def adaboosttrainds(datasrr,classlabels,numit = 40):
    weakclassarr = []                             #弱分类器集合
    m = shape(datasrr)[0]                          #数据集样本数
    D = mat(ones((m,1))/m)                          #初始化权重，每个样本的权重一样
    aggclassest = mat(zeros((m,1)))                    #每个样本的预测类别累加值，多个弱分类器集成结果
    for i in range(numit):                                   #循环最大迭代次数
        beststump,error,classest = buildstump(datasrr,classlabels,D)            #得到最优决策树桩相关信息
        alpha = float(0.5 * log((1-error)/max(error,1e-16)))                    #根据错误率error计算alpha，max函数确保在错误率为0时，不会有除零溢出
        beststump["alpha"] = alpha                                               #每次的弱分类器中，加入“alpha”项，记录alpha
        weakclassarr.append(beststump)                                           #将完整的弱分类器信息append到weakclassarr中
        expon = multiply(-1 * alpha * mat(classlabels).T,classest)               #计算exp的指数项
        D = multiply(D,exp(expon))                                               #更新权重
        D = D /D.sum()                                                            #调整权重，每一个除以权重之和，使权重和始终为1
        aggclassest += alpha * classest                                          #每个样本的预测类别累加值，或多个弱分类器集成结果
        aggerror = multiply(sign(aggclassest)!=mat(classlabels).T,ones((m,1)))   #将预测累加值对比真实标签，计算错误个数
        errorrate = aggerror.sum()/m                                             #计算错误率
        print("error rate is ",errorrate)
        if errorrate==0.0:               #如果错误率等于0，跳出迭代
            break
    return weakclassarr,aggclassest


def adaclassify(dattoclass , classifierarr):
    datamatrix = mat(dattoclass)
    m = shape(datamatrix)[0]
    aggclassest = mat(zeros((m,1)))
    for i in range(len(classifierarr)):
        classest = stumpclassify(datamatrix,classifierarr[i]["dim"],classifierarr[i]["thresh"],classifierarr[i]["ineq"])
        aggclassest +=classifierarr[i]["alpha"] * classest
        # print(aggclassest)
    return sign(aggclassest)


'''
datamat,classlabel = loadsimpdata()
weakclassarr = adaboosttrainds(datamat,classlabel)
result = adaclassify([2,2],weakclassarr)
print(result)
'''

'''
从文档中加载数据
'''
def loaddata(filename):
    numfeat = len(open(filename).readline().strip().split("\t"))
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = []
        allfeat = line.strip().split("\t")
        for i in range(numfeat-1):
            linearr.append(float(allfeat[i]))
        datamat.append(linearr)
        labelmat.append(float(allfeat[-1]))
    return datamat,labelmat

datamat,labelmat = loaddata("D://MLInAction//ch7//horseColicTraining2.txt")
weakclassarr,aggclassest = adaboosttrainds(datamat,labelmat)
# testdata,testlabel = loaddata("D://MLInAction//ch7//horseColicTest2.txt")
# result = adaclassify(testdata,weakclassarr)

'''
count =0
for i in range(len(testlabel)):
    if testlabel[i]!=result[i]:
        count+=1
print(count)
'''

'''
画ROC图像
'''

def plotROC(predstrengths,classlabels):
    import matplotlib.pyplot as plt
    cur = (1,1)                              #初始点在右上角（1,1）
    ysum = 0                                       #用来计算AUC面积的变量，整体面积可以划分为多个竖型的矩形，矩形上下的长度为每一次坐标点横向移动的长度，也就是Xstep，此变量则是所有矩形高度之和
    numposclas = sum(array(classlabels)==1)        #计算真实类别为正的数量
    Ystep = 1/float(numposclas)                    #分母不受影响，所以这每一次竖向改变坐标的长度
    Xstep = 1/float(len(classlabels)-numposclas)   #每一次横向改变坐标的长度
    sortedindicies = predstrengths.argsort()       #将预测值排序，从小到大，返回位置索引
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedindicies.tolist()[0]:        #索引转化为list进行遍历
        if classlabels[index]==1:                   #如果真实结果为正，改变纵轴，即真正率
            delX = 0
            delY = Ystep                             #改变量为Ystep
        else:
            delX = Xstep
            delY = 0
            ysum +=cur[1]                             #只有坐标点横向改变，才会产生计算面积的小矩形，矩形的高度全部加在此变量上
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c = "b")               #在画板上画出上一次的点和更新后的点之间的连线，蓝色
        cur = (cur[0]-delX,cur[1]-delY)               #更新后的点作为下一次更新的起始点
    ax.plot([0,1],[0,1],'b--')                        #画对角线
    plt.xlabel("FPR")                     #横轴标题：假正率
    plt.ylabel("TPR")                     #纵轴标题：真正率
    plt.axis([0,1,0,1])                   #四个变量分别表示横轴最小值、最大值，纵轴最小值、最大值
    plt.show()
    print("the AUC is ",ysum*Xstep)           #计算AUC面积

plotROC(aggclassest.T,labelmat)