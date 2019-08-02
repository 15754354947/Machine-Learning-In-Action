from numpy import *



'''
读取文件数据
输入：文件路径
输出：datamat特征数据集,labelmat样本类别集
'''
def loaddataset(filename):
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split('\t')
        datamat.append([float(linearr[0]),float(linearr[1])])
        labelmat.append(float(linearr[2]))
    return datamat,labelmat


'''
随机选择另一个alpha的下标
输入：已选alpha的下标i，alpha可选的长度
输出：随机生成的另一个alpha的下标j
'''
def selectjrand(i,m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

'''
修剪alpha[j]的值，不能使其超过最低值L和最高值H
'''
def clipalpha(aj,H,L):
    if aj>H:
        aj = H
    if aj < L:
        aj = L
    return aj

'''
简化版SMO算法
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))                    # 创建一个包含m个alpha值的向量，全部初始化为0
    iter = 0
    while (iter < maxIter):  # 外循环，迭代次数小于最大迭代次数
        alphaPairsChanged = 0
        for i in range(m):   # 内循环，遍历数据集中的每个数据向量，每一次都进行两个alpha的优化
            # 计算预测类别 fXi
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  # multiply 实现对应位置元素相乘
            # 计算误差，预计类别值-真实类别值
            Ei = fXi - float(labelMat[i])  #if checks if an example violates KKT conditions
            # 如果误差很大，则该数据实例所对应的alpha值要进行优化
            # 这里本来是 C>=alpha>=0，不取等号是因为等号代表数据实例位于边界上，不值得优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectjrand(i,m)  # 选择第二个alpha值 alphas[j]
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy(); # 用copy方法拷贝旧值，防止Python传递列表采用的是传引用方式
                if (labelMat[i] != labelMat[j]):  # 控制 alphas[j] 调整到0到C之间
                    L = max(0, alphas[j] - alphas[i])  # 下届不能小于0
                    H = min(C, C + alphas[j] - alphas[i])  # 上界不能大于C
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: continue   # 如果上下界相等，不做任何改变，continue进行下一次循环
                # eta是alphas[j]的最优修改量，因为如果eta为0，计算新的alphas[j]比较麻烦，所以对SMO做了简化，不做处理，continue进行下一次循环
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: continue
                # 对alphas[j]的上下界进行调整
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipalpha(alphas[j],H,L)
                # 检查alpha[j]是否有轻微改变，如果是，continue进行下一次循环
                if (abs(alphas[j] - alphaJold) < 0.00001): continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j]) # 更新alphas[i]值，与alphas[j]值大小一样，但方向相反，即一个增加，另外一个减少
                # 设置常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1  # 标记这一对alpha值进行了改变
                print(("iter: %d i:%d, pairs changed %d") % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1 # 如果这一对alpha值没有改变，则继续迭代。这里是所有alpha不发生修改，程序才会停止退出while循环
        else: iter = 0  # 如果这一对alpha值改变了，iter设置为0重新进行迭代
        print(("iteration number: %d") % iter)
    return b,alphas  # 返回常数项b和更新后的alphas向量



'''
利用类的方式创建一种数据结构
'''
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler,ktup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))    #用来存储所有误差
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kerneltrans(self.X,self.X[i,:],ktup)


'''
计算误差E值并返回
'''
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


'''
选择第二个alpha或者说内循环的alpha值
'''
def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectjrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


'''
计算误差值，并存入缓存中
'''
def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: return 0
        eta = 2.0 * oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta >= 0: return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipalpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter,ktup):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,ktup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #遍历全部数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:          #遍历非边界的alpha
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]    #返回alpha中数值在0-C间的数的索引位置
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

'''
计算决策超平面的w
'''
def calcWs(alpha,dataarr,classlabels):
    X = mat(dataarr)
    labelmat = mat(classlabels).T
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alpha[i]*labelmat[i],X[i,:].T)
    return w


'''
利用各种核函数计算核矩阵
'''
def kerneltrans(X,A,ktup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if ktup[0] =='lin':
        K = X * A.T
    elif ktup[0] =='rbf':
        for j in range(m):
            detalrow = X[j,:] - A
            K[j] = detalrow * detalrow.T
        K = exp(K/(-1 * ktup[1] **2 ))
    else:
        raise NameError("NO FUNCTION")
    return K


def testrbf(k1 = 1.3):
    dataarr, labelarr = loaddataset('D://MLInAction//ch6//testSetRBF.txt')
    b, alpha = smoP(dataarr, labelarr, 200, 0.0001,10000,('rbf',k1))
    datamat = mat(dataarr)
    labelmat = mat(labelarr).transpose()
    svind = nonzero(alpha>0)[0]       #取alpha中大于零，也就是支持向量的索引位置
    svs = datamat[svind]
    labelsv = labelmat[svind]
    print('there are %d support vectors' %shape(svs)[0])
    m,n = shape(datamat)
    errorcount = 0
    for i in range(m):
        kerneleval = kerneltrans(svs,datamat[i,:],('rbf',k1))
        perdict = kerneleval.T * multiply(alpha[svind],labelsv) + b
        if sign(perdict)!= sign(labelarr[i]):
            errorcount+=1
    print("the training error rate is %f"%(float(errorcount)/m))
    dataarr, labelarr = loaddataset('D://MLInAction//ch6//testSetRBF2.txt')
    errorcount = 0
    datamat = mat(dataarr)
    labelmat = mat(labelarr).transpose()
    m,n = shape(datamat)
    for i in range(m):
        kerneleval = kerneltrans(svs,datamat[i,:],('rbf',k1))
        perdict = kerneleval.T * multiply(alpha[svind], labelsv) + b
        if sign(perdict) != sign(labelarr[i]):
            errorcount += 1
    print("the test error rate is %f" % (float(errorcount) / m))


def img2vector(filename):
    returnvect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j] = int(linestr[j])
    return returnvect


def loadimage(dirname):
    from os import listdir
    hwlabels = []
    trainingfilelist = listdir(dirname)
    m = len(trainingfilelist)
    trainingmat = zeros((m,1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        if classnumstr == 9:
            hwlabels.append(-1)
        else:
            hwlabels.append(1)
        trainingmat[i,:] = img2vector("%s//%s"%(dirname,filenamestr))
    return trainingmat,hwlabels


def testdigits(ktup = ('rbf',10)):
    dataarr, labelarr = loadimage("D://MLInAction//ch6//trainingDigits")
    b, alpha = smoP(dataarr, labelarr, 200, 0.0001,10000,ktup)
    datamat = mat(dataarr)
    labelmat = mat(labelarr).transpose()
    svind = nonzero(alpha>0)[0]       #取alpha中大于零，也就是支持向量的索引位置
    svs = datamat[svind]
    labelsv = labelmat[svind]
    print('there are %d support vectors' %shape(svs)[0])
    m,n = shape(datamat)
    errorcount = 0
    for i in range(m):
        kerneleval = kerneltrans(svs,datamat[i,:],ktup)
        perdict = kerneleval.T * multiply(alpha[svind],labelsv) + b
        if sign(perdict)!= sign(labelarr[i]):
            errorcount+=1
    print("the training error rate is %f"%(float(errorcount)/m))
    dataarr, labelarr = loadimage("D://MLInAction//ch6//testDigits")
    errorcount = 0
    datamat = mat(dataarr)
    labelmat = mat(labelarr).transpose()
    m,n = shape(datamat)
    for i in range(m):
        kerneleval = kerneltrans(svs,datamat[i,:],ktup)
        perdict = kerneleval.T * multiply(alpha[svind], labelsv) + b
        if sign(perdict) != sign(labelarr[i]):
            errorcount += 1
    print("the test error rate is %f" % (float(errorcount) / m))


testdigits()
'''
测试数据集的代码
datamat,labelmat = loaddataset('D://MLInAction//ch6//testSet.txt')
b,alpha = smoP(datamat,labelmat,0.6,0.001,40)
w = calcWs(alpha,datamat,labelmat)
testnum = input("测试序号")
testnum = int(testnum)
testdata = mat(datamat[testnum])
print(testdata)
print(testdata*mat(w)+b)
print(labelmat[testnum])

testrbf()
'''
