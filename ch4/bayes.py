from numpy import *

def loaddataset():
    postinglist = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]   # 每一行代表一篇文档
    classvec = [0,1,0,1,0,1]         #1表示存在侮辱文字
    return postinglist,classvec

'''
创建词字典
'''
def createvocablist(dataset):
    vocabset = set([])                #set结果每次不一定一样
    for document in dataset:
        vocabset = vocabset | set(document)    #集合并操作
    return list(vocabset)

'''
生成每篇文档的词向量
'''
def bagofwords2vec(vocablist,inputset):
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] += 1
    return returnvec

'''
输入：trainmatrix文档矩阵，每一列为一个文档的词向量,traincategory每篇文档标签所构成的向量
输出： p0vect：
        p1vect：
        pabusive：一篇文档为侮辱文档的概率
'''

def trainnb0(trainmatrix,traincategory):
    numtraindocs = len(trainmatrix)       #训练的文档数
    numwords = len(trainmatrix[0])        #词向量的长度
    pabusive = sum(traincategory)/float(numtraindocs)
    p0num = ones(numwords)                 #原代码为初始化零，利用拉普拉斯进行修改，计算概率的分子初始化为一
    p1num = ones(numwords)
    p0denom = 2                            #原代码为初始化零，利用拉普拉斯进行修改，计算概率的分目初始化为标签种类数
    p1denom = 2
    for i in range(numtraindocs):
        if traincategory[i] ==1:
            p1num += trainmatrix[i]                  #p1num统计侮辱文档中各单词出现的频数
            p1denom += sum(trainmatrix[i])           #p1denom统计侮辱文档中所有单词的数目
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1vect = log(p1num/p1denom)                      #计算侮辱文档中各单词的条件概率
    p0vect = log(p0num/p0denom)
    return p0vect,p1vect,pabusive


def classifynb(vec2classify,p0vec,p1vec,pclass1):
    p1 = sum(vec2classify * p1vec) + log(pclass1)     #因为前文计算p1vec使用了log，所用原始计算式为连乘，现在要用连加
    p0 = sum(vec2classify * p0vec) + log(pclass1)
    if p1>p0:
        return 1
    else:
        return 0


def testingnb(testentry):
    listOposts,listclasses = loaddataset()
    myvocablist = createvocablist(listOposts)
    trainmat = []              #每组数据对应的词向量
    for postindoc in listOposts:
        trainmat.append(bagofwords2vec(myvocablist,postindoc))
    p0,p1,pab = trainnb0(array(trainmat),array(listclasses))
    thisdoc = array(bagofwords2vec(myvocablist,testentry))
    print(testentry," classifid as ",classifynb(thisdoc,p0,p1,pab))

'''
利用正则表达式将邮件内容中对分类影响不大的字符过滤掉，此处做法为先将内容按照非（A-Z|a-z|0-9|_）字符进行切片，再在其中只取长度大于2的内容，
假如想要提高效果，可以之后在进行内容对比，舍去停用词
'''
def textparse(bigstring):
    import re
    listoftakens = re.split(r'\W*',bigstring)
    return [tok.lower() for tok in listoftakens if len(tok)>2]

def spamtest():
    doclist = []
    classlist = []
    fulllist = []
    for i in range(1,26):
        wordlist = textparse(open("D://MLInAction//ch4//email//spam//%d.txt"%i,errors='ignore').read())
        # wordlist = textparse(open('email/spam/%d.txt' % i,errors='ignore').read())
        doclist.append(wordlist)
        fulllist.extend(wordlist)
        classlist.append(1)
        wordlist = textparse(open("D://MLInAction//ch4//email//ham//%d.txt"%i,errors='ignore').read())
        # wordlist = textparse(open('email/ham/%d.txt' % i,errors='ignore').read())
        doclist.append(wordlist)
        fulllist.extend(wordlist)
        classlist.append(0)
    vocablist = createvocablist(doclist)
    trainingset = list(range(50))
    testset = []
    for i in range(10):
        randindex = int(random.uniform(0,len(trainingset)))
        testset.append(randindex)
        del(trainingset[randindex])
    trainmat = []
    trainclasses = []
    for docindex in trainingset:
        trainmat.append(bagofwords2vec(vocablist,doclist[docindex]))
        trainclasses.append(classlist[docindex])
    p0v,p1v,pspam = trainnb0(array(trainmat),array(trainclasses))
    errorcount = 0
    for docindex in testset:
        wordvector = bagofwords2vec(vocablist,doclist[docindex])
        if classifynb(wordvector,p0v,p1v,pspam)!= classlist[docindex]:
            errorcount+=1
            print(doclist[docindex])
    print("error rate is ",float(errorcount)/len(testset))

# spamtest()

def calcmostfreq(vocablist,fulltext):
    import operator
    freqdict = {}
    for token in vocablist:
        freqdict[token] = fulltext.count(token)
    sortedfreq = sorted(freqdict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedfreq[:30]

