import matplotlib.pyplot as plt

decisionnode = dict(boxstyle = 'sawtooth',fc = '0.8')
leafnode = dict(boxstyle = 'round4',fc = '0.8')
arrow_args = dict(arrowstyle = '<-')

def plotnode(nodetxt,centerpt,parentpt,nodetype):
    createPlot.ax1.annotate(nodetxt,xy = parentpt,xycoords = 'axes fraction',xytext = centerpt,textcoords = 'axes fraction',va ='center',ha = 'center',
                           bbox = nodetype,arrowprops = arrow_args)          #https://blog.csdn.net/leaf_zizi/article/details/82886755参数讲解
#                          第一个参数注释文本内容，xy箭头尾坐标点，xycoords坐标系属性，xytext注释文本的坐标点，bbox注释文本边框



# def createPlot():
#     fig = plt.figure(1,facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111,frameon = False)
#     plotnode('决策节点',(0.1,0.5),(0.7,0.1),decisionnode)
#     plotnode('叶节点',(0.8,0.1),(0.3,0.8),leafnode)
#     plt.show()

def tolist(notlist):
    islist = []
    for i in notlist:
        islist.append(i)
    return islist


def getnumleafs(mytree):
    numleafs = 0
    firststr = tolist(mytree.keys())[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            numleafs +=getnumleafs(seconddict[key])
        else:
            numleafs +=1
    return numleafs


def gettreedepth(mytree):
    maxdepth = 0
    firststr = tolist(mytree.keys())[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            thisdepth = 1 + gettreedepth(seconddict[key])
        else:
            thisdepth = 1
        if thisdepth>maxdepth:
            maxdepth = thisdepth
    return maxdepth


def plotmidtext(cntrpt,parentpt,txtstring):
    xMid = (parentpt[0] - cntrpt[0]) / 2.0 + cntrpt[0]
    yMid = (parentpt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    createPlot.ax1.text(xMid, yMid, txtstring)

def plottree(mytree,parentpt,nodetxt):
    numleafs = getnumleafs(mytree)
    depth = gettreedepth(mytree)
    firststr = tolist(mytree.keys())[0]
    cntrpt = (plottree.xOff+(1.0 +float(numleafs))/2.0/plottree.totalW,plottree.yOff)
    plotmidtext(cntrpt,parentpt,nodetxt)
    plotnode(firststr,cntrpt,parentpt,decisionnode)
    seconddict = mytree[firststr]
    plottree.yOff = plottree.yOff - 1.0/plottree.totalD
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ =='dict':
            plottree(seconddict[key],cntrpt,str(key))
        else:
            plottree.xOff = plottree.xOff + 1.0/plottree.totalW
            plotnode(seconddict[key],(plottree.xOff,plottree.yOff),cntrpt,leafnode)
            plotmidtext((plottree.xOff,plottree.yOff),cntrpt,str(key))
    plottree.yOff = plottree.yOff + 1.0/plottree.totalD


def createPlot(intree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks= [])
    createPlot.ax1 = plt.subplot(111,frameon = False,**axprops)
    plottree.totalW = float(getnumleafs(intree))
    plottree.totalD = float(gettreedepth(intree))
    plottree.xOff = -0.5/plottree.totalW
    plottree.yOff = 1.0
    plottree(intree,(0.5,1.0)," ")
    plt.savefig('testtree.jpg')
    plt.show()


# createPlot()