from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import ch9.regtree as regtree
from numpy import *
from tkinter import *

def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a=reDraw.f.add_subplot(111)
    if chkbtnvar.get():
        if tolN<2:
            tolN=2
        mytree = regtree.creattree(reDraw.rawdat,regtree.modelleaf,regtree.modelerr,ops=(tolS,tolN))
        yhat = regtree.creatforecast(mytree,reDraw.testdat,regtree.modeltreeeval)
    else:
        mytree = regtree.creattree(reDraw.rawdat,ops=(tolS, tolN))
        yhat = regtree.creatforecast(mytree, reDraw.testdat)
    reDraw.a.scatter(reDraw.rawdat[:,0],reDraw.rawdat[:,1],s=5)
    reDraw.a.plot(reDraw.testdat,yhat,linewidth=2.0)
    reDraw.canvas.show()

def getinputs():
    try:
        tolN = int(tolnentry.get())
    except:
        tolN = 10
        print("请输入正确的tolN值")
        tolnentry.delete(0,END)
        tolnentry.insert(0,"10")
    try:
        tolS = float(tolsentry.get())
    except:
        tolS = 1
        print("请输入正确的tolS值")
        tolsentry.delete(0, END)
        tolsentry.insert(0, "1.0")
    return tolN,tolS

def drawnewtree():
    tolN,tolS = getinputs()
    reDraw(tolS,tolN)



root = Tk()

reDraw.f = Figure(figsize=(5,4),dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)
# Label(root,text='Plot Place Holder').grid(row = 1,columnspan = 3)
Label(root,text='tolN').grid(row = 1,column=0)
tolnentry = Entry(root)
tolnentry.grid(row=1,column=1)
tolnentry.insert(0,'10')
Label(root,text='tolS').grid(row = 2,column=0)
tolsentry = Entry(root)
tolsentry.grid(row=2,column=1)
tolsentry.insert(0,'1.0')
Button(root,text='ReDraw',command=drawnewtree).grid(row=1,column=2,rowspan=3)
chkbtnvar = IntVar()
chkbtn = Checkbutton(root,text='Model Tree',variable = chkbtnvar)
chkbtn.grid(row=3,column=0,columnspan=2)
reDraw.rawdat = mat(regtree.loaddataset("D://MLInAction//ch9//sine.txt"))
reDraw.testdat = arange(min(reDraw.rawdat[:,0]),max(reDraw.rawdat[:,0]),0.01)
Button(root,text='Quit',command=root.quit).grid(row=4,column=1)
reDraw(1.0,10)
root.mainloop()