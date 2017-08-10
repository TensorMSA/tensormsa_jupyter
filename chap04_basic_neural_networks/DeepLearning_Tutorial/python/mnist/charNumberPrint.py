from pylab import *
from numpy import *
from mnist import load_mnist
images,labels=load_mnist('training',digits=[2])

s=""
n=0
for h in range(3):
    for i in images[h]:
        for j in i:
            if j>250:
                c="X"
            elif j>150:
                c="-"
            else:
                c=" "
            s=s+c
        print (s)
        s=""
