from pylab import *
from numpy import *
from mnist import load_mnist
sum=0
for i in range(10):
    images, labels = load_mnist('training', digits=[i])
    num=0
    sum=sum+len(images)
    print ('%d - %d'%(i,len(images)))
    print ('-------')
    for img in images:
        imsave('./all_images/%d_%d.png'%(i,num),img)
        num=num+1
print ('total=%d'%sum)
# print labels

