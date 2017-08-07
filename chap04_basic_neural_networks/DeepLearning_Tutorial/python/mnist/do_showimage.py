from pylab import *
from numpy import *
from mnist import load_mnist
images, labels = load_mnist('training', digits=[2])
imshow(images.mean(axis=0), cmap=cm.gray)
show()
# num=0
# for img in images:
#     imsave('./all_images/'+str(num)+".png",img)
#     num=num+1
# print labels

