#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylab import *
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
import struct
import os 

def save_mnist(                                       \
        fname_img="images.idx3-ubyte",                \
        fname_lbl="labels.idx1-ubyte",                \
        digits=np.arange(10),                         \
        path=".",                                     \
        sourceFolder="./all_images"                   \
    ):
    """
    Saves MNIST files from files
    Adapted from : http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    Made by : nowage@gmail.com
    """
    
    files=os.listdir(sourceFolder)
    # print file names.
    # for i in files:
    #     print i
    fileList=map(lambda b:b.split("_"), map(lambda a:a.replace(".png",""),files))
    
    size=len(fileList)
    magic_nr=2049

    flbl = open(fname_lbl, 'wb')
    flbl.write(struct.pack(">II",magic_nr, size  ) )
    for i in fileList:
        flbl.write(struct.pack(">b",int(i[0])))
    flbl.close()



    # ind = [ k for k in range(size) if lbl[k] in digits ]
    ind = [ k for k in range(size) if fileList[k][0] in digits ]
    N = len(ind)
    fimg = open(fname_img, 'wb')
    rows,cols=(28,28)

    
    # 헤더 부분을 쓰는 부분
    fimg.write(struct.pack(">IIII",magic_nr, size, rows, cols  ) )
    for fl in fileList:
        #파일에서 읽어서 파일에 쓰기
        print ('./all_images/%s_%s.png'%(fl[0],fl[1]))    
        img=imread('./all_images/%s_%s.png'%(fl[0],fl[1]))
        for r in img:
            fm='b'*cols
            # print len(r), r.__class__ 
            # print r[:,2], r[:,2][0].__class__
            fimg.write(struct.pack(fm,*r[:,2]))
        #print img.__class__
        #print img.shape
    fimg.close()    

    
    



def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    print ("%s -------------------- %d"%(magic_nr, size))
    lbl = pyarray("b", flbl.read())
    # for i in lbl:
    #      print '%d %s'%(i,i.__class__)
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    # print "%d,%d,%d,%d"%(magic_nr, size, rows, cols)
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels



