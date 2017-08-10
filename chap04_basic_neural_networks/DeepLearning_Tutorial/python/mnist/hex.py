import struct
from array import array as pyarray
import numpy as np
def hexRead():
    f = open('a.hex','rb')
    n,s = struct.unpack(">II",f.read(8))
    print "%d %d"%(n,s)
    arr=pyarray("b", f.read())
    for i in arr:
        print i
    f.close()

def hexWrite():
    f = open('a.hex','wb')
    f.write(struct.pack(">II",1,1024))
    f.write(struct.pack(">b",1))
    f.write(struct.pack(">b",2))
    f.write(struct.pack(">b",3))
    f.write(struct.pack(">b",4))
    f.close()

def hexArrayWrite():
    f=open("a.hex","wb")
    mydata=np.arange(10)
    print(mydata)
    myfmt='B'*len(mydata)
    #  You can use 'd' for double and < or > to force endinness
    bin=struct.pack(myfmt,*mydata)
    print(bin)
    f.write(bin)
    f.close()

if __name__ == "__main__":
    # hexRead()
    # hexWrite()
    hexArrayWrite()


























