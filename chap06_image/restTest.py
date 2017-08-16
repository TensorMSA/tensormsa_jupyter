import tensorflow as tf
from lenet.predict import lenet_predict as lp

class restTest:
    def print_test(self, a,b):
        return_val = lp.predict(self)
        test_string = "predict result {0}".format(return_val)
        return test_string

if __name__ == '__main__':
    lp2 = lp()
    lp2.predict()