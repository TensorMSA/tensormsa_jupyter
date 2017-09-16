import keras
import h5py
import numpy as np
from keras.utils import np_utils

batch_size = 32
nb_classes = 2

h5file = h5py.File('/jupyter/chap06_image/air_car.hdf5', mode='r')
X_test = h5file['image_features']
y_test = h5file['targets']

x_batch = np.zeros((len(X_test), 3072))
i = 0
for j in X_test:
    j = j.tolist()
    x_batch[i] = j
    i += 1

labels = ['airplane','car']
y_batch = []
i = 0
for j in y_test:
    j = j.decode('UTF-8')
    k = labels.index(j)
    y_batch.append(k)
    i += 1

# Convert class vectors to binary class matrices.
Y_test = np_utils.to_categorical(y_batch, nb_classes)

x_batch = np.reshape(x_batch, (-1, 32, 32, 3))

X_test = x_batch.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_test, axis=0)
#X_test -= mean_image
#X_test /= 128.

model = keras.models.load_model('resnet.mdl')
return_val = model.evaluate(X_test, Y_test,batch_size)
print(return_val)