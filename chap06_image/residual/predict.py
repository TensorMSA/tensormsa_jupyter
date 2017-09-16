import keras
from PIL import Image
import numpy as np
import h5py

h5file = h5py.File('../air_car.hdf5', mode='r')
X_test = h5file['image_features']

x_batch = np.zeros((len(X_test), 3072))
i = 0
for j in X_test:
    j = j.tolist()
    x_batch[i] = j
    i += 1

x_batch = np.reshape(x_batch, (-1, 32, 32, 3))

X_test = x_batch.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_test, axis=0)

#img = Image.open('/jupyter/chap06_image/air_car/airplane/55sm.jpg')
#img = Image.open('/jupyter/chap06_image/air_car/car/513456060_1267fdde77.jpg')
#img = Image.open('../air_car/airplane/55sm.jpg')
img = Image.open('../air_car/car/513456060_1267fdde77.jpg')
longer_side = max(img.size)
horizontal_padding = (longer_side - img.size[0]) / 2
vertical_padding = (longer_side - img.size[1]) / 2
img = img.crop(
    (
        -horizontal_padding,
        -vertical_padding,
        img.size[0] + horizontal_padding,
        img.size[1] + vertical_padding
    )
)
img = img.resize((32, 32), Image.ANTIALIAS)
img = np.array(img)
img = img.reshape([-1, 32, 32, 3])
img = img.astype('float32')
mean_image = np.mean(img, axis=0)
#img -= mean_image
#img /= 128.

model = keras.models.load_model('resnet.mdl')
return_val = model.predict(img)
print(return_val)