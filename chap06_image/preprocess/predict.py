import keras
from PIL import Image
import numpy as np

img = Image.open('/jupyter/chap06_image/air_car/airplane/55sm.jpg')
#img = Image.open('/jupyter/chap06_image/air_car/car/513456060_1267fdde77.jpg')
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
img /= 128.

model = keras.models.load_model('resnet.mdl')
return_val = model.predict(img)
print(return_val)