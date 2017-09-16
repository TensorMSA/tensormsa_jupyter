from PIL import Image
import numpy as np
import h5py

image_arr = []
shape_arr = []
lable_arr = []
img = Image.open('resize.jpeg')
img = np.array(img)
shape_arr.append(img.shape)
img = img.flatten()
image_arr.append(img)
lable_arr.append('elephant'.encode('utf8'))

output_path = 'resize.hdf5'
h5file = h5py.File(output_path, mode='w')
dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
hdf_features = h5file.create_dataset('image_features', (1,), dtype=dtype)
hdf_shapes = h5file.create_dataset('image_features_shapes', (1, 3), dtype='int32')
hdf_labels = h5file.create_dataset('targets', (1,), dtype='S240')

# Attach shape annotations and scales
hdf_features.dims.create_scale(hdf_shapes, 'shapes')
hdf_features.dims[0].attach_scale(hdf_shapes)

hdf_shapes_labels = h5file.create_dataset('image_features_shapes_labels', (3,), dtype='S7')
hdf_shapes_labels[...] = ['channel'.encode('utf8'),
                          'height'.encode('utf8'),
                          'width'.encode('utf8')]
hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
hdf_features.dims[0].attach_scale(hdf_shapes_labels)

# Add axis annotations
hdf_features.dims[0].label = 'batch'

for i in range(len(image_arr)):
    hdf_features[i] = image_arr[i]
    #hdf_shapes[i] = shape_arr[i]
    hdf_labels[i] = lable_arr[i]

h5file.flush()
h5file.close()

h5file = h5py.File(output_path, mode='r')
rawdata = h5file['image_features']
targets = h5file['targets']
print()