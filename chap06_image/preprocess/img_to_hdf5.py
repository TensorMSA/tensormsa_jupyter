import os
import h5py
import numpy as np
from PIL import Image

directory = '/air_car'
folderlist = os.listdir(directory)
folderlist.sort()
filecnt = 0
image_arr = []
shape_arr = []
lable_arr = []
processcnt = 1

for folder in folderlist:
    try:
        filelist = os.listdir(directory + '/' + folder)
    except Exception as e:
        print(e)
        continue

    for filename in filelist:
        try:
            img = Image.open(directory + '/' + folder + '/' + filename)
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
            img.save(directory + '/' + str(filecnt) + '.jpg')
            img = np.array(img)
            shape_arr.append(img.shape)
            img = img.reshape([-1, 32, 32, 3])
            img = img.flatten()

            image_arr.append(img)
            lable_arr.append(folder.encode('utf8'))
            print(str(len(shape_arr))+' '+str(len(image_arr)))
            filecnt += 1

            print("Processcnt=" + str(processcnt) + " File=" + directory + " forder=" + folder + "  name=" + filename)
        except Exception as e:
            print(e)
            print("Processcnt=" + str(
                processcnt) + " ErrorFile=" + directory + " forder=" + folder + "  name=" + filename)
        processcnt += 1

    if filecnt > 0:
        output_path = '/air_car/hdf5'
        h5file = h5py.File(output_path, mode='w')
        dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
        hdf_features = h5file.create_dataset('image_features', (filecnt,), dtype=dtype)
        hdf_shapes = h5file.create_dataset('image_features_shapes', (filecnt, 3), dtype='int32')
        hdf_labels = h5file.create_dataset('targets', (filecnt,), dtype='S240')

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
