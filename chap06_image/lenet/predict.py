from PIL import Image
import tensorflow as tf
import numpy as np

img = Image.open('./air_car/car/513456060_1267fdde77.jpg')
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
img = img.reshape([-1, 3072])
img = img.astype('float32')

tf.reset_default_graph()

# placeholder is used for feeding data.
x = tf.placeholder("float", shape=[None, 3072],
                   name='x')  # none represents variable length of dimension. 784 is the dimension of MNIST data.

# reshape input data
x_image = tf.reshape(x, [-1, 32, 32, 3], name="x_image")

# Build a convolutional layer and maxpooling with random initialization
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1),
                      name="W_conv1")  # W is [row, col, channel, feature]
b_conv1 = tf.Variable(tf.zeros([32]), name="b_conv1")
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name="h_conv1")
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="h_pool1")

# Repeat again with 64 number of filters
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),
                      name="W_conv2")  # W is [row, col, channel, feature]
b_conv2 = tf.Variable(tf.zeros([64]), name="b_conv2")
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name="h_conv2")
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="h_pool2")

# Build a fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64], name="h_pool2_flat")
W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024], stddev=0.1), name='W_fc1')
b_fc1 = tf.Variable(tf.zeros([1024]), name='b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

# Dropout Layer
keep_prob = tf.placeholder("float", name="keep_prob")
h_fc1 = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

# Build a fully connected layer with softmax
W_fc2 = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.1), name='W_fc2')
b_fc2 = tf.Variable(tf.zeros([2]), name='b_fc2')
#y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name="y")
y = tf.matmul(h_fc1, W_fc2) + b_fc2
prd_y = tf.argmax(y, 1)

# Create Session
sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)))  # open a session which is a envrionment of computation graph.
sess.run(tf.global_variables_initializer())  # initialize the variables
saver = tf.train.Saver()
saver.restore(sess, save_path='./model/CNN')

# training the MLP
y = sess.run(prd_y, feed_dict={x: img,
                                    keep_prob: 1.0})  # placeholder's none length is replaced by i:i+100 indexes

print(y)