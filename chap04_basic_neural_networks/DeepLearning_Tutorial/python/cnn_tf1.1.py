import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()

def CNN():
    # download the mnist data.
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 


    # placeholder is used for feeding data.
    x = tf.placeholder("float", shape=[None, 784], name = 'x') # none represents variable length of dimension. 784 is the dimension of MNIST data.
    y_target = tf.placeholder("float", shape=[None, 10], name = 'y_target') # shape argument is optional, but this is useful to debug.


    
    # reshape input data
    x_image = tf.reshape(x, [-1,28,28,1], name="x_image")
    
    # Build a convolutional layer and maxpooling with random initialization
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="W_conv1") # W is [row, col, channel, feature]
    b_conv1 = tf.Variable(tf.zeros([32]), name="b_conv1")
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name="h_conv1")
    h_pool1 = tf.nn.max_pool( h_conv1 , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = "h_pool1")
    
    # Repeat again with 64 number of filters
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="W_conv2") # W is [row, col, channel, feature]
    b_conv2 = tf.Variable(tf.zeros([64]), name="b_conv2")
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name="h_conv2")
    h_pool2 = tf.nn.max_pool( h_conv2 , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = "h_pool2")
    
    # Build a fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="h_pool2_flat")
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1), name = 'W_fc1')
    b_fc1 = tf.Variable(tf.zeros([1024]), name = 'b_fc1')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

    
    # Dropout Layer
    keep_prob = tf.placeholder("float", name="keep_prob")
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")
    
    # Build a fully connected layer with softmax 
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name = 'W_fc2')
    b_fc2 = tf.Variable(tf.zeros([10]), name = 'b_fc2')
    y=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name="y")
    



    # define the Loss function
    cross_entropy = -tf.reduce_sum(y_target*tf.log(y), name = 'cross_entropy')
    

    # define optimization algorithm
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
    # correct_prediction is list of boolean which is the result of comparing(model prediction , data)


    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
    # tf.cast() : changes true -> 1 / false -> 0
    # tf.reduce_mean() : calculate the mean



    # create summary of parameters
    tf.summary.histogram('weights_1', W_conv1)
    tf.summary.histogram('weights_2', W_conv2)
    tf.summary.histogram('y', y)
    tf.summary.scalar('cross_entropy', cross_entropy)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("/tmp/cnn")

    
    # Create Session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))   # open a session which is a envrionment of computation graph.
    sess.run(tf.global_variables_initializer())# initialize the variables


    # training the MLP
    for i in range(5001): # minibatch iteraction
        batch = mnist.train.next_batch(100) # minibatch size
        sess.run(train_step, feed_dict={x: batch[0], y_target: batch[1], keep_prob: 0.5}) # placeholder's none length is replaced by i:i+100 indexes

        if i%500 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_target: batch[1], keep_prob: 1})
            print ("step %d, training accuracy: %.3f"%(i, train_accuracy))

            # calculate the summary and write.
            summary = sess.run(merged, feed_dict={x:batch[0], y_target: batch[1], keep_prob: 1})
            summary_writer.add_summary(summary , i)

    # for given x, y_target data set
    print  ("test accuracy: %g"% sess.run(accuracy, feed_dict={x: mnist.test.images[0:250], y_target: mnist.test.labels[0:250], keep_prob: 1}))
    sess.close()
CNN()
