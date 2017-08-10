import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

def MLP():
    # download the mnist data.
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 


    # placeholder is used for feeding data.
    x = tf.placeholder("float", shape=[None, 784], name = 'x') # none represents variable length of dimension. 784 is the dimension of MNIST data.
    y_target = tf.placeholder("float", shape=[None, 10], name = 'y_target') # shape argument is optional, but this is useful to debug.


    # all the variables are allocated in GPU memory
    W1 = tf.Variable(tf.zeros([784, 256]), name = 'W1')   # create (784 * 256) matrix
    b1 = tf.Variable(tf.zeros([256]), name = 'b1')        # create (1 * 256) vector
    h1 = tf.sigmoid(tf.matmul(x, W1) + b1, name = 'h1')   # compute --> sigmoid(weighted summation)

    # Repeat again
    W2 = tf.Variable(tf.zeros([256, 10]), name = 'W2')     # create (256 * 10) matrix
    b2 = tf.Variable(tf.zeros([10]), name = 'b2')          # create (1 * 10) vector
    y = tf.nn.softmax(tf.matmul(h1, W2) + b2, name = 'y')  # compute classification --> softmax(weighted summation)


    # define the Loss function
    cross_entropy = -tf.reduce_sum(y_target*tf.log(y), name = 'cross_entropy')


    # define optimization algorithm
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
    # correct_prediction is list of boolean which is the result of comparing(model prediction , data)


    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
    # tf.cast() : changes true -> 1 / false -> 0
    # tf.reduce_mean() : calculate the mean



    # create summary of parameters
    tf.summary.histogram('weights_1', W1)
    tf.summary.histogram('weights_2', W2)
    tf.summary.histogram('y', y)
    tf.summary.scalar('cross_entropy', cross_entropy)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("/tmp/mlp")


    # Create Session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) # open a session which is a envrionment of computation graph.
    sess.run(tf.global_variables_initializer())# initialize the variables

    # training the MLP
    for i in range(5001): # minibatch iteraction
        batch = mnist.train.next_batch(100) # minibatch size
        sess.run(train_step, feed_dict={x: batch[0], y_target: batch[1]}) # placeholder's none length is replaced by i:i+100 indexes

        if i%500 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_target: batch[1]})
            print ("step %d, training accuracy: %.3f"%(i, train_accuracy))



            # calculate the summary and write.
            summary = sess.run(merged, feed_dict={x:batch[0], y_target: batch[1]})
            summary_writer.add_summary(summary , i)

    # for given x, y_target data set
    print  ("test accuracy: %g"% sess.run(accuracy, feed_dict={x: mnist.test.images, y_target: mnist.test.labels}))
    sess.close()

MLP()