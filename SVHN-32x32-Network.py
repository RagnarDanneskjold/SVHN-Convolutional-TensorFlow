#build a simple convolutional network to classify the 32x32 images
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

num_X_train = house_number_X_train / 255.
num_y_train = house_number_y_train
num_X_test = house_number_X_test / 255.
num_y_test = house_number_y_test

#build a validation set
num_X_test, num_X_val, num_y_test, num_y_val = train_test_split(num_X_test, num_y_test, test_size = 0.5, random_state = 7)

epochs_3 = 10000
num_examples_3 = 73257
batch_size_3 = 1024

start1 = timer()

model3 = tf.Graph()
with model3.as_default():
    with tf.device('/gpu:0'):

        W3_conv0 = tf.Variable(tf.truncated_normal(shape = [32, 32, 3, 32], stddev = 0.1), name = "W_0_3")
        W3_conv1 = tf.Variable(tf.truncated_normal(shape =[5, 5, 32, 64], stddev = 0.1), name = "W_1_3")
        #W3_conv2 = tf.Variable(tf.truncated_normal(shape =[7, 7, 64, 128], stddev = 0.1), name = "W_2_3")
        W3_conv3 = tf.Variable(tf.truncated_normal(shape =[5, 5, 64, 64], stddev = 0.1), name = "W_3_3")
        W3_fc1 = tf.Variable(tf.truncated_normal(shape =[4*4*64, 384], stddev = 0.1), name = "W_4_3")
        W3_fc2 = tf.Variable(tf.truncated_normal(shape =[384, 192], stddev = 0.1), name = "W_5_3")
        #W3_fc3 = tf.Variable(tf.truncated_normal(shape =[200, 100], stddev = 0.1), name = "W_6_3")
        W3_fc4 = tf.Variable(tf.truncated_normal(shape =[192, 10], stddev = 0.1), name = "W_7_3")

        b3_conv0 = tf.Variable(tf.constant(0.1, shape = [32]), name = "b_0_3")
        b3_conv1 = tf.Variable(tf.constant(0.1, shape = [64]), name = "b_1_3")
        #b3_conv2 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_2_3")
        b3_conv3 = tf.Variable(tf.constant(0.1, shape = [64]), name = "b_3_3")
        b3_fc1 = tf.Variable(tf.constant(0.1, shape = [384]), name = "b_4_3")
        b3_fc2 = tf.Variable(tf.constant(0.1, shape = [192]), name = "b_5_3")
        #b3_fc3 = tf.Variable(tf.constant(0.1, shape = [100]), name = "b_6_3")
        b3_fc4 = tf.Variable(tf.constant(0.1, shape = [10]), name = "b_7_3")

        x_3 = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
        y_3 = tf.placeholder(tf.float32, shape = [None, 10])
        keep_prob_3 = tf.placeholder(tf.float32)
        #x_image = tf.reshape(x_1, [-1,28,28,1])

        layer_0_3 = conv2d(x_3, W3_conv0)
        layer_0_3 = tf.add(layer_0_3, b3_conv0)
        layer_0_3 = tf.nn.relu(layer_0_3)
        layer_0_3 = max_pool_2x2(layer_0_3)
        #layer_0_3 = tf.nn.lrn(layer_0_3, 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)

        #layer_0_3 = tf.nn.dropout(layer_0_3, keep_prob_3)

        layer_1_3 = conv2d(layer_0_3, W3_conv1)
        layer_1_3 = tf.add(layer_1_3, b3_conv1)
        layer_1_3 = tf.nn.relu(layer_1_3)
        layer_1_3 = max_pool_2x2(layer_1_3)

        #layer_2_3 = conv2d(layer_1_3, W3_conv2)
        #layer_2_3 = tf.add(layer_2_3, b3_conv2)
        #layer_2_3 = tf.nn.relu(layer_2_3)
        #layer_2_3 = max_pool_2x2(layer_2_3)

        layer_3_3 = conv2d(layer_1_3, W3_conv3)
        layer_3_3 = tf.add(layer_3_3, b3_conv3)
        layer_3_3 = tf.nn.relu(layer_3_3)
        #layer_3_3 = tf.nn.lrn(layer_3_3, 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)
        layer_3_3 = max_pool_2x2(layer_3_3)

        layer_3_3 = tf.nn.dropout(layer_3_3, keep_prob_3)

        layer_3_3 = tf.reshape(layer_3_3, [-1, 4*4*64])

        layer_4_3 = tf.add(tf.matmul(layer_3_3, W3_fc1), b3_fc1)
        layer_4_3 = tf.nn.relu(layer_4_3)

        #layer_4_3 = tf.nn.dropout(layer_4_3, keep_prob_3)

        layer_5_3 = tf.add(tf.matmul(layer_4_3, W3_fc2), b3_fc2)
        layer_5_3 = tf.nn.relu(layer_5_3)

        #layer_6_3 = tf.add(tf.matmul(layer_5_3, W3_fc3), b3_fc3)
        #layer_6_3 = tf.nn.relu(layer_6_3)

        layer_7_3 = tf.add(tf.matmul(layer_5_3, W3_fc4), b3_fc4)

        cost_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_7_3, labels = y_3))
        #optimizer_3 = tf.train.GradientDescentOptimizer(0.1).minimize(cost_3)
        optimizer_3 = tf.train.AdamOptimizer(1e-4).minimize(cost_3)

        correct_prediction_3 = tf.equal(tf.argmax(layer_7_3, 1), tf.argmax(y_3, 1))
        accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, tf.float32))
    
        init = tf.global_variables_initializer()
        
tf.reset_default_graph()

def save3():
    save_file_3 = './train_model3.ckpt'
    saver3 = tf.train.Saver({"W_0_3": W3_conv0,
                            "W_1_3": W3_conv1,
                            #"W_2_3": W3_conv2,
                            "W_3_3": W3_conv3,
                            "W_4_3": W3_fc1,
                            "W_5_3": W3_fc2,
                            #"W_6_3": W3_fc3,
                            "W_7_3": W3_fc4,
                            "b_0_3": b3_conv0,
                            "b_1_3": b3_conv1,
                            #"b_2_3": b3_conv2,
                            "b_3_3": b3_conv3,
                            "b_4_3": b3_fc1,
                            "b_5_3": b3_fc2,
                            #"b_6_3": b3_fc3,
                            "b_7_3": b3_fc4})
    return saver3, save_file_3

def train3(tfgraph, tfepochs, tfbatch_size, xtrain, ytrain, xtest, ytest, xval, yval, saver, save_file):


    with tf.Session(graph = tfgraph) as sess:
        sess.run(init)
        num_examples_3 = len(xtrain)
        print("Training...")

        for epoch in range(tfepochs):
            xtrain, ytrain = shuffle(xtrain, ytrain)
            #for offset in range(0, num_examples_3, batch_size_3):
                #end3 = offset + batch_size_3
                #batch_X_3, batch_y_3 = num_X_train[offset:end], num_y_train[offset:end]

                #optimizer_3.run(feed_dict = {x_3: batch_X_3, y_3: batch_y_3, keep_prob_3: 0.5})
            optimizer_3.run(feed_dict = {x_3: xtrain[0:batch_size_3], y_3: ytrain[0:batch_size_3], keep_prob_3: 0.5})


            if epoch % 50 == 0:
                xval, yval = shuffle(xval, yval)
                train_accuracy3 = accuracy_3.eval(feed_dict = {x_3: xval[0:batch_size_3], y_3: yval[0:batch_size_3], keep_prob_3: 1.})
                print("step %d, validation accuracy %g"%(epoch, train_accuracy3))

        xtest, ytest = shuffle(xtest, ytest)
        print("test accuracy %g"%accuracy_3.eval(feed_dict = {x_3: xtest[0:batch_size_3], y_3: ytest[0:batch_size_3], keep_prob_3: 1.}))

        saver.save(sess, save_file)
        print("")
        print("trained model saved")   
    
    end1 = timer()
    print("time: ", end1 - start1)
    
#train3(model3, epochs_3, batch_size_3, num_X_train, num_y_train, num_X_test, num_y_test, num_X_val, num_y_val, save3()[0], save3()[1])
tf.reset_default_graph()

#MODEL 3 TESTING SET ACCURACY:
with tf.Session(graph = model3) as sess:
    save3()[0].restore(sess, save3()[1])
    feed_dict = {x_3: num_X_test[:1000], y_3: num_y_test[:1000], keep_prob_3: 1.}
    print("Test Accuracy: ", accuracy_3.eval(feed_dict = feed_dict))
tf.reset_default_graph()
