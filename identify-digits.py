#this code is to look for digits in a full-sized multi-digit image.

#next step:
#identify digits y/n
#how to:
#in each 50x150 image:
#pull out random 32x32 squares from along the top of the image --> 0
image_samples = []
image_labels = []

for i in range(9000):
    if i%90 == 0:
            print(str(i / 90), "percent complete. ", sep = ' ', end='', flush=True)
    
    if i in range(0, 4500):
        random_image_pick_svhn = random.randint(0, len(house_number_X_train))
        temp_svhn_image = house_number_X_train[random_image_pick_svhn]
        temp_svhn_image = temp_svhn_image.ravel()
        image_samples.append(temp_svhn_image)
        image_labels.append([1, 0])
        
    else:
        random_image_pick_svhn = random.randint(0, len(train_house_sequence))
        temp_svhn_image = train_house_sequence[random_image_pick_svhn]
        random_top_picker = random.randint(0, 9)
        cropped_svhn = temp_svhn_image[0:15, random_top_picker: random_top_picker+15, 0:3]
        cropped_svhn = misc.imresize(cropped_svhn, (32, 32, 3))
        cropped_svhn = cropped_svhn.ravel()
        image_samples.append(cropped_svhn)
        image_labels.append([0, 1])
        
image_samples = np.array(image_samples)
image_labels = np.array(image_labels)

print(image_samples.shape)
print(image_labels.shape)

plotImage(image_samples[0].reshape(32, 32, 3)) # a digit
plotImage(image_samples[5421].reshape(32, 32, 3)) # not a digit

#a fourth network for determining if something is a digit or not
svhndigit_X_train = image_samples / 255.
svhndigit_y_train = image_labels

#build a test set
svhndigit_X_train, svhndigit_X_test, svhndigit_y_train, svhndigit_y_test = train_test_split(svhndigit_X_train, svhndigit_y_train, test_size = 0.20, random_state = 7)
#build a validation set
svhndigit_X_test, svhndigit_X_val, svhndigit_y_test, svhndigit_y_val = train_test_split(svhndigit_X_test, svhndigit_y_test, test_size = 0.5, random_state = 7)

print(svhndigit_X_train.shape)
print(svhndigit_X_test.shape)
print(svhndigit_X_val.shape)

svhndigit_X_train = svhndigit_X_train.reshape(7200, 32, 32, 3)
svhndigit_X_test = svhndigit_X_test.reshape(900, 32, 32, 3)
svhndigit_X_val = svhndigit_X_val.reshape(900, 32, 32, 3)

epochs_4 = 1000
num_examples_4 = 7200
batch_size_4 = 1024

start1 = timer()

model4 = tf.Graph()
with model4.as_default():
    with tf.device('/gpu:0'):

        W4_conv0 = tf.Variable(tf.truncated_normal(shape = [32, 32, 3, 32], stddev = 0.1), name = "W_0_4")
        W4_conv1 = tf.Variable(tf.truncated_normal(shape =[5, 5, 32, 64], stddev = 0.1), name = "W_1_4")
        #W4_conv2 = tf.Variable(tf.truncated_normal(shape =[7, 7, 64, 128], stddev = 0.1), name = "W_2_4")
        W4_conv3 = tf.Variable(tf.truncated_normal(shape =[5, 5, 64, 64], stddev = 0.1), name = "W_3_4")
        W4_fc1 = tf.Variable(tf.truncated_normal(shape =[4*4*64, 384], stddev = 0.1), name = "W_4_4")
        W4_fc2 = tf.Variable(tf.truncated_normal(shape =[384, 192], stddev = 0.1), name = "W_5_4")
        #W4_fc3 = tf.Variable(tf.truncated_normal(shape =[200, 100], stddev = 0.1), name = "W_6_4")
        W4_fc4 = tf.Variable(tf.truncated_normal(shape =[192, 2], stddev = 0.1), name = "W_7_4")

        b4_conv0 = tf.Variable(tf.constant(0.1, shape = [32]), name = "b_0_4")
        b4_conv1 = tf.Variable(tf.constant(0.1, shape = [64]), name = "b_1_4")
        #b4_conv2 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_2_4")
        b4_conv3 = tf.Variable(tf.constant(0.1, shape = [64]), name = "b_3_4")
        b4_fc1 = tf.Variable(tf.constant(0.1, shape = [384]), name = "b_4_4")
        b4_fc2 = tf.Variable(tf.constant(0.1, shape = [192]), name = "b_5_4")
        #b4_fc3 = tf.Variable(tf.constant(0.1, shape = [100]), name = "b_6_4")
        b4_fc4 = tf.Variable(tf.constant(0.1, shape = [2]), name = "b_7_4")

        x_4 = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
        y_4 = tf.placeholder(tf.float32, shape = [None, 2])
        keep_prob_4 = tf.placeholder(tf.float32)
        #x_image = tf.reshape(x_1, [-1,28,28,1])

        layer_0_4 = conv2d(x_4, W4_conv0)
        layer_0_4 = tf.add(layer_0_4, b4_conv0)
        layer_0_4 = tf.nn.relu(layer_0_4)
        layer_0_4 = max_pool_2x2(layer_0_4)
        #layer_0_4 = tf.nn.lrn(layer_0_4, 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)

        #layer_0_4 = tf.nn.dropout(layer_0_4, keep_prob_4)

        layer_1_4 = conv2d(layer_0_4, W4_conv1)
        layer_1_4 = tf.add(layer_1_4, b4_conv1)
        layer_1_4 = tf.nn.relu(layer_1_4)
        layer_1_4 = max_pool_2x2(layer_1_4)

        #layer_2_4 = conv2d(layer_1_4, W4_conv2)
        #layer_2_4 = tf.add(layer_2_4, b4_conv2)
        #layer_2_4 = tf.nn.relu(layer_2_4)
        #layer_2_4 = max_pool_2x2(layer_2_4)

        layer_3_4 = conv2d(layer_1_4, W4_conv3)
        layer_3_4 = tf.add(layer_3_4, b4_conv3)
        layer_3_4 = tf.nn.relu(layer_3_4)
        #layer_3_4 = tf.nn.lrn(layer_3_4, 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)
        layer_3_4 = max_pool_2x2(layer_3_4)

        layer_3_4 = tf.nn.dropout(layer_3_4, keep_prob_4)

        layer_3_4 = tf.reshape(layer_3_4, [-1, 4*4*64])

        layer_4_4 = tf.add(tf.matmul(layer_3_4, W4_fc1), b4_fc1)
        layer_4_4 = tf.nn.relu(layer_4_4)

        #layer_4_4 = tf.nn.dropout(layer_4_4, keep_prob_3)

        layer_5_4 = tf.add(tf.matmul(layer_4_4, W4_fc2), b4_fc2)
        layer_5_4 = tf.nn.relu(layer_5_4)

        #layer_6_4 = tf.add(tf.matmul(layer_5_4, W4_fc3), b4_fc3)
        #layer_6_4 = tf.nn.relu(layer_6_4)

        layer_7_4 = tf.add(tf.matmul(layer_5_4, W4_fc4), b4_fc4)

        cost_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_7_4, labels = y_4))
        #optimizer_4 = tf.train.GradientDescentOptimizer(0.1).minimize(cost_4)
        optimizer_4 = tf.train.AdamOptimizer(1e-4).minimize(cost_4)

        correct_prediction_4 = tf.equal(tf.argmax(layer_7_4, 1), tf.argmax(y_4, 1))
        accuracy_4 = tf.reduce_mean(tf.cast(correct_prediction_4, tf.float32))
    
        init = tf.global_variables_initializer()
        
tf.reset_default_graph()

def save4():
    save_file_4 = './train_model4.ckpt'
    saver4 = tf.train.Saver({"W_0_4": W4_conv0,
                            "W_1_4": W4_conv1,
                            #"W_2_4": W4_conv2,
                            "W_3_4": W4_conv3,
                            "W_4_4": W4_fc1,
                            "W_5_4": W4_fc2,
                            #"W_6_4": W4_fc3,
                            "W_7_4": W4_fc4,
                            "b_0_4": b4_conv0,
                            "b_1_4": b4_conv1,
                            #"b_2_4": b4_conv2,
                            "b_3_4": b4_conv3,
                            "b_4_4": b4_fc1,
                            "b_5_4": b4_fc2,
                            #"b_6_4": b4_fc3,
                            "b_7_4": b4_fc4})
    return saver4, save_file_4

def train4(tfgraph, tfepochs, tfbatch_size, xtrain, ytrain, xtest, ytest, xval, yval, saver, save_file):

    with tf.Session(graph = tfgraph) as sess:
        sess.run(init)
        num_examples_4 = len(xtrain)
        print("Training...")

        for epoch in range(tfepochs):
            xtrain, ytrain = shuffle(xtrain, ytrain)
            #for offset in range(0, num_examples_3, batch_size_3):
                #end3 = offset + batch_size_3
                #batch_X_3, batch_y_3 = num_X_train[offset:end], num_y_train[offset:end]

                #optimizer_3.run(feed_dict = {x_3: batch_X_3, y_3: batch_y_3, keep_prob_3: 0.5})
            optimizer_4.run(feed_dict = {x_4: xtrain[0:batch_size_4], y_4: ytrain[0:batch_size_4], keep_prob_4: 0.5})


            if epoch % 50 == 0:
                xval, yval = shuffle(xval, yval)
                train_accuracy4 = accuracy_4.eval(feed_dict = {x_4: xval[0:batch_size_4], y_4: yval[0:batch_size_4], keep_prob_4: 1.})
                print("step %d, validation accuracy %g"%(epoch, train_accuracy4))

        xtest, ytest = shuffle(xtest, ytest)
        print("test accuracy %g"%accuracy_4.eval(feed_dict = {x_4: xtest[0:batch_size_4], y_4: ytest[0:batch_size_4], keep_prob_4: 1.}))

        saver.save(sess, save_file)
        print("")
        print("trained model saved")   
    
    end1 = timer()
    print("time: ", end1 - start1)
    
#train4(model4, epochs_4, batch_size_4, svhndigit_X_train, svhndigit_y_train, svhndigit_X_test, svhndigit_y_test, svhndigit_X_val, svhndigit_y_val, save4()[0], save4()[1])
tf.reset_default_graph()

#MODEL 4 TESTING SET ACCURACY:
with tf.Session(graph = model4) as sess:
    save4()[0].restore(sess, save4()[1])
    feed_dict = {x_4: svhndigit_X_test, y_4: svhndigit_y_test, keep_prob_4: 1.}
    print("Test Accuracy: ", accuracy_4.eval(feed_dict = feed_dict))
tf.reset_default_graph()
