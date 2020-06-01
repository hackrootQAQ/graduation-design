import tensorflow as tf
import databatch
import define
import numpy as np
import config
import pickle 
import os

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CFG = config.CONFIG()
CFG.Print()

def init_w(s, name = None):
    return tf.Variable(tf.truncated_normal(shape = s, stddev = 0.1), name = name)

def init_b(s, name = None):
    return tf.Variable(tf.constant(0.1, shape = s), name = name)

def conv2d(x, W, strides = [1, 1, 1, 1], downsize = False, name = None):
    if downsize: strides = [1, 4, 1, 1]
    return tf.nn.conv2d(x, W, strides = strides, padding = "SAME", name = name)

def res_block(input, input_channel_num, output_channel_num, name, downsize = False, 
    attention = False, training = True):
    with tf.variable_scope(name):
        ''' Add attention layer. '''
        if attention == True:
            ''' N * H * W * C'''
            s = input.shape.as_list()
            W_a_softmax = tf.nn.softmax(init_w([s[2], s[3]]), axis = 0, name = "W_a")
            W_a_reshape = tf.reshape(W_a_softmax, [1, 1, s[2], s[3]], name = "W_a_reshape")
            W_a_tile = tf.tile(W_a_reshape, [s[0], s[1], 1, 1], name = "W_a_tile")
            input = tf.multiply(input, W_a_tile)

        W_c1 = init_w([3, 1, input_channel_num, output_channel_num], name = "W_c1")
        h_c1 = conv2d(input, W_c1, downsize = downsize, name = "h_c1")
        h_c1_bn = tf.layers.batch_normalization(h_c1, training = True, name = "h_c1_bn")
        h_c1_relu = tf.nn.relu6(h_c1_bn)
        W_c2 = init_w([3, 1, output_channel_num, output_channel_num], name = "W_c2")
        h_c2 = conv2d(h_c1_relu, W_c2, name = "h_c2")
        h_c2_bn = tf.layers.batch_normalization(h_c2, training = True, name = "h_c2_bn")
        if input_channel_num == output_channel_num: 
            h_c2_add = tf.add(h_c2_bn, input, name = "h_c2_add")
            h_c2_relu = tf.nn.relu6(h_c2_add, name = "h_c2_relu")
        else: 
            W_up = init_w([1, 1, input_channel_num, output_channel_num], name = "W_up")
            h_c2_up = conv2d(input, W_up, downsize = downsize, name = "h_c2_up")
            h_c2_up_bn = tf.layers.batch_normalization(h_c2_up, training = True, name = "h_c2_up_bn")
            h_c2_add = tf.add(h_c2_bn, h_c2_up_bn, name = "h_c2_add")
            h_c2_relu = tf.nn.relu6(h_c2_add, name = "h_c2_relu")
        return h_c2_relu

"""
def get_sentences_vector(batch_size = CFG.batch_size, D = None):
    G = databatch.get_batch(batch_size, D)
    
    while True:
        tmp, fr = next(G)
        x, y = [], []
        for items in tmp:
            y.append(items[1])
            sens = [comment.content for comment in items[0]]
            x.append(bc.encode(sens))
        yield x, y, fr
"""

def reshape_matmul(A, B):
    return tf.einsum("aij,jk->aik", A, B)

if __name__ == "__main__":
    input_X = tf.placeholder(tf.float32, 
        [CFG.batch_size, CFG.num_comment, CFG.embedding_size],
        name = "input_comment")
    input_Y = tf.placeholder(tf.int64,
        [CFG.batch_size],
        name = "input_label")
    input_p = tf.placeholder(tf.float32,
        name = "dropout")
    input_L = tf.placeholder([CFG.batch_size],
        name = "sequen_length"
    )

    cell = tf.nn.rnn_cell.LSTMCell(CFG.num_comment, state_is_tuple = True)
    Cell = tf.contrib.rnn.DropoutWrapper(cell, input_p)
    output, _ = tf.nn.dynamic_rnn(cell = Cell, 
        inputs = input_X, 
        sequence_length = input_L, 
        dtype = tf.float32)
    output = tf.reshape(output, [CFG.batch_size, CFG.num_comment, CFG.embedding_size, 1])
    
    #output = tf.reshape(input_X, [CFG.batch_size, CFG.num_comment, CFG.embedding_size, 1])

    #192 * 192
    conv1_x_1 = res_block(output, 1, 4, "conv1_x_1", downsize = True, attention = CFG.attention)
    conv1_x_2 = res_block(conv1_x_1, 4, 4, "conv1_x_2")

    #48 * 48
    conv2_x_1 = res_block(conv1_x_2, 4, 16, "conv2_x_1", downsize = True, attention = CFG.attention)
    conv2_x_2 = res_block(conv2_x_1, 16, 16, "conv2_x_2")

    #12 * 12
    conv3_x_1 = res_block(conv2_x_2, 16, 64, "conv3_x_1", downsize = True, attention = CFG.attention)
    conv3_x_2 = res_block(conv3_x_1, 64, 64, "conv3_x_2")
    conv3_x_p = tf.nn.avg_pool(value = conv3_x_2,
        ksize = [1, 12, 1, 1],
        strides = [1, 12, 1, 1],
        padding = "SAME"
    )

    ret = tf.reshape(conv3_x_p, [-1, 768, 64])
    W_f1 = init_w([64, 1], name = "W_f1")
    b_f1 = init_b([768], name = "b_f1")
    f1 = tf.add(tf.reshape(reshape_matmul(ret, W_f1), [-1, 768]), b_f1)
    f1 = tf.nn.relu6(tf.layers.batch_normalization(f1, training = True))
    W_f2 = init_w([768, 96], name = "W_f2")
    b_f2 = init_b([96], name = "b_f2")
    f2 = tf.add(tf.matmul(f1, W_f2), b_f2)
    f2 = tf.nn.relu6(tf.layers.batch_normalization(f2, training = True))
    W_f3 = init_w([96, 12], name = "W_f3")
    b_f3 = init_b([12], name = "b_f3")
    out_ = tf.add(tf.matmul(f2, W_f3), b_f3)   
   
    loss_list = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = input_Y,
        logits = out_)
    loss = tf.reduce_mean(loss_list)
    predict_ans = tf.argmax(out_, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(predict_ans, input_Y), tf.float64))

    global_step = tf.Variable(0, trainable = False)
    lr = tf.train.exponential_decay(
        learning_rate = CFG.lr_base,
        global_step = global_step,
        decay_steps = CFG.lr_step,
        decay_rate = CFG.lr_decay,
        staircase = True
    )
    with tf.name_scope("train_op"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step = global_step)
                
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    saver = tf.train.Saver(max_to_keep = 0)
   
    _L = np.array([CFG.num_comment for i in range(CFG.batch_size)])
    with open("./train", "rb") as f: train_D = pickle.load(f)
    with open("./test", "rb") as f: test_D = pickle.load(f)
    S_train = databatch.get_mean_batch(batch_size = CFG.batch_size, D = train_D)
    ret_loss = []; min_loss = 1000
    ret_acc = []; max_acc = 0

    for step in range(CFG.max_steps):
        X, Y, fr = next(S_train)
        X_, Y_, fr_ = next(S_train)

        X = np.array(X)
        #X = X.reshape([-1, CFG.num_comment, CFG.embedding_size, 1])
        X_ = np.array(X_)
        #X_ = X_.reshape([-1, CFG.num_comment, CFG.embedding_size, 1])
        _ = sess.run(
            train_op,
            feed_dict = {input_X : X, input_Y : Y, input_p : 0.5, input_L : _L}
        )
        with tf.control_dependencies(update_ops):
            l, a = sess.run(
                [loss, acc],
                feed_dict = {input_X : X_, input_Y : Y_, input_p : 1, input_L : _L}
            )
        print("step %d, loss %.4f, acc %.4f" % (step, l, a))
        #ret_loss.append(l); min_loss = min(l, min_loss)
        #ret_acc.append(a); max_acc = max(a, max_acc)
        #if min(ret_loss[-min(len(ret_loss), 600):]) > min_loss: break
        #if max(ret_acc[-max(len(ret_acc), 600):]) < max_acc: break

        #if min(ret_loss[-min(len(ret_loss), 500):]) > min_loss or min_loss == l:
        #if max(ret_acc[-max(len(ret_acc), 500):]) < max_acc or max_acc == a:
        if (step + 1) % 2000 == 0:
            S_test = databatch.get_mean_batch(batch_size = CFG.batch_size, D = test_D)
            predict_a, predict_l = 0, 0
            for i in range(len(test_D) // CFG.batch_size):
                X, Y, fr = next(S_test)
                X = np.array(X)
                #X = X.reshape([-1, CFG.num_comment, CFG.embedding_size, 1])
                l, a = sess.run(
                    [loss, acc],
                    feed_dict = {input_X : X, input_Y : Y, input_p : 1, input_L : _L}
                )
                predict_a += a; predict_l += l
                
            num = (len(test_D) // CFG.batch_size)
            print("predict_loss %.4f, predict_acc %.4f" % (predict_l / num, predict_a / num))
    saver.save(sess, "./save/{}".format(CFG.model_name), global_step = global_step)
    