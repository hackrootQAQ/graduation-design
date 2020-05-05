import tensorflow as tf
import databatch
from bert_serving.client import BertClient
import define
import numpy as np
import json

class CONFIG(object):
    def __init__(self):
        with open("config.json", "r") as f:
            config_dic = json.load(f)
        self.batch_size = config_dic["batch_size"]
        self.embedding_size = config_dic["embedding_size"]
        self.num_comment = config_dic["num_comment"]
        self.num_class = config_dic["num_class"]
        self.lr_base = config_dic["lr_base"]
        self.lr_decay = config_dic["lr_decay"]
        self.lr_step = config_dic["lr_step"]
        self.max_steps = config_dic["max_steps"]
        self.test_size = config_dic["test_size"]

CFG = CONFIG()
bc = BertClient(check_length = False)

def init_w(s, name = None):
    return tf.Variable(tf.truncated_normal(shape = s, stddev = 0.1), name = name)

def init_b(s, name = None):
    return tf.Variable(tf.constant(0.1, shape = s), name = name)

def conv2d(x, W, strides = [1, 1, 1, 1], downsize = False, name = None):
    if downsize: strides = [1, 4, 4, 1]
    return tf.nn.conv2d(x, W, strides = strides, padding = "SAME", name = name)

def res_block(input, input_channel_num, output_channel_num, name, downsize = False, training = True):
    with tf.variable_scope(name):
        W_c1 = init_w([3, 3, input_channel_num, output_channel_num], name = "W_c1")
        h_c1 = conv2d(input, W_c1, downsize = downsize, name = "h_c1")
        h_c1_bn = tf.layers.batch_normalization(h_c1, training = training, name = "h_c1_bn")
        h_c1_relu = tf.nn.relu6(h_c1_bn)
        W_c2 = init_w([3, 3, output_channel_num, output_channel_num], name = "W_c2")
        h_c2 = conv2d(h_c1_relu, W_c2, name = "h_c2")
        h_c2_bn = tf.layers.batch_normalization(h_c2, training = training, name = "h_c2_bn")
        if input_channel_num == output_channel_num: 
            h_c2_add = tf.add(h_c2_bn, input, name = "h_c2_add")
            h_c2_relu = tf.nn.relu6(h_c2_add, name = "h_c2_relu")
        else: 
            W_up = init_w([1, 1, input_channel_num, output_channel_num], name = "W_up")
            h_c2_up = conv2d(input, W_up, downsize = downsize, name = "h_c2_up")
            h_c2_up_bn = tf.layers.batch_normalization(h_c2_up, training = training, name = "h_c2_up_bn")
            h_c2_add = tf.add(h_c2_bn, h_c2_up_bn, name = "h_c2_add")
            h_c2_relu = tf.nn.relu6(h_c2_add, name = "h_c2_relu")
        return h_c2_relu

def get_sentences_vector(batch_size = CFG.batch_size, D = None):
    G = databatch.get_batch(batch_size, D)
    
    while True:
        tmp, fr = next(G)
        x, y = [], []
        for items in tmp:
            y.append(items[1])
            sens = [comment.content for comment in items[0]]
            for i in range(len(sens)): 
                if sens[i] is None: sens[i] == "23333"
            x.append(bc.encode(sens))
        yield x, y, fr

if __name__ == "__main__":
    input_X = tf.placeholder(tf.float32, 
        [None, CFG.num_comment, CFG.embedding_size, 1],
        name = "input_comment")
    input_Y = tf.placeholder(tf.int64,
        [None],
        name = "input_label")

    #192 * 192
    conv1_x_1 = res_block(input_X, 1, 4, "conv1_x_1", downsize = True)
    conv1_x_2 = res_block(conv1_x_1, 4, 4, "conv1_x_2")

    #48 * 48
    conv2_x_1 = res_block(conv1_x_2, 4, 16, "conv2_x_1", downsize = True)
    conv2_x_2 = res_block(conv2_x_1, 16, 16, "conv2_x_2")

    #12 * 12
    conv3_x_1 = res_block(conv2_x_2, 16, 64, "conv3_x_1", downsize = True)
    conv3_x_2 = res_block(conv3_x_1, 64, 64, "conv3_x_2")
    conv3_x_p = tf.nn.avg_pool(value = conv3_x_2,
        ksize = [1, 12, 12, 1],
        strides = [1, 12, 12, 1],
        padding = "SAME"
    )

    ret = tf.reshape(conv3_x_p, [-1, 64])
    W_f1 = init_w([64, 12], name = "W_f1")
    b_f1 = init_b([12], name = "b_f1")
    out_ = tf.add(tf.matmul(ret, W_f1), b_f1)

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
    
    train_D, test_D = databatch.cut_two_parts(CFG.test_size)

    for step in range(CFG.max_steps):
        S_train = get_sentences_vector(batch_size = CFG.batch_size, D = train_D)
        X, Y, fr = next(S_train)
        
        try:
            X = np.array(X)
            X = X.reshape([-1, CFG.num_comment, CFG.embedding_size, 1])
            l, a, _ = sess.run(
                [loss, acc, train_op],
                feed_dict = {input_X : X, input_Y : Y}
            )
        except:
            with open("wrong_data.txt", "a+") as f:
                f.write(str(fr))
        
        print("step %d, loss %.4f, acc %.4f" % (step, l, a))
        
        if (step + 1) % 2000 == 0:
            S_test = get_sentences_vector(batch_size = CFG.batch_size, D = test_D)
            predict_a, predict_l = 0, 0
            for i in range(len(test_D) // CFG.batch_size):
                X, Y, fr = next(S_test)
                
                try:
                    X = np.array(X)
                    X = X.reshape([-1, CFG.num_comment, CFG.embedding_size, 1])
                    l, a = sess.run(
                        [loss, acc],
                        feed_dict = {input_X : X, input_Y : Y}
                    )
                    predict_a += a; predict_l += l
                except: 
                    with open("wrong_data.txt", "a+") as f:
                        f.write(str(fr))
                
            num = (len(test_D) // CFG.batch_size)
            print("predict_loss %.4f, predict_acc %.4f" % (predict_l / num, predict_a / num))