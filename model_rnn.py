import tensorflow as tf
import databatch
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class RnnModel(object):

    def __init__(self):
        self.input_x = tf.placeholder(tf.float32, shape=[None, 768, 768], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, 12], name='input_y')
        self.seq_length = tf.placeholder(tf.float32, shape=[None], name='sequen_length')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.rnn()

    def rnn(self):

        with tf.name_scope('cell'):
            cell = tf.nn.rnn_cell.LSTMCell(768)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            cells = [cell for _ in range(2)]
            Cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


        with tf.name_scope('rnn'):
            #hidden一层 输入是[batch_size, seq_length, hidden_dim]
            #hidden二层 输入是[batch_size, seq_length, 2*hidden_dim]
            #2*hidden_dim = embendding_dim + hidden_dim
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=self.input_x, sequence_length=self.seq_length, dtype=tf.float32)
            output = tf.reduce_sum(output, axis=1)
            #output:[batch_size, seq_length, hidden_dim]

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([768, 12], stddev=0.1), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[12]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(0.001)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            #global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    def feed_data(self, x_batch, y_batch, seq_len, keep_prob):

        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.seq_length: seq_len,
                     self.keep_prob: keep_prob}

        return feed_dict

if __name__ == "__main__":
    with open("./train", "rb") as f: train_D = pickle.load(f)
    with open("./test", "rb") as f: test_D = pickle.load(f)
    S_train = databatch.get_rnn_batch(batch_size = 128, 
        max_length = 768, 
        num_class = 12, 
        eb_size = 768, 
        D = train_D)

    model = RnnModel()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    saver = tf.train.Saver(max_to_keep = 0)

    for step in range(20000):
        _X, _Y, _L = next(S_train)
        feed_dict = model.feed_data(_X, _Y, _L, 0.5)
        _, global_step, train_loss, train_accuracy = sess.run(
            [model.optimizer, model.global_step, model.loss, model.accuracy],
            feed_dict = feed_dict
        )
        print("step %d, loss %.4f, acc %.4f" % (step, train_loss, train_accuracy))

        if (step + 100) % 1 == 0:
            S_test = databatch.get_rnn_batch(batch_size = 128, 
                max_length = 768, 
                num_class = 12, 
                eb_size = 768, 
                D = test_D)
            predict_a, predict_l = 0, 0
            for i in range(len(test_D) // 128):
                _X, _Y, _L = next(S_test)
                feet_dict = model.feed_data(_X, _Y, _L, 1.0)
                global_step, train_loss, train_accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    feed_dict = feed_dict
                )
            predict_a += train_accuracy
            predict_l += train_loss

            num = (len(test_D) // 128)
            print("predict_loss %.4f, predict_acc %.4f" % (predict_l / num, predict_a / num))    