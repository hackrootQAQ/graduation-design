import tensorflow as tf
import databatch
from bert_serving.client import BertClient
import define
import numpy as np
import config

CFG = config.CONFIG_B()
bc = BertClient(check_length = False)

def get_sentences_vector(batch_size = CFG.batch_size, D = None):
    G = databatch.get_raw_batch(batch_size, D)

    while True:
        tmp, fr = next(G)
        x, y = [], []
        for items in tmp:
            y.append(items[1])
            sens = [comment.content for comment in items[0]]
            bx = np.array(bc.encode(sens))
            x.append(bx.mean(0))
        yield x, np.eye(CFG.num_class)[y], fr

def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [CFG.num_class, CFG.batch_size, 1])
    return tf.einsum("aij,ajk->aik", v2, v1)

def Gaussian_Kernel(gamma, data):
    g = tf.constant(gamma) 
    dist = tf.reshape(tf.reduce_sum(tf.square(data), 1), [CFG.batch_size, 1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(data, tf.transpose(data))), tf.transpose(dist)))
    return tf.exp(tf.multiply(g, tf.abs(sq_dists)))

if __name__ == "__main__":
    X = tf.placeholder(
        shape = [CFG.batch_size, CFG.embedding_size],
        dtype = tf.float32
    )
    Y = tf.placeholder(
        shape = [CFG.num_class, CFG.batch_size],
        dtype = tf.float32
    )
    P = tf.placeholder(
        shape = [CFG.batch_size, CFG.embedding_size], 
        dtype = tf.float32
    ) 
    b = tf.Variable(tf.random_normal(shape = [CFG.num_class, CFG.batch_size]))

    K = Gaussian_Kernel(CFG.gamma, X)

    model_output = tf.matmul(b, K)
    loss_1 = tf.reduce_sum(b)
    bvec_cross = tf.matmul(tf.transpose(b), b)
    Y_cross = reshape_matmul(Y)
    loss_2 = tf.reduce_sum(tf.multiply(K, tf.multiply(bvec_cross, Y_cross)), [1, 2])
    loss = tf.reduce_sum(tf.negative(tf.subtract(loss_1, loss_2)))

    rA = tf.reshape(tf.reduce_sum(tf.square(X), 1), [CFG.batch_size, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(P), 1), [CFG.batch_size, 1])
    P_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(X, tf.transpose(P)))), tf.transpose(rB)) 
    P_K = tf.exp(tf.multiply(CFG.gamma, tf.abs(P_sq_dist)))

    predout = tf.matmul(tf.matmul(Y, b), P_K) 
    pred = tf.arg_max(predout - tf.expand_dims(tf.reduce_mean(predout, 1), 1), 0) 
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(Y, 0)), tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    with tf.name_scope("train_op"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(CFG.lr).minimize(loss)

    train_D, test_D = databatch.cut_two_parts(CFG.test_size)
    S_train = get_sentences_vector(batch_size = CFG.batch_size, D = train_D)

    for step in range(CFG.max_steps):
        input_X, input_Y, fr = next(S_train)
        _loss, _ = sess.run([loss, train_op], feed_dict = {X : input_X, Y : input_Y})
        _acc = sess.run(acc, feed_dict = {X : input_X, Y : input_Y, P : input_X})
        print("step %d, loss %.4f, acc %.4f" % (step, _loss, _acc))

        if (step + 1) % CFG.test_interval == 0:
            S_test = get_sentences_vector(batch_size = CFG.batch_size, D = test_D)
            predict_a, predict_l = 0, 0
            for i in range(len(test_D) // CFG.batch_size):
                input_X, input_Y, fr = next(S_test)
                _loss, _acc = sess.run([loss, acc], feed_dict = {X : input_X, Y : input_Y, P : input_X})
                predict_a += _acc; predict_l += _loss

            num = (len(test_D) // CFG.batch_size)
            print("predict_loss %.4f, predict_acc %.4f" % (predict_l / num, predict_a / num))