'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

"""
Reference From 
https://blog.csdn.net/amanfromearth/article/details/81057577
https://blog.csdn.net/leviopku/article/details/78510977
https://blog.csdn.net/weixin_33972649/article/details/87505050
https://blog.csdn.net/guotong1988/article/details/74748806
https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
"""

import time
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import os, re
# from keras.models import Sequential
# from keras.layers import *
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
# from keras.utils import multi_gpu_model
# from create_1hot_encoding import get_node_1hot
from lstm_ae import *
from numpy import array, argmax, savetxt
# import sys

start_time = time.time()

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=True)

    parser.add_argument('--firstFile', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--lastFile', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    return parser.parse_args()

def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

#create the walks from a single graph as input to main_n2v
def get_walks_n2v():
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    new_walks = []
    for walk in walks:
        walk = [str(i) for i in walk]
        new_walks.append(walk)
    #walks = [map(str, walk) for walk in walks]

    return new_walks

def get_seqs_lstm(curr_file, seq_dir, one_hot_dir, L):

    seqs_1hot = []
    f_curr = open(seq_dir + curr_file, 'r')
    one_hot_list = get_1hot_list(curr_file, one_hot_dir)
    for line in f_curr:
        seq = line.rstrip().split(',')
        seq_1hot = get_seq_1hot_fromList(seq, one_hot_list)
        seqs_1hot.append(seq_1hot)

    return array(seqs_1hot), one_hot_list

# The function from "cifar10_multi_gpu_train.py"
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers(models).
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def create_lstm(seqs, lstm_one_hot_list, L):
    encoded_length = len(lstm_one_hot_list)
    n_in = L
    n_out = L

    # print('seqs',seqs)

    X = seqs
    # print('shape',np.array(X).shape)
    print('shape',X.shape)
    cell_fun = tf.contrib.rnn.BasicLSTMCell
    def getcell(emb_size):
        return cell_fun(emb_size)

    input_data = tf.placeholder(tf.float32, [None, n_in, encoded_length])
    output_targets = tf.placeholder(tf.int32, [None, n_out, encoded_length])


	# all GPU in the server need to share the network variable
    with tf.variable_scope('change_weigths',reuse=tf.AUTO_REUSE):
        cell = tf.contrib.rnn.MultiRNNCell([getcell(emb_size) for _ in range(2)], state_is_tuple=True)



        initial_state = cell.zero_state(len(seqs), tf.float32)
        output_data, _ = tf.nn.dynamic_rnn(cell, input_data, initial_state=initial_state, dtype=tf.float32)

    output = tf.reshape(output_data, [-1, emb_size])

    weights = tf.Variable(tf.truncated_normal([emb_size, encoded_length]))
    bias = tf.Variable(tf.zeros(shape=[encoded_length]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)  # one fully connected layer

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=output_targets, logits=logits)

    total_loss = tf.reduce_mean(loss)


    # train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss)  # adam

    opt = tf.train.AdamOptimizer(learning_rate=0.005)
    tower_grads = []

    for i in range(N_GPU):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                # cur_loss = get_loss(sess,epoch,total_loss,X,X)
                # tf.get_variable 
                # tf.get_variable_scope().reuse_variables()
                # use the current GPU to calculate the gradients for all variables

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    # updated by average gradient
    apply_gradient_op = opt.apply_gradients(grads)
    # cell1 = tf.contrib.rnn.MultiRNNCell([getcell(emb_size)] , state_is_tuple=True)
    # cell2 = tf.contrib.rnn.MultiRNNCell([getcell(emb_size)])
    #
    # input_data = tf.placeholder(tf.float32, [None, n_in, encoded_length])
    # output_targets = tf.placeholder(tf.int32, [None, n_out, encoded_length])
    #
    # initial_state = cell1.zero_state(len(seqs), tf.float32)
    #
    # with tf.variable_scope('lstm1'):
    #     middle_data, middle_state = tf.nn.dynamic_rnn(cell1, input_data, initial_state=initial_state, dtype=tf.float32)
    #
    # # print('shape',middle_data.shape)
    #
    # middle_data = tf.reshape(middle_data, [-1, n_out, emb_size])
    # with tf.variable_scope('lstm2'):
    #
    #     output_data, _ = tf.nn.dynamic_rnn(cell2, middle_data, initial_state=middle_state, dtype=tf.float32)



    # calculate the average gradient
	# variables used in different GPU may cause extremely long execution or unusual complication errors.
	#Use tf.ConfigProto(allow_soft_placement=True) automatically choose one existing and usable device to make execution

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config = tf.ConfigProto(allow_soft_placement=True)# solve extremely long execution or unusual complication errors, make sure there is usable device in the server
    with tf.Session(config=config).as_default() as sess :
        sess.run(init_op)
        for epoch in range(500):
            loss, _, = sess.run([total_loss,apply_gradient_op], feed_dict={input_data:X, output_targets:X})
            print('[INFO] Epoch: %d ,training loss: %.6f' % (epoch, loss))



            # n_chunk = len(X) // n_in
            # for batch in range(n_chunk):
            # loss, _, = sess.run([total_loss,train_op], feed_dict={input_data:X, output_targets:X})

            #
        saver.save(sess, './model/auto_encoder.ckpt')
    # sess = tf.get_default_session()
    # saver.restore(sess,tf.train.latest_checkpoint('./model'))
    model_reader = pywrap_tensorflow.NewCheckpointReader(r"./model/auto_encoder.ckpt")
    # var_dict = model_reader.get_variable_to_shape_map()
    # for key in var_dict:
    #     print("variable name: ", key)
    #     print(model_reader.get_tensor(key).shape)
    weights = model_reader.get_tensor('change_weigths/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel')# 259*512


    # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=)
    return weights

def update_lstm(n2v_nodes, n2v_embs, seqs, lstm_one_hot_list, model):
    # print('lstm_embs1')
    # print(model.layers[0].get_weights()[0][lstm_one_hot_list.index('1444'), :emb_size])

    encoded_length = len(lstm_one_hot_list)
    # initialize the weights with n2v weights
    array = np.random.rand(encoded_length, 4 * emb_size)

    # use n2v weights as initials in lstm
    for node in n2v_nodes:
        array[lstm_one_hot_list.index(node), :emb_size] = n2v_embs[n2v_nodes.index(node)]

    array[:, emb_size:] = model.layers[0].get_weights()[0][:, emb_size:]

    new_weights = []
    new_weights.append(array)
    new_weights.append(model.layers[0].get_weights()[1])
    new_weights.append(model.layers[0].get_weights()[2])

    # lstm_model.layers[0].get_weights()[0][:, 0:emb_size]
    model.layers[0].set_weights(new_weights)

    X = seqs
    model.fit(X, X, epochs=500, verbose=2, shuffle=False)

def create_initialize_lstm(seqs, lstm_one_hot_list, L, n2v_nodes, n2v_embs):
    encoded_length = len(lstm_one_hot_list)
    n_in = L
    n_out = L

    cell_fun = tf.contrib.rnn.BasicLSTMCell

    def getcell(emb_size):
        return cell_fun(emb_size)

    input_data = tf.placeholder(tf.float32, [None, n_in, encoded_length])
    output_targets = tf.placeholder(tf.int32, [None, n_out, encoded_length])

        


    with tf.variable_scope('change_weigths',reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.MultiRNNCell([getcell(emb_size) for _ in range(2)], state_is_tuple=True)



            initial_state = cell.zero_state(len(seqs), tf.float32)
            output_data, _ = tf.nn.dynamic_rnn(cell, input_data, initial_state=initial_state, dtype=tf.float32)


    output = tf.reshape(output_data, [-1, emb_size])

    weights = tf.Variable(tf.truncated_normal([emb_size, encoded_length]))
    bias = tf.Variable(tf.zeros(shape=[encoded_length]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)  # fully connected layer


    loss = tf.nn.softmax_cross_entropy_with_logits(labels=output_targets, logits=logits)

    total_loss = tf.reduce_mean(loss)

    # train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss)  #  adam

    opt = tf.train.AdamOptimizer(learning_rate=0.005)
    tower_grads = []

    for i in range(N_GPU):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                # cur_loss = get_loss(sess,epoch,total_loss,X,X)
                # tf.get_variable的命名空间
                # tf.get_variable_scope().reuse_variables()
                # use current GPU to calculate the gradient for all variables

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    # updated by average gradient
    apply_gradient_op = opt.apply_gradients(grads)


    saver = tf.train.Saver()



    array = np.random.rand(encoded_length, 4 * emb_size)  # 131*512

    # use n2v weights as initials in lstm
    print(n2v_nodes)
    for node in lstm_one_hot_list:
        if node in n2v_nodes:
            # print("********************")
            array[lstm_one_hot_list.index(node), :emb_size] = n2v_embs[n2v_nodes.index(node)]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config = tf.ConfigProto(allow_soft_placement=True)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:

        sess.run(init_op)
        for v in tf.trainable_variables():
            print(v.name)
            if v.name == 'change_weigths/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0':
                print('shape',v.shape)

                sess.run(tf.assign(v[:encoded_length,:emb_size],array[:,:emb_size]))

        for epoch in range(500):
            # n_chunk = len(X) // n_in
            # for batch in range(n_chunk):
            loss, _, = sess.run([total_loss, apply_gradient_op], feed_dict={input_data: seqs, output_targets: seqs})

            print('[INFO] Epoch: %d ,training loss: %.6f' % (epoch, loss))
        saver.save(sess, './model/auto_encoder.ckpt')
    # sess = tf.get_default_session()
    # saver.restore(sess,tf.train.latest_checkpoint('./model'))
    model_reader = pywrap_tensorflow.NewCheckpointReader(r"./model/auto_encoder.ckpt")
    #var_dict = model_reader.get_variable_to_shape_map()
    #for key in var_dict:
    #    print("variable name: ", key)
    weights = model_reader.get_tensor('change_weigths/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel')


    return weights

def create_n2v(walks,lstm_one_hot_list, lstm_embs):
    model = Word2Vec(size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)

    # print(walks)
    model.build_vocab(walks)
    for node in model.wv.index2word:
        model.wv.syn0[model.wv.index2word.index(node)] = lstm_embs[lstm_one_hot_list.index(node)]

    model.train(walks, total_examples=model.corpus_count, epochs=1)

    return model


#def update_n2v(walks, model):
def update_n2v(lstm_one_hot_list, lstm_embs, walks, model):
    # print('n2v_embs1')
    # print(model.wv.syn0[model.wv.index2word.index('1444')])

    #use lstm weights as initials in n2v
    for node in model.wv.index2word:
        model.wv.syn0[model.wv.index2word.index(node)] = lstm_embs[lstm_one_hot_list.index(node)]
        #model.wv.syn0[model.wv.index2word.index(node)] = model.wv.syn0[model.wv.index2word.index(node)]

    # print('n2v_embs2')
    # print(model.wv.syn0[model.wv.index2word.index('1444')])
    model.train(walks, total_examples=model.corpus_count, epochs=1)
    # print('n2v_embs3')
    # print(model.wv.syn0[model.wv.index2word.index('1444')])


if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    N_GPU = len([x.name for x in local_device_protos if x.device_type == 'GPU'])






    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #dir = 'data/dnc/test/'
    dir1 = './l10/'
    edge_dir = dir1 + 'edgeList_w/'

    dir2 = './l10/'#'../lf_any_2step_init/radoslaw/l10/'
    emb_dir = dir2 + 'node_emb/'

    dir3 = './l10/'
    seq_dir = dir3 + 'seqs/'
    one_hot_dir = dir3 + 'onehots/'

    #fnames = os.listdir(edge_dir)
    emb_size = 128
    args = parse_args()
    L = 10

    lstm_embs = []
    n2v_nodes = []
    n2v_embs = []

    count = 0
    for i in range(args.firstFile, args.lastFile + 29):# for all graph node
        tf.reset_default_graph()
    #for i in range(50, 51):
        curr_file = 'graph_' + str(i)
        print(curr_file)

        args.input = edge_dir + curr_file
        args.output = emb_dir + curr_file

        seqs, lstm_one_hot_list = get_seqs_lstm(curr_file, seq_dir, one_hot_dir, L)
        # print('shape',seqs.shape)
        print('length',len(lstm_one_hot_list))

        num_seqs = len(seqs)

        walks = get_walks_n2v()
        num_walks = len(walks)

        print('num_seqs')
        print(num_seqs)
        print('num_walks')
        print(num_walks)

        if count == 0:
            #print('Hi')
            # First we create models and then update their weights
            weights = create_lstm(seqs[0:num_seqs], lstm_one_hot_list, L)
            lstm_embs = weights[:, 0:emb_size]

            n2v_model = create_n2v(walks, lstm_one_hot_list, lstm_embs)
            n2v_embs = n2v_model.wv.syn0  # weights of the nodes, N*d
            n2v_nodes = n2v_model.wv.index2word  # node itself with the order of syn0, N*1

            n2v_model.wv.save_word2vec_format(args.output)
            count += 1
            #print(n2v_nodes)

        else:

            weights = create_initialize_lstm(seqs[0:num_seqs], lstm_one_hot_list, L, n2v_nodes, n2v_embs)
            lstm_embs = weights[:, 0:emb_size]

            n2v_model = create_n2v(walks[0:num_walks], lstm_one_hot_list, lstm_embs)
            n2v_embs = n2v_model.wv.syn0  # weights of the nodes, N*d
            n2v_nodes = n2v_model.wv.index2word  # node itself with the order of syn0, N*1

            n2v_model.wv.save_word2vec_format(args.output)

print("---Running time as %s seconds ---" % (time.time() - start_time))