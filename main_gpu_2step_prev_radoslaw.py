'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

'''
For some parts in the following code, reference from 

Copyright (c) 2018, by the Authors: Amir H. Abdi

This script is freely available under the MIT Public License.
Please see the License file in the root for details.
The following code snippet will convert the keras model files
to the freezed .pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
'''


'''
Referenced websites from 

https://blog.csdn.net/weixin_33972649/article/details/87505050

'''

##import tensorflow as tf
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import os, re
from keras.models import Sequential
from keras.layers import *
from keras.utils import multi_gpu_model
# from create_1hot_encoding import get_node_1hot
from lstm_ae import *
from numpy import array, argmax, savetxt
import sys

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
	
	##自己写的
	# args = parser.parse_args()
	# if args.theano_backend is True and args.quantize is True:
		# raise ValueError("Quantize feature does not work with theano backend.")

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
    walks = [map(str, walk) for walk in walks]

    return walks

def get_seqs_lstm(curr_file, seq_dir, one_hot_dir, L):

    seqs_1hot = []
    f_curr = open(seq_dir + curr_file, 'r')
    one_hot_list = get_1hot_list(curr_file, one_hot_dir)
    for line in f_curr:
        seq = line.rstrip().split(',')
        seq_1hot = get_seq_1hot_fromList(seq, one_hot_list)
        seqs_1hot.append(seq_1hot)

    return array(seqs_1hot), one_hot_list

#def create_lstm(n2v_nodes, n2v_embs, seqs, lstm_one_hot_list):
def create_lstm(seqs, lstm_one_hot_list, L):
    encoded_length = len(lstm_one_hot_list)
    n_in = L
    n_out = L

    model = Sequential()
    # model.add(LSTM(emb_size, batch_input_shape=(batch_size, n_in, encoded_length)))
    #model.add(CuDNNLSTM(emb_size, batch_input_shape=(len(seqs), n_in, encoded_length)))
    model.add(CuDNNLSTM(emb_size, input_shape=(n_in, encoded_length)))
    model.add(RepeatVector(n_out))
    model.add(CuDNNLSTM(emb_size, return_sequences=True))
    model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))

    print(model.summary())
    #parallel_model = multi_gpu_model(model, gpus=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    #parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # train by sequences

    X = seqs
    model.fit(X, X, epochs=500, verbose=2, shuffle=False)
    #print(X)
    #parallel_model.fit(X, X, epochs=500, verbose=2, shuffle=False)
    #model.set_weights(parallel_model.get_weights())
    #print(model.layers[0].get_weights())
    return model

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

    model = Sequential()
    model.add(CuDNNLSTM(emb_size, input_shape=(n_in, encoded_length)))
    model.add(RepeatVector(n_out))
    model.add(CuDNNLSTM(emb_size, return_sequences=True))
    model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    array = np.random.rand(encoded_length, 4 * emb_size)

    # use n2v weights as initials in lstm
    #print(n2v_nodes)
    for node in lstm_one_hot_list:
        if node in n2v_nodes:
            #print("********************")
            array[lstm_one_hot_list.index(node), :emb_size] = n2v_embs[n2v_nodes.index(node)]

    #print(array[lstm_one_hot_list.index('3'), :emb_size])
    #print(n2v_embs[n2v_nodes.index('3')])

    array[:, emb_size:] = model.layers[0].get_weights()[0][:, emb_size:]

    new_weights = []
    new_weights.append(array)
    new_weights.append(model.layers[0].get_weights()[1])
    new_weights.append(model.layers[0].get_weights()[2])

    #print(new_weights)
    model.layers[0].set_weights(new_weights)

    X = seqs
    model.fit(X, X, epochs=500, verbose=2, shuffle=False)
    return model

def create_n2v(walks,lstm_one_hot_list, lstm_embs):
    model = Word2Vec(size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    #dir = 'data/dnc/test/'
    dir1 = 'C:/Users/Mimo/Desktop/l10/'
    edge_dir = dir1 + 'edgeList_w/'

    dir2 = 'C:/Users/Mimo/Desktop/l10/'
    emb_dir = dir2 + 'node_emb/'

    dir3 = 'C:/Users/Mimo/Desktop/l10/'
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
    for i in range(args.firstFile, args.lastFile + 1):
    #for i in range(50, 51):
        curr_file = 'graph_' + str(i)
        print(curr_file)

        args.input = edge_dir + curr_file
        args.output = emb_dir + curr_file

        seqs, lstm_one_hot_list = get_seqs_lstm(curr_file, seq_dir, one_hot_dir, L)
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
            lstm_model = create_lstm(seqs[0:num_seqs], lstm_one_hot_list, L)
            lstm_embs = lstm_model.layers[0].get_weights()[0][:, 0:emb_size]

            n2v_model = create_n2v(walks[0:num_walks], lstm_one_hot_list, lstm_embs)
            n2v_embs = n2v_model.wv.syn0  # weights of the nodes, N*d
            n2v_nodes = n2v_model.wv.index2word  # node itself with the order of syn0, N*1

            n2v_model.wv.save_word2vec_format(args.output)
            count += 1
            #print(n2v_nodes)

        else:
            #print('Hi22')
            #print(n2v_nodes)
            lstm_model = create_initialize_lstm(seqs[0:num_seqs], lstm_one_hot_list, L, n2v_nodes, n2v_embs)
            lstm_embs = lstm_model.layers[0].get_weights()[0][:, 0:emb_size]

            n2v_model = create_n2v(walks[0:num_walks], lstm_one_hot_list, lstm_embs)
            n2v_embs = n2v_model.wv.syn0  # weights of the nodes, N*d
            n2v_nodes = n2v_model.wv.index2word  # node itself with the order of syn0, N*1

            n2v_model.wv.save_word2vec_format(args.output)
