import sys
sys.path.append('./')
import model
import numpy as np
import _pickle as pkl
import math
import tensorflow as tf
from cvrnn.config import *
import os
from sampling import generateSample
from gensim.models.word2vec import Word2Vec



def generateCodeMatrix(dataset_path, logdir, word2vec, dump_path):  # 构建代码向量
    # innitial the network
    nodes, children, code_vector, prediction = model.init_net(embedding_size, label_size)
    sess = tf.Session()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        saver.restore(sess, logdir)
    correct_labels = []
    # make predictions from the input
    predictions = []
    step = 1

    # 存储代码向量的列表
    code_vectors = []
    # 标记
    labels = []
    for nodes_node, children_node, label_vector in generateSample(dataset_path, word2vec):
        try:
            code_vector_element = sess.run([code_vector], feed_dict={
                nodes: nodes_node,
                children: children_node,
            })
        except Exception:
            continue

        code_vectors.append(code_vector_element[0][0])
        labels.append(np.argmax(label_vector) + 1)

    with open(dump_path, "wb") as f:
        assert len(code_vectors) == len(labels)
        pkl.dump((code_vectors, labels), f)


if __name__ == "__main__":
    dataset_path='dataset/84-20/20.pkl'
    logdir='log/tbcnn_84_20/tbcnn.ckpt-30'
    word2vec = Word2Vec.load('dataset/104/withLeave/node_embedding_128').wv
    dump_path='dataset/84-20/20_tbcnn_vectors.pkl'
    generateCodeMatrix(dataset_path,logdir,word2vec,dump_path)


