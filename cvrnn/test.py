import os
import sys
sys.path.append('./')

import faiss
import numpy as np
from cvrnn.config import *
import cvrnn.model as cmodel
from cvrnn.sampling import *
import tensorflow as tf
import sklearn.cluster as sc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import math
from gensim.models.word2vec import Word2Vec
from sampling import generateSample  # 这里使用tbcnn的采样方法


if __name__ == "__main__":
    log_dir='log/84-20/cvrnn.ckpt-1'
    word2vec = Word2Vec.load('dataset/104/withLeave/node_embedding_128').wv
    # innitial the network
    nodes_node, children_node, statement_len_list, code_vector, logits = cmodel.init_net(embedding_size, label_size)
    sess = tf.Session()

    # load
    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        saver.restore(sess, log_dir)

    out_node = cmodel.out_layer(logits)
    # 计算测试精度
    test_path='dataset/84-20/20.pkl'
    correct_labels = []
    predictions = []
    for nodes, children, statement_len, label_vector in generateSample(test_path, word2vec):
        try:
            output = sess.run(out_node, feed_dict={
                nodes_node: nodes,
                children_node: children,
                statement_len_list: statement_len
            })
        except Exception:
            continue

        correct_labels.append(np.argmax(label_vector))
        predictions.append(np.argmax(output))
    labels = []
    for i in range(1, label_size + 1):
        labels.append(str(i))
    valid_acc = accuracy_score(correct_labels, predictions)
    print('Accuracy:', valid_acc)  