from sampling import generateSample  # 这里使用tbcnn的采样方法
from gensim.models.word2vec import Word2Vec
import math
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
import sklearn.cluster as sc
import tensorflow as tf
import model as cmodel
from cvrnn.config import *
import numpy as np
import faiss
import os
import sys
sys.path.append('./')


if __name__ == "__main__":
    log_dir = 'log/tbcnn_84_20/tbcnn.ckpt-30'
    word2vec = Word2Vec.load('dataset/104/withLeave/node_embedding_128').wv
    test_path = 'dataset/84-20/train/train1.pkl'
    # innitial the network
    nodes_node, children_node, code_vector, logits = cmodel.init_net(embedding_size, label_size)
    sess = tf.Session()

    # load
    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        saver.restore(sess, log_dir)

    out_node = cmodel.out_layer(logits)
    # 计算测试精度
    correct_labels = []
    predictions = []
    for nodes, children, label_vector in generateSample(test_path, word2vec):
        try:
            output = sess.run(out_node, feed_dict={
                nodes_node: nodes,
                children_node: children,
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
