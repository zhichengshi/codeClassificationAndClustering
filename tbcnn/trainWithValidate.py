import sys
sys.path.append('./')
from gensim.models.word2vec import Word2Vec
import logging
from cvrnn.config import *
import model
import tensorflow as tf
import os
from sampling import generateSample  # 这里使用tbcnn的采样方法
from sklearn.metrics import classification_report, accuracy_score
import _pickle as pkl
import numpy as np



def buildLogger(log_file, part):
    logger = logging.getLogger(part)
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    # FileHandler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def train_model(logdir, train_paths, valid_path, word2vec):
    logger = buildLogger("log/tbcnn_84_20/tbcnnTrain.log", "train")

    # init the network
    nodes_node, children_node, code_vector, logits = model.init_net(
        embedding_size, label_size
    )

    # for calculate the training accuracy
    out_node = model.out_layer(logits)

    label_node, loss = model.loss_layer(logits, label_size)

    global_ = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_, decay_steps, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    tf.summary.scalar('loss', loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)

    checkfile = os.path.join(logdir, 'tbcnn.ckpt')

    step = 1
    error_sum = 0
    max_valid_acc = 0
    for epoch in range(1, epochs+1):
        for train_path in train_paths:
            for nodes, children, label_vector in generateSample(train_path, word2vec):
                try:
                    _, _, summary, error, out = sess.run([learning_rate, train_step, summaries, loss, out_node], feed_dict={
                        nodes_node: nodes,
                        children_node: children,
                        label_node: label_vector,
                        global_: step
                    })
                except Exception:
                    continue
                error_sum += error
                if step % 10000 == 0:
                    print('Epoch:', epoch, 'Step:', step, 'Loss:', error_sum/10000)
                    error_sum = 0
                    writer.add_summary(summary, step)

                step += 1
        if epoch % 5 == 0:
            saver.save(sess, checkfile, epoch)
        # # 计算验证精度
        # valid_path='dataset/84-20/train/train3.pkl'
        # if valid_path and epoch % 1 == 0:
        #     correct_labels = []
        #     predictions = []
        #     print("Epoch", epoch, 'Computing validate accuracy...')
        #     for nodes, children, label_vector in generateSample(valid_path, word2vec):
        #         try:
        #             output = sess.run(out_node, feed_dict={
        #                 nodes_node: nodes,
        #                 children_node: children,
        #             })
        #         except Exception:
        #             continue

        #         correct_labels.append(np.argmax(label_vector))
        #         predictions.append(np.argmax(output))
        #     labels = []
        #     for i in range(1, label_size + 1):
        #         labels.append(str(i))
        #     validata_acc = accuracy_score(correct_labels, predictions)
        #     logger.info(f'Epoch:{epoch},ValidAccuracy:{validata_acc}')
        #     print('Accuracy:', validata_acc)  # 保存验证精度最高的模型

        #     saver.save(sess, checkfile, epoch)


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    word2vec = Word2Vec.load('dataset/104/withLeave/node_embedding_128').wv

    train_paths = []
    for root, dirs, files in os.walk('dataset/84-20/train'):
        for file in files:
            train_paths.append(os.path.join(root, file))

    valid_path = None
    sess_graph_path = "log/tbcnn_84_20"

    train_model(sess_graph_path, train_paths, valid_path, word2vec)
