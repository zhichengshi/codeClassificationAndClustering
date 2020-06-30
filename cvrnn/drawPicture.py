# from utils import draw2DPicture
import _pickle as pkl
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def draw(code_vectors,labels):
    # 降维
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    x_true = tsne.fit_transform(np.asarray(code_vectors))
    y_true=np.asarray(labels)
    color=[item+10 for item in y_true]

    # 绘制出所生成的数据
    plt.figure(figsize= (10, 10))
    plt.scatter(x_true[:, 0], x_true[:, 1], c= color, s= 2)
    plt.show()


path='dataset/104/withLeave/test_vector.pkl'
with open(path,'rb') as f:
    dataset=pkl.load(f)
    labels=dataset[1]
    code_vectors=dataset[0]

draw(code_vectors,labels)

