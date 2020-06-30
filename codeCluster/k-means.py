from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl 
from tqdm import tqdm 


def calculateK(code_vectors,labels,draw_len,K):

    #定义一个随机数组，控制打印代码向量的个数
    shuffle_array=np.arange(draw_len) 
    # 降维
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    x_true = tsne.fit_transform(np.asarray(code_vectors)[shuffle_array])
    y_true=np.asarray(labels)[shuffle_array]
    color=[item+10 for item in y_true]




    # 绘制出所生成的数据
    plt.figure(figsize= (20, 20))
    plt.scatter(x_true[:, 0], x_true[:, 1], c= color, s= 10)
    plt.title("Origin data")
    plt.show()


    # 实例化k-means分类器
    clf = KMeans(n_clusters=K)
    y_predict = clf.fit_predict(x_true)
    
    # 绘制分类结果
    plt.figure(figsize= (6, 6))
    plt.scatter(x_true[:, 0], x_true[:, 1], c= y_predict, s= 10)
    plt.title("n_clusters= {}".format(20))
    
    ex = 0.5
    step = 0.01
    xx, yy = np.meshgrid(np.arange(x_true[:, 0].min() - ex, x_true[:, 0].max() + ex, step),
                        np.arange(x_true[:, 1].min() - ex, x_true[:, 1].max() + ex, step))
    
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz.shape = xx.shape
    
    plt.contourf(xx, yy, zz, alpha= 0.1)
    
    plt.show()
    


if __name__ == "__main__":
    path='dataset/84-20/20_vector.pkl'
    with open(path,'rb') as f:
        dataset=pkl.load(f)
        labels=dataset[1]
        embeddings=dataset[0]
        calculateK(embeddings,labels,len(labels),20)
