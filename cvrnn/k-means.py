from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples,davies_bouldin_score,fowlkes_mallows_score,jaccard_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl 
from tqdm import tqdm 
from utils import generateCodeMatrix
from munkres import Munkres, print_matrix
from sklearn.metrics import accuracy_score
from gensim.models.word2vec import Word2Vec
from sampling import generateSample

'''
该函数解决分类标签与实际标签不一致的问题
'''
# 根据L1映射L2的标记
def maplabels(L1, L2):
    L2 = L2+1
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass1 = len(Label1)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2*ind_cla1)

    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    index = index+1
    newL2 = np.zeros(L2.shape, dtype=int)
    for i in range(nClass2):
        for j in range(len(L2)):
            if L2[j] == index[i, 0]:
                newL2[j] = index[i, 1]

    return newL2



# 绘制聚类图
def drawCluster(code_vectors,labels,K):

    # 降维
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    x_true = tsne.fit_transform(np.asarray(code_vectors))
    y_true=np.asarray(labels)
    color=[item+10 for item in y_true]

    # # 绘制出所生成的数据
    plt.figure(figsize= (10, 10))
    plt.scatter(x_true[:, 0], x_true[:, 1], c= color, s= 2)
    plt.title("Origin data")
    plt.show()


    # 实例化k-means分类器
    clf = KMeans(n_clusters=K)
    y_predict = clf.fit_predict(x_true) # !!!这里可以可以使用未降维之前的向量作为输入

    
    # 将预测的类标记与实际标记之间实现映射
    y_predict=maplabels(y_true,y_predict)
    


    # # 外部指标
    JC=jaccard_score(y_true= y_true,y_pred=y_predict,average='macro')
    FMI=fowlkes_mallows_score(labels_true=y_true,labels_pred=y_predict)

    # 准确率
    ACC=accuracy_score(y_true,y_predict)

    print(f'JC:{JC} FMI:{FMI} ACC:{ACC}')

    
    # # 绘制分类结果
    # plt.figure(figsize= (6, 6))
    # plt.scatter(x_true[:, 0], x_true[:, 1], c= y_true, s= 10)
    # plt.title("n_clusters= {}".format(20))
    
    # ex = 0.5
    # step = 0.01
    # xx, yy = np.meshgrid(np.arange(x_true[:, 0].min() - ex, x_true[:, 0].max() + ex, step),
    #                     np.arange(x_true[:, 1].min() - ex, x_true[:, 1].max() + ex, step))
    
    # zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # zz.shape = xx.shape
    
    # plt.contourf(xx, yy, zz, alpha= 0.1)
    
    # plt.show()

def calMetric(code_vectors,labels,K,n):
    JCs,FMIs,ACCs=[],[],[]
    for i in tqdm(range(n)):
        # 降维
        tsne = TSNE(n_components=2, init='pca', verbose=1)
        x_true = tsne.fit_transform(np.asarray(code_vectors))
        y_true=np.asarray(labels)

        # 实例化k-means分类器
        clf = KMeans(n_clusters=K)
        y_predict = clf.fit_predict(x_true) # !!!这里可以可以使用未降维之前的向量作为输入

        # 将预测的类标记与实际标记之间实现映射
        y_predict=maplabels(y_true,y_predict)
        
        # # 外部指标
        JC=jaccard_score(y_true= y_true,y_pred=y_predict,average='macro')
        FMI=fowlkes_mallows_score(labels_true=y_true,labels_pred=y_predict)

        # 准确率
        ACC=accuracy_score(y_true,y_predict)

        JCs.append(JC)
        FMIs.append(FMI)
        ACCs.append(ACC)
        
    print(f"JC:{np.mean(JCs)},FMI:{np.mean(FMIs)},ACC:{np.mean(ACCs)}")

if __name__ == "__main__":
    ast_path='dataset/84-20/20.pkl'
    vector_path='dataset/84-20/20_tbcnn_vectors.pkl'
    log_dir='log/84-20/cvrnn.ckpt-10'

    # word2vec = Word2Vec.load('dataset/104/withLeave/node_embedding_128').wv

    # generateCodeMatrix(ast_path,log_dir,word2vec,vector_path,generateSample)

    with open(vector_path,'rb') as f:
        (code_vectors,labels)=pkl.load(f)
    
    calMetric(code_vectors,labels,20,10)


    

