'''
根据AST的先序遍历获得AST节点序列，然后使用word2vec获得词向量
'''
import pandas as pd 
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import _pickle as pkl 

class Embedding:
    def __init__(self,dataset,embedding_dim):
        self.dataset=dataset
        self.embedding_dim=embedding_dim
    
    def generateAstSeq(self):
        def preVisit(root,seq):
            seq.append(root.tag.lower())
            for child in root:
                preVisit(child,seq)
            return seq
        
        corpus=[]
        for data in tqdm(self.dataset):
            corpus.append(preVisit(data[1],[])) # dfs即深度优先获得AST节点序列
        
        w2v = Word2Vec(corpus, size=self.embedding_dim, workers=16, sg=1, min_count=5)
        w2v.save('dataset/104'+'/node_embedding_'+str(self.embedding_dim))

if __name__ == "__main__":
    # with open('dataset/104/withLeave/train.pkl','rb') as f:
    #     dataset=pkl.load(f)
    
    # Embedding=Embedding(dataset,128)
    # Embedding.generateAstSeq()
    word2vec = Word2Vec.load('dataset/104/withLeave/node_embedding_128').wv
    vocab = word2vec.vocab
    # max_token = word2vec.syn0.shape[0]
    print(vocab['unit'].index)

        



