import sys
sys.path.append('./')
from tqdm import tqdm
import _pickle as pkl
import numpy as np
from cvrnn.sampling import processTree, _pad
import random
from cvrnn.config import label_size



def generateSample(path, word2vec):
    with open(path, "rb") as f:
        datas = pkl.load(f)
        random.shuffle(datas)

    for data in tqdm(datas):
        label = data[0]
        tree = data[1]
        nodes, children, max_children_size = processTree([tree], word2vec)
        # 根据标记获得对应于该标记的向量
        label_vector = np.eye(label_size, dtype=int)[int(label) - 1]

        yield [nodes], [children], [label_vector]


