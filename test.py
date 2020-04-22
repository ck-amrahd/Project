import pickle
import numpy as np

coco_adj = pickle.load(open('data/coco_adj.pkl', 'rb'))
nums = coco_adj['nums']
adj = coco_adj['adj']

nums = nums[:, np.newaxis]

adj = adj / nums

adj[adj < 0.5] = 0
adj[adj >= 0.5] = 1

adj = adj * 0.25 / (np.sum(adj, axis=0, keepdims=True) + 1e-6)
A = adj + np.identity(80, np.int)

D = np.power(np.sum(A, axis=1), -0.5)
D = np.diag(D)

adj = np.matmul(np.matmul(A, D).T, D)
print(adj)