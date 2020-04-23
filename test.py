import pickle
import numpy as np

num_classes = 80

adj = pickle.load(open('adj.pickle', 'rb'))
adj = adj / np.max(adj)

adj = adj + np.identity(num_classes)
print(adj)