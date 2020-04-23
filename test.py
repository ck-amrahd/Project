from sklearn.metrics import f1_score
import numpy as np

y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([0, 0, 1, 0, 0])

print(f1_score(y_true, y_pred))