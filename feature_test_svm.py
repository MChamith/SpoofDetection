import pickle

import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with open("train_data.pickle", "rb") as myfile:
    X_train = pickle.load(myfile)
    label = pickle.load(myfile)

pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['x1', 'x2', 'x3',
                                      'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
# print(principalDf.head())

# scatter_matrix(principalDf, alpha=0.2, figsize=(10, 10), diagonal='kde', c = ["r" if y == 0 else "b" for y in label])
plt.show()
