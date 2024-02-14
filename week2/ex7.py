import statsmodels.api as sm
from sklearn import datasets #importing the database
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pandas.tools.plotting import scatter_matrix
iris = datasets.load_iris()
# Let's convert to dataframe
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
columns= iris['feature_names'] + ['species'])
# scatter_matrix(iris, figsize=(10, 10))
# use suptitle to add title to all sublots
plt.suptitle("Pair Plot", fontsize=20)