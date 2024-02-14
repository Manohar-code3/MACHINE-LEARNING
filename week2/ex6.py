import statsmodels.api as sm
from sklearn import datasets #importing the database
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# Let's convert to dataframe
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
columns= iris['feature_names'] + ['species'])
corr = iris.corr()
print(corr)
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()