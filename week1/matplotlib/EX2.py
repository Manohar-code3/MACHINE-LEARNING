import pandas as pd
df = pd.read_csv("Data/iris.csv")
df.hist()# Histogram

df.plot() # Line Graph
df.boxplot()