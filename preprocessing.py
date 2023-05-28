import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

print(X)
print(Y)