import pandas as pd


dataset = pd.read_csv('madelon_csv.csv')

print(dataset.head(10))
print(dataset.count)


X= dataset.iloc[:,:-1].values

print(X)