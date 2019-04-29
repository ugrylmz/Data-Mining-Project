import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  Imputer, LabelEncoder, OneHotEncoder

dataset = pd.read_csv('forestfires.csv')

#print(dataset.count)
#print(dataset.head(10))


X= dataset.iloc[:,:-1].values

print(X)

y= dataset.iloc[:,3].values

print(y)


imputer = Imputer(missing_values='NaN', strategy='mean',axis=0, verbose=0, copy=True)

imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

print(X)

labelencoder_X = LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])
print(X)

onehateencoder = OneHotEncoder(categorical_features=[0])

X= onehateencoder.fit_transform(X).toarray()

print(X)

y= onehateencoder.X.fit_transform(y)

print(y)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)