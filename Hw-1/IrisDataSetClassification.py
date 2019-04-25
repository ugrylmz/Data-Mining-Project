from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


iris = load_iris()
X= iris.data
y= iris.target

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)


nearest_komsu = KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=1,p=2,weights='uniform')

nearest_komsu.fit(X_train,y_train)
y_pred = nearest_komsu.predict(X_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)

