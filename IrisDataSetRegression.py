from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


iris =load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)


nearest_neighbors = KNeighborsClassifier()
nearest_neighbors=KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=1,p=2,weights='uniform')

nearest_neighbors.fit(X_train,y_train)
y_pred = nearest_neighbors.predict(X_test)

cm = confusion_matrix(y_test , y_pred)
print(cm)

lda = LinearDiscriminantAnalysis(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)



X_set, y_set = X_test, y_test

aranged_pc1 = np.arange(start=X_set[:, 0].min(), stop=X_set[:, 0].max(), step=0.01)
aranged_pc2 = np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max(), step=0.01)

X1, X2 = np.meshgrid(aranged_pc1, aranged_pc2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.5, cmap=ListedColormap(('orange', 'blue', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Linear Discriminant Analysis')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

aranged_pc1 = np.arange(start=X_set[:, 0].min(), stop=X_set[:, 0].max(), step=0.01)
aranged_pc2 = np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max(), step=0.01)

X1, X2 = np.meshgrid(aranged_pc1, aranged_pc2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.5, cmap=ListedColormap(('orange', 'blue', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Linear Discriminant Analysis')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

