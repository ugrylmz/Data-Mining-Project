from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import mean_squared_error


boston = load_boston()
X = boston.data
y = boston.target

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)

print(X_train.shape)
print(X_test.shape)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

for idx, col_name in enumerate(X_train.columns):
    intercept = regression_model.intercept_[0]
    print("The intercept for our model is {}".format(intercept))

regression_model.score(X_test, y_test)

y_predict = regression_model.predict(X_test)

regression_model_mse = mean_squared_error(y_predict, y_test)

print(regression_model_mse)