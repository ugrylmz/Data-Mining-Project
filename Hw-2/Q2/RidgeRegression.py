import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.model_selection import train_test_split

data = pd.read_csv('forestfires.csv')
#print(data.info())
#print(data.shape)

from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
X=data.loc[:,['FFMC','DMC']]
X_train, X_cv, y_train, y_cv = train_test_split(X,data.DC)

lreg.fit(X_train,y_train)

pred = lreg.predict(X_cv)

mse = np.mean((pred - y_cv)**2)

coeff = DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = Series(lreg.coef_)

print(mse)

coeff = DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = Series(lreg.coef_)
print(coeff)

lreg.score(X_cv,y_cv)

x = data.loc[:,['FFMC','DMC','ISI']]

X_train, X_cv, y_train, y_cv = train_test_split(X,data.DC)
lreg.fit(X_train,y_train)

pred = lreg.predict(X_cv)

mse = np.mean((pred - y_cv)**2)
print(mse)
pred_cv = lreg.predict(X_cv)

from sklearn.linear_model import Ridge


ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)

pred = ridgeReg.predict(X_cv)

mse = np.mean((pred_cv - y_cv)**2)

## calculating score
ridgescore = ridgeReg.score(X_cv,y_cv)
print(ridgescore)

from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(X_train,y_train)

pred = lassoReg.predict(X_cv)

print(pred)
# calculating mse

mse = np.mean((pred_cv - y_cv)**2)
print(mse)


lassoscore = lassoReg.score(X_cv,y_cv)
print(lassoscore)