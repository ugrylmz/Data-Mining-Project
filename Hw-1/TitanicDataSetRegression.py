import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
train.head()

missing_values = train.isnull()
missing_values
sns.heatmap(data=missing_values, yticklabels=False, cbar=False, cmap='viridis')
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', data=train, hue='Pclass')


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

    train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)


sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train, sex, embark], axis=1)
train.drop(['Sex', 'Embarked', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
X = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
