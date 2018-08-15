# Titanic
import pandas as pd
df = pd.read_csv('./train.csv')

# pre process data
X_train = df[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
y_train = df['Survived']

from sklearn.preprocessing import OneHotEncoder
