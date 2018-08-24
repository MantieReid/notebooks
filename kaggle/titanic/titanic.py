# Titanic
import numpy as np
import pandas as pd
df = pd.read_csv('./train.csv')

class DataProcessor:
    
    def __init__(self):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
        self.sex_lbb = LabelBinarizer()
        self.embarked_lbb = LabelBinarizer()

    def Process(self, X, isTest = False):        
        X = X.reset_index()
        # sex
        if isTest:
            Xoh = self.sex_lbb.transform(X['Sex'].values.reshape(-1,1))
        else:
            Xoh = self.sex_lbb.fit_transform(X['Sex'].values.reshape(-1,1))
        dfOneHot = pd.DataFrame(Xoh, columns = ["Sex_" + str(int(i)) for i in range(Xoh.shape[1])])
        X = pd.concat([X, dfOneHot], axis=1)
        # embarked
        X['Embarked'].fillna('', inplace=True)
        if isTest:
            Xoh = self.embarked_lbb.transform(X['Embarked'].values.reshape(-1,1))
        else:
            Xoh = self.embarked_lbb.fit_transform(X['Embarked'].values.reshape(-1,1))
        dfOneHot = pd.DataFrame(Xoh, columns = ["Embarked_" + str(int(i)) for i in range(Xoh.shape[1])])
        X = pd.concat([X, dfOneHot], axis=1)
        # drop columns
        X.drop(['index', 'Sex', 'Embarked'], axis=1, inplace=True)
        # n/a
        X.fillna(X.mean(), inplace=True)
        return X

# initial features
Xi = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]   #'Cabin'
yi = df['Survived']
# split training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=0.20)
# pre process data 
processor = DataProcessor()
X_train = processor.Process(X_train, isTest=False)
X_test = processor.Process(X_test, isTest=True)

# *********************************************************
# models
# *********************************************************
# support vector classifier
from sklearn.svm import SVC
clf = SVC()
# random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, n_estimators=10, 
                             min_samples_split=3, min_samples_leaf=2)

clf.fit(X_train, y_train)
print('Score on train set: ', clf.score(X_train, y_train))
print('Score on test set: ', clf.score(X_test, y_test))

# ************************************************************
# neural network with tensorflow
# ************************************************************
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import ELU, PReLU
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
from keras import backend as keras_backend
#print(keras_backend.tensorflow_backend._get_available_gpus())

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, 
                    kernel_initializer='normal', 
                    activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(64, 
                    kernel_initializer='normal', 
                    activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, 
                    kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

step_scaler = StandardScaler()
step_regressor = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=32, verbose=2)
steps = []
steps.append(('step_scaler', step_scaler))
steps.append(('step_regressor', step_regressor))
pipeline = Pipeline(steps)
clf = pipeline

clf.fit(X_train, y_train)
score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)
print('Accuracy on training set: {:.2f}'.format(score_train))
print('Accuracy on test set: {:.2f}'.format(score_test))