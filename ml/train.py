import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
os.chdir("C:/Users/rhome/github/notebooks/ml/")

# load data
df_macro = pd.read_csv('data/macro.csv').set_index('date').dropna(axis=1, how='all')
df_sector = pd.read_csv('data/sector.csv').set_index('date').dropna(axis=1, how='all')
df_gdx = pd.read_csv('data/GDXd.csv').set_index('date').dropna(axis=1, how='all')
df_xlk = pd.read_csv('data/XLKd.csv').set_index('date').dropna(axis=1, how='all')
df_all = df_macro.join(df_sector, how='inner').join(df_gdx, how='inner').join(df_xlk, how='inner')

security = 'VZ_US'
target = security + '.ivol'
start_training = pd.to_datetime('2007/01/01')
end_training = pd.to_datetime('2017/01/01')

# compute correls of implied vol
df_ivols = df_all[[x for x in list(df_all.columns) if x.endswith('.ivol')]]
correls = df_ivols.corr()
correls_neighbors = correls[security + '.ivol'].nlargest(11)
# plot
'''
correls_neighbors = correls_neighbors.iloc[::-1]
plt.barh(correls_neighbors.index, correls_neighbors, align='center', alpha=0.8)
plt.title("Correls of ivol @ " + security)
plt.show()
'''

# features
# ***************************************************
# macro
features_macro = [
        'CL1.rvol10D', 'CL1.spot',
        'DXY.rvol10D', 'DXY.spot',
        'SPX.ivol', 'SPX.putcallvolumeratio', 'SPX.spot', 'SPX.volume',
        'USSW10.rvol10D', 'USSW10.spot',
        'VIX.putcallvolumeratio', 'VIX.rvol10D', 'VIX.spot',
        'XAG.rvol10D', 'XAG.spot',
        'XAU.rvol10D', 'XAU.spot']

features_sectors = [
        'EEM.ivol', 'EEM.putcallvolumeratio',
        'EEM.rvol10D', 'EEM.rvol30D', 'EEM.spot', 'EEM.volume',
        'FDN.shortintratio',
        'IBB.shortintratio',
        'XLE.putcallvolumeratio', 'XLE.shortintratio', 'XLE.spot',
        'XLF.putcallvolumeratio', 'XLF.shortintratio', 'XLF.spot',
        'XLI.putcallvolumeratio', 'XLI.shortintratio', 'XLI.spot',
        'XLK.putcallvolumeratio', 'XLK.shortintratio', 'XLK.spot',
        'XLP.putcallvolumeratio', 'XLP.shortintratio', 'XLU.shortint',
        'XLV.putcallvolumeratio', 'XLV.shortintratio', 'XLV.spot',
        'XLY.putcallvolumeratio', 'XLY.shortintratio', 'XLY.spot']

features_securities = [s + x 
       for s in [security]
       for x in ['.ivol', '.putcallvolumeratio', '.shortintratio', '.spot']]
# features_securities += [x for x in list(correls_neighbors.index) if x != security + '.ivol']

df_all = df_all[features_securities + features_macro + features_sectors]

# compute zscores
print('Computing zscore...')
for day in [30, 60, 90, 260]:
    df_all[security + '.ivol.zscore' + str(day) + 'd'] = \
        df_all[security + '.ivol'].rolling(day).apply( \
              lambda x: (x[-1] - x.mean())/x.std(), raw=True) \
              .fillna(method='bfill')

# compute momentum
print('Computing momentum...')
for feature in list(df_all.columns):
    if '.spot' in feature:
        for day in [2, 5, 10]:
            df_all[feature + '.momentum' + str(day) + 'd'] = \
                df_all[feature].rolling(day).apply( \
                      lambda x: (x[-1] - x.mean())/x.std(), raw=True) \
                      .fillna(method='bfill')
  
df_all[security + '.ivol.zscore' + str(day) + 'd'] = \
        df_all[security + '.ivol'].rolling(day).apply(lambda x: (x[-1] - x.mean())/x.std(), raw=True).fillna(method='bfill')

df_all = df_all[np.isfinite(df_all[target])]
df_dates = pd.to_datetime(df_all.reset_index()['date'])
df_target = df_all[target].reset_index()
df_features = df_all.drop([target], axis=1)
df_features = df_features.dropna(axis=1, how='any').select_dtypes(include='float64')

# train decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

X_train = df_features[list((df_dates >= start_training) & (df_dates < end_training))].values
y_train = df_target[list((df_dates >= start_training) & (df_dates < end_training))][target].values
d_train = df_dates[list((df_dates >= start_training) & (df_dates < end_training))]
X_test = df_features[list(df_dates >= end_training)].values
y_test = df_target[list(df_dates >= end_training)][target].values
d_test = df_dates[df_dates >= end_training]

# decision tree
clf = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, min_samples_split=5)

# random forest
#clf = RandomForestRegressor(max_depth=8, n_estimators=20, max_leaf_nodes=50, n_jobs=-1)

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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    model.add(Dense(1, 
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
# kfold = KFold(n_splits=3, random_state=seed)
# cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=kfold)
# Results: -19376733.79 (49264686.37) MSE (no scaling)
# Results: -144.28 (148.91) MSE (relu, [62*1])
# Results: -59.37 (44.19) MSE (sigmoid, [62*1])
# Results: -87.21 (28.69) MSE (sigmoid, [62,32,1]) - deeper
# Results: -53.58 (32.51) MSE (sigmoid, [512,1]) - wider

clf.fit(X_train, y_train)
score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)
print('Accuracy on training set: {:.2f}'.format(score_train))
print('Accuracy on test set: {:.2f}'.format(score_test))
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# plot training and test sets
df_train_plot = pd.DataFrame({'date': d_train, 'ivol': y_train, 'model': y_train_pred})
df_test_plot = pd.DataFrame({'date': d_test, 'ivol': y_test, 'model': y_test_pred})
plt.figure()
plt.plot(df_train_plot['date'], df_train_plot['ivol'], label='ivol', c="cornflowerblue", linewidth=2)
plt.plot(df_train_plot['date'], df_train_plot['model'], label='model', c="darkorange", linewidth=2)
plt.plot(df_test_plot['date'], df_test_plot['ivol'], label='', c="cornflowerblue", linewidth=2)
plt.plot(df_test_plot['date'], df_test_plot['model'], label='predictions', c="green", linewidth=2)
plt.title(str(clf).split('(')[0] + " @ " + target)
plt.legend()
plt.text(6, 6, 'test', fontsize=15)
plt.show()
plt.savefig(target + "_plot.png")

# show tree
'''
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                feature_names=df_features.head().columns,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
img = Image(graph.create_png())
with open(target + "_tree.png", "wb") as png:
    png.write(img.data)
'''