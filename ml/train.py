import pandas as pd
import numpy as np
import os 
os.chdir("C:/Users/rhome/github/notebooks/ml/")

# load data
df_macro = pd.read_csv('data/macro.csv').set_index('date').dropna(axis=1, how='all')
df_sector = pd.read_csv('data/sector.csv').set_index('date').dropna(axis=1, how='all')
df_xlk = pd.read_csv('data/XLKd.csv').set_index('date').dropna(axis=1, how='all')
df_all = df_macro.join(df_sector, how='inner')

df_all = df_all[['GDX.ivol',
        'CL1.rvol10D', 
        'CL1.spot',
        'DXY.rvol10D',
        'DXY.spot',
        'EEM.ivol',
        'EEM.ivol90',
        'EEM.pucallopeninterestratio',
        'EEM.putcallvolumeratio',
        'EEM.rvol10D',
        'EEM.rvol30D',
        'EEM.spot',
        'EEM.volume',
        'FDN.shortint',
        'FDN.shortintratio',
        'GDX.putcallvolumeratio',
        'GDX.shortint',
        'GDX.shortintratio',
        'GDX.spot',
        'IBB.shortint',
        'IBB.shortintratio',
        'SPX.ivol',
        'SPX.putcallvolumeratio',
        'SPX.spot',
        'SPX.volume',
        'USSW10.rvol10D',
        'USSW10.spot',
        'VFH.shortint',
        'VGT.putcallvolumeratio',
        'VGT.shortint',
        'VGT.shortintratio',
        'VHT.shortint',
        'VIX.putcallvolumeratio',
        'VIX.rvol10D',
        'VIX.spot',
        'VNQ.shortint',
        'XAG.rvol10D',
        'XAG.spot',
        'XAU.rvol10D',
        'XAU.spot',
        'XLE.putcallvolumeratio',
        'XLE.shortint',
        'XLE.shortintratio',
        'XLE.spot',
        'XLF.putcallvolumeratio',
        'XLF.shortint',
        'XLF.shortintratio',
        'XLF.spot',
        'XLI.putcallvolumeratio',
        'XLI.shortint',
        'XLI.shortintratio',
        'XLI.spot',
        'XLK.putcallvolumeratio',
        'XLK.shortint',
        'XLK.shortintratio',
        'XLK.spot',
        'XLP.putcallvolumeratio',
        'XLP.shortint',
        'XLP.shortintratio',
        'XLU.shortint',
        'XLV.putcallvolumeratio',
        'XLV.shortint',
        'XLV.shortintratio',
        'XLV.spot',
        'XLY.putcallvolumeratio',
        'XLY.shortint',
        'XLY.shortintratio',
        'XLY.spot']]

target = 'GDX.ivol'
end_training = pd.to_datetime('2017/01/01')

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

X_train = df_features[list(df_dates < end_training)].values
y_train = df_target[list(df_dates < end_training)][target].values
d_train = df_dates[df_dates < end_training]
X_test = df_features[list(df_dates >= end_training)].values
y_test = df_target[list(df_dates >= end_training)][target].values
d_test = df_dates[df_dates >= end_training]

# decision tree
clf = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, min_samples_split=5)

# random forest
#clf = RandomForestRegressor(max_depth=8, n_estimators=20, max_leaf_nodes=50, n_jobs=-1)

# neural network
'''
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_train)
clf = MLPRegressor(hidden_layer_sizes=(1024, 1024), solver="sgd", activation="tanh",
                   alpha=0.0001, batch_size=64, max_iter=10000, tol=0.00001, shuffle=True)
'''

clf.fit(X_train, y_train)
print('Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# plot training and test sets
import matplotlib.pyplot as plt
df_train_plot = pd.DataFrame({'date': d_train, 'ivol': y_train, 'model': y_train_pred})
df_test_plot = pd.DataFrame({'date': d_test, 'ivol': y_test, 'model': y_test_pred})
plt.figure()
plt.plot(df_train_plot['date'], df_train_plot['ivol'], label='ivol', c="cornflowerblue", linewidth=2)
plt.plot(df_train_plot['date'], df_train_plot['model'], label='model', c="darkorange", linewidth=2)
plt.plot(df_test_plot['date'], df_test_plot['ivol'], label='', c="cornflowerblue", linewidth=2)
plt.plot(df_test_plot['date'], df_test_plot['model'], label='predictions', c="green", linewidth=2)
plt.title(str(clf).split('(')[0] + " @ " + target)
plt.legend()
#plt.show()
plt.savefig(target + "_plot.png")

# show tree
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
    
# ************************************************************
# neural network with tensorflow
# ************************************************************
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

def neural_net_model(X_data, input_dim):
    """
    neural_net_model is function applying 2 hidden layer feed forward neural net.
    Weights and biases are abberviated as W_1, W_2 and b_1, b_2 
    These are variables with will be updated during training.
    """ 
    W_1 = tf.Variable(tf.random_uniform([input_dim, 10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
    # layer 1 multiplying and adding bias then activation function
    W_2 = tf.Variable(tf.random_uniform([10,10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # layer 2 multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([10,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2, W_O), b_O)
    # O/p layer multiplying and adding bias then activation function
    # notice output layer has one node only since performing regression
    return output

xs = tf.placeholder("float")
ys = tf.placeholder("float")
output = neural_net_model(xs, 3)
# Cost function to minimize: mean squared error
cost = tf.reduce_mean(tf.square(output-ys))
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# Gradinent Descent optimization for updating weights and biases

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(100):
        for j in range(X_train.shape[0]):
            sess.run([cost,train],feed_dict=    {xs:X_train[j,:].reshape(1,3), ys:y_train[j]})
            # Run cost and train with each sample
        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :', i, 'Cost :', c_t[i])
    pred = sess.run(output, feed_dict={xs:X_test})
    # predict output of test data after training
    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    #saver.save(sess,'tfmodel.ckpt')