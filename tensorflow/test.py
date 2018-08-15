import tensorflow as tf
import pandas as pd

# XOR classifier
df = pd.DataFrame({'x0': [0, 0, 1, 1], 
                   'x1': [0, 1, 0, 1],
                   'y': [0, 1, 1, 0]})

n_inputs = 2
n_hidden1 = 2
n_outputs = 2
learning_rate = 0.01

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
     hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation='relu')
     logits = tf.layers.dense(hidden1, n_outputs, name='outputs')
     
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
    

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
n_epochs = 5
batch_size = 1

with tf.Session() as sess:
    init.run()
    X_batch = df[['x0','x1']].values
    y_batch = df['y'].values
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print('Epoch ', epoch, '; Accuracy=', acc_train)