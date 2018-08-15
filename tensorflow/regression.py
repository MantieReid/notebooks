import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from datetime import datetime

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones(m), housing.data]
tf.reset_default_graph()

# folder for tensorboard
now = datetime.now().strftime('%Y%m%d%H%M%S')
root_logdir = './tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)
# start tensorboard using:
# tensorboard --logdir ./tf_logs/

def RunNormalEquation():
    print('RunNormalEquation...')
    X = tf.constant(housing_data_plus_bias, dtype='float32', name='X')
    y = tf.constant(housing.target.reshape(-1, 1), dtype='float32', name='y')
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    with tf.Session():
        theta_value = theta.eval()
    print('End. Theta=', theta_value)

# run
# RunNormalEquation()
    
def RunGradientDescent():
    print('RunGradientDescent...')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(housing_data_plus_bias)
    n_epochs = 1000
    learning_rate = 0.01
    X = tf.constant(X_scaled, dtype='float32', name='X')
    y = tf.constant(housing.target.reshape(-1, 1), dtype='float32', name='y')
    theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error))
    # compute gradients manually
    # gradients = 2/m * tf.matmul(tf.transpose(X), error)
    # compute gradients with tensorflow autodiff
    # gradients = tf.gradients(mse,[theta])[0]
    # training_op = tf.assign(theta, theta - learning_rate * gradients)
    # use built-in optimizers
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    init = tf.global_variables_initializer()
    # create saver to save model on disk
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print('Epoch ', epoch,'; MSE=', mse.eval())
            summary_str = mse_summary.eval()
            file_writer.add_summary(summary_str, epoch)
            sess.run(training_op)
        theta_value = theta.eval()
        print('End. Theta=', theta_value)
        file_writer.close()
        # Save model. Use saver.restore() in place of tf.global_variables_initializer
        save_path = saver.save(sess, './ckpt/regression.ckpt')
        print('Model saved in ', save_path)

# run
RunGradientDescent()
