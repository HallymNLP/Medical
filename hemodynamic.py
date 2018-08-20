import tensorflow as tf
import pandas as pd

# for reproducibility
tf.set_random_seed(777)

# hyper parameters
learning_rate = 0.01
epochs = 601

input_nodes = 14
neural_nodes = 10
nb_classes = 2  #  True / False

# Load Dataset
train_df = pd.read_csv('medical_train.csv', header=None)
test_df = pd.read_csv('medical_test.csv', header=None)

# Make readability uses dataframe
train_dataset = train_df.values
test_dataset = test_df.values

train_X = train_dataset[:, 0:-1].astype(float)
train_Y = train_dataset[:, [-1]]
test_X = test_dataset[:, 0:-1].astype(float)
test_Y = test_dataset[:, [-1]]

# placeholders for a tensor
X = tf.placeholder(tf.float32, shape=[None, input_nodes])
Y = tf.placeholder(tf.int32, shape=[None, 1])

# One-Hot Encoding
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# Deep Neural Network (Deep Learning) Structure with TensorFlow
# Input Layer
input_W = tf.Variable(tf.random_normal([input_nodes, neural_nodes]))
input_b = tf.Variable(tf.random_normal([neural_nodes]))
in_layer = tf.nn.relu(tf.matmul(X, input_W) + input_b)

# Hidden Layer (1)
W1 = tf.Variable(tf.random_normal([neural_nodes, neural_nodes]))
b1 = tf.Variable(tf.random_normal([neural_nodes]))
h1_layer = tf.nn.relu(tf.matmul(in_layer, W1) + b1)

# # Hidden Layer (2)
# W2 = tf.Variable(tf.random_normal([neural_nodes, neural_nodes]))
# b2 = tf.Variable(tf.random_normal([neural_nodes]))
# h2_layer = tf.nn.relu(tf.matmul(h1_layer, W2) + b2)

# Output Layer
output_W = tf.Variable(tf.random_normal([neural_nodes, nb_classes]))
output_b = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(h1_layer, output_W) + output_b
hypothesis = tf.nn.softmax(logits)

# Cost / Loss Function
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

# Setup the Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch Graph for TensorFlow
with tf.Session() as sess:
	# Initialize global variables
	sess.run(tf.global_variables_initializer())

	# Train the model
	for e in range(epochs):
		c, _ = sess.run([cost, optimizer], feed_dict={X: train_X, Y: train_Y})

		if e % 10 == 0:
			print('\nEpoch : {} / Cost : {}'.format(e, c))

	# for 'Prediction' and 'Accuracy'
	predict_answer = tf.argmax(hypothesis, 1)
	real_answer = tf.argmax(Y_one_hot, 1)
	check_predict = tf.equal(predict_answer, real_answer)
	accuracy = tf.reduce_mean(tf.cast(check_predict, tf.float32))

	# Test the model's result and Check Accuracy
	print('\nPrediction : ', sess.run(predict_answer, feed_dict={X: test_X}))
	print('Real Answer :', sess.run(real_answer, feed_dict={Y: test_Y}))
	print('Accuracy : %f' % sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))