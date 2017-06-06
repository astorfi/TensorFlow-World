import numpy as np
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
num_labels = len(set(iris.target))

data = iris.data.astype(np.float32)
labels = (np.arange(num_labels) == np.array(iris.target)[:,None]).astype(np.float32)
print(data.shape, labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


feature_size = data.shape[1]
delta = 1.0
regulation_rate = 5e-4
graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.constant(data)
    tf_train_labels = tf.constant(labels)

    weights = tf.Variable(tf.truncated_normal([feature_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + biases
    # TODO better way as numpy's: np.choose(data.target, logits.T)
    y = tf.reduce_sum(logits * tf_train_labels, 1, keep_dims=True)
    loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, logits - y + delta), 1)) - delta
    loss += regulation_rate * tf.nn.l2_loss(weights)

    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
    train_prediction = tf.nn.softmax(logits)


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(10001):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if step % 500 == 0:
            print('step:{} loss:{:.6f} accuracy: {:.2f}'.format(
                    step, l, accuracy(predictions, labels)))