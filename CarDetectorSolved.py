import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import math
import random
import scipy
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from PIL import Image


# Load images and tags:
def LoadData(path):

    # Initialize arrays for images and tags:
    images = []
    labels = []

    # Search the images in the image folder:
    files = [f for f in listdir(path) if isfile(join(path, f))]

    # For each image...
    for i in range(len(files)):

        # Get path:
        imagePath = path + str(files[i])

        # Load image:
        image = Image.open(imagePath)

        # Resize image:
        image = scipy.misc.imresize(image, size=(64, 64))

        # Create tag:
        name = str(files[i])
        object = name.split(" ")[0]
        color = name.split(" ")[1]

        if object=="car":
            if color=="blue":
                label = [0, 1, 0, 0, 0, 0, 0, 0]
            elif color=="cyan":
                label = [0, 0, 1, 0, 0, 0, 0, 0]
            elif color == "green":
                label = [0, 0, 0, 1, 0, 0, 0, 0]
            elif color == "grey":
                label = [0, 0, 0, 0, 1, 0, 0, 0]
            elif color == "orange":
                label = [0, 0, 0, 0, 0, 1, 0, 0]
            elif color == "purple":
                label = [0, 0, 0, 0, 0, 0, 1, 0]
            elif color == "red":
                label = [0, 0, 0, 0, 0, 0, 0, 1]
        else:
            label = [1, 0, 0, 0, 0, 0, 0, 0]

        # Add image and tag to lists:
        images.append(image)
        labels.append(label)

    return images, labels

# Create datasets:
def CreateSets(images, labels):

    # Initialize datasets:
    X_train = [] # Images for training.
    Y_train = [] # Detections for prediction.
    X_test = [] # Images for training.
    Y_test = [] # Detections for prediction.

    # Create datasets:
    for i in range(len(images)):
        if random.random() > 0.1: # Train (90%) / Test (10%)
            X_train.append(images[i])
            Y_train.append(labels[i])
        else:
            X_test.append(images[i])
            Y_test.append(labels[i])

    return X_train, Y_train, X_test, Y_test

# Random mini batches:
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = len(X)  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))

    shuffled_X = []
    shuffled_Y = []

    for i in range(len(permutation)):
        shuffled_X.append(X[permutation[i]])
        shuffled_Y.append(Y[permutation[i]])

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:][(k) * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:][(k) * mini_batch_size: (k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:][(num_complete_minibatches) * mini_batch_size: (num_complete_minibatches + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:][(num_complete_minibatches) * mini_batch_size: (num_complete_minibatches + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# TODO: Create placeholders:
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name='Y')

    return X, Y

# TODO: Initialize parameters:
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters

# TODO: Forward propagation:
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    s = 1
    Z1 = tf.nn.conv2d(X, W1, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    f = 8
    s = 8
    P1 = tf.nn.max_pool(A1, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    s = 1
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    f = 4
    s = 4
    P2 = tf.nn.max_pool(A2, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 8, activation_fn=None)

    return Z3

# TODO: Compute cost:
def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost

# Train model:
def model(X_train, Y_train, X_test, Y_test, path, learning_rate=0.001, num_epochs=1000, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    m = len(X_train)
    (n_H0, n_W0, n_C0) = X_train[0].shape
    n_y = len(Y_train[0])
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    Y_pred = tf.nn.softmax(Z3, name="Y_PRED")

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1, name="predict_op")
        tf.add_to_collection("predict_op", predict_op)

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        saver = tf.train.Saver()

        saver.save(sess, path + '/model', global_step=num_epochs)

        return train_accuracy, test_accuracy, parameters

# Make prediction:
def Predict(path1, path2):

    # Initialize variables:
    init = tf.global_variables_initializer()

    # Create session:
    with tf.Session() as sess:

        # Load variables in session:
        sess.run(init)

        # Initialize random seed:
        tf.set_random_seed(1)

        # Load saved model:
        saver = tf.train.import_meta_graph(path1)

        # Get last checkpoint:
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Load image:
        image = Image.open(path2)

        image = scipy.misc.imresize(image, size=(64, 64))

        # Get graph:
        graph = tf.get_default_graph()

        # Load inputs and outputs:
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        Y_pred = graph.get_tensor_by_name("Y_PRED:0")

        # Make prediction:
        y_pred = sess.run(Y_pred, feed_dict={X: [image], Y: np.asarray([[0,0,0,0,0,0,0,0]])})

        prediction = np.argmax(y_pred, axis=1)

        result = 'no car'

        if prediction == 1:
            result = 'blue'
        elif prediction == 2:
            result = 'cyan'
        elif prediction == 3:
            result = 'green'
        elif prediction == 4:
            result = 'grey'
        elif prediction == 5:
            result = 'orange'
        elif prediction == 6:
            result = 'purple'
        elif prediction == 7:
            result = 'red'


        print('Result: ' + result)

# Global main:
if __name__ == '__main__':

    dir = os.path.dirname(__file__)

    models_path = dir
    model_path = os.path.join(dir, 'model-100.meta')
    images_path = os.path.join(dir, 'Images/')
    image_path = os.path.join(images_path, 'car blue size 25 250.png') # You can change it!

    train = True # True for training, False for prediction

    if train:

        images, labels = LoadData(images_path)

        X_train, Y_train, X_test, Y_test = CreateSets(images, labels)

        _, _, parameters = model(X_train, Y_train, X_test, Y_test, models_path, learning_rate= 0.001, minibatch_size=64, num_epochs=1000)

    else:

        Predict(model_path, image_path)