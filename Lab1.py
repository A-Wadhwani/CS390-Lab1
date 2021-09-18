import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"
ALGORITHM = "tf_conv_net" # Gets us to 99% accuracy


class NeuralNetwork_2Layer:
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.01):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (np.power(np.e, -1 * x) + 1)

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def __loss(self, true_val, pred_val):
        return np.square(pred_val - true_val) / 2.

    def __lossDerivative(self, true_val, pred_val):
        return pred_val - true_val

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=5000, minibatches=True, mbs=128):
        xGen = self.__batchGenerator(xVals, mbs)
        yGen = self.__batchGenerator(yVals, mbs)
        for i in range(0, epochs):
            x = next(xGen, None) if minibatches else xVals
            y = next(yGen, None) if minibatches else yVals

            # Restart batch gen if complete
            if x is None:
                xGen = self.__batchGenerator(xVals, mbs)
                yGen = self.__batchGenerator(yVals, mbs)
                x = next(xGen, None)
                y = next(yGen, None)

            # Make into np arrays
            x = np.array(x)
            y = np.array(y)

            # Forward Propagation
            l1out, l2out = self.__forward(x)

            # Layer 2 Error and Adjustment
            l2error = self.__lossDerivative(y, l2out)
            l2delta = l2error * self.__sigmoidDerivative(np.dot(l1out, self.W2))
            l2adj = self.lr * np.transpose(l1out).dot(l2delta)
            self.W2 = self.W2 - l2adj

            # Layer 1 Error and Adjustment
            l1error = np.dot(l2delta, self.W2.transpose())
            l1delta = l1error * self.__sigmoidDerivative(np.dot(x, self.W1))
            l1adj = self.lr * x.transpose().dot(l1delta)
            self.W1 = self.W1 - l1adj

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        # Postprocessing to final 1 and 0 outputs
        ans = []
        for entry in layer2:
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            pred[np.argmax(entry)] = 1
            ans.append(pred)
        return np.array(ans)


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw

    # Casting
    xTrain = xTrain.astype(float)
    xTest = xTest.astype(float)

    # Rescaling
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    # Flattening
    xTrain = xTrain.reshape((xTrain.shape[0], IMAGE_SIZE))
    xTest = xTest.reshape((xTest.shape[0], IMAGE_SIZE))

    # One-hot encoding
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        model = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, 30, learningRate=0.07)
        model.train(xTrain, yTrain, epochs=5000, minibatches=True, mbs=128)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=40, batch_size=128)
        return model
    elif ALGORITHM == "tf_conv_net":
        print("Building and training TF_CONV_NN.")
        xTrain = xTrain.reshape((xTrain.shape[0], 28, 28))
        xTrain = np.expand_dims(xTrain, -1)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32,
                                   kernel_size=3,
                                   activation=tf.nn.relu,
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(pool_size=2,
                                      padding='valid'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64,
                                   kernel_size=3,
                                   activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=2,
                                      padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        def scheduler(epoch, lr):
            if epoch < 3:
                return lr
            elif epoch < 6:
                return 0.005
            return 0.001

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
        model.fit(xTrain, yTrain, epochs=10, callbacks=[reduce_lr], batch_size=256)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        results = model.predict(data)
        # Postprocessing to final 1 and 0 outputs
        ans = []
        for entry in results:
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            pred[np.argmax(entry)] = 1
            ans.append(pred)
        return np.array(ans)
    elif ALGORITHM == "tf_conv_net":
        print("Testing TF_CONV_NN.")
        data = data.reshape((data.shape[0], 28, 28))
        data = np.expand_dims(data, -1)
        results = model.predict(data)
        # Postprocessing to final 1 and 0 outputs
        ans = []
        for entry in results:
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            pred[np.argmax(entry)] = 1
            ans.append(pred)
        return np.array(ans)
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    # Accuracy
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):
            acc = acc + 1
    accuracy = acc / preds.shape[0]

    # Confusion Matrix
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
