import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest, ytest) = fashion.load_data()

imgIndex = 9
image = xtrain[imgIndex]
print("Image Label :",ytrain[imgIndex])
plt.imshow(image)

Now let’s have a look at the shape of both the training and test data:

print(xtrain.shape)
print(xtest.shape)

Building a Neural Network Architecture

Now I will build a neural network architecture with two hidden layers:

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
print(model.summary())

Before training our model, I will split the training data into training and validation sets:

xvalid, xtrain = xtrain[:5000]/255.0, xtrain[5000:]/255.0
yvalid, ytrain = ytrain[:5000], ytrain[5000:]

Training a Classification Model with Neural Networks:

Now here’s how we can train a neural network for the task of image classification:

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
history = model.fit(xtrain, ytrain, epochs=30, 
                    validation_data=(xvalid, yvalid))

Now let’s have a look at the predictions:

new = xtest[:5]
predictions = model.predict(new)
print(predictions)

Here is how we can look at the predicted classes:

classes = np.argmax(predictions, axis=1)
print(classes)

