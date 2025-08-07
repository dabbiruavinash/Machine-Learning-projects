# Image Classification Model with Deep Learning

Introducing the Dataset: Fashion MNIST
Fashion MNIST is a dataset of 70,000 grayscale images of clothing items. Each image is 28×28 pixels and labelled into one of 10 categories, such as T-shirts, Trousers, Sneakers, and more. This dataset contains:

60,000 images for training
10,000 images for testing
10 classes in total

Building an Image Classification Model with Deep Learning
We’ll be using TensorFlow + Keras, one of the most beginner-friendly and powerful libraries for deep learning. So, let’s get started by importing the necessary Python libraries:

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

Let’s take a look at the shape of our data:

print("Training shape:", x_train.shape)
print("Test shape:", x_test.shape)

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=5, subplot_titles=[class_names[y_train[i]] for i in range(10)])

for i in range(10):
    row = i // 5 + 1
    col = i % 5 + 1

    fig.add_trace(go.Heatmap(z=x_train[i], colorscale='gray'), row=row, col=col)

    fig.update_xaxes(showticklabels=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, row=row, col=col)

fig.update_layout(height=600, width=1000, title_text="Sample Images")
fig.show()

Preprocessing the Data
Before feeding data into a neural network, you need to normalize and reshape it. So, let’s preprocess the data with these steps:

x_train = x_train / 255.0
x_test = x_test / 255.0
​
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

Building a Convolutional Neural Network

CNNs are designed to work with images. They extract patterns like edges, textures, and shapes, making them perfect for tasks like this. So, let’s build a CNN:

def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

model = create_model()
model.summary()

Now, let’s add callbacks. Callbacks like EarlyStopping and ModelCheckpoint make training smarter by stopping when the model stops improving and saving the best version of it:

early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint])

Evaluating the Model
Let’s test how well the model performs on unseen data:

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

The output tells us that the model achieved an accuracy of around 90.8%, which means it correctly classified about 9 out of every 10 test images,  pretty solid for a baseline CNN on Fashion MNIST. The loss value of 0.2523 reflects how confident or calibrated the model’s predictions were; lower is better.

Now, let’s have a look at the classification report:

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
​
print(classification_report(y_test, y_pred_classes, target_names=class_names)

