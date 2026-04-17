import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from PIL import Image
from tensorflow import keras
import numpy as np
import gradio as gr
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

x_test = x_test.reshape(-1, 28, 28,1)
x_train = x_train.reshape(-1, 28, 28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D( pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 10, batch_size = 500, validation_data = (x_test, y_test))

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'],loc = 'upper left')
#plt.show()

def predict(img):

    img = img['composite']
    img = Image.fromarray(img).convert("L")
    img = np.array(img)
    img = np.invert(img)

    coords = np.argwhere(img > 0)
    y0, x0 = coords.min(axis = 0)
    y1, x1 = coords.max(axis = 0)
    img = img[y0:y1+1, x0:x1+1]

    img = Image.fromarray(img)
    img.thumbnail((20,20))
    centeredimage = Image.new("L", (28,28), 0)
    offset = ((28 - img.width) // 2, (28 - img.height) // 2)
    centeredimage.paste(img, offset)
    img = np.array(centeredimage)

    img = tf.keras.utils.normalize(img, axis = 1)
    img = img.reshape(-1,28,28,1)
    prediction = model.predict(img)
    return int(np.argmax(prediction))

interface = gr.Interface(
    fn = predict,
    inputs = gr.Sketchpad(),
    outputs = gr.Label(num_top_classes = 3),
    title = "🔢 Digit Recognizer 🔢",
    description = "Welcome to digit recognizer. Draw a number (0-9) and the model will predict it!",
    css="""
        .gradio-container {
            font-size: 18px;
            font-weight: bold;
            padding: 30px;
            background-color: #0f3460;
        }
        h1 {
            font-size: 36px !important;
            text-align: center;
            color: white;
        }

        .gradio-container p {
            font-size: 16px;
            font-weight: 600;
            text-align: center;
            color: white;
        }

        p, span, .prose p, .description {
        color: white !important;
        }

        button.clear {
        display: none !important;
    }

    """
    )
interface.launch(theme = gr.themes.Ocean())