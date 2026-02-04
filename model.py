%%writefile /content/drive/MyDrive/NeuroTwin/project/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# --------------------------------------------------
# ANN Model for NeuroTwin
# --------------------------------------------------

def build_ann_model(input_shape, num_classes):

    model = Sequential([
        Input(shape=input_shape),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    return model

# --------------------------------------------------
# Compile Model
# --------------------------------------------------

def compile_model(model, learning_rate=0.001):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
