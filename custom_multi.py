# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from skimage import io

"Dataset Loading"
dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Project\Data2",
    shuffle=True,
    image_size=(128,128),
    batch_size=150
)

class_names=dataset.class_names
print(class_names)
for image_batch, label_batch in dataset.take(1):
    numpy_images=image_batch.numpy()
    numpy_labels=label_batch.numpy()

def get_data_partition(data,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=5000):
     if shuffle:
        data=data.shuffle(shuffle_size,seed=12)
     size=len(data)
     train_size=int(train_split*size)
     val_size=int(val_split*size)
     data_train=data.take(train_size)
     data_val=data.skip(train_size).take(val_size)
     data_test=data.skip(train_size).skip(val_size)
     
     return data_train,data_val,data_test

data_train, data_val, data_test = get_data_partition(dataset)
from tensorflow.keras import layers,models

"Custom model layer definition"
resize_rescale= tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(128,128),
    layers.experimental.preprocessing.Rescaling(1.0/255),
    ])
data_aug= tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
    ])

model=keras.models.Sequential([
resize_rescale,
data_aug,
tf.keras.layers.Conv2D(96, (11, 11), strides = (4, 4), activation = "relu"),
tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
tf.keras.layers.BatchNormalization(axis = 3),
tf.keras.layers.Conv2D(256, (5, 5), padding = "same",activation = "relu"),
tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
tf.keras.layers.BatchNormalization(axis = 3),
tf.keras.layers.Conv2D(256, (3, 3), padding = "same",activation = "relu"),
tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(4096, activation = "relu",),
tf.keras.layers.Dense(4096, activation = "relu"),
tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)
"Model training"
history=model.fit(
    data_train,
    epochs=15,
    batch_size=150,
    verbose=1,
    validation_data=data_val
)

import os.path
if os.path.isfile("C:\Project/models/malaria_custom_multiclass.h5") is False:
     model.save("C:\Project/models/malaria_custom_multiclass.h5")   
from tensorflow.keras.models import load_model
model= load_model("C:\Project/models/malaria_custom_multiclass.h5")

"Model results"
l=[]
l1=[]
for images,labels in data_test.take(2):
    for j in range(0,140):
        k=labels[j].numpy()
        l.append(k)
        batch_pred=model.predict(images)
        l1.append(np.argmax(batch_pred[j]))
        print(k,np.argmax(batch_pred[j]))
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(l,l1))
print(confusion_matrix(l,l1))

"Loss-epoch plot"
plt.plot(history.history['loss'], 'r', history.history['val_loss'],'black')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

"accuracy-epoch plot"
plt.plot(history.history['accuracy'], 'r', history.history['val_accuracy'],'black')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
model.evaluate(data_test)

class_names=dataset.class_names
print(class_names)
"Custom testing"
image = tf.keras.preprocessing.image.load_img("C:/Project/Data1/red blood cell/temp3178.png")
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions = model.predict(input_arr)
print(np.argmax(predictions))
