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
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras import layers, models
import os.path
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix,classification_report

base_model = MobileNetV3Large(input_shape=(128,128,3), classes=6,weights="imagenet", pooling=None, include_top=False)
base_model.trainable = False ## Not trainable weights
base_model.summary()


flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(6, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "C:\Project\Data2",
    shuffle=True,
    image_size=(128,128),
    batch_size=150
)
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


history=model.fit(
    data_train,
    epochs=20,
    batch_size=150,
    verbose=1,
    validation_data=data_val
)
model.evaluate(data_test)

if os.path.isfile("C:\Project/models/malaria__Mobile_net.h5") is False:
     model.save("C:\Project/models/malaria__Mobile_net.h5")

model= load_model("C:\Project\models\malaria__Mobile_net.h5")

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
image = tf.keras.preprocessing.image.load_img("C:/Project/Data1/ring/temp6849.png")
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions = model.predict(input_arr)
plt.imshow(image)
plt.title(f"{class_names[np.argmax(predictions)]}")


