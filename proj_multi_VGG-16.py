# -*- coding: utf-8 -*-

"Importing packages"
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from skimage import io
from sklearn.metrics import confusion_matrix,classification_report
"dataset operations"
train=json.loads(open("C:/Project/malaria/training.json").read())
d={'path':[],
   'min_r':[],
   'min_c':[],
   'max_r':[],
   'max_c':[],
   'type':[]
   }
d1={}
for i in train:
    for j in i['objects']:
        d['path'].append(i['image']['pathname'])
        d['min_r'].append(j['bounding_box']['minimum']['r'])
        d['min_c'].append(j['bounding_box']['minimum']['c'])
        d['max_r'].append(j['bounding_box']['maximum']['r'])
        d['max_c'].append(j['bounding_box']['maximum']['c'])
        d['type'].append(j['category'])
        d1[j['category']]=1
print(d1)
data=pd.DataFrame(d)
data.to_csv("Data.csv",encoding='utf-8',index=False)
val=0
y1=[]
i=0
for j in range(0,70001):
  k=f"malaria/{d['path'][j]}"
  img=cv2.imread(f"C:/Project/malaria/{d['path'][j]}")   
  i+=1
  x=d['min_c'][i]
  y=d['min_r'][i]
  h=d['max_c'][i]-d['min_c'][i]
  w=h=d['max_r'][i]-d['min_r'][i]
  crop_img = img[y:y+h,x:x+w]
  crop_img=cv2.resize(crop_img,(128,128),interpolation=cv2.INTER_AREA)
  cv2.imwrite(f"C:/Project/Data1/{d['type'][i]}/temp{val}.png",crop_img)
  y1.append(d['type'][i])
  val+=1
  j+=1
  
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
    
print(numpy_images)
print(numpy_labels)

"VGG16 base model loading"
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128,128,3))
base_model.trainable = False ## Not trainable weights
base_model.summary()

from tensorflow.keras import layers, models
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

"Splitting data into 80-10-10 ratio for training testing validation"
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

"Model training"
history=model.fit(
    data_train,
    epochs=20,
    batch_size=150,
    verbose=1,
    validation_data=data_val
)
model.evaluate(data_test)

import os.path
if os.path.isfile("C:\Project/models/malaria_VGG16_multiclass.h5") is False:
     model.save("C:\Project/models/malaria_custom_VGG16_multiclass.h5")
from tensorflow.keras.models import load_model
model= load_model("C:\Project/models/malaria_custom_VGG16_multiclass.h5")

"Model results"
l=[]
l1=[]
for images,labels in data_test.take(2):
    for j in range(0,100):
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

"custom testing"
image = tf.keras.preprocessing.image.load_img("C:/Project/Data1/red blood cell/temp3178.png")
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions = model.predict(input_arr)
print(np.argmax(predictions))


