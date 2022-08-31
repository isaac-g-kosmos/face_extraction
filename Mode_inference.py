from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D

from tensorflow.keras.applications import VGG16
from keras.models import Model, load_model

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
import wandb
from keras.utils import plot_model
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pre_process import augmented_cut
#%%
shape = (256, 256)
def load_and_preprocess_image(path):
    image_string = tf.io.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # Channels needed because some test images are b/w
    image_resized = tf.image.resize(image_decoded, [shape[0], shape[1]])

    return tf.cast(image_resized, tf.float32)
#%%
df=pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset1.csv')

df=df[df['one']==1]
#reset the index of df
df=df.reset_index(drop=True)
df['path']=df.apply(lambda row: os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',row['path']), axis=1)
# model=load_model(r'C:\Users\isaac\PycharmProjects\face_exctraction\face_extraction-olive-plant-66.h5')

#%%
import random
for x in range(len(df)):
    row=df.loc[x]
    top, left, width, height= augmented_cut(row['pic_width'], row['pic_height'], row['top'], row['left'], row['width'], row['height'], .2)
    df['top'].loc[x]=top
    df['left'].loc[x]=left
    df['width'].loc[x]=width
    df['height'].loc[x]=height
#%%
# model=load_model(r'C:\Users\isaac\PycharmProjects\face_exctraction\face_extraction-firm-serenity-58.h5')
# model=load_model(r'/models/face_extraction-olive-plant-66.h5')
model=load_model(r'face_extraction-icy-snowball-77.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model=load_model(r'C:\Users\isaac\PycharmProjects\face_exctraction\face_extraction-distinctive-pyramid-73.h5')
# model=load_model(r'C:\Users\isaac\PycharmProjects\face_exctraction\face_extraction-frosty-moon-68.h5')
#%%
rand=random.randint(0,len(df))
path=df['path'][rand]
top=df['top'][rand]
left=df['left'][rand]
width=df['width'][rand]
height=df['height'][rand]

image=load_and_preprocess_image(path)
image=tf.expand_dims(image,0)

landmarks,preds=model(image,training=False)

print(landmarks)
from PIL import Image
landmarks=np.array(landmarks[0])*256

fig,ax=plt.subplots(1)
image=Image.open(path)
image=image.resize((256,256))
image=np.array(image)
ax.imshow(image)
ax.add_patch(plt.Rectangle((left*256,top*256),width*256,height*256,fill=False,edgecolor='red',linewidth=3))
ax.add_patch(plt.Rectangle((landmarks[1],landmarks[0]),landmarks[2],landmarks[3],fill=False,edgecolor='blue',linewidth=3))
plt.show()

slice=image[int(landmarks[0]):int(landmarks[0])+int(landmarks[3]),int(landmarks[1]):int(landmarks[1])+int(landmarks[2])]
plt.imshow(slice)
plt.show()