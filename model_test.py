from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D

from tensorflow.keras.applications import MobileNetV3Small
from pre_process import augmented_cut
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.models import Model, load_model

import wandb
from keras.utils import plot_model
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pre_process import augmented_cut, BB_Metrics
shape = (256, 256)
epochs = 1
lr=1e-3
df = pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset_augmentations.csv')
#%%
for x in range(len(df)):
    row=df.loc[x]
    top, left, width, height= augmented_cut(row['pic_width'], row['pic_height'], row['top'], row['left'], row['width'], row['height'])
    df['top'].loc[x]=top
    df['left'].loc[x]=left
    df['width'].loc[x]=width
    df['height'].loc[x]=height
#%%
x2=df['top']+df['height']
y2=df['left']+df['width']
df['width']=x2
df['height']=y2
df=df[['path','top','left','width','height','zero','one','many']]
df['path']=df.apply(lambda row: os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',row['path']), axis=1)
#%%
# split the data into train and test
###separe into train test and validation
def load_and_preprocess_image(path):
    image_string = tf.io.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # Channels needed because some test images are b/w
    image_resized = tf.image.resize(image_decoded, [shape[0], shape[1]])

    return tf.cast(image_resized, tf.float32)
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), (label[:4], label[4:])

train, test = train_test_split(df, test_size=0.3, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)
filenames = tf.constant(train.iloc[:, 0].tolist())
labels = tf.constant(train.iloc[:, 1:].values)
val_filenames = tf.constant(val.iloc[:, 0].tolist())
val_labels = tf.constant(val.iloc[:, 1:].values)
test_labels = tf.constant(test.iloc[:, 1:].values)
test_filenames = tf.constant(test.iloc[:, 0].tolist())
labels=tf.cast(labels, tf.float32)
val_labels=tf.cast(val_labels, tf.float32)
test_labels=tf.cast(test_labels, tf.float32)
# %%
model=load_model('face_extraction-elated-glitter-75.h5')

wandb.init(project="preprocessing model",config={"epochs":epochs,"shape":shape,"filter_size":filter_size,"maxpool_size":maxpool_size,"dr":dr})
name=wandb.run.name
wandb.run.name='face _extraction_test'+wandb.run.name
#%%
giou_loss = tfa.losses.GIoULoss()

val_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
train_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
test_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
bb_metrics=BB_Metrics()
cat_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
test = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test = test.map(load_and_preprocess_from_path_label)
test=test.batch(1)
test_landmarks_loss=0
test_categorical_loss=0
bb_metrics=BB_Metrics()
table = wandb.Table(columns=['picture', 'landmarks loss', 'Class','Correct class','Accurately classified'])
for step, (x_batch_test, y_batch_test) in enumerate(test):
    landmarks, people_count = model(x_batch_test, training=False)

    test_categorical_loss += cat_loss(y_batch_test[1], people_count)
    test_people_accuracy(y_batch_test[1], people_count)
    numpy_landmarks = np.array(landmarks)[0]
    ground_truth = np.array(y_batch_test[0])[0]
    fig,ax=plt.subplots(1,1)

    class1=np.argmax(people_count)
    correct_class=np.argmax(y_batch_test[1])
    if (correct_class==0 or correct_class==2):
        pass
    else:
        bb_metrics.update(numpy_landmarks.reshape(1,4), ground_truth.reshape(1,4))
        ax.imshow(x_batch_test.numpy()[0].astype(int))
        ax.add_patch(plt.Rectangle((numpy_landmarks[0]*256,numpy_landmarks[1]*256)
                                   ,(numpy_landmarks[2]*256),
                                   (numpy_landmarks[3]*256),fill=False,edgecolor='r',linewidth=3))
        ax.add_patch(plt.Rectangle((ground_truth[0]*256, ground_truth[1]*256), ground_truth[2]*256 , ground_truth[3]*256 , fill=False, edgecolor='blue', linewidth=3))
        test_landmarks_loss = giou_loss(y_batch_test[0], landmarks)
        if step==0:
            cum_loss=test_landmarks_loss
        else:
            cum_loss+=test_landmarks_loss
        print('testing step:',step)


        # fig.show()
    table.add_data(wandb.Image(fig), float(test_landmarks_loss),class1,correct_class,class1==correct_class)

wandb.log({'test landmarks loss': cum_loss,
            'test categorical loss': test_categorical_loss})
wandb.log({'test people accuracy': test_people_accuracy.result()})
wandb.log(bb_metrics.get_metrics('test_BB'))