from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D

from tensorflow.keras.applications import VGG16 ,MobileNetV3Small
from pre_process import augmented_cut
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
shape = (256, 256)
epochs = 10
lr=1e-3
df = pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset_augmentations.csv')

#%%
for x in range(len(df)):
    row=df.loc[x]
    top, left, width, height= augmented_cut(row['pic_width'], row['pic_height'], row['top'], row['left'], row['width'], row['height'], .2)
    df['top'].loc[x]=top
    df['left'].loc[x]=left
    df['width'].loc[x]=width
    df['height'].loc[x]=height
#%%
df=df[['path','top','left','width','height','zero','one','many']]
df['path']=df.apply(lambda row: os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',row['path']), axis=1)
#%
# split the data into train and test
###separe into train test and validation
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
#%%
# import albumentations as alb
# cropper=alb.RandomCrop(width=100, height=100, p=1)
# augmentor = alb.Compose([alb.RandomCrop(width=150, height=150),
#                         alb.HorizontalFlip(p=0.5),
#                          alb.RandomBrightnessContrast(p=0.2),
#                          alb.RandomGamma(p=0.2),
#                          alb.RGBShift(p=0.2),
#                          alb.Flip(p=0.2),
#                          alb.VerticalFlip(p=0.5)],
#                        bbox_params=alb.BboxParams(format='coco',
#                                                   label_fields=['class_labels']))
#%%
# from PIL import Image
# import numpy as np
# import random
# # df = pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset1.csv')
# df['path']=df.apply(lambda row: os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',row['path']), axis=1)
# df=df[df['one']==1]
# number=random.randint(0,len(df)-1)
# df.reset_index(drop=True, inplace=True)
# path=df['path'].iloc[number]
# cords=np.zeros(4)
# for x in range(len(df)):
#     row=df.loc[x]
#     top, left, width, height= augmented_cut(row['pic_width'], row['pic_height'], row['top'], row['left'], row['width'], row['height'], .2)
#     df['top'].loc[x]=top
#     df['left'].loc[x]=left
#     df['width'].loc[x]=width
#     df['height'].loc[x]=height
# cords[0]=df['top'].iloc[number]
# cords[1]=df['left'].iloc[number]
# cords[2]=df['width'].iloc[number]
# cords[3]=df['height'].iloc[number]
# img=Image.open(path)
# img=np.array(img)
# plt.imshow(img)
# plt.show()
# cropped_image = cropper(image=img, bboxes=[cords])['image']
# plt.imshow(cropped_image)
# plt.show()
# image_string = tf.io.read_file(path)
# image_decoded = tf.image.decode_jpeg(image_string, channels=3)
# image_decoded = np.array(image_decoded)
#
# augmented= augmentor(image=image_decoded, bboxes=[cords], class_labels=['face'])
# augmented_img = augmented['image']
# augmented_bboxes = augmented['bboxes']
# augmented_img = tf.convert_to_tensor(augmented_img)
# print(augmented_bboxes)
# augmented_img=tf.image.resize(augmented_img, [shape[0], shape[1]])

# %%
def load_and_preprocess_image(path):
    image_string = tf.io.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # Channels needed because some test images are b/w
    image_resized = tf.image.resize(image_decoded, [shape[0], shape[1]])

    return tf.cast(image_resized, tf.float32)
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), (label[:4], label[4:])


ds = tf.data.Dataset.from_tensor_slices((filenames, labels), )

#%%
# The tuples are unpacked into the positional arguments of the mapped function


ds = ds.map(load_and_preprocess_from_path_label)
val = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
val = val.map(load_and_preprocess_from_path_label)

# %%
filter_size = (5, 5)
maxpool_size = (2, 2)
dr = 0.3

inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
# MobileNetV3Small(input_shape=(inputs.s),include_top=False, weights='imagenet')(inputs)
vgg = MobileNetV3Small(include_top=False)(inputs)
vgg.trainable = False
#flatten
flat = Flatten()(vgg)
classification = Dense(256, activation=tf.keras.layers.LeakyReLU())(flat)
classification = Dense(128, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dense(64, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dense(32, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dense(3, activation='softmax')(classification)

regression = Dense(256, activation=tf.keras.layers.LeakyReLU())(flat)
regression = Dense(128, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dense(64, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dense(32, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dense(16, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dense(4, activation=tf.keras.layers.LeakyReLU())(regression)





model = Model(inputs=inputs,
              outputs=[regression,
                       classification])
model.summary()
# model.compile(optimizer='rmsprop',
#               loss={'landmarks': 'mse',
#                     'people': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
#               loss_weights={'landmarks': .001,
#                             'lighting': .001,
#                             'face_cover': .001,
#                             'hat': .001,
#                             'glasses': .001,
#                             'dark_glasses': .001})



# %%
ds = ds.batch(4)
val = val.batch(4)
cat_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
mse_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#%%
wandb.init(project="preprocessing model",config={"epochs":epochs,"shape":shape,"filter_size":filter_size,"maxpool_size":maxpool_size,"dr":dr})
name=wandb.run.name
wandb.run.name='face _extraction_'+wandb.run.name
#%%
val_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
train_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
test_people_accuracy = tf.keras.metrics.CategoricalAccuracy()

# %%
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(ds):
        # Open a GradientTape to record the operations run
        print("\nStart of iteration %d" % (step,))
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            tape.watch(x_batch_train)

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            landmarks,people_count = model(x_batch_train,training=True)  # Logits for this minibatch
            empty= tf.zeros_like(landmarks)
            # Compute the loss value for this minibatch.
            booleans=tf.math.reduce_sum(tf.cast(empty == y_batch_train[0], dtype=tf.float32), axis=1) != 4
            train_slice=tf.boolean_mask(y_batch_train[0],booleans)
            train_landmarks=tf.boolean_mask(landmarks,booleans)
            landmarks_loss = mse_fn(train_slice, train_landmarks)
            categorical_loss = cat_loss(y_batch_train[1], people_count)
            train_people_accuracy(y_batch_train[1], people_count)
            wandb.log({'landmarks loss': landmarks_loss})
            wandb.log({'categorical loss': categorical_loss})
        wandb.log({'people accuracy': train_people_accuracy.result()})



        grads = tape.gradient([landmarks_loss,
                               categorical_loss],
                              model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for step, (x_batch_val, y_batch_val) in enumerate(val):
        landmarks, people_count = model(x_batch_val, training=False)
        empty = tf.zeros_like(landmarks)
        booleans = tf.math.reduce_sum(tf.cast(empty == y_batch_val[0], dtype=tf.float32), axis=1) != 4
        train_slice = tf.boolean_mask(y_batch_val[0], booleans)
        train_landmarks = tf.boolean_mask(landmarks, booleans)
        val_landmarks_loss = mse_fn(train_slice, train_landmarks)
        val_categorical_loss = cat_loss(y_batch_val[1], people_count)
        val_people_accuracy(y_batch_val[1], people_count)
        wandb.log({'val landmarks loss': val_landmarks_loss,
                   'val  categorical loss': val_categorical_loss})
    wandb.log({'val people accuracy': val_people_accuracy.result()})



    train_people_accuracy.reset_states()
    val_people_accuracy.reset_states()
model.save(r'C:\Users\isaac\PycharmProjects\face_exctraction\face_extraction-'+name+'.h5')
#%%
# wandb.init(project="preprocessing model", resume=True)
#hot encode poeple


# %%

#%%
test = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test = test.map(load_and_preprocess_from_path_label)
test=test.batch(1)
test_landmarks_loss=0
test_categorical_loss=0
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
        ax.imshow(x_batch_test.numpy()[0].astype(int))
        ax.add_patch(plt.Rectangle((numpy_landmarks[0]*256,numpy_landmarks[1]*256)
                                   ,(numpy_landmarks[2]*256),
                                   (numpy_landmarks[3]*256),fill=False,edgecolor='r',linewidth=3))
        ax.add_patch(plt.Rectangle((ground_truth[0]*256, ground_truth[1]*256), ground_truth[2]*256 , ground_truth[3]*256 , fill=False, edgecolor='blue', linewidth=3))
        print('hey')
        test_landmarks_loss += mse_fn(train_slice, train_landmarks)
        # fig.show()
    table.add_data(wandb.Image(fig), float(test_landmarks_loss),class1,correct_class,class1==correct_class)

wandb.log({'test landmarks loss': test_landmarks_loss,
            'test categorical loss': test_categorical_loss})
wandb.log({'test people accuracy': test_people_accuracy.result()})
#%%
model.save(r'C:\Users\isaac\PycharmProjects\face_exctraction\models\face_extraction-'+name+'.h5')
wandb.log({'test table': table})
wandb.finish()
