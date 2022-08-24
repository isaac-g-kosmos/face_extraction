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
shape = (300, 300)
epochs = 4
df = pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset.csv')
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
#%%
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

main_branch = Conv2D(16, kernel_size=filter_size, padding="same")(inputs)
main_branch = Activation("relu")(main_branch)
main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
main_branch = Dropout(dr)(main_branch)

main_branch = Conv2D(8, kernel_size=filter_size, padding="same")(main_branch)
main_branch = Activation("relu")(main_branch)
main_branch = MaxPooling2D(pool_size=maxpool_size)(main_branch)
main_branch = Dropout(dr)(main_branch)

main_branch = Flatten()(main_branch)
main_branch = Dense(32)(main_branch)
main_branch = Activation('relu')(main_branch)
main_branch = Dropout(dr)(main_branch)

land_marks = Dense(4, activation='relu', name='landmarks')(main_branch)
people_count = Dense(3, activation='softmax', name='people')(main_branch)

model = Model(inputs=inputs,
              outputs=[land_marks,
                       people_count])
# model.summary()
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
ds = ds.batch(28)
val = val.batch(28)
cat_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
mse_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
#%%
wandb.init(project="preprocessing model",config={"epochs":epochs,"shape":shape,"filter_size":filter_size,"maxpool_size":maxpool_size,"dr":dr})
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
#%%
# wandb.init(project="preprocessing model", resume=True)
test = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test = test.map(load_and_preprocess_from_path_label)
test=test.batch(28)
for step, (x_batch_test, y_batch_test) in enumerate(test):
    landmarks, people_count = model(x_batch_test, training=False)
    empty = tf.zeros_like(landmarks)
    booleans = tf.math.reduce_sum(tf.cast(empty == y_batch_test[0], dtype=tf.float32), axis=1) != 4
    train_slice = tf.boolean_mask(y_batch_test[0], booleans)
    train_landmarks = tf.boolean_mask(landmarks, booleans)
    test_landmarks_loss = mse_fn(y_batch_test[0], landmarks)
    test_categorical_loss = cat_loss(y_batch_test[1], people_count)
    test_people_accuracy(y_batch_test[1], people_count)
    wandb.log({'test landmarks loss': test_landmarks_loss,
                'test categorical loss': test_categorical_loss})
wandb.log({'test people accuracy': test_people_accuracy.result()})
