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
from pre_process import augmented_cut, BB_Metrics
shape = (256, 256)
epochs = 25
lr=1e-3
df = pd.read_csv(r'/home/ubuntu/face_extraction/dataset_augmentations_linux.csv')

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
df['path']=df.apply(lambda row: os.path.join(r'/home/ubuntu/face_extraction/dlib_face_detection_dataset',row['path']), axis=1)
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
ds = ds.map(load_and_preprocess_from_path_label)
val = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
val = val.map(load_and_preprocess_from_path_label)

# %%
filter_size = (5, 5)
maxpool_size = (2, 2)
dr = 0.5

inputs = Input(shape=(shape[0], shape[1], 3), name='main_input')
# MobileNetV3Small(input_shape=(inputs.s),include_top=False, weights='imagenet')(inputs)
vgg = MobileNetV3Small(include_top=False)(inputs)
#[4, 8, 8, 576]
vgg.trainable = False
#flatten
classification=tf.keras.layers.ZeroPadding2D(padding=1)(vgg)
classification = Conv2D(180, kernel_size=(5, 5), padding="valid")(classification)
classification = Activation("LeakyReLU")(classification)
classification = MaxPooling2D(pool_size=(2, 2))(classification)
classification=tf.keras.layers.ZeroPadding2D(padding=1)(classification)
classification = Conv2D(180, kernel_size=(3, 3), padding="valid")(classification)
classification = Activation("LeakyReLU")(classification)
classification = MaxPooling2D(pool_size=(2, 2))(classification)
classification = Flatten()(classification)
classification = Dense(480, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dropout(.2)(classification)
classification = Dense(256, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dropout(.2)(classification)
classification = Dense(128, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dropout(.2)(classification)
classification = Dense(64, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dropout(.2)(classification)
classification = Dense(32, activation=tf.keras.layers.LeakyReLU())(classification)
classification = Dropout(.2)(classification)
classification = Dense(3, activation='softmax')(classification)

regression=tf.keras.layers.ZeroPadding2D(padding=1)(vgg)
regression = Conv2D(560, kernel_size=(3, 3), padding="valid")(regression)
regression = Activation("LeakyReLU")(regression)
regression = MaxPooling2D(pool_size=(2, 2))(regression)
regression=tf.keras.layers.ZeroPadding2D(padding=1)(regression)
regression = Conv2D(500, kernel_size=(3, 3), padding="valid")(regression)
regression = Activation("LeakyReLU")(regression)
regression = MaxPooling2D(pool_size=(2, 2))(regression)
regression = Flatten()(regression)
regression = Dense(540, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dropout(dr)(regression)
regression = Dense(256, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dropout(dr)(regression)
regression = Dense(128, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dropout(dr)(regression)
regression = Dense(64, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dropout(dr)(regression)
regression = Dense(32, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dropout(dr)(regression)
regression = Dense(16, activation=tf.keras.layers.LeakyReLU())(regression)
regression = Dense(4, activation=tf.keras.layers.LeakyReLU())(regression)






model = Model(inputs=inputs,
              outputs=[regression,
                       classification])
model.summary()


# %%
ds = ds.batch(10)
val = val.batch(10)
cat_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# mse_fn =tf.keras.losses.Huber ()
mse_fn =tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#%%
wandb.init(project="preprocessing model",config={"epochs":epochs,"shape":shape,"filter_size":filter_size,"maxpool_size":maxpool_size,"dr":dr
                                                  ,"dataset":'Augmented',
                                                 'loss':'MSE'})
name=wandb.run.name
wandb.run.name='face _extraction_'+wandb.run.name
#%%
val_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
train_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
test_people_accuracy = tf.keras.metrics.CategoricalAccuracy()
bb_metrics=BB_Metrics()
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
            bb_metrics.update(train_landmarks.numpy(), train_slice.numpy(),True)
            if step == 0:
                combined_loss = landmarks_loss
            else:
                combined_loss += landmarks_loss
            wandb.log({'landmarks loss': landmarks_loss})
            wandb.log({'categorical loss': categorical_loss})
        grads = tape.gradient([landmarks_loss,
                               categorical_loss],
                              model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))



    wandb.log(bb_metrics.get_metrics('train_BB'))
    bb_metrics.clear_data()
    wandb.log({'Train combined loss': combined_loss})
    wandb.log({'Train accuracy': train_people_accuracy.result()})
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    for step, (x_batch_val, y_batch_val) in enumerate(val):
        landmarks, people_count = model(x_batch_val, training=False)
        empty = tf.zeros_like(landmarks)
        booleans = tf.math.reduce_sum(tf.cast(empty == y_batch_val[0], dtype=tf.float32), axis=1) != 4
        train_slice = tf.boolean_mask(y_batch_val[0], booleans)
        train_landmarks = tf.boolean_mask(landmarks, booleans)
        val_landmarks_loss = mse_fn(train_slice, train_landmarks)
        val_categorical_loss = cat_loss(y_batch_val[1], people_count)
        val_people_accuracy(y_batch_val[1], people_count)
        bb_metrics.update(train_landmarks.numpy(), train_slice.numpy())
        if step == 0:
            combined_loss = val_landmarks_loss
        else:
            combined_loss += val_landmarks_loss
        wandb.log({'val landmarks loss': val_landmarks_loss,
                   'val  categorical loss': val_categorical_loss})
    wandb.log({'Val accuracy': val_people_accuracy.result()})
    wandb.log({'Combined val loss': combined_loss})
    wandb.log(bb_metrics.get_metrics('val_BB'))
    bb_metrics.clear_data()


    train_people_accuracy.reset_states()
    val_people_accuracy.reset_states()
model.save(r'/home/ubuntu/face_extraction/models/face_extraction-'+name+'.h5')
#%%
# wandb.init(project="preprocessing model", resume=True)
#hot encode poeple


# %%

#%%
#%%
test = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test = test.map(load_and_preprocess_from_path_label)
test=test.batch(10)
test_landmarks_loss=0
test_categorical_loss=0
bb_metrics=BB_Metrics()
for step, (x_batch_val, y_batch_val) in enumerate(test):
    print("Start of val %d" % (step,))
    landmarks, people_count = model(x_batch_val, training=False)
    empty = tf.zeros_like(landmarks)
    booleans = tf.math.reduce_sum(tf.cast(empty == y_batch_val[0], dtype=tf.float32), axis=1) != 4
    train_slice = tf.boolean_mask(y_batch_val[0], booleans)
    train_landmarks = tf.boolean_mask(landmarks, booleans)
    val_landmarks_loss = mse_fn(train_slice, train_landmarks)
    val_categorical_loss = cat_loss(y_batch_val[1], people_count)
    test_people_accuracy(y_batch_val[1], people_count)
    bb_metrics.update(train_landmarks.numpy(), train_slice.numpy())
    if step == 0:
        combined_loss = val_landmarks_loss
    else:
        combined_loss += val_landmarks_loss
    wandb.log({'val landmarks loss': val_landmarks_loss,
               'val  categorical loss': val_categorical_loss})


wandb.log({'test landmarks loss': combined_loss,
            'test categorical loss': val_categorical_loss})
wandb.log({'test people accuracy': test_people_accuracy.result()})
wandb.log(bb_metrics.get_metrics('test_BB'))
#%%
model.save(r'home/ubuntu/face_extraction/models/face_extraction-'+name+'.h5')
# wandb.log({'test table': table})
wandb.save('/home/ubuntu/face_extraction/vgg16_experiment.py')
wandb.finish()
print(r'home/ubuntu/face_extraction/models/face_extraction-'+name+'.h5')