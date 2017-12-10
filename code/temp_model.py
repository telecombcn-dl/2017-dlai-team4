#coding=utf-8

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    import keras
except:
    pass

try:
    from keras.applications import VGG19
except:
    pass

try:
    from keras.models import Model, Sequential
except:
    pass

try:
    from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
except:
    pass

try:
    from keras.preprocessing.image import ImageDataGenerator
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    import random
except:
    pass

try:
    import os
except:
    pass

try:
    from tqdm import tqdm
except:
    pass

try:
    from sklearn import preprocessing
except:
    pass

try:
    from sklearn.model_selection import train_test_split
except:
    pass

try:
    import cv2
except:
    pass

try:
    from subprocess import check_output
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

print('Getting data')
df_train = pd.read_csv('../input/labels.csv')
df_test = pd.read_csv('../input/sample_submission.csv')

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)

im_size = 90;
x_train = []
y_train = []
x_test = []

i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread('../input/train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
num_class = y_train_raw.shape[1]

print('Splitting into training/validation')
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)



def keras_fmin_fnct(space):

    print('Creating model')
     #with tf.device('/device:GPU:0'):
    batchSize = 64;
    
    dropout = space['dropout'];
    
    print()
    print('dropout=',dropout)
    print('Ndrop=',Ndrop)
    print()
    
    stepsPerEpoch = round( len(X_train)/batchSize );
    
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(im_size,im_size,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu')) # after batch norm
    model.add(Conv2D(32, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu')) # after batch norm
    model.add(Conv2D(64, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    # Fully connected layer

    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(120,activation='softmax'))
    #model.add(Conv2D(10,(3,3)))
    #model.add(GlobalAveragePooling2D('none'))
    
    
    #predictions = Dense(num_class, activation='softmax')(x)

    # This is the model we will train
    #model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    #for i in range(len(base_model.layers)):
    #    base_model.layers[i].trainable = False

    # predetermined optimizer
    lr=0.00020389590556056983;
    beta_1=0.9453158868247398;
    beta_2=0.9925872692991417;
    decay=0.000821336141287018;
    adam = keras.optimizers.Adam(lr=lr,beta_1=beta_1,beta_2=beta_2,decay=decay)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam, 
                  metrics=['accuracy'])

    callbacks_list = [];
    callbacks_list.append(keras.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=10,
        verbose=1));


    # data augmentation & fitting
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=True);
    
    model.fit_generator(
        datagen.flow(X_train,Y_train,batch_size=batchSize),
        steps_per_epoch=stepsPerEpoch,
        epochs=150,
        verbose=1,
        validation_data=(X_valid,Y_valid),
        workers=2,
        shuffle=True,
        callbacks=callbacks_list)
#     model.fit(X_train, Y_train,
#       epochs=100,
#       batch_size = batchSize,
#       validation_data=(X_valid, Y_valid),
#       verbose=1,
#       callbacks=callbacks_list)

    score, acc = model.evaluate(X_valid, Y_valid, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'dropout': hp.uniform('dropout', 0.4,0.9),
    }
