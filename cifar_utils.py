import numpy as np
from keras.models import Model
from keras.layers import Dropout,Dense, Activation, Input## layers of the model
from tensorflow.keras.optimizers import SGD,Adam ## for learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential ## for building the model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout, AvgPool2D, Conv2D, BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint
import random


##load cifar10
def load_CIFAR10_data():
  ## import CIFAR-10
  (train_images, train_labels),(test_images, test_labels)= cifar10.load_data()
  x_train = train_images.astype("float32") / 255
  x_test = test_images.astype("float32") / 255

  return x_train,train_labels,x_test,test_labels


## classification model for cifar10
def build_model_base_CNN(input_shape):
  input = Input(shape=input_shape)
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
  model.add(BatchNormalization())
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2)))
  model.add(Dropout(0.3))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2)))
  model.add(Dropout(0.3))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2)))
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dropout(0.3))
  model.add(Dense(10, activation='softmax'))
  # compile model
  opt = SGD(lr=0.01, momentum=0.9)
  op = Adam(lr=0.01)
  model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
  return model


loss_object = tf.keras.losses.CategoricalCrossentropy()
### Gradient of the cost for FSGM attack
def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_object(prediction, label)
    
    gradient = tape.gradient(loss, image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad




## A generator to generate FGSM adversarial attacks from train
def generate_adversarials_train_FGSM(batch_size,eps):


    if batch_size == None :
      indices_chosen = list(range(len(x_train)))
    else :
      indices_chosen = np.random.choice(x_train.shape[0],batch_size, replace = False)

    labels = y_train[indices_chosen]
    ori_images = x_train[indices_chosen]
    
    perturbations = adversarial_pattern(ori_images, labels).numpy()    

    adversarial = ori_images + eps * perturbations 

 
    return adversarial, labels


## A generator to generate FGSM adversarial attacks from test
def generate_adversarials_test_FGSM(batch_size,eps):

    if batch_size == None :
      indices_chosen = list(range(len(x_test)))
    else :
      indices_chosen = np.random.choice(x_test.shape[0],batch_size, replace = False)

    labels = y_test[indices_chosen]
    ori_images = x_test[indices_chosen]
    perturbations = adversarial_pattern(ori_images, labels).numpy()    

    adversarial = ori_images + eps * perturbations 

 
    return adversarial, labels

