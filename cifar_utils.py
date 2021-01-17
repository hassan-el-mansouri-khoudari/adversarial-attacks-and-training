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

import tensorflow as tf


    
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
def adversarial_pattern2(image, label,model):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_object(prediction, label)
    
    gradient = tape.gradient(loss, image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad

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
def generate_adversarials_train_FGSM(batch_size,eps,model):
  max_mem = 2000
  indices_chosen = np.random.choice(x_train.shape[0],batch_size, replace = False)
  labels = y_train[indices_chosen]
  ori_images = x_train[indices_chosen]
  adversarial = None
  if batch_size > max_mem :
    perturbations = adversarial_pattern2(ori_images[0:max_mem], labels[0:max_mem],model).numpy()
    adversarial = ori_images[0:max_mem] + eps * perturbations 
    for cpt in range(1, (batch_size//max_mem)+1):
      start_interval = cpt*max_mem
      end_interval = min(batch_size, (cpt+1)*max_mem)
      pert = adversarial_pattern2(ori_images[start_interval:end_interval], labels[start_interval:end_interval], model).numpy()    
      adv = ori_images[start_interval:end_interval] + eps * pert 
      adversarial = np.concatenate((adversarial, adv))
  else :
    perturbations = adversarial_pattern2(ori_images, labels, model).numpy()    
    adversarial = ori_images + eps * perturbations 
  return adversarial, labels


## A generator to generate FGSM adversarial attacks from test
# def generate_adversarials_test_FGSM(batch_size,eps, model):

#     if batch_size == None :
#       indices_chosen = list(range(len(x_test)))
#     else :
#       indices_chosen = np.random.choice(x_test.shape[0],batch_size, replace = False)

#     labels = y_test[indices_chosen]
#     ori_images = x_test[indices_chosen]s
#     perturbations = adversarial_pattern2(ori_images, labels, model).numpy()    

#     adversarial = ori_images + eps * perturbations 

 
    # return adversarial, labels

## A generator to generate FGSM adversarial attacks from test
def generate_adversarials_test_FGSM(batch_size,eps,model):
  max_mem = 2000
  indices_chosen = np.random.choice(x_test.shape[0],batch_size, replace = False)
  labels = y_test[indices_chosen]
  ori_images = x_test[indices_chosen]
  adversarial = None
  if batch_size > max_mem :
    perturbations = adversarial_pattern2(ori_images[0:max_mem], labels[0:max_mem],model).numpy()
    adversarial = ori_images[0:max_mem] + eps * perturbations 
    for cpt in range(1, (batch_size//max_mem)+1):
      start_interval = cpt*max_mem
      end_interval = min(batch_size, (cpt+1)*max_mem)
      pert = adversarial_pattern2(ori_images[start_interval:end_interval], labels[start_interval:end_interval], model).numpy()    
      adv = ori_images[start_interval:end_interval] + eps * pert 
      adversarial = np.concatenate((adversarial, adv))
  else :
    perturbations = adversarial_pattern2(ori_images, labels, model).numpy()    
    adversarial = ori_images + eps * perturbations 
  return adversarial, labels

def attack_PGD(model,x, y, delta, step_size, max_iter=100, verbose = 0):
    pert_x = x
    cpt = 0
    while True :
        cpt += 1
        pert = adversarial_pattern2(pert_x, y, model)
        new_pert_x = pert_x + step_size*pert
        #First clip serves to keep values between 0 and 1 and second clip for attacked examples to stay within the original examples l_inf sphere
        new_pert_x = np.clip(np.clip(new_pert_x, x-delta, x+delta),0,1)

        if cpt == max_iter :
            pert_x = new_pert_x 
            if verbose :
              print("max iter break")
            break
        
        pert_x = new_pert_x
    
    return pert_x


## A generator to generate PGD adversarial attacks from train
def generate_adversarials_train_PGD(model, batch_size,delta, step_size, max_iter , seed=None, verbose = 0):
    random.seed(seed)
    max_mem = 3000
    x = []
    y = []
    indices_chosen = []

    indices_chosen = np.random.choice(x_train.shape[0],batch_size, replace = False)

    labels = y_train[indices_chosen]
    ori_images = x_train[indices_chosen]
    
    adversarial = None
    if batch_size > max_mem :
      adversarial = attack_PGD(model, ori_images[0:max_mem],labels[0:max_mem], delta, step_size, max_iter, verbose)
      for cpt in range(1, (batch_size//max_mem)+1):
        start_interval = cpt*max_mem
        end_interval = min(batch_size, (cpt+1)*max_mem)

        adversarial = np.concatenate((adversarial, attack_PGD(model,ori_images[start_interval:end_interval],labels[start_interval:end_interval], delta, step_size, max_iter, verbose)), axis = 0)
    else : 
      adversarial = attack_PGD(model,ori_images, labels, delta, step_size, max_iter, verbose)
    
    return indices_chosen, adversarial, labels
## A generator to generate PGD adversarial attacks from test set

def generate_adversarials_test_PGD(model,batch_size,delta, step_size, max_iter , seed=None, verbose = 0):
    random.seed(seed)
    max_mem = 3000
    x = []
    y = []
    indices_chosen = []

    indices_chosen = np.random.choice(x_test.shape[0],batch_size, replace = False)

    labels = y_test[indices_chosen]
    ori_images = x_test[indices_chosen]
    
    adversarial = None
    if batch_size > max_mem :
      adversarial = attack_PGD(model,ori_images[0:max_mem],labels[0:max_mem], delta, step_size, max_iter, verbose)
      for cpt in range(1, (batch_size//max_mem)+1):
        start_interval = cpt*max_mem
        end_interval = min(batch_size, (cpt+1)*max_mem)

        adversarial = np.concatenate((adversarial, attack_PGD(model,ori_images[start_interval:end_interval],labels[start_interval:end_interval], delta, step_size, max_iter, verbose)), axis = 0)
    else : 
      adversarial = attack_PGD(model,ori_images, labels, delta, step_size, max_iter, verbose)

    return indices_chosen, adversarial, labels
##### test of denoising feature maps
def denoising(name, l, embed=True, softmax=True):
    """
    Feature Denoising, Fig 4 & 5.
    """
    with tf.variable_scope(name):
        f = non_local_op(l, embed=embed, softmax=softmax)
        f = Conv2D('conv', f, l.shape[1], 1, strides=1, activation=get_bn(zero_init=True))
        l = l + f
    return l


def non_local_op(l, embed, softmax):
    """
    Feature Denoising, Sec 4.2 & Fig 5.
    Args:
        embed (bool): whether to use embedding on theta & phi
        softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
    """
    n_in, H, W = l.shape.as_list()[1:]
    if embed:
        theta = Conv2D(n_in / 2, 1,
                       strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
        phi = Conv2D(n_in / 2, 1,
                     strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(l)
        g = l
    else:
        theta, phi, g = l, l, l
    if n_in > H * W or softmax:
        f = tf.einsum('niab,nicd->nabcd', theta, phi)
        if softmax:
            orig_shape = tf.shape(f)
            f = tf.reshape(f, [-1, H * W, H * W])
            f = f / tf.sqrt(tf.cast(theta.shape[1], theta.dtype))
            f = tf.nn.softmax(f)
            f = tf.reshape(f, orig_shape)
        f = tf.einsum('nabcd,nicd->niab', f, g)
    else:
        f = tf.einsum('nihw,njhw->nij', phi, g)
        f = tf.einsum('nij,nihw->njhw', f, theta)
    if not softmax:
        f = f / tf.cast(H * W, f.dtype)
    return tf.reshape(f, tf.shape(l))

