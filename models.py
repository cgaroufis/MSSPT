#containers for the standalone convolutional frontend, as well as the TUne+ adaptation

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Multiply,Reshape,Permute,BatchNormalization,ReLU,Flatten,Dense,Conv2D,AveragePooling2D,MaxPooling2D,LeakyReLU,Concatenate,Dropout,Conv2DTranspose

def get_tail_network(dataset):

  NtailBls = 7
  Nfilts = np.asarray([128,128,256,256,256,256,512,512])//2
  epsilon = 10e-5

  x = Input((384,256)) #2 for stereo
  m = x
  melscaler = tf.signal.linear_to_mel_weight_matrix(128, 256, 16000, 100, 7800)
  x = tf.tensordot(x,melscaler,1)
  x = tf.math.log(x+epsilon)/2.3025
  x = tf.expand_dims(x,axis=-1)

  for i in range(0,NtailBls):

    for j in range(0,2):
      x = Conv2D(Nfilts[i],(3,3),padding='same')(x)
      x = BatchNormalization()(x) 
      x = ReLU()(x)
    if i != NtailBls - 1:
      x = MaxPooling2D((2,2))(x)
    else:
      x = MaxPooling2D((3,2))(x)

  x = Flatten()(x)
  x = Dense(512)(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Dropout(0.5)(x)
  if dataset == 'mtat':
    x = Dense(50,activation='sigmoid')(x)
  elif dataset == 'fma':
    x = Dense(16,activation='softmax')(x)

  return tf.keras.Model(inputs=m,outputs=x)

def get_tune_network(dataset,n_src,skips):
  

  multisource = (n_src == 4)
  ceskips = []
  etskips = []
  NencBls = 6
  NdecBls = 6
  NtailBls = 7
  Nfilts = np.asarray([32,64,128,256,384,384])//2
  NfiltsT = np.asarray([128,256,256,256,256,256,512,512])//2

  epsilon = 10e-5

  x = Input((384,256,1)) #2 for stereo
  m = x #non-log input

  x = tf.math.log(x+epsilon)/2.3025
  mm = x

  ceskips.append(x)
  for i in range(0,NencBls):
    for j in range(0,2):
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU(0.01)(x)
      x = tf.keras.layers.Conv2D(Nfilts[i],(3,3),padding='same')(x)
  
    x = MaxPooling2D((2,2))(x)
    if i < 5: #and skip:
      ceskips.append(x)

  for i in range(0,NdecBls):
    if i > 0: #and skip:
      x = Concatenate()([x,ceskips[6-i]])
    x = Conv2DTranspose(Nfilts[5-i],(2,2),strides=(2,2),padding='same')(x)
    for j in range(0,2):
      x = BatchNormalization()(x)
      x = LeakyReLU(0.01)(x)
      x = Conv2D(Nfilts[5-i],(3,3),padding='same')(x)
    if i < NdecBls-1:
      zz = MaxPooling2D((1,2))(x)
      etskips.append(zz)

  x = Concatenate()([x,ceskips[0]])

  for j in range(0,2): 
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(32,(3,3),padding='same')(x)

  if multisource:
    x = Conv2D(4,(3,3),activation='sigmoid',padding='same')(x)
  else:
    x = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x) #mask
  x = Multiply()([m,x]) #magnitude estimate of the source, multiplied.

  if not multisource:
    x = Reshape((384,256))(x)
  melscaler = tf.signal.linear_to_mel_weight_matrix(128, 256, 16000, 100, 7800)
  if multisource:
    x1 = tf.tensordot(x[:,:,:,0],melscaler,1)
    x2 = tf.tensordot(x[:,:,:,1],melscaler,1)
    x3 = tf.tensordot(x[:,:,:,2],melscaler,1)
    x4 = tf.tensordot(x[:,:,:,3],melscaler,1)
    x = tf.concat([tf.expand_dims(x1,axis=-1),tf.expand_dims(x2,axis=-1),tf.expand_dims(x3,axis=-1),tf.expand_dims(x4,axis=-1)],axis=-1)
  else:
    x = tf.tensordot(x,melscaler,1)
  x = tf.math.log(x+epsilon)/2.3025
  if not multisource:
    x = tf.expand_dims(x,axis=-1)

#also map the input src

  mm = Reshape((384,256))(mm)
  mm = tf.tensordot(mm,melscaler,1)
  mm = tf.expand_dims(mm,axis=-1)
  z = Concatenate(axis=3)([x,mm])
  x = z
# Tail

  for i in range(0,NtailBls):

    if i > 0 and i < (skips+1):
      x = Concatenate()([x,etskips[5-i]])

    for j in range(0,2):
      x = Conv2D(NfiltsT[i],(3,3),padding='same')(x)
      x = BatchNormalization()(x) 
      x = ReLU()(x)
    if i != NtailBls - 1:
      x = MaxPooling2D((2,2))(x)
    else:
      x = MaxPooling2D((3,2))(x)

  x = Flatten()(x)
  x = Dense(512)(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Dropout(0.5)(x)
  if dataset == 'mtat':
    x = Dense(50,activation='sigmoid')(x)
  elif dataset == 'fma':
    x = Dense(16,activation='softmax')(x)

  return tf.keras.Model(inputs=m,outputs=x)


