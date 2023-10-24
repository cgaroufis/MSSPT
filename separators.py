#container for the unisource and multisource separation U-Nets

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Multiply,Reshape,Permute,BatchNormalization,Conv2D,AveragePooling2D,MaxPooling2D,LeakyReLU,Concatenate,Conv2DTranspose


def get_unisource_separator():

  NencBls = 6
  NdecBls = 6
  epsilon = 10e-5
  Nfilts = np.asarray([32,64,128,256,384,384])//2 #small config: 32, 64, 128, 256, 384, 384

  skips = []
  
  x = Input((61792,)) #2 for stereo
  y = x

#stft
  x = tf.signal.stft(x,512,160,window_fn=tf.signal.hann_window) #hann is the default but nevertheless.
  x = tf.expand_dims(x,axis=-1)
  l = x[:,:,-1,:]
  z = tf.math.angle(x) #angle to get used only after src estimation
  x = tf.math.abs(x) #mag to get processed
  x = x[:,:,:-1,:]
  m = x
  z = z[:,:,:-1,:]

  x = tf.math.log(x+epsilon)/2.3026 #tf.math.log10()

  skips.append(x)
  for i in range(0,NencBls):
    for j in range(0,2):
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU(0.01)(x)
      x = tf.keras.layers.Conv2D(Nfilts[i],(3,3),padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    if i < 5:
      skips.append(x)

  for i in range(0,NdecBls):
    if i > 0:
      x = Concatenate()([x,skips[6-i]])
    x = Conv2DTranspose(Nfilts[5-i],(2,2),strides=(2,2),padding='same')(x)
    for j in range(0,2):
      x = BatchNormalization()(x)
      x = LeakyReLU(0.01)(x)
      x = Conv2D(Nfilts[5-i],(3,3),padding='same')(x)

  x = Concatenate()([x,skips[0]])

  for j in range(0,2): 
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(32,(3,3),padding='same')(x)

  x = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x) #mask
  x = Multiply()([m,x]) #magnitude estimate of the source

  # ! this works for inverse stft!
  x = tf.cast(x,tf.complex64)
  x = Multiply()([x,tf.math.exp(1j*tf.cast(z,tf.complex64))])
  x = tf.concat([x,tf.expand_dims(l,axis=-1)],axis=2)
  x = tf.squeeze(x,axis=-1) #inverse_stft_window_fn is necessary to autocalc. the correction factor
  x = tf.signal.inverse_stft(x,512,160,window_fn=tf.signal.inverse_stft_window_fn(160,forward_window_fn=tf.signal.hann_window))
  #UNet = Model(inputs=y,outputs=x)

  return tf.keras.Model(inputs=y,outputs=x)

def get_multisource_separator():

  epsilon = 10e-5

  # model definition

  NencBls = 6
  NdecBls = 6
  Nfilts = np.asarray([32,64,128,256,384,384])//2
  skips = []


  #if td_loss:
  
  x = Input((61792,)) #2 for stereo
  y = x

  #stft: keeping up w/ the given params (for 44.1khz) as far as possible.
  x = tf.signal.stft(x,512,160,window_fn=tf.signal.hann_window) #hann is the default but nevertheless.
  x = tf.expand_dims(x,axis=-1)
  l = x[:,:,-1:,:]
  l = tf.tile(l,tf.constant([1,1,1,4]))#,[1,1,1,2]) #expand last dim per number of srcs to multiply with the src_wise mask
  z = tf.math.angle(x) #angle to get used only after src estimation
  x = tf.math.abs(x) #mag to get processed
  x = x[:,:,:-1,:]
  m = tf.tile(x,tf.constant([1,1,1,4]))
  z = z[:,:,:-1,:]

  x = tf.math.log(x+epsilon)/2.3026 #tf.math.log10()


  skips.append(x)
  for i in range(0,NencBls):
    for j in range(0,2):
      x = BatchNormalization()(x)
      x = LeakyReLU(0.01)(x)
      x = Conv2D(Nfilts[i],(3,3),padding='same')(x)
    x = AveragePooling2D((2,2))(x)
    if i < 5:
      skips.append(x)

  for i in range(0,NdecBls):
    if i > 0:
      x = Concatenate()([x,skips[6-i]])
    x = Conv2DTranspose(Nfilts[5-i],(2,2),strides=(2,2),padding='same')(x)
    for j in range(0,2):
      x = BatchNormalization()(x)
      x = LeakyReLU(0.01)(x)
      x = Conv2D(Nfilts[5-i],(3,3),padding='same')(x)

  x = Concatenate()([x,skips[0]])

  for j in range(0,2): #additional layers?!
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(32,(3,3),padding='same')(x)

  x = Conv2D(4,(3,3),activation='sigmoid',padding='same')(x) #masks
  x = Multiply()([m,x]) #magnitude estimate of the source

  x = tf.cast(x,tf.complex64)
  x = Multiply()([x,tf.math.exp(1j*tf.cast(z,tf.complex64))])
  x = tf.concat([x,l],axis=2)
  x1 = tf.signal.inverse_stft(x[:,:,:,0],512,160,window_fn=tf.signal.inverse_stft_window_fn(160,forward_window_fn=tf.signal.hann_window))
  x2 = tf.signal.inverse_stft(x[:,:,:,1],512,160,window_fn=tf.signal.inverse_stft_window_fn(160,forward_window_fn=tf.signal.hann_window))
  x3 = tf.signal.inverse_stft(x[:,:,:,2],512,160,window_fn=tf.signal.inverse_stft_window_fn(160,forward_window_fn=tf.signal.hann_window))
  x4 = tf.signal.inverse_stft(x[:,:,:,3],512,160,window_fn=tf.signal.inverse_stft_window_fn(160,forward_window_fn=tf.signal.hann_window))
  x = tf.concat([tf.expand_dims(x1,axis=-1),tf.expand_dims(x2,axis=-1),tf.expand_dims(x3,axis=-1),tf.expand_dims(x4,axis=-1)],axis=-1)

  return tf.keras.Model(inputs=y,outputs=x)


