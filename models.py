#containers for the standalone backbones, as well as the TUne+ adaptation for both the CNN/AST backbone cases.
#code for helper functions for class tokens and positional embeddings gladly borrowed from 

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Multiply,Reshape,Permute,BatchNormalization,ReLU,Flatten,Dense,Conv2D,AveragePooling2D,MaxPooling2D,LeakyReLU,Concatenate,Dropout,Conv2DTranspose, MultiHeadAttention, LayerNormalization, Layer

class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_transformer_encoder(dataset,embeddings):

  epsilon = 10e-5
  embedding_dim = 768

  x = Input((768,256)) #2 for stereo
  m = x
  melscaler = tf.signal.linear_to_mel_weight_matrix(128, 256, 16000, 100, 7800) #128 mel bands
  x = tf.tensordot(x,melscaler,1)
  x = tf.math.log(x+epsilon)/2.3025 #log-mel spectogram
 
  x = 2*(x-tf.math.reduce_min(x))/(tf.math.reduce_max(x)-tf.math.reduce_min(x))- 1 #batch-wise [-1, 1] scaling
  x = tf.expand_dims(x,axis=-1)
  
  x = Conv2D(embedding_dim,kernel_size=(16,16),strides=(16,16),padding='valid')(x)
  x = Reshape((x.shape[1]*x.shape[2],embedding_dim))(x)

  #prepend class embedding (trainable)

  x = ClassToken()(x)
  x = AddPositionEmbs()(x)

  Nlayers = 12
  Nheads = 12

  for i in range(0,Nlayers):

    z = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=Nheads,key_dim=64)(x,x) #MHA+addnorm
    x = x+z #keydim = (embedding_size/headnum)
    
    z = x
    x = LayerNormalization()(x)
    x = Dense(3072)(x) #FFN+addnorm (inner MLP dim separately denoted)
    x = tf.keras.activations.gelu(x)
    x = Dense(embedding_dim)(x)
    x = Dropout(0.1)(x)
    x = x+z

  x = LayerNormalization()(x)
  x = x[:,0,:] #cls token
  if not embeddings: #1-layer classification mlp
    if dataset=='mtat':
      x = Dense(50,activation='sigmoid')(x)
    elif dataset=='fma':
      x = Dense(16,activation='softmax')(x)
    elif dataset=='gtzan':
      x = Dense(10,activation='softmax')(x)

  return tf.keras.Model(inputs=m,outputs=x)

def get_tail_network(dataset,embeddings):

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
  if not embeddings:
    if dataset == 'mtat':
      x = Dense(50,activation='sigmoid')(x)
    elif dataset == 'fma':
      x = Dense(16,activation='softmax')(x) #16
    elif dataset=='gtzan':
      x = Dense(10,activation='softmax')(x)

  return tf.keras.Model(inputs=m,outputs=x)

def get_transformer_tune(dataset,adapter,n_src,skips,embeddings):

  embedding_dim = 768

  multisource = (n_src == 4)
  ceskips = []
  etskips = []
  NencBls = 6
  NdecBls = 6
  Nfilts = np.asarray([32,64,128,256,384,384])//2
  NfiltsT = np.asarray([128,128,256,256,256,256,512,512])//2

  epsilon = 10e-5

  x = Input((768,256,1)) #2 for stereo
  m = x #non-log input

  mm = x
  x = tf.math.log(x+epsilon)/2.3025

  ceskips.append(x)
  for i in range(0,NencBls):
    for j in range(0,2):
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU(0.01)(x)
      x = tf.keras.layers.Conv2D(Nfilts[i],(3,3),padding='same')(x)
  
    x = MaxPooling2D((2,2))(x)
    if i < 5: 
      ceskips.append(x)

  for i in range(0,NdecBls):
    if i > 0: 
      x = Concatenate()([x,ceskips[6-i]])
    x = Conv2DTranspose(Nfilts[5-i],(2,2),strides=(2,2),padding='same')(x)
    for j in range(0,2):
      x = BatchNormalization()(x)
      x = LeakyReLU(0.01)(x)
      x = Conv2D(Nfilts[5-i],(3,3),padding='same')(x)
    if i < NdecBls-1:
      if adapter:
        if (i == 0):   
          zz = Conv2DTranspose(embedding_dim,(2,2),strides=(2,1),padding='same')(x) #+ strided conv
        else:
          zz = Conv2D(embedding_dim,kernel_size=(2**(i-1),2**i),strides=(2**(i-1),2**i),padding='same')(x) #patchification attempt
        zz = Reshape((384,768))(zz)
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
    x = Reshape((768,256))(x)
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

  mm = Reshape((768,256))(mm)
  mm = tf.tensordot(mm,melscaler,1)
  mm = tf.math.log(mm+epsilon)/2.3025
  mm = tf.expand_dims(mm,axis=-1)
  z = Concatenate(axis=3)([x,mm])
  x = z

# Tail

  x = 2*(x-tf.math.reduce_min(x))/(tf.math.reduce_max(x)-tf.math.reduce_min(x))- 1 #[-1, 1] scaling
  
  x = Conv2D(embedding_dim,kernel_size=(16,16),strides=(16,16),padding='valid')(x)
  x = Reshape((x.shape[1]*x.shape[2],embedding_dim))(x)

  #prepend class embedding (trainable)

  x = ClassToken()(x)
  x = AddPositionEmbs()(x)

  Nlayers = 12
  Nheads = 12

  for i in range(0,Nlayers):

    skipct = 0
    if (i > 0) and (i %2 == 0) and (skipct < skips): #skip connection.
      if adapter:
        x = tf.concat((x[:,0:1,:],x[:,1:,:]+etskips[5-i//2]),axis=1) #x dims: 192x768
      skipct += 1  
    
    z = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=Nheads,key_dim=64)(x,x) #MHA+addnorm
    x = x+z #keydim = (embedding_size/headnum)
    
    z = x
    x = LayerNormalization()(x)
    x = Dense(3072)(x) #FFN+addnorm (inner MLP dim separately denoted)
    x = tf.keras.activations.gelu(x)
    x = Dense(embedding_dim)(x)
    x = Dropout(0.1)(x)
    x = x+z

  x = LayerNormalization()(x)
  x = x[:,0,:] #cls token

  if not embeddings:
    if dataset == 'mtat':
      x = Dense(50,activation='sigmoid')(x)
    elif dataset == 'fma':
      x = Dense(16,activation='softmax')(x)
    elif dataset == 'gtzan':
      x = Dense(10,activation='softmax')(x)

  return tf.keras.Model(inputs=m,outputs=x)

def get_tune_network(dataset,adapter,n_src,skips,embeddings):
  
  multisource = (n_src == 4)
  ceskips = []
  etskips = []
  NencBls = 6
  NdecBls = 6
  NtailBls = 7
  Nfilts = np.asarray([32,64,128,256,384,384])//2
  NfiltsT = np.asarray([128,128,256,256,256,256,512,512])//2

  epsilon = 10e-5

  x = Input((384,256,1)) #2 for stereo
  m = x #non-log input

  mm = x
  x = tf.math.log(x+epsilon)/2.3025
  #mm = x

  ceskips.append(x)
  for i in range(0,NencBls):
    for j in range(0,2):
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU(0.01)(x)
      x = tf.keras.layers.Conv2D(Nfilts[i],(3,3),padding='same')(x)
  
    x = MaxPooling2D((2,2))(x)
    if i < 5:
      ceskips.append(x)

  for i in range(0,NdecBls):
    if i > 0: 
      x = Concatenate()([x,ceskips[6-i]])
    x = Conv2DTranspose(Nfilts[5-i],(2,2),strides=(2,2),padding='same')(x)
    for j in range(0,2):
      x = BatchNormalization()(x)
      x = LeakyReLU(0.01)(x)
      x = Conv2D(Nfilts[5-i],(3,3),padding='same')(x)
    if i < NdecBls-1:
      zz = MaxPooling2D((1,2))(x)
      if adapter:
        zz = Conv2D(NfiltsT[4-i],(1,1),padding='same')(zz) #1x1 conv to resize
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
  mm = tf.math.log(mm+epsilon)/2.3025
  mm = tf.expand_dims(mm,axis=-1)
  z = Concatenate(axis=3)([x,mm])
  x = z
# Tail

  for i in range(0,NtailBls):

    if i > 0 and i < (skips+1):
      if adapter:
        x = x+etskips[5-i]
      else:
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
  if not embeddings:
    if dataset == 'mtat':
      x = Dense(50,activation='sigmoid')(x)
    elif dataset == 'fma':
      x = Dense(16,activation='softmax')(x)
    elif dataset == 'gtzan':
      x = Dense(10,activation='softmax')(x)

  return tf.keras.Model(inputs=m,outputs=x)


