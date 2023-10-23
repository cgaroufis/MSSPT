# reimplementation of a U-Net for music source separation
# implementation details based on the U-Net baseline described in Q. Kong et al, Proc. ISMIR 2021
# usage: python3 train_separator.py path_to_musdb18_data model_path source {out of: bass, drums, multisource, other, vocal}
# eg: python3 train_separator.py /data/musdb18/npys models/myexperiment vocal

import os
import gc
import sys
import numpy as np
import tensorflow as tf
import argparse
import separators
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Multiply,Reshape,Permute,BatchNormalization,Conv2D,MaxPooling2D,LeakyReLU,Concatenate,Conv2DTranspose

parser = argparse.ArgumentParser()
parser.add_argument('datapath',type=str)
parser.add_argument('checkpoint_path',type=str) #directory to store the models
parser.add_argument('source',type=str)

arguments = parser.parse_args()
data_path = arguments.datapath
checkpoint_path = arguments.checkpoint_path
source = arguments.source

datapath = data_path+'/train'
valid_datapath = data_path+'/valid'

os.makedirs('./'+checkpoint_path,exist_ok=True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

if source == 'multisource':
  UNet = separators.get_multisource_separator()
else:
  UNet = separators.get_unisource_separator() 


if source == 'vocal':  
  lr_ = 0.001
elif source == 'bass':
  lr_ = 0.0001
elif source == 'drums':
  lr_ = 0.0002
else:
  lr_ = 0.0003

UNet.compile(optimizer=Adam(lr=lr_),loss='mae')
checkpoint = tf.train.Checkpoint(UNet)

print('Current optimizer lr value', K.get_value(UNet.optimizer.lr))

print(UNet.summary())
gc.collect()

nFiles = len(os.listdir(datapath))//5
nValFiles = len(os.listdir(valid_datapath))//5
nSteps = 300000
input_len = 61792
minibatch = 8

running_loss = 0

for i in range(0,nSteps):

  ids = np.random.permutation(nFiles)[:640]
  mix = np.empty((minibatch,input_len))
  if source == 'multisource':
    seps = np.empty((minibatch,input_len,4))
  else:
    seps = np.empty((minibatch,input_len))
  for k in range(0,minibatch):
    
    if source == 'multisource':
    
      v1 = np.load(datapath+'/vocal_seg_'+str(ids[7*k])+'.npy') #augmentation scheme: mixing 2x the same source (bar bass)
      v1 = (v1[:,0] + v1[:,1])/2
      v2 = np.load(datapath+'/vocal_seg_'+str(ids[7*k+1])+'.npy')
      v2 = (v2[:,0] + v2[:,1])/2
      a1 = np.load(datapath+'/other_seg_'+str(ids[7*k+2])+'.npy')
      a1 = (a1[:,0] + a1[:,1])/2
      a2 = np.load(datapath+'/other_seg_'+str(ids[7*k+3])+'.npy')
      a2 = (a2[:,0] + a2[:,1])/2
      d1 = np.load(datapath+'/drums_seg_'+str(ids[7*k+4])+'.npy')
      d1 = (d1[:,0] + d1[:,1])/2
      d2 = np.load(datapath+'/drums_seg_'+str(ids[7*k+5])+'.npy')
      d2 = (d2[:,0] + d2[:,1])/2
      b1 = np.load(datapath+'/bass_seg_'+str(ids[7*k+6])+'.npy')
      b1 = (b1[:,0] + b1[:,1])/2

      offset = int(np.random.uniform(0,96000-input_len))
      mix[k,:] = v1[offset:offset+input_len]+v2[offset:offset+input_len]+a1[offset:offset+input_len]+a2[offset:offset+input_len]+d1[offset:offset+input_len]+d2[offset:offset+input_len]+b1[offset:offset+input_len]
      seps[k,:,0] = v1[offset:offset+input_len]+v2[offset:offset+input_len]
      seps[k,:,1] = a1[offset:offset+input_len]+a2[offset:offset+input_len]
      seps[k,:,2] = d1[offset:offset+input_len]+d2[offset:offset+input_len]
      seps[k,:,3] = b1[offset:offset+input_len]
  
    else:
      v1 = np.load(datapath+'/'+source+'_seg_'+str(ids[4*k])+'.npy') #augmentation scheme: mix 2 vocals + 2 accompaniments, sep 2 vocals.
      v1 = (v1[:,0] + v1[:,1])/2
      v2 = np.load(datapath+'/'+source+'_seg_'+str(ids[4*k+1])+'.npy')
      v2 = (v2[:,0] + v2[:,1])/2
      a1 = np.load(datapath+'/full_seg_'+str(ids[4*k+2])+'.npy')-np.load(datapath+'/'+source+'_seg_'+str(ids[4*k+2])+'.npy')
      a1 = (a1[:,0] + a1[:,1])/2
      a2 = np.load(datapath+'/full_seg_'+str(ids[4*k+3])+'.npy')-np.load(datapath+'/'+source+'_seg_'+str(ids[4*k+3])+'.npy')
      a2 = (a2[:,0] + a2[:,1])/2
      offset = int(np.random.uniform(0,96000-input_len))
      mix[k,:] = v1[offset:offset+input_len]+a1[offset:offset+input_len]+v2[offset:offset+input_len]+a2[offset:offset+input_len]
      seps[k,:] = v1[offset:offset+input_len]+v2[offset:offset+input_len]

  evals = UNet.fit(x=mix,y=seps,epochs=1,batch_size=minibatch,verbose=0)
  gc.collect()
  running_loss += evals.history['loss'][0]

  if (i%100) == 99:
    print('Average training loss at step', i+1, running_loss/100)
    running_loss = 0

  # validation loader + loss
  
  if (i%1000) == 999:
  
    val_loss = 0  
    if source == 'multisource':
      ministep = 13
      mix = np.zeros((ministep,input_len))
      seps = np.zeros((ministep,input_len,4))
    else:
      ministep = 91
      mix = np.zeros((ministep,input_len))
      seps = np.zeros((ministep,input_len))
    
    for j in range(0,nValFiles,ministep):
      for k in range(j,j+ministep):
        if source != 'multisource':
          y = np.load(valid_datapath+'/full_seg_'+str(k)+'.npy')
          y = (y[:,0] + y[:,1])/2
          x = np.load(valid_datapath+'/'+source+'_seg_'+str(k)+'.npy')
          x = (x[:,0] + x[:,1])/2
          offset = int(np.random.uniform(0,96000-input_len))
          mix[k-j,:] = y[offset:offset+input_len]
          seps[k-j,:] = x[offset:offset+input_len]
        
        else:
          y = np.load(valid_datapath+'/full_seg_'+str(k)+'.npy')
          y = (y[:,0] + y[:,1])/2
          x = np.load(valid_datapath+'/vocal_seg_'+str(k)+'.npy')
          x = (x[:,0] + x[:,1])/2
          z = np.load(valid_datapath+'/other_seg_'+str(k)+'.npy')
          z = (z[:,0] + z[:,1])/2
          v = np.load(valid_datapath+'/drums_seg_'+str(k)+'.npy')
          v = (v[:,0] + v[:,1])/2
          w = np.load(valid_datapath+'/bass_seg_'+str(k)+'.npy')
          w = (w[:,0] + w[:,1])/2

          offset = int(np.random.uniform(0,96000-input_len))
          mix[k-j,:] = y[offset:offset+input_len]
          seps[k-j,:,0] = x[offset:offset+input_len]
          seps[k-j,:,1] = z[offset:offset+input_len]
          seps[k-j,:,2] = v[offset:offset+input_len]
          seps[k-j,:,3] = w[offset:offset+input_len]


      evals=UNet.evaluate(x=mix,y=seps,verbose=0)
      val_loss+=evals

    print('Running validation loss at step',i+1,val_loss/(nValFiles//ministep))
     
  if (i%15000 == 14999):
    checkpoint.save('./'+checkpoint_path+'/checkpoint')
    K.set_value(UNet.optimizer.lr, 0.9*K.get_value(UNet.optimizer.lr))
    print('Current optimizer lr value', K.get_value(UNet.optimizer.lr))  
