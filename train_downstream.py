# Script for jointly finetuning a pre-trained separation network, along with a convolutional classification head
# Usage: python3 train_downstream.py dataset path-to-dataset model-directory [-- unet --pretrain separation-model-directory --skips  num_of_skips --multisource]
# --unet: whether a U-Net is prepended to the convolutional frontend
# --pretrain separation-model-directory: whether the prepended U-Net has been pre-trained with a separation objective (weights saved at the separation-model-directory)
# --skips num_of_skips: number of skip connections between the U-net and the convolutional frontend (defaults to 5)
# --multisource: if provided, the pre-trained U-Net has been pre-trained with a multi-source separation objective
# eg: python3 train_downstream.py mtat /data/MTT/ myexperiment/ --unet --pretrain models/separators/vocal/ (vocal pre-training)
# python3 train_downstream.py fma /data/FMA/ myexperiment2/ --unet --pretrain models/separators/multisource/ --multisource (multisource pre-training)
# python3 train_downstream.py mtat /data/MTT/ myexperiment3/ --unet (complete architecture, no pre-training)
# python3 train_downstream.py mtat /data/MTT/ myexperiment4/ (just the convolutional frontend baseline)

# imports

import gc
import os
import sys
import argparse
import data
import models
import separators
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Multiply,Dropout,Flatten,Reshape,Permute,BatchNormalization,Dense,Conv2D,MaxPooling2D,ReLU,LeakyReLU,Concatenate,Conv2DTranspose, LayerNormalization,Activation

def expand_labels(Y,n):
  Y = np.tile(Y,(n,1,1))
  Y = np.transpose(Y,(1,0,2))
  Yx = np.reshape(Y,(Y.shape[1]*Y.shape[0],Y.shape[2]))
  return Yx

def prepare_standard_example(x, is_training, is_infering):
    """Creates an example for supervised training."""
    if is_training:
        s = int(np.random.uniform(0,x.shape[0]-384))
        x = x[s:s+384,:]
    elif is_infering:
        x = tf.signal.frame(x,frame_length=384,frame_step=192,axis=0,pad_end=False)
    else:
        x = tf.signal.frame(x,frame_length=384,frame_step=384,axis=0,pad_end=False)   
    return np.expand_dims(x,axis=-1)


parser = argparse.ArgumentParser()
parser.add_argument('dataset',type=str) #name of the dataset to train (candidates: mtat, fma)
parser.add_argument('datapath',type=str) #directory to load the data from
parser.add_argument('model_dir',type=str) #directory to store the model to
parser.add_argument('--unet',required=False,action='store_true')
parser.add_argument('--pretrain',required=False,type=str)
parser.add_argument('--skips',required=False,type=int, default=5)
parser.add_argument('--multisource',default=False,action='store_true')

args = parser.parse_args()

dataset = args.dataset
datapath = args.datapath
model_dir = args.model_dir
pretrain = args.pretrain
multisource = args.multisource
skips = args.skips
unet = args.unet

if not unet:
  pretrain = False
if multisource:
  n_src = 4
else:
  n_src = 1

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

if unet:
  TUneTF = models.get_tune_network(dataset,n_src,skips)
else:
  TUneTF = models.get_tail_network(dataset)

# Load pretrained weights from separator

if pretrain:
  if multisource:
    UNet = separators.get_multisource_separator()
  else:
    UNet = separators.get_unisource_separator()

  UNet.compile()
  UNet.load_weights(tf.train.latest_checkpoint(pretrain)).expect_partial()

  for i in range(8,102): #don't ask that's the correspondence :P
    TUneTF.layers[i-4].set_weights(UNet.layers[i].get_weights())  

  TUneTF.layers[99].set_weights(UNet.layers[104].get_weights())
  if multisource:
    TUneTF.layers[100].set_weights(UNet.layers[107].get_weights())
  else:
    TUneTF.layers[100].set_weights(UNet.layers[106].get_weights())

print(TUneTF.summary())

# Data Loader 

if dataset == 'mtat':
  train_keys, train_labels = data.get_mtat_subset(datapath,'train')
  valid_keys, valid_labels = data.get_mtat_subset(datapath,'valid')
elif dataset == 'fma':
  train_keys, train_labels = data.get_fma_subset(datapath,'train')
  valid_keys, valid_labels = data.get_fma_subset(datapath,'valid')
else:
  print('New dataset! See the code on dataset.py to write a compatible loader!')

if dataset == 'mtat':
  lossfunc = tf.keras.losses.BinaryCrossentropy()
  metrics_ = [tf.keras.metrics.BinaryAccuracy()]
  valsplit = 139 #batch size for the validation loader
elif dataset == 'fma':
  lossfunc = tf.keras.losses.CategoricalCrossentropy()
  metrics_ = [tf.keras.metrics.CategoricalAccuracy()]
  valsplit = 71

train_numel = len(train_keys)
valid_numel = len(valid_keys)

print(len(train_keys), len(valid_keys))

TUneTF.compile(
         optimizer=tf.keras.optimizers.Adam(0.0001),
          loss=lossfunc,
          metrics=metrics_)

checkpoint = tf.train.Checkpoint(TUneTF)

gc.collect()
phase = 0
best_val_acc = 0
best_val_loss = 999
patience = 0
epochs = 200
for k in range(0,epochs):
  running_loss = 0
  running_acc = 0

  batchcomp = np.random.permutation(train_numel)
  batch_cnt = 0
  batchSize = 1024
  ct = 0

  for i in range(0,batchSize*(train_numel-batchSize)//batchSize,batchSize):

    batch_comp = batchcomp[i:i+batchSize]
    specs = np.empty((batchSize,384,256,1))
    for n in range(0,batchSize):
      filename = datapath+'/stft_npys/train/'+train_keys[batch_comp[n]]
      y_raw = np.load(filename) 
      specs[n,:,:,:] = prepare_standard_example(y_raw,True,False)
    
    labels = train_labels[batch_comp[:batchSize],:]
    evals = TUneTF.fit(x=specs,y=labels,batch_size=16,epochs=k+1,initial_epoch=k,verbose=0)

    running_loss += evals.history["loss"][0]
    running_acc += evals.history[list(evals.history.keys())[1]][0]
    ct += 1

  running_loss = running_loss/ct
  running_acc = running_acc/ct
  print("Epoch",k+1,": Training Loss:", "{:.5f}".format(running_loss),"Training Accuracy:","{:.5f}".format(running_acc))

  batch_comp = np.random.permutation(valid_numel)
  val_loss = 0
  val_acc = 0
  batch_comp = np.random.permutation(valid_numel)
  for q in range(0,valid_numel,valsplit):
    specs = np.empty((valsplit*7,384,256,1))
    cct = 0
    for n in range(q,q+valsplit):
      filename = datapath+'/stft_npys/valid/'+valid_keys[n]
      y_raw = np.load(filename)
      specs[cct:cct+7,:,:,:] = prepare_standard_example(y_raw,False,False)
      cct += 7

    labels_ = valid_labels[q:q+valsplit,:]
    labels_ = expand_labels(np.expand_dims(labels_,0),7)
    evals = TUneTF.evaluate(x=specs,y=labels_,batch_size=16,verbose=0)
    val_loss += evals[0]
    val_acc += evals[1]
  print("Validation Loss:","{:.5f}".format(val_loss/(valid_numel/valsplit)),"Validation Accuracy","{:.5f}".format(val_acc/(valid_numel/valsplit)))
  gc.collect()
  
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_val_acc = val_acc
    patience = 0
    save_path = checkpoint.save('./'+model_dir+'/checkpoint')
  else:
    patience += 1
    if patience > 10:
      TUneTF.load_weights(tf.train.latest_checkpoint(model_dir)).expect_partial()
      patience = 0
      phase += 1
      if phase == 1:
        print('Switching to SGD training')
        TUneTF.compile(
         optimizer=tf.keras.optimizers.SGD(0.001),
          loss=lossfunc,
          metrics=metrics_)
      elif phase == 2:
        print('LR plateau #1 for SGD training')
        TUneTF.compile(
         optimizer=tf.keras.optimizers.SGD(0.0001),
          loss=lossfunc,
          metrics=metrics_)
      elif phase == 3:
        print('LR plateau #2 for SGD training')
        TUneTF.compile(
         optimizer=tf.keras.optimizers.SGD(0.00001),
          loss=lossfunc,
          metrics=metrics_)
      else:
        print('Epochs required for convergence of downstream model:',k-10)
        print('Performance of the downstream model in the training set', "{:.5f}".format(running_acc))
        print('Performance of the downstream model in the validation set', "{:.5f}".format(best_val_acc))
        break

