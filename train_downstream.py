# Script for jointly finetuning a pre-trained separation network, along with a classification head
# Usage: python3 train_downstream.py dataset path-to-dataset model-directory [-- unet --pretrain separation-model-directory --skips  num_of_skips --multistage [pretrained-backbone-directory] --multisource] -- frontend [type of frontend]
# --unet: whether a U-Net is prepended to the convolutional frontend
# --pretrain separation-model-directory: whether the prepended U-Net has been pre-trained with a separation objective (weights saved at the separation-model-directory)
# --skips num_of_skips: number of skip connections between the U-Net and the convolutional frontend (defaults to 5)
# --multisource: if provided, the pre-trained U-Net has been pre-trained with a multi-source separation objective
# --multistage [initial-backbone-weights]: if provided, an adaptation module is inserted between the U-Net and the backbone. 
# --frontend: determines whether a convolutional ('cnn') or AST-based ('transformer') backbone network will be used.
# The multistage argument is necessary for using pre-initialized backbones; in this case, provide the pre-trained backbone weights as a second argument.
# Usage examples:
# python3 train_downstream.py mtat /data/MTT/ myexperiment1/ --frontend cnn(just the convolutional frontend baseline)
# python3 train_downstream.py mtat /data/MTT/ myexperiment2/ --frontend transformer(just the AST baseline)
# python3 train_downstream.py fma /data/FMA/ myexperiment3/ --unet --frontend cnn (the complete architecture, no pre-trained elements, CNN backend)
# python3 train_downstream.py mtat /data/MTT/ myexperiment4/ --unet --pretrain models/separators/vocal/ --multistage models/downstream_models/ShortChunkCNN-based/MTAT/tail/ --frontend cnn (vocal pre-training, randomly initialized CNN backend)
# python3 train_downstream.py mtat /data/MTT/ myexperiment5/ --unet --pretrain models/separators/vocal/ --multistage models/downstream_models/ShortChunkCNN-based/MTAT/tail/ --frontend cnn (vocal pre-training, pre-trained CNN backend)
# python3 train_downstream.py fma /data/FMA/ myexperiment6/ --unet --pretrain models/separators/multisource/ --multistage models/downstream_models/ShortChunkCNN-based/FMA/tail/ --frontend cnn --multisource (multisource pre-training, pre-trained CNN backend)
# python3 train_downstream.py mtat /data/MTT/ myexperiment7/ --unet --pretrain models/separators/other/ --multistage models/downstream_models/AST-based/MTAT/imagenet_weights --frontend transformer (accompaniment pre-training, ImageNet-pretrained AST backend)
# python3 train_downstream.py mtat /data/MTT/ myexperiment8/ --unet --pretrain models/separators/other/ --multistage models/downstream_models/AST-based/MTAT/tail/ --frontend transformer (accompaniment pre-training, pre-trained AST backend)


# imports

import gc
import os
import sys
import argparse
import data
import models
import weight_loaders
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

def prepare_standard_example(x, is_training, is_infering,inlen):
    """Creates an example for supervised training."""
    if is_training:
        s = int(np.random.uniform(0,x.shape[0]-inlen))
        x = x[s:s+inlen,:]
    elif is_infering:
        x = tf.signal.frame(x,frame_length=inlen,frame_step=inlen//2,axis=0,pad_end=False)
    else:
        x = tf.signal.frame(x,frame_length=inlen,frame_step=inlen,axis=0,pad_end=False)   
    return np.expand_dims(x,axis=-1)


parser = argparse.ArgumentParser()
parser.add_argument('dataset',type=str) #name of the dataset to train (candidates: mtat, fma, gtzan)
parser.add_argument('datapath',type=str) #directory to load the data from
parser.add_argument('model_dir',type=str) #directory to store the model to
parser.add_argument('--frontend',type=str)
parser.add_argument('--unet',required=False,action='store_true')
parser.add_argument('--pretrain',required=False,type=str)
parser.add_argument('--skips',required=False,type=int, default=5)
parser.add_argument('--multisource',default=False,action='store_true')
parser.add_argument('--multistage',type=str,required=False)

args = parser.parse_args()

dataset = args.dataset
datapath = args.datapath
model_dir = args.model_dir
pretrain = args.pretrain
multisource = args.multisource
multistage = args.multistage
skips = args.skips
unet = args.unet
frontend = args.frontend

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
  if frontend == 'transformer':
    TUneTF = models.get_transformer_tune(dataset,multistage,n_src,skips,False)
  elif frontend == 'cnn':
    TUneTF = models.get_tune_network(dataset,multistage,n_src,skips,False)
  else:
    print('Incompatible frontend option; use "transformer" or "cnn"')
else:
  if frontend == 'transformer':
    TUneTF = models.get_transformer_encoder(dataset,False) #tail network (dataset,False)
  elif frontend == 'cnn':
    TUneTF = models.get_tail_network(dataset,False)
  else:
    print('Incompatible frontend option; use "transformer" or "cnn"') 


if frontend == 'transformer':
  init_lr = 5e-05
  miniBatch = 4
  inlen = 768
  segnum = 3
elif frontend == 'cnn':
  init_lr = 1e-04
  miniBatch = 16
  inlen = 384
  segnum = 7
else:
  print('Incompatible frontend option; use "transformer" or "cnn"')

# Load pretrained weights from separator

if pretrain:
  if multisource:
    UNet = separators.get_multisource_separator()
  else:
    UNet = separators.get_unisource_separator()

  UNet.compile()
  print(UNet.summary())
  
  UNet.load_weights(tf.train.latest_checkpoint(pretrain)).expect_partial()

  print('this is fine :)')
  for i in range(8,102): #don't ask that's the correspondence :P
    TUneTF.layers[i-4].set_weights(UNet.layers[i].get_weights())  

  TUneTF.layers[99].set_weights(UNet.layers[104].get_weights())
  if multisource:
    TUneTF.layers[100].set_weights(UNet.layers[107].get_weights())
  else:
    TUneTF.layers[100].set_weights(UNet.layers[106].get_weights())

# Load pre-trained weights for the tail

if multistage:

  if frontend == 'transformer':
    TailNet = models.get_transformer_encoder(dataset,False)
    TailNet.compile()
    TailNet.load_weights(tf.train.latest_checkpoint(multistage)).expect_partial() 
    TUneTF.layers[len(TUneTF.layers)-145].set_weights([tf.tile(TailNet.layers[14].get_weights()[0],tf.constant([1,1,n_src+1,1])),TailNet.layers[14].get_weights()[1]])
    for i in range(16,34):
      TUneTF.layers[i+len(TUneTF.layers)-(129+5*skips+5)].set_weights(TailNet.layers[i].get_weights()) #163  
    for i in range(36,52):
      TUneTF.layers[i+len(TUneTF.layers)-(129+4*skips+4)].set_weights(TailNet.layers[i].get_weights()) #156
    for i in range(54,70):
      TUneTF.layers[i+len(TUneTF.layers)-(129+3*skips+3)].set_weights(TailNet.layers[i].get_weights())
    for i in range(72,88):
      TUneTF.layers[i+len(TUneTF.layers)-(129+2*skips+2)].set_weights(TailNet.layers[i].get_weights())
    for i in range(90,106):
      TUneTF.layers[i+len(TUneTF.layers)-(129+skips+1)].set_weights(TailNet.layers[i].get_weights()) #135
    for i in range(108,len(TailNet.layers)):
      TUneTF.layers[i+len(TUneTF.layers)-129].set_weights(TailNet.layers[i].get_weights())

    TUneTF = weight_loaders.load_adapter_weights(TUneTF)

  else:
    TailNet = models.get_tail_network(dataset,False)
    TailNet.compile()
    TailNet.load_weights(tf.train.latest_checkpoint(multistage)).expect_partial()
  
    print(len(TUneTF.layers))
    TUneTF.layers[len(TUneTF.layers)-70].set_weights([tf.tile(TailNet.layers[6].get_weights()[0],tf.constant([1,1,n_src+1,1])),TailNet.layers[6].get_weights()[1]])
    for i in range(7,12):
      TUneTF.layers[i+len(TUneTF.layers)-76].set_weights(TailNet.layers[i].get_weights()) #106
    for i in range(13,19):
      TUneTF.layers[i+len(TUneTF.layers)-73].set_weights(TailNet.layers[i].get_weights())
    for i in range(20,26):
      TUneTF.layers[i+len(TUneTF.layers)-70].set_weights(TailNet.layers[i].get_weights())
    for i in range(27,33):
      TUneTF.layers[i+len(TUneTF.layers)-67].set_weights(TailNet.layers[i].get_weights())
    for i in range(34,40):
      TUneTF.layers[i+len(TUneTF.layers)-64].set_weights(TailNet.layers[i].get_weights())
    for i in range(41,len(TailNet.layers)):
      TUneTF.layers[i+len(TUneTF.layers)-61].set_weights(TailNet.layers[i].get_weights())
  
# Data Loader 

if dataset == 'mtat':
  train_keys, train_labels = data.get_mtat_subset(datapath,'train')
  valid_keys, valid_labels = data.get_mtat_subset(datapath,'valid')
elif dataset == 'fma':
  train_keys, train_labels = data.get_fma_subset(datapath,'train')
  valid_keys, valid_labels = data.get_fma_subset(datapath,'valid')
elif dataset == 'gtzan':
  train_keys, train_labels = data.get_gtzan_subset(datapath,'train')
  valid_keys, valid_labels = data.get_gtzan_subset(datapath,'valid')
else:
  print('New dataset! See the code on dataset.py to write a compatible loader!')

if dataset == 'mtat':
  lossfunc = tf.keras.losses.BinaryCrossentropy()
  metrics_ = [tf.keras.metrics.BinaryAccuracy()]
  valsplit = 139 #batch size for the validation loader
elif dataset == 'fma' or dataset == 'gtzan':
  lossfunc = tf.keras.losses.CategoricalCrossentropy()
  metrics_ = [tf.keras.metrics.CategoricalAccuracy()]
  if dataset == 'fma':
    valsplit = 8 #check this value for gtzan
  else:
    valsplit = 98

train_numel = len(train_keys)
valid_numel = len(valid_keys)

TUneTF.compile(
         optimizer=tf.keras.optimizers.Adam(init_lr),
          loss=lossfunc,
          metrics=metrics_)

print(TUneTF.summary())
if frontend == 'transformer' and not multistage:
  TUneTF = weight_loaders.load_transformer_weights(TUneTF) #only for backend training

TUneTF.compile(
         optimizer=tf.keras.optimizers.Adam(init_lr),
          loss=lossfunc,
          metrics=metrics_)

checkpoint = tf.train.Checkpoint(TUneTF)

gc.collect()
phase = 0


if multistage:
 
  val_loss = 0
  val_acc = 0 
  for q in range(0,valid_numel,valsplit):
    specs = np.empty((valsplit*segnum,inlen,256,1))
    cct = 0
    for n in range(q,q+valsplit):
      filename = datapath+'/stft_npys/valid/'+valid_keys[n]
      y_raw = np.load(filename)
      specs[cct:cct+segnum,:,:,:] = prepare_standard_example(y_raw,False,False,inlen)
      cct += segnum

    labels_ = valid_labels[q:q+valsplit,:]
    labels_ = expand_labels(np.expand_dims(labels_,0),segnum) #(len//inlen)
    evals = TUneTF.evaluate(x=specs,y=labels_,batch_size=4*miniBatch,verbose=0)
    val_loss += evals[0]
    val_acc += evals[1]
  best_val_acc = val_acc
  best_val_loss = val_loss
  print('current best metrics (accuracy/loss)',best_val_acc/(valid_numel//valsplit),best_val_loss/(valid_numel//valsplit))

else:
  best_val_acc = 0
  best_val_loss = 999

epochs = 200
patience = 0
for k in range(0,epochs):
  running_loss = 0
  running_acc = 0
  batchcomp = np.random.permutation(train_numel)
  batch_cnt = 0
  batchSize = min(1024,train_numel)
  ct = 0

  for i in range(0,batchSize*(train_numel-batchSize)//batchSize,batchSize):
    batch_comp = batchcomp[i:i+batchSize]
    specs = np.empty((batchSize,inlen,256,1))
    for n in range(0,batchSize):
      filename = datapath+'/stft_npys/train/'+train_keys[batch_comp[n]]
      y_raw = np.load(filename) 
      specs[n,:,:,:] = prepare_standard_example(y_raw,True,False,inlen)
    
    labels = train_labels[batch_comp,:]
    evals = TUneTF.fit(x=specs,y=labels,batch_size=miniBatch,epochs=k+1,initial_epoch=k,verbose=0)
    running_loss += evals.history["loss"][0]
    running_acc += evals.history[list(evals.history.keys())[1]][0]
    ct += 1

  running_loss = running_loss/ct 
  running_acc = running_acc/ct
  print("Epoch",k+1,": Training Loss:", "{:.5f}".format(running_loss),"Training Accuracy:","{:.5f}".format(running_acc))
 
  val_loss = 0
  val_acc = 0
  
  for q in range(0,valid_numel,valsplit):
    specs = np.empty((valsplit*segnum,inlen,256,1))
    cct = 0
    for n in range(q,q+valsplit):
      filename = datapath+'/stft_npys/valid/'+valid_keys[n]
      y_raw = np.load(filename)
      specs[cct:cct+segnum,:,:,:] = prepare_standard_example(y_raw,False,False,inlen)
      cct += segnum

    labels_ = valid_labels[q:q+valsplit,:]
    labels_ = expand_labels(np.expand_dims(labels_,0),segnum) 
    evals = TUneTF.evaluate(x=specs,y=labels_,batch_size=4*miniBatch,verbose=0)
    val_loss += evals[0]
    val_acc += evals[1]

  print("Validation Loss:","{:.5f}".format(val_loss/(valid_numel/valsplit)),"Validation Accuracy","{:.5f}".format(val_acc/(valid_numel/valsplit)))
  gc.collect()
  
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_val_acc = val_acc
    patience = 0
    if k > 0: 
      os.remove(tf.train.latest_checkpoint(model_dir)+'.data-00000-of-00001')
      os.remove(tf.train.latest_checkpoint(model_dir)+'.index')
    save_path = checkpoint.save('./'+model_dir+'/checkpoint')
    old_filename = tf.train.latest_checkpoint(model_dir)
    print("Current checkpoint directory",old_filename)
  else:
    patience += 1
    
  if frontend == 'transformer':
      
    if (phase == 0 and k%5 == 4) or (phase > 0 and k%2 == 0):
      
      curr_lr = K.get_value(TUneTF.optimizer.lr)
      TUneTF.load_weights(tf.train.latest_checkpoint(model_dir)).expect_partial()
      if phase < 5:
          
        print('Reducing LR...')
        TUneTF.compile(
          optimizer=tf.keras.optimizers.Adam(curr_lr),
          loss=lossfunc,
          metrics=metrics_)
        K.set_value(TUneTF.optimizer.lr,0.5*K.get_value(TUneTF.optimizer.lr))
        phase += 1
        patience = 0

      elif phase == 5:
        print('Switching to SGD training')
        TUneTF.compile(
          optimizer=tf.keras.optimizers.SGD(0.001),
            loss=lossfunc,
            metrics=metrics_)
        patience = 0
        phase += 1
        
      else:
        print('Performance of the downstream model in the training set', "{:.5f}".format(running_acc))
        print('Performance of the downstream model in the validation set', "{:.5f}".format(best_val_acc/(valid_numel/valsplit)))
        break


  else:
      
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
      elif phase == 2 and dataset != 'gtzan':
        print('LR plateau #1 for SGD training')
        TUneTF.compile(
        optimizer=tf.keras.optimizers.SGD(0.0001),
          loss=lossfunc,
          metrics=metrics_)
      elif phase == 3 and dataset != 'gtzan':
        print('LR plateau #2 for SGD training')
        TUneTF.compile(
          optimizer=tf.keras.optimizers.SGD(0.00001),
            loss=lossfunc,
            metrics=metrics_)
      else:
        print('Performance of the downstream model in the training set', "{:.5f}".format(running_acc))
        print('Performance of the downstream model in the validation set', "{:.5f}".format(best_val_acc/(valid_numel/valsplit)))
        break