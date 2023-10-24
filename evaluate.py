# Script for the evaluation of already trained models
# Usage: python3 evaluate.py dataset datapath model_dir --unet --skips num_skips --multisource
# Arguments operate similar to the train_downstream.py script
# eg: python3 evaluate.py mtat /data/MTT/ myexperiment_n/ --unet (single source case)

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
parser.add_argument('dataset',type=str)
parser.add_argument('datapath',type=str)
parser.add_argument('model_dir',type=str) #directory to load the model to
parser.add_argument('--unet',required=False,action='store_true')
parser.add_argument('--skips',required=False,type=int, default=5)
parser.add_argument('--multisource',default=False,action='store_true')

args = parser.parse_args()

dataset = args.dataset
datapath = args.datapath
model_dir = args.model_dir
multisource = args.multisource
skips = args.skips
unet = args.unet

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

print(TUneTF.summary())


if dataset == 'mtat':
  test_keys, test_labels = data.get_mtat_subset(datapath,'test')
elif dataset == 'fma':
  test_keys, test_labels = data.get_fma_subset(datapath,'test')
else:
  print('New dataset! See the code on dataset.py to write a compatible loader!')
test_numel = len(test_labels)

TUneTF.load_weights(tf.train.latest_checkpoint(model_dir[:-1])).expect_partial()

if dataset == 'mtat':
  lossfunc = tf.keras.losses.BinaryCrossentropy()
  metrics_ = [tf.keras.metrics.BinaryAccuracy()]
elif dataset == 'fma':
  lossfunc = tf.keras.losses.CategoricalCrossentropy()
  metrics_ = [tf.keras.metrics.CategoricalAccuracy()]


# dummy
TUneTF.compile(
         optimizer=tf.keras.optimizers.Adam(0.0001),
          loss=lossfunc,
          metrics=metrics_)

evals_ = np.zeros((test_numel,test_labels.shape[1]))
ct = 0

for kk in range(0,test_numel):
  
  filename = datapath+'/stft_npys/test/'+test_keys[kk]
  y_raw = np.load(filename)
  segnum = (len(y_raw)-192)//192
  specs = prepare_standard_example(y_raw,False,True)
  evals = TUneTF.predict(specs,batch_size=32)
  evals_[kk,:] = np.sum(evals,axis=0)/segnum

if dataset == 'mtat':
  print('roc-auc score',roc_auc_score(test_labels,evals_,average=None))
  print('pr-auc score',average_precision_score(test_labels,evals_,average=None))
  print('macro roc-auc score',roc_auc_score(test_labels,evals_))
  print('macro pr-auc score',average_precision_score(test_labels,evals_))
elif dataset == 'fma':
  evals_oh = (evals_ == np.expand_dims(np.max(evals_,axis=1),1)).astype(int)
  print(sklearn.metrics.confusion_matrix(np.argmax(evals_oh,axis=1), np.argmax(test_labels,axis=1)))
  print('WA', np.sum(evals_oh*test_labels)/len(test_labels))

                                        

