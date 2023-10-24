# Script for preprocessing the downstream datasets, by means of extracting the STFT magnitude corresponding to each song excerpt
# Usage: python3 preprocess.py dataset path-to-dataset subset
# dataset can be either mtat or fma, subset one of train, valid or test
# eg: python3 preprocess.py mtat /data/MTAT/ train

import os
import sys
import numpy as np
import pandas as pd
import data
import tensorflow as tf
import librosa

def expand_labels(Y,n):
  Y = np.tile(Y,(n,1,1))
  Y = np.transpose(Y,(1,0,2))
  Yx = np.reshape(Y,(Y.shape[1]*Y.shape[0],Y.shape[2]))
  return Yx

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()

dataset = sys.argv[1]
datapath = sys.argv[2]
subset = sys.argv[3]

target_dir = datapath+'/stft_npys/'
os.makedirs(target_dir+'/'+subset,exist_ok=True)
if dataset == 'mtat':
  valid_paths = data.get_valid_paths_mtat(datapath,subset)
  sr = 16000
elif dataset == 'fma':
  valid_paths = data.get_valid_paths_fma(datapath,subset)
  sr = 44100  

ct = 0
print(len(valid_paths))
for full_name in valid_paths:
  filename = full_name.split('/')[-1][:-4]
  y,fs=librosa.load(full_name,sr=sr)
  y = librosa.resample(y,orig_sr=fs,target_sr=16000)
  x = tf.signal.stft(y,512,160,window_fn=tf.signal.hann_window) #hann is the default but nevertheless.
  x = x[:,:-1]
  x = tf.math.abs(x) #magnitude to get processed
  dest_filename = target_dir+subset+'/'+filename+'_stft.npy'
  np.save(dest_filename,x)
  ct += 1
  print(ct,'stfts saved')
