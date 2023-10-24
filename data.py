# contains data-related (paths corresponding to each subset of the dataset, as well as the respective STFT magnitudes)
# for MTAT and FMA datasets

import os
import numpy as np
import pandas as pd

def get_valid_paths_mtat(path,split):
  
  subfolds = {'train':['0','1','2','3','4','5','6','7','8','9','a','b'], 'valid': ['c'], 'test':['d','e','f']}
  pathlist = []  

  for fold in subfolds[split]:
    wavs = os.listdir(path+'/Audio/'+fold)
    pathlist += [path+'/Audio/'+fold+'/'+x for x in wavs]

  return pathlist

def get_valid_paths_fma(path,split):

  data_ = pd.read_csv(path+'/fma_metadata/tracks.csv')
  data_medium = data_.loc[data_.index[data_['set.1'] == 'medium']]
  subsets = {'train': 'training', 'valid': 'validation', 'test':'test'}

  data_sub = data_medium.loc[data_medium.index[data_medium['set'] == subsets[split]]]
  idlist = data_sub["Unnamed: 0"].tolist()
  pathlist = []
  
  for k in range(0,len(idlist)):
    filename = (str(idlist[k]).zfill(6))
    full_name = path+'/fma_medium/'+filename+'.mp3'
    if filename not in ['001486','005574','065753','080391','098558','098559','098560','098565','098566','098567','098568','098569','098571','099134','105247','108924','108925','126981','127336','133297','143992']:
      pathlist.append(full_name)
  
  return pathlist
  
def get_mtat_subset(path,split):

  datapath = path+'/stft_npys/'+split
  keys_full = np.load(os.getcwd()+'/data-split/'+split+'_keys.npy')
  keys_filt = [x[2:] for x in keys_full]

  dict_ = os.listdir(datapath)
  numel = len(dict_)
  keys = []
  full_labels = np.load(os.getcwd()+'/data-split/y_'+split+'_pub.npy')
  labels = np.zeros((numel,50))
  ct = 0

  for key in dict_:
    key_filt=key[:-9]
    try:
        idx = keys_filt.index(key_filt)
    except:
        continue

    labels[ct,:] = full_labels[idx,:]
    keys.append(key)
    ct+=1

  numel = ct
  labels = labels[:numel,:]
  keys = keys[:numel]
  return keys,labels

def get_fma_subset(path,split):

  set_ = {'train': 'training', 'valid':'validation', 'test':'test'}
  datapath = path+'/stft_npys/'+split
  genre_dict = ['Electronic', 'Instrumental', 'Hip-Hop', 'Country', 'Spoken', 'Old-Time / Historic', 'Classical', 'Blues', 'International', 'Pop', 'Folk', 'Jazz', 'Experimental', 'Easy Listening', 'Rock', 'Soul-RnB']
  csv_path = path+'/fma_metadata/'
  data_ = pd.read_csv(csv_path+'/tracks.csv')
  data_medium = data_.loc[data_.index[data_['set.1'] == 'medium']]

  dirs = os.listdir(datapath)
  numel = len(dirs)-1
  keys = []
  cnt = 0
  data__ = data_medium.loc[data_medium.index[data_medium['set'] == set_[split]]]
  ids = data__["Unnamed: 0"]
  genre_labels = data__['track.7'].tolist()
  idlist = ids.tolist()
  idlist_str = [str(x).zfill(6) for x in idlist]
  genres = np.zeros((numel,16))

  for dir_ in dirs:
    try:
      idx = idlist_str.index(dir_.split('_')[0])
    except:
      continue
    gidx = genre_dict.index(genre_labels[idx])
    keys.append(dir_[:-4])
    genres[cnt,gidx] = 1
    cnt += 1

  keys_ = [key+'.npy' for key in keys] 
  labels = genres[:cnt,:]
  return keys_,labels

