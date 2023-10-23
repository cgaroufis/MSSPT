# creates overlapping 6-sec segments from musdb18
# for splitting/training an MSS network
# usage: python3 preprocess_mss.py path_to_musdb18 subset (train,valid,test)
# eg: python3 preprocess_mss.py /data/musdb18 train

import stempeg
import librosa
import numpy as np
import os
import sys

path = sys.argv[1]+'Audio'
subset = sys.argv[2]
dest_path = sys.argv[1]+'/npys/'+subset
os.makedirs(dest_path,exist_ok=True)
src_path = path+'/'+subset

ct = 0
for stem in os.listdir(src_path):
 
  print(stem)

  y,fs = stempeg.read_stems(src_path+'/'+stem,stem_id=0,sample_rate=44100) #full stem
  yv,fs = stempeg.read_stems(src_path+'/'+stem,stem_id=4,sample_rate=44100)
  yd,fs = stempeg.read_stems(src_path+'/'+stem,stem_id=1,sample_rate=44100)
  yb,fs = stempeg.read_stems(src_path+'/'+stem,stem_id=2,sample_rate=44100)
  yo,fs = stempeg.read_stems(src_path+'/'+stem,stem_id=3,sample_rate=44100)
  for i in range(0,len(yd)-(6*44100),3*44100):
    y_seg = y[i:i+6*44100,:]
    y_seg = librosa.resample(y_seg.T,orig_sr=44100,target_sr=16000)
    filename = dest_path+'/full_seg_'+str(ct)+'.npy'
    np.save(filename,y_seg.T)
    yv_seg = yv[i:i+6*44100,:]
    yv_seg = librosa.resample(yv_seg.T,orig_sr=44100,target_sr=16000)
    filename = dest_path+'/vocal_seg_'+str(ct)+'.npy'
    np.save(filename,yv_seg.T)    
    yd_seg = yd[i:i+6*44100,:]
    yd_seg = librosa.resample(yd_seg.T,orig_sr=44100,target_sr=16000)
    filename = dest_path+'/drums_seg_'+str(ct)+'.npy'
    np.save(filename,yd_seg.T)
    yb_seg = yb[i:i+6*44100,:]
    yb_seg = librosa.resample(yb_seg.T,orig_sr=44100,target_sr=16000)
    filename = dest_path+'/bass_seg_'+str(ct)+'.npy'
    np.save(filename,yb_seg.T)
    yo_seg = yo[i:i+6*44100,:]
    yo_seg = librosa.resample(yo_seg.T,orig_sr=44100,target_sr=16000)
    filename = dest_path+'/other_seg_'+str(ct)+'.npy'
    np.save(filename,yo_seg.T)
    ct += 1

  print('segments isolated',ct)
