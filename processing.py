from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def PreProcessing(args):
  files=[]
  ann_files1 = pd.read_csv('double.csv',encoding='cp949')
  for str in ann_files1["fullText"]:
    str='/'.join(str.split('\n'))
    files.append(str)
  label=list(ann_files1["conItemId"])
  print(label)

  skf = StratifiedKFold(n_splits = 5, random_state=42, shuffle=True)

  folds = []
  for idx, (train_idx, val_idx) in enumerate(skf.split(files, label)):
      folds.append((train_idx, val_idx))

  train_idx, val_idx = folds[0]
  train_ann_files = pd.DataFrame(np.array([files, label])[:, train_idx].transpose())
  val_ann_files = pd.DataFrame(np.array([files, label])[:, val_idx].transpose())
  print('train data length: ', len(train_ann_files))
  print('val data length: ', len(val_ann_files))
  train_ann_files.columns = ['text', 'label']
  val_ann_files.columns = ['text', 'label']

  ann_files = pd.DataFrame(np.array([files, label]).transpose())
  ann_files.columns = ['text', 'label']
  #ann_files.to_csv("double_data.csv")
  return train_ann_files, val_ann_files
