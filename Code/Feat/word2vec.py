import os
import cPickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine

def proc(s):
  s = str(s)
  try:
    us = s.decode('utf-8')
  except:
    us = s
  words = word_tokenize(us)
  ret = [w for w in words if w not in stop_words]
  return ret

def sent2vec(s):
  words = proc(s)
  words = [w for w in words if w.isalpha()]
  M = []
  for w in words:
    try:
      M.append(model[w])
    except:
      continue
  if len(M) == 0:
    return [0.0 for _ in range(300)]
  M = np.array(M)
  v = M.sum(axis=0)
  try:
    ret = v / np.sqrt((v ** 2).sum())
  except:
    ret = v
  return ret

def cosd(model, v1, v2):
  v1 = np.nan_to_num(v1)
  v2 = np.nan_to_num(v2)
  return cosine(v1, v2)

def wmd(model, s1, s2):
  s1 = proc(s1)
  s2 = proc(s2)
  return model.wmdistance(s1, s2)

def load_csv(fn):
  fn+='.csv'
  print 'Loading csv file %s'%fn
  return pd.read_csv(fn)

def save_csv(data, fn, labels):
  fn += '.csv'
  print 'Saving data in csv file %s'%fn
  data.to_csv(fn, columns=labels, index=False)

def get_feat(data, model, func, cols, featname, labels):
  print "Getting feature %s"%featname
  cols.append(featname)
  data[featname] = data.apply(lambda x: func(model, x[labels[0]], x[labels[1]]), axis=1)

if __name__ == '__main__':
  columns = [list(('id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate')),
          list(('test_id', 'question1', 'question2'))]
  drop_columns = [["question1", "question2", "is_duplicate", "qid1", "qid2"],
          ["question1", "question2"]]
  data_dir = '../../Output/'
  fdata = ['train.processed', 'test.processed']
  fword2vec='../../Data/GoogleNews-vectors-negative300.bin'

  print "Getting stopwords"
  stop_words = stopwords.words('english')

  print "Loading google word2vec model"
  model=KeyedVectors.load_word2vec_format(fword2vec, binary=True)

  for fn, cols in zip(fdata, columns):
    fn=os.path.join(data_dir, fn)
    data=load_csv(fn)
    # wmd
    get_feat(data, model, wmd, cols, 'wmd', ['question1', 'question2'])
    # save tmp file
    save_csv(data, "%s.tmp"%fn, cols)

  print "Getting norm word2vec model"
  norm_model = model
  norm_model.init_sims(replace=True)
  for fn, cols, drop_cols in zip(fdata, columns, drop_columns):
    fn=os.path.join(data_dir, fn)
    data=load_csv("%s.tmp"%fn)
    # norm_wmd
    get_feat(data, norm_model, wmd, cols, 'norm_wmd', ['question1', 'question2'])
    # sentence vector
    print "Getting sentence vector"
    vector1 = [sent2vec(q) for i, q in tqdm(enumerate(data.question1.values))]
    vector2 = [sent2vec(q) for i, q in tqdm(enumerate(data.question2.values))]
    data['vector1']=vector1
    data['vector2']=vector2
    # sentence vector cosine distance
    get_feat(data, None, cosd, cols, 'cosd', ['vector1', 'vector2'])
    # drop unnecessary columns
    data = data.drop(drop_cols, axis=1)
    for c in drop_cols:
      cols.remove(c)
    drop_cols += ['vector1', 'vector2']
    # save data with feata
    save_csv(data, "%s.wmd_cosd"%fn, cols)
    # save sentence vector
    print 'Save sentence vector file in %s.snt2vec.pkl'%fn
    cPickle.dump([vector1, vector2], open("%s.snt2vec.pkl"%fn, 'wb'), 2)
    # remove tmp file
    os.remove("%s.tmp.csv"%fn)
