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
  s = str(s).lower()
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
  M = np.array(M)
  v = M.sum(axis=0)
  return v / np.sqrt((v ** 2).sum())

def cosd(model, v1, v2):
  v1 = np.nan_to_num(v1)
  v2 = np.nan_to_num(v2)
  return cosine(sent2vec(s1), sent2vec(s2))

def wmd(model, s1, s2):
  s1 = proc(s1)
  s2 = proc(s2)
  return model.wmdistance(s1, s2)

def get_feat(data, model, func, cols, featname):
  print "Getting feature %s"%featname
  cols.append(featname)
  labels=['question1', 'question2']
  if featname=='cosd':
      labels=['vector1', 'vector2']
  data[featname] = data.apply(lambda x: func(model, x[labels[0]], x[labels[1]]), axis=1)

if __name__ == '__main__':
  columns = [list(('id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate')),
          list(('test_id', 'question1', 'question2'))]
  data_dir = '../../Output/'
  fdata = ['train.processed.csv', 'test.processed.csv']
  fword2vec='../../Data/GoogleNews-vectors-negative300.bin'

  print "Getting stopwords"
  stop_words = stopwords.words('english')

  print "Loading google word2vec model"
  model=KeyedVectors.load_word2vec_format(fword2vec, binary=True)
'''
  for fn, cols in zip(fdata, columns):
    fn=data_dir+fn
    print "Loading data %s"%fn
    data = pd.read_csv(fn)
    # wmd
    get_feat(data, model, wmd, cols, 'wmd')
    # save tmp file
    data.to_csv("%s.tmp"%fn, columns=cols, index=False)
'''
  print "Getting norm word2vec model"
  norm_model = model
  norm_model.init_sims(replace=True)
  for fn, cols in zip(fdata, columns):
    fn=data_dir+fn
    print "Loading data %s.tmp"%fn
    data = pd.read_csv("%s.tmp"%fn)
    os.remove("%s.tmp"%fn)
    # norm_wmd
    get_feat(data, norm_model, wmd, cols, 'norm_wmd')
    # sentence vector
    print "Getting sentence vector"
    cols += ['vector1', 'vector2']
    data['vector1'] = data.apply(lambda x: sent2vec(x['question1']))
    data['vector2'] = data.apply(lambda x: sent2vec(x['question2']))
    # sentence vector cosine distance
    get_feat(data, None, cosd, cols, 'cosd')
    # save data with feata
    foutput = os.path.join(output_dir, "%s.word2vec", fn)
    print "Saving data with features in %s"%foutput
    data.to_csv(foutput, columns=cols, index=False)
