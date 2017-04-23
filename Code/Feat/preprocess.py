
"""
__file__

    preprocess.py

__description__

    This file preprocesses data.

__author__

    Chenglong Chen

"""

import re
import sys
import csv
import cPickle
import numpy as np
import pandas as pd

replace_dict = {
    "nutri system": "nutrisystem",
    "soda stream": "sodastream",
    "playstation's": "ps",
    "playstations": "ps",
    "playstation": "ps",
    "(ps 2)": "ps2",
    "(ps 3)": "ps3",
    "(ps 4)": "ps4",
    "ps 2": "ps2",
    "ps 3": "ps3",
    "ps 4": "ps4",
    "coffeemaker": "coffee maker",
    "k-cups": "k cup",
    "k-cup": "k cup",
    "4-ounce": "4 ounce",
    "8-ounce": "8 ounce",
    "12-ounce": "12 ounce",
    "ounce": "oz",
    "button-down": "button down",
    "doctor who": "dr who",
    "2-drawer": "2 drawer",
    "3-drawer": "3 drawer",
    "in-drawer": "in drawer",
    "hardisk": "hard drive",
    "hard disk": "hard drive",
    "harley-davidson": "harley davidson",
    "harleydavidson": "harley davidson",
    "e-reader": "ereader",
    "levi strauss": "levi",
    "levis": "levi",
    "mac book": "macbook",
    "micro-usb": "micro usb",
    "screen protector for samsung": "screen protector samsung",
    "video games": "videogames",
    "game pad": "gamepad",
    "western digital": "wd",
    "eau de toilette": "perfume",
}

class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return [self.word_map.get(w, w) for w in word]

class CsvWordReplacer(WordReplacer):
    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            if word.startswith("#"):
                continue
            word_map[word] = syn
        super(CsvWordReplacer, self).__init__(word_map)

data_dir = '../../Data/'
output_dir = '../../Output/'
replacer = CsvWordReplacer(data_dir+'synonyms.csv')
def clean_text(line):
    names = ["question1", "question2"]
    for name in names:
        l = line[name]
        l = l.lower()
        ## replace gb
        for vol in [16, 32, 64, 128, 500]:
            l = re.sub("%d gb"%vol, "%dgb"%vol, l)
            l = re.sub("%d g"%vol, "%dgb"%vol, l)
            l = re.sub("%dg "%vol, "%dgb "%vol, l)
        ## replace tb
        for vol in [2]:
            l = re.sub("%d tb"%vol, "%dtb"%vol, l)

        ## replace other words
        for k,v in replace_dict.items():
            l = re.sub(k, v, l)
        l = l.split(" ")

        ## replace synonyms
        l = replacer.replace(l)
        l = " ".join(l)
        line[name] = l
    return line

###############
## Load Data ##
###############
print("Load data...")
dfTrain = pd.read_csv(data_dir+'train.csv').fillna("")
dfTest = pd.read_csv(data_dir+'test.csv').fillna("")
print("Done.")

######################
## Pre-process Data ##
######################
print("Pre-process data...")

## clean text
clean = lambda line: clean_text(line)
dfTrain = dfTrain.apply(clean, axis=1)
dfTest = dfTest.apply(clean, axis=1)
print("Done.")


###############
## Save Data ##
###############
print("Save data...")
dfTrain.to_csv(output_dir+'train.processed.csv', columns=list(('id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate')), index=False)
dfTest.to_csv(output_dir+'test.processed.csv', index=False, columns=list(('test_id', 'question1', 'question2')))
print("Done.")
