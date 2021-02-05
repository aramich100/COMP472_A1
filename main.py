from __future__ import division
from codecs import open
from sklearn.datasets import load_files
import pandas as pd

# COMP 472
# WINTER 2021

# Michael Arabian - 40095854
# Thomas Le       - xxxxxxxx
# Andre Saad      - xxxxxxxx


# ---------------- Task 0 --------------------- #
def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]


# ---------------- Task 1 --------------------- #

