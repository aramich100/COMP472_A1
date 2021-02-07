from __future__ import division
from codecs import open
from collections import Counter
from sklearn.datasets import load_files
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np


# COMP 472
# WINTER 2021

# Michael Arabian - 40095854
# Thomas Le       - 40096120
# Andre Saad      - 40076579


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
#print(all_labels[1])


# ---------------- Task 1 --------------------- #
frequency = Counter()

for doc in all_docs:
    for word in doc:
        frequency.update(word)
print(frequency.values())
plt.bar(frequency.keys(), frequency.values())
plt.title("Distribution Plot")
plt.xlabel("Word/Letter")
plt.ylabel("Frequency")
plt.show()

# ---------------- Task 2 --------------------- #
letter = list(frequency.keys())
letter_array = np.array(letter)
le = preprocessing.LabelEncoder()
letter_encoder = le.fit(letter_array)
print (letter_encoder)
values = list(frequency.values())
values_array = np.array(values)

features = zip(letter_array, values_array)
print(features)

model = GaussianNB()
