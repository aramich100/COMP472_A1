# COMP 472
# WINTER 2021

# Michael Arabian - 40095854
# Thomas Le       - 40096120
# Andre Saad      - 40076579

from sklearn.datasets import load_files
from sklearn import *
from sklearn.naive_bayes import *
from collections import Counter
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy


# ---------------- Task 0 --------------------- #

def read_documents(docName):
    docs = []
    label = []
    with open(docName, encoding="utf8") as f:
        for line in f:
            split_line = re.findall(r'\w+', line)
            label.append(split_line[1])
            docs.append(" ".join(split_line[4:]))
    return docs, label


all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

# ---------------- Task 1 --------------------- #

frequency = Counter()

# Distribution for words
# Takes too long to load and plot
# for doc in all_docs:
#     frequency.update(doc)

# # Distribution for positive/negative
# for doc in all_labels:
#     frequency[doc] += 1
#
# plt.bar(frequency.keys(), frequency.values())
# plt.title("Distribution Plot")
# plt.xlabel("Label")
# plt.ylabel("Frequency")
# plt.show()

# ---------------- Task 2 --------------------- #
gnb = GaussianNB()
reshape_docs = numpy.array(train_docs).reshape(-1,1)
    #.reshape((len(train_docs)),1)
gnb.fit(reshape_docs, train_labels)

predicted = gnb.predict(eval_docs)
accuracy = metrics.accuracy_score(predicted,eval_labels)
print(accuracy*100)