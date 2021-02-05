# COMP 472
# WINTER 2021

# Michael Arabian - 40095854
# Thomas Le       - 40096120
# Andre Saad      - 40076579

from sklearn.datasets import load_files
import pandas as pd

# ---------------- Task 0 --------------------- #

def read_documents (docName):
    docs = []
    label = []
    with open(docName, encoding="utf8") as f:
        for line in f:
            split = line.split()
            label.append(split[1])
            docs.append(' '.join(split[3:]))
    return docs, label

all_docs, all_labels = read_documents ('all_sentiment_shuffled.txt')
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

print(all_labels[1])



# ---------------- Task 1 --------------------- #

