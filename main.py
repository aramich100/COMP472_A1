# COMP 472
# WINTER 2021

# Michael Arabian - 40095854
# Thomas Le       - 40096120
# Andre Saad      - 40076579

from sklearn.datasets import load_files
from sklearn import *
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split
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

# Naives Bayes
gnb = MultinomialNB()
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',ngram_range = (-1,1), stop_words='english')
train_docs_Vec = cv.fit_transform(train_docs)
gnb.fit(train_docs_Vec, train_labels)


eval_docs_Vec = cv.transform(eval_docs)
predicted = gnb.predict(eval_docs_Vec)
accuracy = metrics.accuracy_score(predicted,eval_labels)
print("NB: ",accuracy*100)

f = open("NaiveBayes_all_sentiment_shuffled.txt", "w")
outputFile1 = "Accuracy : " + str(accuracy*100)
f.write(outputFile1)
f.close()



# Base DT
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion ='entropy')
clf.fit(train_docs_Vec, train_labels)
y_pred = clf.predict(eval_docs_Vec)
acc= metrics.accuracy_score(y_pred,eval_labels)
print("DT: ", acc*100)

f2 = open("Base_DT_all_sentiment_shuffled.txt", "w")
outputFile2 = "Accuracy : " + str(acc*100)
f2.write(outputFile2)
f2.close()

# Better DT
clfb = tree.DecisionTreeClassifier(criterion ='gini', splitter = 'random')
clfb.fit(train_docs_Vec, train_labels)
y_predb = clfb.predict(eval_docs_Vec)
accb= metrics.accuracy_score(y_predb,eval_labels)
print("Better DT : ",accb*100)

f3 = open("Best_DT_all_sentiment_shuffled.txt", "w")
outputFile3= "Accuracy : " + str(accb*100)
f3.write(outputFile3)
f3.close()