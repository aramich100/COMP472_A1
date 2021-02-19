# COMP 472
# WINTER 2021

# Michael Arabian - 40095854
# Thomas Le       - 40096120
# Andre Saad      - 40076579

from sklearn.feature_extraction.text import *
from sklearn.datasets import load_files
from sklearn import *
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
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',ngram_range = (1,1), stop_words='english')
train_docs_Vec = cv.fit_transform(train_docs)
gnb.fit(train_docs_Vec, train_labels)

eval_docs_Vec = cv.transform(eval_docs)
predictedGnb = gnb.predict(eval_docs_Vec)

# part 3a)
row = len(train_docs) + 1
f = open("NaiveBayes-all_sentiment_shuffled.txt", "w")
for result in predictedGnb:
    index = 1
    if result == "neg":
        index = 0
    f.write(str(row) + ", " + str(index) + "\n")
    row += 1

# part 3c) d)
precisionGnb = metrics.precision_score(predictedGnb, eval_labels, average = None)
recallGnb = metrics.recall_score(predictedGnb, eval_labels, average = None)
f1ScoreGnb = metrics.f1_score(predictedGnb, eval_labels, average = None)
accuracyGnb = metrics.accuracy_score(predictedGnb, eval_labels)
print('Naives Bayes Precision: ' + str(precisionGnb))
print('Naives Bayes Recall: ' + str(recallGnb))
print('Naives Bayes f1_score: ' + str(f1ScoreGnb))
print('Naives Bayes Accuracy: ' + str(accuracyGnb*100))
f.write('Naives Bayes Precision: ' + str(precisionGnb) +
        '\nNaives Bayes Recall: ' + str(recallGnb) +
        '\nNaives Bayes f1_score: ' + str(f1ScoreGnb) +
        '\nNaives Bayes Accuracy: ' + str(accuracyGnb*100))

cmGnb = numpy.array2string(metrics.confusion_matrix(eval_labels, predictedGnb))
print(cmGnb)
f.write('\n Confusion Matrix: \n' + cmGnb )

f.close()

# Decision tree -------------------------- #

decisionTree = tree.DecisionTreeClassifier(criterion= 'entropy')
decisionTree.fit(train_docs_Vec, train_labels)
predictedDt = decisionTree.predict(eval_docs_Vec)

# part 3a)
row = len(train_docs) + 1
f = open("DecisionTree-all_sentiment_shuffled.txt", "w")
for result in predictedDt:
    index = 1
    if result == "neg":
        index = 0
    f.write(str(row) + ", " + str(index) + "\n")
    row += 1

# part 3c) d)
precisionDt = metrics.precision_score(predictedGnb, eval_labels, average = None)
recallDt = metrics.recall_score(predictedDt, eval_labels, average = None)
f1ScoreDt = metrics.f1_score(predictedDt, eval_labels, average = None)
accuracyDt = metrics.accuracy_score(predictedDt, eval_labels)
print('Decision Tree Precision: ' + str(precisionDt))
print('Decision Tree Recall: ' + str(recallDt))
print('Decision Tree f1_score: ' + str(f1ScoreDt))
print('Decision Tree Accuracy: ' + str(accuracyDt*100))

f.write('Decision Tree Precision: ' + str(precisionDt) +
        '\nDecision Tree Recall: ' + str(recallDt) +
        '\nDecision Tree f1_score: ' + str(f1ScoreDt) +
        '\nDecision Tree Accuracy: ' + str(accuracyDt*100))

cmDt = numpy.array2string(metrics.confusion_matrix(eval_labels, predictedDt))
print(cmDt)
f.write('\n Confusion Matrix: \n' + cmDt )

f.close()

# ---------------- Task 4 --------------------- #

index = 0
listOfString = []
realLabel = []
predictedLabel = []

while index < len(predictedGnb):
    if predictedGnb[index] != eval_labels[index]:
        listOfString.append(eval_docs[index])
        realLabel.append(eval_labels[index])
        predictedLabel.append(predictedGnb[index])
        index += 1
        if len(listOfString) > 10:
            break
    else:
        index += 1

i = 0
f = open("misclassified.txt", 'w')
while i < len(listOfString):
    print(listOfString[i])
    print("Real Value: " + realLabel[i])
    print("Predicted Value: " + predictedLabel[i] + "\n")
    f.write(listOfString[i] + "\n")
    f.write(("Real Value: " + realLabel[i]) + "\n")
    f.write(("Predicted Value: " + predictedLabel[i] + "\n\n"))
    i += 1

f.close()

