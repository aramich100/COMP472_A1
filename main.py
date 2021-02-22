# COMP 472
# WINTER 2021

# Michael Arabian   -   40095854
# Thomas Le         -   40096120
# Andre Saad        -   40076579


# Sklearn Imports
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


# -------------------------------------- Task 0 --------------------------------------- #
#
#   You first need to remove the document identifier, and also the topic label, which you don't need.
#   Then, split the data into a training and an evaluation part. For instance, we may use 80% for training and the
#   remainder for evaluation.
#


def read_documents(docName):
    """
    read_documents takes in a file name properly reads and sorts all data.

    :param docName: Name of inputed file
    
    """

    docs = []
    label = []
    with open(docName, encoding="utf8") as f:
        for line in f:
            split_line = re.findall(r'\w+', line)
            label.append(split_line[1])
            docs.append(" ".join(split_line[4:]))
    return docs, label


all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')


split_point = int(0.80 * len(all_docs))   # Uses 80% of data set for training.


train_docs = all_docs[:split_point]         # Training Documents        
train_labels = all_labels[:split_point]     # Training Labels

eval_docs = all_docs[split_point:]          # Testing Documents
eval_labels = all_labels[split_point:]      # Testing Labels



# ------------------------------ Task 1 ------------------------------- #
#
#   Plot the distribution of the number of the instances in each class.
#
#

frequency = Counter()   # Counter Initialization for frequency

# Distribution for Positive/Negative Data
for doc in all_labels:
    frequency[doc] += 1

# Plot Creation
plt.bar(frequency.keys(), frequency.values())
plt.title("Distribution Plot")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.show()


# ------------------------------ Task 2 -------------------------------- #
#   Run 3 different ML models.


#   a)    
#   Naives Bayes

gnb = MultinomialNB()   # Use of Multinomial Naive Bayes Classifier

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',ngram_range = (1,1), stop_words='english')
train_docs_Vec = cv.fit_transform(train_docs)
gnb.fit(train_docs_Vec, train_labels)
eval_docs_Vec = cv.transform(eval_docs)
predictedGnb = gnb.predict(eval_docs_Vec)


# Task 3 :
# Part a)
# Writing the row number of the instance, followed by a comma, followed by the index of the predicted class of that instance to a file

row = len(train_docs) + 1

f = open("NaiveBayes-all_sentiment_shuffled.txt", "w")
for result in predictedGnb:
    index = 1
    if result == "neg":
        index = 0
    f.write(str(row) + ", " + str(index) + "\n")
    row += 1


# Task 3 
# Part c) & Part d)

precisionGnb = metrics.precision_score(eval_labels, predictedGnb, average = None)   # Calculation of Precision
recallGnb = metrics.recall_score(eval_labels, predictedGnb, average = None)         # Calculation of Recall
f1ScoreGnb = metrics.f1_score(eval_labels, predictedGnb, average=None)              # Calculation of F1-Measure
accuracyGnb = metrics.accuracy_score(eval_labels, predictedGnb)                     # Calculation of Accuracy


print('Naives Bayes Precision: ' + str(precisionGnb))
print('Naives Bayes Recall: ' + str(recallGnb))
print('Naives Bayes f1_score: ' + str(f1ScoreGnb))
print('Naives Bayes Accuracy: ' + str(accuracyGnb*100))
f.write('Naives Bayes Precision: ' + str(precisionGnb) +
        '\nNaives Bayes Recall: ' + str(recallGnb) +
        '\nNaives Bayes f1_score: ' + str(f1ScoreGnb) +
        '\nNaives Bayes Accuracy: ' + str(accuracyGnb*100))


cmGnb = numpy.array2string(metrics.confusion_matrix(eval_labels, predictedGnb))     # Generating Confusion Matrix
print(cmGnb)

f.write('\n Confusion Matrix: \n' + cmGnb )     # Writing to generated file.
f.close()                                       # Closing the file.



# b ) 
# Decision tree 

decisionTree = tree.DecisionTreeClassifier(criterion= 'entropy')    # Use of Decision Tree Classifier with Criterion set to 'entropy'
decisionTree.fit(train_docs_Vec, train_labels)
predictedDt = decisionTree.predict(eval_docs_Vec)


# Task 3
# Part a)

row = len(train_docs) + 1

f = open("DecisionTree-all_sentiment_shuffled.txt", "w")
for result in predictedDt:
    index = 1
    if result == "neg":
        index = 0
    f.write(str(row) + ", " + str(index) + "\n")
    row += 1


# Task 3
# Part c) & Part d)

precisionDt = metrics.precision_score(eval_labels, predictedDt, average=None)   # Calculation of Precision
recallDt = metrics.recall_score(eval_labels, predictedDt, average=None)         # Calculation of Recall
f1ScoreDt = metrics.f1_score(eval_labels, predictedDt, average=None)            # Calculation of F1-Measure
accuracyDt = metrics.accuracy_score(predictedDt, eval_labels)                   # Calculation of Accuracy

print('Decision Tree Precision: ' + str(precisionDt))
print('Decision Tree Recall: ' + str(recallDt))
print('Decision Tree f1_score: ' + str(f1ScoreDt))
print('Decision Tree Accuracy: ' + str(accuracyDt*100))

f.write('Decision Tree Precision: ' + str(precisionDt) +
        '\nDecision Tree Recall: ' + str(recallDt) +
        '\nDecision Tree f1_score: ' + str(f1ScoreDt) +
        '\nDecision Tree Accuracy: ' + str(accuracyDt*100))

cmDt = numpy.array2string(metrics.confusion_matrix(eval_labels, predictedDt))   # Generating Confusion Matrix
print(cmDt)


f.write('\n Confusion Matrix: \n' + cmDt )  # Writing to generated file.
f.close()                                   # Closing File.





# c ) 
# Better Decision Tree 

betterDecisionTree = tree.DecisionTreeClassifier(splitter= 'random')    # Use of Decision Tree Classifier with Splitter set to 'random'
betterDecisionTree.fit(train_docs_Vec, train_labels)
predictedBdt = betterDecisionTree.predict(eval_docs_Vec)


# Task
# Part 3a)
row2 = len(train_docs) + 1

f = open("BetterDecisionTree-all_sentiment_shuffled.txt", "w")
for result in predictedBdt:
    index2 = 1
    if result == "neg":
        index2 = 0
    f.write(str(row) + ", " + str(index2) + "\n")
    row2 += 1


# Task 3
# Part c) & Part d)
precisionBdt = metrics.precision_score(eval_labels, predictedBdt, average=None)
recallBdt = metrics.recall_score(eval_labels, predictedBdt, average=None)
f1ScoreBdt = metrics.f1_score(eval_labels, predictedBdt, average=None)
accuracyBdt = metrics.accuracy_score(eval_labels, predictedBdt)


print('Better Decision Tree Precision: ' + str(precisionBdt))
print('Better Decision Tree Recall: ' + str(recallBdt))
print('Better Decision Tree f1_score: ' + str(f1ScoreBdt))
print('Better Decision Tree Accuracy: ' + str(accuracyBdt*100))

f.write('Better Decision Tree Precision: ' + str(precisionBdt) +
        '\nBetter Decision Tree Recall: ' + str(recallBdt) +
        '\nBetter Decision Tree f1_score: ' + str(f1ScoreBdt) +
        '\nBetter Decision Tree Accuracy: ' + str(accuracyBdt*100))

cmBdt = numpy.array2string(metrics.confusion_matrix(eval_labels, predictedBdt))     # Generating Confusion Matrix
print(cmBdt)

f.write('\n Confusion Matrix: \n' + cmBdt)  # Writing to generated file.
f.close()                                   # Closing File.



# ------------------------------ Task 4 -------------------------------- #
#
#   Error Analysis
#
# Find the few misclassified documents and comment on why you think they were hard to classify. For
# instance, you may select a few short documents where the probabilities were particularly high in the wrong
# direction.


# index = 0
# listOfString = []
# realLabel = []
# predictedLabel = []
#
# while index < len(predictedGnb):
#     if predictedGnb[index] != eval_labels[index]:
#         listOfString.append(eval_docs[index])
#         realLabel.append(eval_labels[index])
#         predictedLabel.append(predictedGnb[index])
#         index += 1
#         if len(listOfString) > 10:
#             break
#     else:
#         index += 1
#
# i = 0
# f = open("misclassified.txt", 'w')
# while i < len(listOfString):
#     print(listOfString[i])
#     print("Real Value: " + realLabel[i])
#     print("Predicted Value: " + predictedLabel[i] + "\n")
#     f.write(listOfString[i] + "\n")
#     f.write(("Real Value: " + realLabel[i]) + "\n")
#     f.write(("Predicted Value: " + predictedLabel[i] + "\n\n"))
#     i += 1
#
# f.close()

