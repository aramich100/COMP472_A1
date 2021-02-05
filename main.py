# COMP 472
# WINTER 2021

# Michael Arabian - 40095854
# Thomas Le       - 40096120
# Andre Saad      - xxxxxxxx


from sklearn.datasets import load_files
import pandas as pd



# ---------------- Task 0 --------------------- #

#  Counting Total number of lines in the file
# fname = "all_sentiment_shuffled.txt"
# count = 0
# with open(fname, encoding="utf8") as f:
#     for line in f:
#         count += 1
#
# print("Total number of lines is:", count)
# split_point = int(0.80*count)
#
# # Loops through each line of the file, splitting the line with a " "
# f = open("all_sentiment_shuffled.txt", encoding="utf8")
# review = ""
# for line in f:
#   fields = line.split(" ")
#   topic = fields[0]
#   sentiment = fields[1]
#   identifier = fields[2]
#   review = fields[3] + " " + fields[4] + fields[5] # Something needs to be added here so that after the 3rd space it reads the remaining of the line as one field.
#
#   print(topic + "\t" + sentiment + "\t" + identifier + "\t" + review)
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

print(all_docs[1])



# ---------------- Task 1 --------------------- #

