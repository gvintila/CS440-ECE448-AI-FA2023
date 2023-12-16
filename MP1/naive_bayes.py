# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)
    
    num_words = 0 # Total number of words
    num_pos_words = 0 # Total number of positive words
    num_neg_words = 0 # Total number of negative words
    prob_map = {} # Maps positive count and negative count for each word type

    for i in tqdm(range(len(train_set)), disable=silently):
        for word in train_set[i]:
            num_words += 1 # Increase word count by 1
            if word not in prob_map: # If word type has not been seen, add to map
                prob_map[word] = [0, 0] # [Pos, Neg]
            if train_labels[i] == 1: # Check if review is positive or negative
                num_pos_words += 1
                prob_map[word][0] += 1
            else:
                num_neg_words += 1
                prob_map[word][1] += 1

    prob_pos_rev = pos_prior # Probability of a review being positive
    prob_neg_rev = 1 - prob_pos_rev # Probability of a review being negative
    num_types = len(prob_map.keys()) # Counts number of types within training data

    for key in tqdm(prob_map, disable=silently):
        prob_map[key][0] = (prob_map[key][0] + laplace) / (num_pos_words + (laplace * (num_types + 1))) # Calculate pos prob for each word
        prob_map[key][1] = (prob_map[key][1] + laplace) / (num_neg_words + (laplace * (num_types + 1))) # Calculate neg prob for each word
    prob_unk = laplace / (num_words + (laplace * (num_types + 1))) # Probability of an unknown word

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        # Use logs instead of multiplication to prevent underflow
        prob_pos_sen = math.log(prob_pos_rev)
        prob_neg_sen = math.log(prob_neg_rev)
        for word in doc:
            if word not in prob_map: # If word is unknown
                prob_pos_sen += math.log(prob_unk)
                prob_neg_sen += math.log(prob_unk)
            else: # If word is known
                prob_pos_sen += math.log(prob_map[word][0])
                prob_neg_sen += math.log(prob_map[word][1])
        if prob_pos_sen > prob_neg_sen: # Check if probability of review being positive is greater than it being negative
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats
