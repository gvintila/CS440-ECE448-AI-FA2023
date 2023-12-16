# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.2, bigram_laplace=0.2, bigram_lambda=0.2, pos_prior=0.2, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # Unigram Model

    num_words_uni = 0 # Total number of words for unigram model
    num_pos_words_uni = 0 # Total number of positive words
    num_neg_words_uni = 0 # Total number of negative words
    prob_map_uni = {} # Maps positive count and negative count for each word type

    for i in tqdm(range(len(train_set)), disable=silently):
        for word in train_set[i]:
            num_words_uni += 1 # Increase word count by 1
            if word not in prob_map_uni: # If word type has not been seen, add to map
                prob_map_uni[word] = [0, 0] # [Pos, Neg]
            if train_labels[i] == 1: # Check if review is positive or negative
                num_pos_words_uni += 1
                prob_map_uni[word][0] += 1
            else:
                num_neg_words_uni += 1
                prob_map_uni[word][1] += 1

    num_types_uni = len(prob_map_uni.keys()) # Counts number of types within training data

    for key in tqdm(prob_map_uni, disable=silently):
        prob_map_uni[key][0] = (prob_map_uni[key][0] + unigram_laplace) / (num_pos_words_uni + (unigram_laplace * (num_types_uni + 1))) # Calculate pos prob for each word
        prob_map_uni[key][1] = (prob_map_uni[key][1] + unigram_laplace) / (num_neg_words_uni + (unigram_laplace * (num_types_uni + 1))) # Calculate neg prob for each word
    prob_unk_uni = unigram_laplace / (num_words_uni + (unigram_laplace * (num_types_uni + 1))) # Probability of an unknown word

    # Bigram Model

    num_words_bi = 0 # Total number of words for bigram model
    num_pos_words_bi = 0 # Total number of positive words
    num_neg_words_bi = 0 # Total number of negative words
    prob_map_bi = {} # Maps positive count and negative count for each word type

    for i in tqdm(range(len(train_set)), disable=silently):
        for prev, curr in zip(train_set[i], train_set[i][1:]):
            num_words_bi += 1 # Increase word count by 1
            if (prev, curr) not in prob_map_bi: # If word type has not been seen, add to map
                prob_map_bi[(prev, curr)] = [0, 0] # [Pos, Neg]
            if train_labels[i] == 1: # Check if review is positive or negative
                num_pos_words_bi += 1
                prob_map_bi[(prev, curr)][0] += 1
            else:
                num_neg_words_bi += 1
                prob_map_bi[(prev, curr)][1] += 1

    num_types_bi = len(prob_map_bi.keys()) # Counts number of types within training data

    for key in tqdm(prob_map_bi, disable=silently):
        prob_map_bi[key][0] = (prob_map_bi[key][0] + bigram_laplace) / (num_pos_words_bi + (bigram_laplace * (num_types_bi + 1))) # Calculate pos prob for each word
        prob_map_bi[key][1] = (prob_map_bi[key][1] + bigram_laplace) / (num_neg_words_bi + (bigram_laplace * (num_types_bi + 1))) # Calculate neg prob for each word
    prob_unk_bi = bigram_laplace / (num_words_bi + (bigram_laplace * (num_types_bi + 1))) # Probability of an unknown word

    # Final Packing

    prob_pos_rev = pos_prior # Probability of a review being positive
    prob_neg_rev = 1 - prob_pos_rev # Probability of a review being negative

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        # Use logs instead of multiplication to prevent underflow
        prob_pos_sen_uni = prob_pos_sen_bi = math.log(prob_pos_rev)
        prob_neg_sen_uni = prob_neg_sen_bi = math.log(prob_neg_rev)
        # Unigram
        for word in doc:
            if word not in prob_map_uni: # If word is unknown
                prob_pos_sen_uni += math.log(prob_unk_uni)
                prob_neg_sen_uni += math.log(prob_unk_uni)
            else: # If word is known
                prob_pos_sen_uni += math.log(prob_map_uni[word][0])
                prob_neg_sen_uni += math.log(prob_map_uni[word][1])
        # Bigram
        for prev, curr in zip(doc, doc[1:]):
            if (prev, curr) not in prob_map_bi: # If word is unknown
                prob_pos_sen_bi += math.log(prob_unk_bi)
                prob_neg_sen_bi += math.log(prob_unk_bi)
            else: # If word is known
                prob_pos_sen_bi += math.log(prob_map_bi[(prev, curr)][0])
                prob_neg_sen_bi += math.log(prob_map_bi[(prev, curr)][1])
        # Mixture
        prob_pos_sen = ((1 - bigram_lambda) * prob_pos_sen_uni) + (bigram_lambda * prob_pos_sen_bi)
        prob_neg_sen = ((1 - bigram_lambda) * prob_neg_sen_uni) + (bigram_lambda * prob_neg_sen_bi)
        if prob_pos_sen > prob_neg_sen: # Check if probability of review being positive is greater than it being negative
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats



