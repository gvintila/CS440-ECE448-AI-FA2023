"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """

    start_count = 0

    tag_count = defaultdict(lambda: [0, 0, 0, 0]) # {init tag: (total_count, unique_word_count, total_tag_trans_count, unique_tag_trans_count)}
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}

    # Unknown probability value dictionary for each tag (for viterbi_2+)
    hapax_dict = defaultdict(lambda: 1) # {init tag: #}
    hapax_total = 0
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    # Iterate through sentences
    for sen in sentences:
        prev_tag = None

        # init_prob
        start_count += 1
        start = sen[0][1]
        if start not in init_prob:
            init_prob[start] = 0
        init_prob[start] += 1

        # Iterate through word tag pairs in each sentence
        for pair in sen:

            word = pair[0]
            tag = pair[1]

            # emit_prob
            # Add tag to word count
            tag_count[tag][0] += 1

            # Add one to unique word count
            if tag not in emit_prob[tag]:
                tag_count[tag][1] += 1

            emit_prob[tag][word] += 1

            # trans_prob
            if prev_tag:
                # Add one to total tag trans count
                tag_count[prev_tag][2] += 1
                # Add one to unique tag trans count
                if tag not in trans_prob[prev_tag]:
                    tag_count[prev_tag][3] += 1
                trans_prob[prev_tag][tag] += 1

            prev_tag = tag
                
    # init_prob
    for tag, count in init_prob.items():
        init_prob[tag] = count / start_count

    # emit_prob
    for tag in emit_prob:
        for word, count in emit_prob[tag].items():
            emit_prob[tag][word] = (count + emit_epsilon) / (tag_count[tag][0] + (emit_epsilon * (tag_count[tag][1] + 1)))
            # If hapax, add count to hapax_dict for current tag
            if count == 1:
                hapax_dict[tag] += 1
                hapax_total += 1
        emit_prob[tag]["UNK"] = (emit_epsilon / (tag_count[tag][0] + (emit_epsilon * (tag_count[tag][1] + 1))))

    # Increase hapax_total by number of tags
    hapax_total += len(emit_prob)
    # Set hapax_dict probability for each tag
    for tag in emit_prob:
        hapax_dict[tag] = hapax_dict[tag] / hapax_total

    # Scale new unknown probability using probability of hapax for that tag
    for tag in emit_prob:
        emit_prob[tag]["UNK"] = hapax_dict[tag] * emit_prob[tag]["UNK"]

    # trans_prob
    for prev_tag in trans_prob:
        for tag, count in trans_prob[prev_tag].items():
            trans_prob[prev_tag][tag] = (count + emit_epsilon) / (tag_count[prev_tag][2] + (emit_epsilon * (tag_count[prev_tag][3] + 1)))
        trans_prob[prev_tag]["UNK"] = (emit_epsilon / (tag_count[prev_tag][2] + (emit_epsilon * (tag_count[prev_tag][3] + 1))))

    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    if i != 0:

        # Iterate through all previous probabilities and tags
        for prev_tag, p_total in prev_prob.items():

            # Iterate through all possible tags to test for current tag
            for cur_tag in emit_prob:

                # Smoothing
                if emit_prob[cur_tag][word] != 0:
                    p_emit = log(emit_prob[cur_tag][word])
                else:
                    # Double check for log errors
                    if emit_prob[cur_tag]["UNK"] != 0:
                        p_emit = log(emit_prob[cur_tag]["UNK"])
                    else:
                        p_emit = 0

                if trans_prob[prev_tag][cur_tag] != 0:
                    p_trans = log(trans_prob[prev_tag][cur_tag])
                else:
                    if trans_prob[prev_tag]["UNK"] != 0:
                        p_trans = log(trans_prob[prev_tag]["UNK"])
                    else:
                        p_trans = 0
                
                # If log_prob has no current tag assigned OR log_prob has a current tag assigned AND it's prob is less than the newly calculated prob, replace it
                test_prob = p_emit + p_trans + p_total
                if (cur_tag not in log_prob) or (cur_tag in log_prob and test_prob > log_prob[cur_tag]):
                    log_prob[cur_tag] = test_prob
                    predict_tag_seq[cur_tag] = prev_predict_tag_seq[prev_tag] + [cur_tag]

    else:
        # START
        
        # Iterate through all tags and initial probabilities in prev_prob
        for tag, initial_p in prev_prob.items():

            # Smoothing
            if emit_prob[tag][word] != 0:
                p_emit = log(emit_prob[tag][word])
            else:
                # Double check for log errors
                if emit_prob[tag]["UNK"] != 0:
                    p_emit = log(emit_prob[tag]["UNK"])
                else:
                    p_emit = 0

            log_prob[tag] = p_emit + initial_p
            predict_tag_seq[tag] = [tag]
        
    return log_prob, predict_tag_seq


def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        high_seq = (None, 0)
        for tag, prob in log_prob.items():
            if prob > high_seq[1] or high_seq[0] is None:
                high_seq = (predict_tag_seq[tag], prob)

        sen_pairs = []
        for i, tag in enumerate(high_seq[0]):
            sen_pairs.append((sentence[i], tag))
        
        predicts.append(sen_pairs)
        
    return predicts